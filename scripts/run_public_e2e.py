"""Run the built image on public data, in separate adapt/eval containers, then score.

Usage: .venv/bin/python scripts/run_public_e2e.py --image metalearning-judge:latest
Successful runs can be repeated with --run-dir to verify restart behavior.
API keys are inherited via named environment variables, never saved to disk.
"""
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from solution.config import Config


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def execute(command, logfile, *, env=None):
    print(json.dumps({"log": str(logfile), "command": command}, ensure_ascii=False), flush=True)
    with logfile.open("a", encoding="utf-8") as stream:
        stream.write("\nRUN " + json.dumps(command) + "\n")
        stream.flush()
        subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default="metalearning-judge:latest")
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args()
    config = Config.from_env()
    if not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
        parser.error("GEMINI_API_KEY or GOOGLE_API_KEY is required")
    if config.budget_usd > 10:
        parser.error("This public verification runner is limited to RUN_BUDGET_USD <= 10")
    folder = (args.run_dir or ROOT / "out/docker-e2e" / datetime.now(ZoneInfo("Europe/Madrid")).strftime("%Y%m%d-%H%M%S")).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    image_id = subprocess.check_output(["docker", "image", "inspect", "--format", "{{.Id}}", args.image], text=True).strip()
    metadata = {"image_id": image_id, "config": asdict(config), "language": "original", "dataset": "public",
                "train_csv_sha256": hashlib.sha256((ROOT / "data/public_train.csv").read_bytes()).hexdigest(),
                "test_csv_sha256": hashlib.sha256((ROOT / "data/public_test.csv").read_bytes()).hexdigest()}
    manifest_path = folder / "e2e_manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != metadata:
        parser.error("Image, data or configuration changed; use a fresh --run-dir")
    write_json(manifest_path, metadata)
    inputs = folder / "input"
    if not inputs.exists():
        execute([sys.executable, "src/prepare_inputs.py", "--train", "data/public_train.csv", "--test", "data/public_test.csv",
                 "--out-dir", str(inputs)], folder / "prepare.log")
    artifacts, output = folder / "artifacts", folder / "output"
    artifacts.mkdir(exist_ok=True)
    output.mkdir(exist_ok=True)
    values = {"MODEL": config.inference.model, "MAX_OUTPUT_TOKENS": str(config.inference.max_output_tokens),
              "REQUEST_TIMEOUT_SECONDS": str(config.inference.timeout_seconds), "ANALYSIS_MODEL": config.analysis.model,
              "ANALYSIS_MAX_OUTPUT_TOKENS": str(config.analysis.max_output_tokens), "SYNTHESIS_MODEL": config.synthesis.model,
              "SYNTHESIS_MAX_OUTPUT_TOKENS": str(config.synthesis.max_output_tokens),
              "SYNTHESIS_TIMEOUT_SECONDS": str(config.synthesis.timeout_seconds), "EXPLANATION_MODEL": config.explanation.model,
              "EXPLANATION_MAX_OUTPUT_TOKENS": str(config.explanation.max_output_tokens), "RUN_BUDGET_USD": str(config.budget_usd),
              "MAX_CONCURRENCY": str(config.concurrency), "REQUEST_INTERVAL_SECONDS": str(config.request_interval_seconds)}
    environment = {**os.environ, **values}
    keys = [key for key in ("GEMINI_API_KEY", "GOOGLE_API_KEY") if os.getenv(key)]
    # -e KEY forwards the value from this process without putting it in command arguments/logs.
    docker = ["docker", "run", "--rm"]
    for key in [*keys, *values]:
        docker.extend(["--env", key])
    adapt_command = [*docker, "--mount", f"type=bind,src={inputs},dst=/input,readonly",
                     "--mount", f"type=bind,src={artifacts},dst=/artifacts", image_id, "adapt"]
    execute([*adapt_command, "--dry-run"], folder / "adapt_dry_run.log", env=environment)
    execute(adapt_command, folder / "adapt.log", env=environment)
    before = {str(path.relative_to(artifacts)): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in artifacts.rglob("*") if path.is_file()}
    # eval sees only test.jsonl, never train.jsonl or a gold-bearing source CSV.
    eval_input = folder / "eval_input"
    eval_input.mkdir(exist_ok=True)
    (eval_input / "test.jsonl").write_bytes((inputs / "test.jsonl").read_bytes())
    eval_command = [*docker, "--mount", f"type=bind,src={eval_input},dst=/input,readonly",
                    "--mount", f"type=bind,src={artifacts},dst=/artifacts,readonly",
                    "--mount", f"type=bind,src={output},dst=/output", image_id, "eval"]
    execute([*eval_command, "--dry-run"], folder / "eval_dry_run.log", env=environment)
    execute(eval_command, folder / "eval.log", env=environment)
    after = {str(path.relative_to(artifacts)): hashlib.sha256(path.read_bytes()).hexdigest()
             for path in artifacts.rglob("*") if path.is_file()}
    if before != after:
        raise RuntimeError("eval modified adaptation artifacts")
    execute([sys.executable, "src/evaluate.py", "--test", "data/public_test.csv", "--predictions", str(output / "predictions.csv"),
             "--out", str(folder / "score.json")], folder / "score.log")
    with (output / "predictions.csv").open(newline="", encoding="utf-8") as stream:
        predictions = list(csv.DictReader(stream))
    with (ROOT / "data/public_test.csv").open(newline="", encoding="utf-8") as stream:
        gold = list(csv.DictReader(stream))
    by_id = {row["item_id"]: row["label"] for row in predictions}
    errors = [{"item_id": row["item_id"], "topic": row["topic"], "gold": row["label"], "prediction": by_id[row["item_id"]]}
              for row in gold if row["label"] != by_id[row["item_id"]]]
    with (folder / "errors.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["item_id", "topic", "gold", "prediction"])
        writer.writeheader()
        writer.writerows(errors)
    report = {"status": "complete", "image_id": image_id, "test_examples": len(predictions), "artifacts_unchanged": True,
              "score": json.loads((folder / "score.json").read_text()),
              "adaptation": json.loads((artifacts / "summary.json").read_text()),
              "inference": json.loads((output / "summary.json").read_text())}
    write_json(folder / "e2e_report.json", report)
    print(json.dumps({"run_directory": str(folder), "macro_topic_accuracy": report["score"]["score"],
                      "accuracy": report["score"]["accuracy"], "pipeline_usd_estimated": report["inference"]["pipeline_accounted_usd"]}), flush=True)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        print(json.dumps({"status": "failed", "returncode": error.returncode, "note": "Inspect the stage log; rerun with the same --run-dir to resume."}), file=sys.stderr)
        raise SystemExit(error.returncode)
