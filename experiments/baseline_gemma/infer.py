"""Run the same baseline pipeline with Gemma 4 and a separate experiment ledger."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from experiments.baseline.infer import main


if __name__ == "__main__":
    raise SystemExit(main(experiment_dir=HERE, experiment_name="baseline_gemma",
                         expected_model="gemini/gemma-4-26b-a4b-it", entrypoint=Path(__file__).resolve()))
