"""python -m solution adapt|eval; also the Docker entry point."""
from pathlib import Path
import argparse
import asyncio
import sys

from . import adapt, eval
from .common import SolutionError, canonical
from .config import Config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    adaptation = sub.add_parser("adapt", help="Learn prompts from labeled JSONL")
    adaptation.add_argument("--train", type=Path, default=Path("/input/train.jsonl"))
    adaptation.add_argument("--artifacts", type=Path, default=Path("/artifacts"))
    adaptation.add_argument("--dry-run", action="store_true", help="Validate inputs and estimate costs without requests/writes")
    inference = sub.add_parser("eval", help="Predict labels from unlabeled JSONL and saved artifacts")
    inference.add_argument("--test", type=Path, default=Path("/input/test.jsonl"))
    inference.add_argument("--artifacts", type=Path, default=Path("/artifacts"))
    inference.add_argument("--output", type=Path, default=Path("/output"))
    inference.add_argument("--dry-run", action="store_true", help="Validate artifacts and estimate costs without requests/writes")
    args = parser.parse_args(argv)
    try:
        config = Config.from_env()
        if args.command == "adapt":
            asyncio.run(adapt.run(args.train.resolve(), args.artifacts.resolve(), config, dry_run=args.dry_run))
        else:
            asyncio.run(eval.run(args.test.resolve(), args.artifacts.resolve(), args.output.resolve(), config, dry_run=args.dry_run))
    except KeyboardInterrupt:
        print(canonical({"status": "interrupted", "note": "Successful responses are saved; rerun the same command to resume."}), file=sys.stderr)
        return 130
    except Exception as exc:
        # Full provider exceptions may contain credentials. Do not print their strings.
        message = str(exc) if isinstance(exc, SolutionError) or type(exc) in {FileNotFoundError, PermissionError} else "Run stopped; inspect progress and request journals, then resume the same command."
        print(canonical({"status": "failed", "error_type": type(exc).__name__, "message": message}), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
