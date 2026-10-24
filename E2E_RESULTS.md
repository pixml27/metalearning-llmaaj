# Docker E2E verification — 24 October 2026

The Docker build and complete public `adapt → eval → official scorer` pipeline passed on `linux/arm64` using Colima. Adaptation started with an empty artifacts directory and processed all 150 public training examples. A separate evaluation container received only the 80 unlabeled test examples and read-only adaptation artifacts. The official scorer accepted exactly 80 predictions.

Image: `metalearning-judge:e2e-20261024`.
Image ID: `sha256:aac071dc8885d17394fe7fa454a49f7974b0bd60a4f70c35fbe11f4436f55132`.

## Public metrics

| Metric | Result |
|---|---:|
| macro_topic_accuracy | **0.8850131752** |
| Accuracy | **0.875000 — 70/80** |
| Cohen's kappa | 0.7435897436 |
| Macro topic kappa | 0.7693450674 |
| Confusion matrix, gold × prediction, good/bad | `[[42, 8], [2, 28]]` |

| Topic | Correct | Accuracy |
|---|---:|---:|
| credit | 14/15 | 0.933333 |
| financial_position | 21/22 | 0.954545 |
| savings | 20/20 | 1.000000 |
| spending | 15/23 | 0.652174 |

The earlier exp1 public run scored **0.922925** with 73/80 correct. This fresh adaptation scored **0.885013** with 70/80 correct, a decrease of **3.79 percentage points** in macro topic accuracy. The baseline, A/B analysis and synthesis templates match exp1 byte for byte, but the newly generated learned prompt artifact differs. Baseline predictions and analyses were also regenerated. This run does not isolate the cause of the quality difference, and the earlier quality was not reproduced by fresh adaptation. No prompt changes or additional paid comparisons were made in response to these test results. Jury interpretability points have not been assigned.

## API calls and estimated cost

| Stage | Responses | Actual response model | Estimated USD |
|---|---:|---|---:|
| Baseline on train | 150 | gemini-3.8-flash | 0.3499065 |
| A/B analyses | 150 | gemini-3.8-flash | 0.5691960 |
| Prompt synthesis | 1 | gemini-4-argon | 0.0931920 |
| Jury explanation | 1 | gemini-4-argon | 0.1553700 |
| Test inference | 80 | gemini-3.8-flash | 0.1997198 |
| **Total** | **382** | | **1.3673843** |

All 382 requests completed without API errors or retries. Costs are LiteLLM estimates from provider usage, not invoices. The total includes fresh baseline generation and the explanation stage. Requested aliases were `gemini/gemini-flash-latest` and `gemini/gemini-4-argon`.

## Verified behavior

- All 68 locked dependencies installed in the Docker build. The image contains only runtime sources, prompt templates and the lock; no `.env`, data or experiment artifacts are included.
- All 17 solution tests passed inside the built image with networking disabled.
- The synthesis input contains all 150 labeled train analyses. No test examples or test labels are used during adaptation.
- `eval` ran in a new container with only `test.jsonl` mounted as input. The artifacts mount was read-only, and every adaptation file remained byte-identical afterward.
- The generated jury explanation contains 12 criteria, 5 boundaries and references to 73 real training examples covering all four topics and both labels. All prompt quotes and referenced IDs passed validation.
- A second complete invocation with the same image, configuration and run directory made **zero additional API attempts** and produced byte-identical predictions.
- Source hashes from the tested image match the current runtime source files.

## Reproduction and saved evidence

Build and run from the repository root with the configured API key:

```sh
docker build -t metalearning-judge:e2e-20261024 .
.venv/bin/python scripts/run_public_e2e.py --image metalearning-judge:e2e-20261024 --run-dir out/docker-e2e/NEW_RUN
```

Use a new run directory for a new adaptation. To resume the verified local run, use `out/docker-e2e/20261024-public-original` with the exact same image/configuration. A rebuild can produce a new image ID and therefore requires a fresh run directory. Full model determinism is not guaranteed.

- [Compact validation record with metrics, costs, source hashes and restart checks](validation/docker-public-20261024.json) — retained in the source tree.
- [Full local E2E report](out/docker-e2e/20261024-public-original/e2e_report.json).
- [Official scorer output](out/docker-e2e/20261024-public-original/score.json).
- [Predictions](out/docker-e2e/20261024-public-original/output/predictions.csv).
- [Ten prediction errors](out/docker-e2e/20261024-public-original/errors.csv).
- [Generated jury explanation](out/docker-e2e/20261024-public-original/artifacts/explanation.md).
- [Learned prompt artifact](out/docker-e2e/20261024-public-original/artifacts/artifact.json).
- [Restart verification](out/docker-e2e/20261024-public-original/restart_verification.json).

Full local artifacts and logs under `out/` are ignored by Git. The runtime uses fresh adaptation when organizers supply their private train; these public learned artifacts are excluded from the Docker image.
