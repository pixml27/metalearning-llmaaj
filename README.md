# A Judge That Learns from an Expert

Infer an expert's evaluation rules from labeled conversations, then assign one of two labels to new conversations: `good` or `bad`. Evaluate the entire conversation. Any violation of an applicable rule makes the conversation `bad`.

**Dataset sizes.** Public: 150 training examples and 80 test examples. Final evaluation: 190 private training examples and 296 private test examples.

The public and private datasets cover the same four topics. The private dataset introduces additional expert requirements whose violations are not represented in the public data, so your solution must adapt again using the training set provided for the final evaluation. The hidden rules are not shared with participants.

## Files

```text
data/
  public_train.csv           # 150 labeled examples for adaptation
  public_test.csv            # 80 labeled examples for local evaluation
src/
  common.py                  # data loading and format validation
  prepare_inputs.py          # CSV -> JSONL; removes labels from prediction inputs
  evaluate.py                # prediction validation and metric calculation
  test_evaluation.py         # evaluation contract tests
README.md
```

The public training set contains 90 `good` and 60 `bad` examples; the public test set contains 50 `good` and 30 `bad` examples. Both sets cover all four topics.

Public test labels are provided for local evaluation. During the final evaluation, your solution receives a labeled private training set and an unlabeled private test set. Adaptation starts with a clean artifacts directory for each public or private run.

## Data format

CSV files use UTF-8 encoding and include a header row. Use a CSV parser: cells may contain quotes, commas, and JSON.

| Column | Contents |
|---|---|
| `item_id` | An opaque, unique example identifier. |
| `topic` | `financial_position`, `savings`, `spending`, or `credit`. |
| `context` | A JSON object containing the source data needed to evaluate the assistant's response. |
| `conversation` | A JSON array of messages with `role` and `content` fields. |
| `label` | The ground-truth label: `good` or `bad`. Omitted from the final prediction input. |
| `input_es` | A Spanish version of the complete input: JSON containing `context` and `conversation`. |
| `input_en` | An English version of the complete input in the same format. |

Translations are alternative representations of the same example, not additional rows or separate tasks. They share the same `item_id`, `topic`, and label. By default, inputs use the original `context` and `conversation`; you can select a translation when preparing inputs. Record the selected language with your results to make experiments comparable.

## Quick start

The scripts in `src/` require Python 3.10 or later. No external libraries or API keys are needed to run these scripts. Run the commands below from the repository root.

Prepare inputs for the two stages of your solution:

```sh
python3 src/prepare_inputs.py --train data/public_train.csv --test data/public_test.csv --out-dir out/public_input
```

The new directory will contain labeled `train.jsonl` and unlabeled `test.jsonl`. Both files contain only the required fields; alternative translations are not duplicated in the inputs. Add `--language es` or `--language en` to select a translation.

During adaptation, use only the training set. During prediction, use the unlabeled test set and the artifacts produced during adaptation. The labeled public test set is intended for scoring and development; the final private test set will be hidden.

Your solution must output **exactly one label for every test ID**. Probabilities and confidence scores are not required. For local evaluation, write predictions as CSV:

```csv
item_id,label
FIRST_EXAMPLE_ID,good
SECOND_EXAMPLE_ID,bad
```

The local scorer also accepts JSONL with exactly the same two fields:

```json
{"item_id":"EXAMPLE_ID","label":"good"}
```

Column names and class labels are case-sensitive. Row order does not matter. Missing, duplicate, or unknown IDs, extra fields, and labels other than `good` or `bad` are rejected with a nonzero exit code.

Measure prediction quality for a complete prediction file:

```sh
python3 src/evaluate.py --test data/public_test.csv --predictions predictions.csv --out out/score.json
```

## Submitting your solution

**Fork this repository and push your complete solution to your team's fork.** Click **Fork** on GitHub, then clone your fork. Replace `YOUR_ACCOUNT` with the account or organization that owns your fork:

```sh
git clone https://github.com/YOUR_ACCOUNT/metalearning-llmaaj.git
cd metalearning-llmaaj
```

Commit the source code, a Dockerfile, dependency specifications, and instructions for building and running both `adapt` and `eval`. Include any files needed to reproduce the solution, and document configuration through environment variables. Keep credentials out of the repository.

After committing your solution, push it to your fork and record the exact commit:

```sh
git push origin HEAD
git rev-parse HEAD
```

Send the organizers **the URL of your fork and the commit SHA**. The organizers will evaluate that submitted version. Make sure they can access your fork and fetch the submitted commit.

Your Docker image must support two commands, `adapt` and `eval`, with this interface:

| Stage | Inputs | Outputs |
|---|---|---|
| `adapt` | Labeled training examples at `/input/train.jsonl`. | Adaptation artifacts in `/artifacts`, including the automatically generated explanation for the jury. |
| `eval` | Unlabeled test examples at `/input/test.jsonl` and the saved `/artifacts` directory. | `/output/predictions.csv` with exactly the columns `item_id,label`, as shown above. |

`eval` runs in a separate container, so persist everything it needs from `adapt` in `/artifacts`.

## Final evaluation

**You submit a runnable solution; the organizers run it themselves and calculate the official result.** Local scores help you develop your pipeline.

1. The organizers clone your fork, check out the submitted commit, and build your Docker image using the supplied instructions.
2. They prepare a new, labeled private training set and an unlabeled private test set. Private test labels remain with the organizers and are never passed to your solution.
3. They run `adapt` on the private training set with a fresh, empty artifacts directory. This stage learns the expert's criteria and automatically produces both prediction artifacts and an explanation for the jury.
4. They run `eval` in a separate container using the private test inputs and those saved artifacts. Your solution writes one `good` or `bad` prediction for every test ID.
5. The organizers validate the prediction file and calculate `acc = macro_topic_accuracy` against the hidden labels. The score is printed and saved to a file.
6. The jury evaluates the explanation generated during that same adaptation run using the five criteria below. The organizers combine `acc` and the jury points using **Final score = 10 × acc + 0.4 × I**.

## Automated metric

The automated metric is **`macro_topic_accuracy`** (`acc`): calculate accuracy separately for each of the four topics, then average those four values. Every topic has equal weight, regardless of its number of examples.

```text
acc = (accuracy_credit + accuracy_financial_position + accuracy_savings + accuracy_spending) / 4
score = macro_topic_accuracy = acc
```

Scores range from 0 to 1; higher is better. For example, topic accuracies of 0.8, 0.7, 0.9, and 0.6 give `acc = 0.75`. Model/API cost does not affect the score, and no cost reports are required for evaluation.

The bundled `src/evaluate.py` calculates this metric directly: its `macro_topic_accuracy` and `score` fields contain the same value. The organizer evaluator uses the same metric. Jury points are added separately as described below.

The scorer also reports overall accuracy, Cohen's kappa, and mean kappa across topics as diagnostics. In confusion matrices, rows represent ground-truth labels and columns represent predictions; the class order is `good`, `bad`. If kappa has a zero denominator, it is reported as 0.

Brier score and log loss are not calculated because solutions output labels only. The scorer evaluates one saved set of predictions without rerunning the model. Identical predictions against the same test set produce identical scores. This does not imply that the solution itself is deterministic.

A difference of a few thousandths in the score is not, by itself, evidence of a statistically significant advantage.

## Project defense and final score

**Final score = 10 × acc + 0.4 × I**, where **acc = `macro_topic_accuracy`** and **I ∈ [0, 5]** is the sum of points in the table below.

The automated component contributes up to 10 points and the jury component up to 2 points. Final scores range from 0 to 12. For example, `acc = 0.80` and `I = 4` give a final score of `10 × 0.80 + 0.4 × 4 = 9.6`.

The jury evaluates an artifact automatically generated by `adapt` on a new training set. Include this explanation alongside the artifacts used for prediction. The examples below are hypothetical illustrations for a text-to-`good`/`bad` task.

The interpretability score **`I` ranges from 0 to 5** and is the sum of five binary criteria:

| Criterion | +1 point | 0 points |
|---|---|---|
| **Clear criteria** | "Unsupported claims indicate a bad response; asking for clarification when essential data are missing indicates a good response." A decision tree with understandable conditions also qualifies. | "The model assesses text quality," or a list of unexplained technical features. |
| **Clear priorities** | "A factual error outweighs completeness." For a decision tree, the order of conditions and thresholds is visible. | Criteria are listed, but it is unclear how conflicting criteria produce the final `good`/`bad` label. |
| **Links to training examples** | Specific texts, their expert labels, and an explanation of which criterion each example illustrates are provided. | Only invented examples are given, or a generic claim such as "the training data support this." |
| **Clear boundaries** | "Labels for short responses are inconsistent: similar texts received different labels," supported by examples from the training set. | "The model sometimes makes mistakes," without specific conditions or examples. |
| **Consistency with the implementation** | The described tree is actually used for prediction; the displayed rules appear in the final prompt; a table of weights matches the model. | A polished explanation is attached separately, but its connection to the working solution is not demonstrated. |

**Rules for the jury:** each row receives exactly 0 or 1 point. The format is unrestricted: text, rules, a tree, a table, or a visualization. Length and literary quality earn no additional points; a classical model can receive all five points.

"Common sense" here means checking whether the explanation is understandable. An unusual expert preference is acceptable when it is supported by the labels.

## Check the evaluation scripts

```sh
python3 -m unittest discover -s src -p 'test_*.py'
```

## Experiments

Compare completed public runs in [experiments/RESULTS.md](experiments/RESULTS.md).

- [baseline](experiments/baseline/README.md): zero-shot judge using Gemini Flash.
- [baseline_gemma](experiments/baseline_gemma/README.md): the identical prompt and parameters using **Gemma 4 (`gemma-4-26b-a4b-it`) through the Gemini API**, integrated via LiteLLM and Pydantic-AI. The API call is in [experiments/llm.py](experiments/llm.py); LiteLLM selects the Gemini API with `gemini/gemma-4-26b-a4b-it` and `GEMINI_API_KEY`.

The comparison table is regenerated after every complete baseline run. See [experiments/README.md](experiments/README.md) for setup and experiment conventions.

## Runnable solution: exp3 prompt pipeline

The standalone implementation is in `solution/`. It retains the baseline and A/B analysis stages from exp1 and uses the exp3 seed, fixed definitions, synthesis with training evidence and a second request that removes evidence from the prediction prompts. It runs from a fresh training set without saved experiment results. An additional explanation stage documents the learned prompts for the jury without changing them. The earlier exp1-based Docker run on 24 October 2026 achieved public `macro_topic_accuracy = 0.885013` and accuracy 87.5% (70/80); those numbers do not evaluate this updated pipeline. See [E2E_RESULTS.md](E2E_RESULTS.md) for that historical run.

`adapt` performs five stages:

1. The configured judge (`MODEL`, Flash by default) predicts every training example without seeing its expert label.
2. `ANALYSIS_MODEL` analyzes every training example, including correct predictions, using its expert label and baseline prediction. A explains the agreement or error; B identifies a conditional lesson for its topic.
3. `SYNTHESIS_MODEL` (Argon by default) receives the exp3 seed, fixed definitions D1–D14 and the complete A/B table. It produces criteria, priorities, boundaries and advice for all four topics, with inline links to training evidence.
4. The same synthesis model removes only evidence spans in a separate request. Offline reports check links, remaining training IDs and preservation of rules; remaining evidence stops adaptation. Both versions and the definitions are saved and hashed.
5. `EXPLANATION_MODEL` documents criteria, priorities, evidence and boundaries. Exact prompt quotes and training IDs are validated. The renderer appends the original referenced training conversations, context and labels, plus all deployed prompt text.

`eval` selects the saved advice using each example's supplied `topic`, appends it after the shared prompt and fixed definitions, and calls the judge with only `context` and the complete `conversation`. Native structured output is validated as exactly `{"label":"good"}` or `{"label":"bad"}`. It does not read the training file or call a scorer. It writes a complete `predictions.csv` only after every prediction succeeds; failures never become default labels. The command interface is unchanged. Artifact version 2 requires a fresh adaptation; old exp1 artifacts are rejected.

### Configuration and dependencies

Use Python 3.12. From the repository root:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock.txt
cp .env.example .env
# Set GEMINI_API_KEY in .env before making live requests.
```

If `.env` already exists, add the settings you need without replacing it. `requirements.txt` lists direct dependencies; `requirements.lock.txt` pins the full dependency set used by Docker.

Configuration precedence is the process environment, `solution/.env`, repository `.env`, then workspace `../.env`. No experiment `.env` files are read. The key is passed to LiteLLM through the environment and is never written to manifests. `.env.example` documents all settings. `MODEL` is the judge used by both commands. `META_MODEL` is the stronger model that learns the prompts during `adapt` — synthesis, evidence stripping and the jury explanation — and is never used at prediction time; `SYNTHESIS_MODEL` and `EXPLANATION_MODEL` override it per stage, and `ANALYSIS_MODEL` defaults to the judge. Keep the same judge model and inference parameters for `adapt` and `eval`; a mismatch is rejected. Language selection happens in `src/prepare_inputs.py` before either command.

`RUN_BUDGET_USD` defaults to 10 and covers adaptation plus one evaluation, including failed attempts and resumed requests. Prices come from the pinned LiteLLM snapshot; they are estimates, not invoices. Each stage estimates pending requests before execution. Unknown prices stop execution. Requests have explicit token/time limits. `MAX_CONCURRENCY` and `REQUEST_INTERVAL_SECONDS` control simultaneous requests and spacing between request starts.

A single transient provider failure does not end a run. Rate limits, timeouts and 5xx responses are retried up to `MAX_ATTEMPTS` times per request (4 including the first), with exponential backoff from `RETRY_BACKOFF_SECONDS` doubling per retry, half jitter and a 60-second cap; `REQUEST_INTERVAL_SECONDS` still spaces every attempt. Rejected requests, invalid or truncated structured output, cost and budget violations are deliberate stops and are never retried. Every attempt is journalled with its error type, and each retry prints the stage, ID, attempt and delay.

### Run on public data locally

The commands below prepare the current public files. Use an empty input directory and a fresh artifacts/output directory for a new configuration.

```sh
.venv/bin/python src/prepare_inputs.py --train data/public_train.csv --test data/public_test.csv --out-dir out/public_input
.venv/bin/python -m solution adapt --train out/public_input/train.jsonl --artifacts out/public_artifacts --dry-run
```

`--dry-run` validates input and configuration and prints estimates without API requests or writing output files. The adaptation dry run estimates the baseline stage; later stages depend on generated answers and are priced immediately before execution. It does not certify the total pipeline cost. The following commands make live requests:

```sh
.venv/bin/python -m solution adapt --train out/public_input/train.jsonl --artifacts out/public_artifacts
.venv/bin/python -m solution eval --test out/public_input/test.jsonl --artifacts out/public_artifacts --output out/public_output --dry-run
.venv/bin/python -m solution eval --test out/public_input/test.jsonl --artifacts out/public_artifacts --output out/public_output
.venv/bin/python src/evaluate.py --test data/public_test.csv --predictions out/public_output/predictions.csv --out out/public_output/score.json
```

On the private data, use the same commands with the new labeled training input, unlabeled test input and fresh output directories. No public training results, IDs, row counts or pre-generated prompt artifacts are required by the runtime.

### Docker interface for organizers

Build one image and run its two commands in separate containers. The mounted input directory contains the prepared JSONL files. The artifacts directory starts empty for each new adaptation. The same `.env` supplies credentials and configuration for both commands.

```sh
docker build -t metalearning-judge .
mkdir -p out/docker_artifacts out/docker_output
docker run --rm --env-file .env \
  -v "$PWD/out/public_input:/input:ro" \
  -v "$PWD/out/docker_artifacts:/artifacts" \
  metalearning-judge adapt
docker run --rm --env-file .env \
  -v "$PWD/out/public_input:/input:ro" \
  -v "$PWD/out/docker_artifacts:/artifacts:ro" \
  -v "$PWD/out/docker_output:/output" \
  metalearning-judge eval
```

The entry point is `python -m solution`. Defaults are `/input/train.jsonl` for `adapt`, `/input/test.jsonl` for `eval`, `/artifacts` for saved adaptation and `/output` for predictions. Both commands also accept the local path arguments shown above. The image copies only runtime sources, prompt templates and the dependency lock. Credentials, data and experiment results are excluded from the build context.

### Artifacts and restarts

`adapt` writes:

```text
/artifacts/
  artifact.json                 # deployed prompt: shared base_prompt + four topic_advice texts
  routed_prompts.json           # exact compiled inference prompt per topic
  definitions.md                # fixed definitions appended to every judge prompt
  linked_artifact.json          # synthesized prompt before its evidence spans were removed
  link_report.json              # evidence links found in the synthesized prompt
  strip_report.json             # evidence removal, remaining training IDs and rule preservation
  explanation.md                # jury explanation with real train evidence
  explanation.json              # validated explanation structure
  baseline_predictions.csv
  analyses.jsonl
  synthesis_input.json
  strip_input.json
  explanation_input.json
  *_prompt.md                   # exact adaptation prompt templates
  *_calls.jsonl                 # model names, raw responses, tokens and estimated costs
  budget.jsonl                  # reservations and settlement for every API attempt
  manifest.json                 # input/config/source hashes and dependency versions
  progress.json
  summary.json                  # complete only when all stages and validation pass
  .run.lock                     # prevents conflicting writers
```

`eval` consumes `/artifacts` read-only and writes `predictions.csv`, request/budget journals, a manifest, progress and summary under `/output`. It requires the saved prediction artifacts (`artifact.json`, `routed_prompts.json`, `definitions.md`, `linked_artifact.json`), the explanation pair, the adaptation manifest/summary, budget and lock file; the original train input and analysis caches are not needed for prediction. The saved definitions must match the runtime copy, so `eval` runs from the same image as `adapt`.

After a failure, rerun the same command with the same directories, inputs and configuration. Matching successful structured responses are recovered from journals; unfinished or invalid requests can be retried. A timeout or server-side failure can still have been billed, so its budget reservation is retained; a request the provider rejected before generating tokens (rate limit, 4xx) releases its reservation, so retries keep their headroom. Changes to input bytes, model parameters, runtime sources or dependency versions require a fresh output directory. Completed adaptation is checked and reused without new requests. Artifact hashes and compiled prompt consistency are verified before prediction.

### Offline verification

```sh
.venv/bin/python -m unittest discover -s tests -v
python3 -m unittest discover -s src -p 'test_*.py'
```

The solution tests run the real adaptation/inference flow with a fake API transport. They cover separate execution with read-only artifacts and no train file, exact routing and label schemas, interruption/resume, cache invalidation, budget accounting, invalid replies, explanation evidence and dry runs. `tests/test_retry.py` covers the transient-failure policy: which errors are retried, backoff bounds, reservation release and the stops that must never be retried. They make no paid API requests.

### Repeat the complete Docker E2E

After building the image and configuring the API key, the host runner prepares public inputs, runs `adapt` and `eval` in separate containers and calls the unchanged official scorer:

```sh
.venv/bin/python scripts/run_public_e2e.py --image metalearning-judge:latest
```

This command makes live API requests under a maximum $10 pipeline budget. It pins the chosen image ID, exposes only `test.jsonl` to `eval`, mounts adaptation artifacts read-only, checks their hashes after evaluation and saves stage logs, metrics and an E2E report under `out/docker-e2e/`. Credentials are forwarded by environment variable name without being saved to logs or manifests.

Use `--run-dir PATH` to resume a run using the same image and configuration. Repeating the completed run is also supported; the verified run reused all responses with zero additional API attempts and byte-identical predictions. Local journals and generated artifacts under `out/` are ignored by Git; the compact validation record is saved in [validation/docker-public-20261024.json](validation/docker-public-20261024.json).
