# System One Bench

System One Bench is a small, reproducible comparison between two approaches to single-choice decision making:

- `typesafe/jev-1.13`, accessed through OpenRouter's Decisions API;
- `lmstudio-community/Qwen3-14B-MLX-4bit`, executed locally with MLX and native thinking enabled.

The benchmark does not argue for either approach. It measures how their accuracy and execution characteristics change as a logical-deduction task becomes more complex.

## Experimental question

The evaluated dataset is BBH Logical Deduction, restricted to:

- `logical_deduction_three_objects` — 300 examples;
- `logical_deduction_five_objects` — 500 examples;
- `logical_deduction_seven_objects` — 700 examples.

This progression provides a simple complexity axis. Jev produces a typed `Choice` decision in a System One-style pass. Qwen can spend autoregressive test-time compute in its native thinking mode before returning a choice.

Both models receive the same problem text, alternatives, and instruction. Neither receives few-shot examples or model-specific task information. The intentional treatment difference is Qwen's native sequential thinking; Jev uses its native Choice interface.

## Reproducibility contract

The loader uses `lighteval/bbh` at the pinned revision:

```text
1b61f099fcbf9e55691ef8cc6b4b8fb431dae097
```

The split is `train`, tasks are loaded in 3/5/7-object order, and alternatives are mapped to canonical keys `A`, `B`, `C`, and so on. Configuration parsing rejects unknown fields. Provider or infrastructure errors stop the run immediately; an unfinished Qwen generation is recorded as an invalid answer and remains part of the benchmark result.

The two checked-in YAML files default to `dry_run: true`. A live run requires changing that value explicitly.

## Metrics

Each completed run reports:

- overall accuracy and macro-F1;
- accuracy per task complexity;
- invalid-output rate;
- p50 and p95 end-to-end latency;
- estimated or provider-reported cost;
- multiclass Brier score and expected calibration error when complete probabilities are available.

Qwen artifacts also preserve its reasoning separately from the final structured answer. Jev artifacts preserve probabilities, usage, cost, and the raw provider response when available.

## Setup

Requirements:

- Python 3.11 or newer;
- [uv](https://docs.astral.sh/uv/);
- Apple Silicon for the local MLX run;
- approximately 10 GB of available unified memory for the 14B 4-bit checkpoint.

On a 16 GB Apple Silicon Mac, close memory-heavy applications before running Qwen. Install the project and development dependencies with:

```bash
make setup
make check
```

The Qwen configuration downloads `lmstudio-community/Qwen3-14B-MLX-4bit` through MLX when it is not already cached. If the same checkpoint already exists locally, set `model_path` in `configs/qwen3-14b-mlx.yaml`.

## Inspecting the plans

These commands parse the YAML and print the execution plan without loading a model or making a paid request:

```bash
make plan-qwen
make plan-jev
```

For a small smoke test, set `dataset.limit: 3`; the limit applies per task, producing nine examples in total. Keep `dry_run: true` while inspecting the plan, then change it to `false` to execute.

## Running Qwen

Qwen uses the checkpoint's chat template with `enable_thinking=True`. Generation is unconstrained during thinking. After `</think>`, a logits processor restricts the final suffix to one JSON object such as `{"choice":"A"}`. Only that final choice is scored.

```bash
make run-qwen
```

The checked-in configuration uses an 8,192-token generation cap and the parameters recorded for this experiment. A generation that reaches the cap before producing final JSON is counted as invalid rather than guessed with fuzzy parsing.

Interrupted local runs are append-only and resumable:

```bash
make resume-qwen RESUME_DIR=results/<run-id>
```

Resume verifies the full configuration, regenerated dataset examples, and choice order before skipping completed identifiers.

## Running Jev

Create a local environment file:

```bash
cp .env.example .env
```

Add an OpenRouter key as `OPENROUTER_API_KEY`, set `dry_run: false` in `configs/jev-1.13-openrouter.yaml`, and run:

```bash
make run-jev
```

The API payload represents each example as a Jev `Choice`: canonical keys are the choices and the original alternative text is used as criteria.

## Artifacts

Every run creates `results/<timestamp>-<adapter>/` containing:

- `metadata.json` — exact parsed configuration and choice order;
- `predictions.jsonl` — one append-only record per completed example;
- `summary.json` — aggregate metrics and estimated cost, written after completion.

`results/`, `.env`, model weights, caches, and virtual environments are excluded from Git.

## Interpretation limits

This benchmark compares complete systems, not parameter-matched architectures. A difference in accuracy cannot be attributed to autoregression alone: training data, objectives, model capacity, quantization, sampling, and provider implementation also differ. Latency is measured in different execution environments and should not be read as a hardware-normalized comparison. The results describe these model versions under this protocol.
