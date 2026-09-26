# Published results

These files are verbatim copies of the small, aggregate artifacts written by the
two completed BBH runs. They make the reported figures independently inspectable
without committing the prediction JSONL files, which contain long Qwen reasoning
traces.

The metadata files preserve the exact parsed configuration emitted at run time.
They use the configuration schema that existed at the time of the runs, so they
include a few legacy no-op fields not present in the current, simplified config.
The dataset revision, task set, instruction, model identity, sampling settings,
and run IDs are the provenance record for the reported figures.

- `jev-1.13-summary.json` and `jev-1.13-metadata.json` — run
  `20260921T195346Z-jev_openrouter`.
- `qwen3-14b-mlx-4bit-summary.json` and
  `qwen3-14b-mlx-4bit-metadata.json` — run `20260923T085321Z-mlx_qwen`.
