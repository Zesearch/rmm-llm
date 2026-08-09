# External baseline policy

The clean RMM repository does not copy implementations of H2O, random/static
pruning, or other comparison methods. Baselines should be executed from their
official repositories under their original licenses.

For every reported baseline, record:

- method and upstream repository URL;
- exact commit or release tag;
- model ID/path and model revision;
- dataset name, configuration, split, and revision;
- launch command and dependency environment;
- prompt, context length, generation arguments, and random seed;
- output artifact used by the shared scorer.

## Summarization interchange format

External methods can use `scripts/score_summaries.py` by exporting one JSON
object per line:

```json
{"id": 0, "prediction": "Generated summary.", "reference": "Gold summary."}
```

Additional fields are allowed. The `prediction` and `reference` field names can
also be overridden on the scoring command line.

## Harness tasks

For MMLU, GSM8K, HumanEval, and RULER, prefer the same pinned
`lm-evaluation-harness` release and task configuration used by RMM. If an
external method requires its own runner, preserve the full harness JSON output
and record the exact task names, few-shot count, batch size, and RULER metadata.

## Dense baseline

RMM keep ratio `1.0` delegates to the unmodified Transformers eager-attention
function and is the dense baseline in the shared runners.

