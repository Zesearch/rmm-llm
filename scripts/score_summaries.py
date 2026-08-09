#!/usr/bin/env python3
"""Score JSONL summaries from RMM or any external baseline implementation."""

import argparse
import json
import re
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Score summarization JSONL output.")
    parser.add_argument("predictions", help="JSONL with prediction/reference fields")
    parser.add_argument("--output", default=None)
    parser.add_argument("--prediction-field", default="prediction")
    parser.add_argument("--reference-field", default="reference")
    parser.add_argument("--language", default="en")
    parser.add_argument("--bertscore-model", default=None)
    parser.add_argument("--skip-bertscore", action="store_true")
    return parser.parse_args()


def load_records(path, prediction_field, reference_field):
    predictions = []
    references = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            try:
                prediction = str(record[prediction_field]).strip()
                reference = str(record[reference_field]).strip()
            except KeyError as exc:
                raise KeyError(f"Missing field on line {line_number}: {exc}") from exc
            predictions.append(prediction)
            references.append(reference)
    if not predictions:
        raise ValueError("No prediction records found")
    return predictions, references


def sentence_lines(text):
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return "\n".join(sentence for sentence in sentences if sentence)


def rouge_metrics(predictions, references):
    from rouge_score import rouge_scorer, scoring

    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL", "rougeLsum"], use_stemmer=True
    )
    aggregator = scoring.BootstrapAggregator()
    for prediction, reference in zip(predictions, references):
        aggregator.add_scores(
            scorer.score(sentence_lines(reference), sentence_lines(prediction))
        )
    aggregate = aggregator.aggregate()
    return {name: value.mid.fmeasure for name, value in aggregate.items()}


def main():
    args = parse_args()
    predictions, references = load_records(
        args.predictions, args.prediction_field, args.reference_field
    )
    metrics = {
        "samples": len(predictions),
        **rouge_metrics(predictions, references),
    }

    from sacrebleu.metrics import CHRF

    metrics["chrf"] = CHRF().corpus_score(predictions, [references]).score

    if not args.skip_bertscore:
        from bert_score import score

        _, _, f1 = score(
            predictions,
            references,
            lang=args.language,
            model_type=args.bertscore_model,
            verbose=True,
        )
        metrics["bertscore_f1"] = float(f1.mean().item())

    output_path = Path(args.output) if args.output else Path(args.predictions).with_suffix(".metrics.json")
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    print(f"Metrics: {output_path}")


if __name__ == "__main__":
    main()

