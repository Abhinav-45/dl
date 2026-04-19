from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
from sklearn.metrics import cohen_kappa_score, f1_score, mean_absolute_error

from .pipeline import ScoreMapPipeline, load_answer_sample, load_rubric
from .schema import AnswerSample, ScoreResult
from .text_utils import safe_divide


def _macro_type_f1(sample: AnswerSample, result: ScoreResult) -> float:
    gold = [region.type_hint or "unknown" for region in sample.regions]
    pred_lookup = {node.node_id: node.predicted_type for node in result.evidence_nodes}
    pred = [pred_lookup.get(region.region_id, "unknown") for region in sample.regions]
    return f1_score(gold, pred, average="macro")


def _item_f1(samples: Sequence[AnswerSample], results: Sequence[ScoreResult]) -> float:
    gold_all: List[int] = []
    pred_all: List[int] = []
    result_lookup = {result.sample_id: result for result in results}
    for sample in samples:
        prediction = {item.item_id: item.hit for item in result_lookup[sample.sample_id].item_results}
        for item_id, gold_value in sample.gold_item_hits.items():
            gold_all.append(1 if gold_value else 0)
            pred_all.append(1 if prediction.get(item_id, False) else 0)
    return f1_score(gold_all, pred_all)


def _evidence_f1(samples: Sequence[AnswerSample], results: Sequence[ScoreResult]) -> float:
    result_lookup = {result.sample_id: result for result in results}
    scores: List[float] = []
    for sample in samples:
        prediction = {item.item_id: set(item.evidence_node_ids) for item in result_lookup[sample.sample_id].item_results}
        for item_id, gold_nodes in sample.gold_evidence.items():
            gold = set(gold_nodes)
            pred = prediction.get(item_id, set())
            tp = len(gold & pred)
            precision = safe_divide(tp, len(pred))
            recall = safe_divide(tp, len(gold))
            if precision + recall == 0:
                scores.append(0.0)
            else:
                scores.append(2 * precision * recall / (precision + recall))
    return mean(scores) if scores else 0.0


def load_split(split_path: Path) -> List[str]:
    return json.loads(split_path.read_text(encoding="utf-8"))


def run_evaluation(
    data_root: Path,
    answers_dir: Path,
    rubrics_dir: Path,
    split_path: Path,
    output_dir: Path,
    variants: Dict[str, ScoreMapPipeline],
) -> Dict[str, Dict[str, float]]:
    sample_ids = load_split(split_path)
    samples = [load_answer_sample(answers_dir / f"{sample_id}.json") for sample_id in sample_ids]
    rubrics = {}
    for sample in samples:
        rubrics[sample.qid] = rubrics.get(sample.qid) or load_rubric(rubrics_dir / f"{sample.qid}.json")

    metrics: Dict[str, Dict[str, float]] = {}
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, str]] = []

    for name, pipeline in variants.items():
        results = [pipeline.run(sample, rubrics[sample.qid]) for sample in samples]
        gold_scores = [sample.gold_total for sample in samples]
        pred_scores = [result.total_score for result in results]
        mae = mean_absolute_error(gold_scores, pred_scores)
        exact = mean(1.0 if gold == pred else 0.0 for gold, pred in zip(gold_scores, pred_scores))
        kappa = cohen_kappa_score(gold_scores, pred_scores, weights="quadratic")
        item_f1 = _item_f1(samples, results)
        evidence_f1 = _evidence_f1(samples, results)
        type_f1 = mean(_macro_type_f1(sample, result) for sample, result in zip(samples, results))

        metrics[name] = {
            "mae": round(mae, 4),
            "exact_match": round(exact, 4),
            "weighted_kappa": round(kappa, 4),
            "rubric_item_f1": round(item_f1, 4),
            "evidence_f1": round(evidence_f1, 4),
            "region_type_macro_f1": round(type_f1, 4),
        }
        rows.append(
            {
                "model": name,
                "mae": f"{mae:.4f}",
                "exact_match": f"{exact:.4f}",
                "weighted_kappa": f"{kappa:.4f}",
                "rubric_item_f1": f"{item_f1:.4f}",
                "evidence_f1": f"{evidence_f1:.4f}",
                "region_type_macro_f1": f"{type_f1:.4f}",
            }
        )

    with (output_dir / "main_results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    figure_path = output_dir / "figures" / "model_comparison.png"
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    _plot_metrics(rows, figure_path)

    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    with (logs_dir / "eval_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    return metrics


def write_ablations(output_dir: Path, metrics: Dict[str, Dict[str, float]]) -> None:
    rows = [
        {
            "ablation": "generic_scorer",
            "removed_component": "type-specific routing + structured executors",
            "mae": metrics["generic"]["mae"],
            "rubric_item_f1": metrics["generic"]["rubric_item_f1"],
            "evidence_f1": metrics["generic"]["evidence_f1"],
        },
        {
            "ablation": "no_graph_constraints",
            "removed_component": "reading-order / prerequisite constraints",
            "mae": metrics["no_graph"]["mae"],
            "rubric_item_f1": metrics["no_graph"]["rubric_item_f1"],
            "evidence_f1": metrics["no_graph"]["evidence_f1"],
        },
        {
            "ablation": "full_scoremap",
            "removed_component": "none",
            "mae": metrics["scoremap"]["mae"],
            "rubric_item_f1": metrics["scoremap"]["rubric_item_f1"],
            "evidence_f1": metrics["scoremap"]["evidence_f1"],
        },
    ]
    with (output_dir / "ablations.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _plot_metrics(rows: Sequence[Dict[str, str]], output_path: Path) -> None:
    models = [row["model"] for row in rows]
    rubric_f1 = [float(row["rubric_item_f1"]) for row in rows]
    evidence_f1 = [float(row["evidence_f1"]) for row in rows]
    mae = [float(row["mae"]) for row in rows]

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    axes[0].bar(models, rubric_f1, color=["#c95c54", "#809848", "#1e88e5"])
    axes[0].set_title("Rubric Item F1")
    axes[0].set_ylim(0, 1)

    axes[1].bar(models, evidence_f1, color=["#c95c54", "#809848", "#1e88e5"])
    axes[1].set_title("Evidence F1")
    axes[1].set_ylim(0, 1)

    axes[2].bar(models, mae, color=["#c95c54", "#809848", "#1e88e5"])
    axes[2].set_title("MAE on Marks")
    axes[2].set_ylim(0, max(mae) + 0.5)

    for ax in axes:
        ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
