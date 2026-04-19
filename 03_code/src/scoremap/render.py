from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from PIL import Image, ImageDraw, ImageFont

from .schema import AnswerSample, ScoreResult


TYPE_COLORS = {
    "definition": "#3366cc",
    "concept": "#5e7c24",
    "algorithm_step": "#d17b0f",
    "complexity": "#8e44ad",
    "code": "#c0392b",
    "diagram": "#00897b",
    "diagram_gantt": "#00897b",
    "final_answer": "#6d4c41",
}


def _font(size: int = 16) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", size=size)
    except OSError:
        return ImageFont.load_default()


def render_overlay(
    image_path: Path,
    result: ScoreResult,
    output_path: Path,
) -> None:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = _font(15)
    hit_by_node: Dict[str, str] = {}
    for item in result.item_results:
        for node_id in item.evidence_node_ids:
            hit_by_node[node_id] = item.item_id

    for node in result.evidence_nodes:
        color = TYPE_COLORS.get(node.predicted_type, "#222222")
        x1, y1, x2, y2 = node.bbox
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        label = f"{node.predicted_type}:{node.confidence:.2f}"
        if node.node_id in hit_by_node:
            label += f" | {hit_by_node[node.node_id]}"
        draw.rectangle([x1, max(0, y1 - 22), min(image.width, x1 + 240), y1], fill=color)
        draw.text((x1 + 4, max(0, y1 - 19)), label, fill="white", font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def export_prediction_json(result: ScoreResult, output_path: Path) -> None:
    payload = {
        "sample_id": result.sample_id,
        "qid": result.qid,
        "total_score": result.total_score,
        "max_marks": result.max_marks,
        "review_flag": result.review_flag,
        "item_results": [
            {
                "item_id": item.item_id,
                "hit": item.hit,
                "score": item.score,
                "confidence": item.confidence,
                "evidence_node_ids": item.evidence_node_ids,
                "rationale": item.rationale,
            }
            for item in result.item_results
        ],
        "typed_evidence_graph": result.extracted_representation,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
