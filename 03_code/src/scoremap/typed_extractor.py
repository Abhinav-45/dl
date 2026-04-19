from __future__ import annotations

from typing import Dict, List

from .schema import AnswerSample, EvidenceNode, Region
from .text_utils import contains_any, normalize_text


TYPE_KEYWORDS = {
    "diagram_gantt": ["gantt", "timeline", "p1", "p2", "p3", "segment", "|"],
    "diagram": ["diagram", "queue", "state", "chart", "->", "|"],
    "complexity": ["o(", "theta(", "omega(", "complexity", "time", "space"],
    "code": ["for", "while", "if", "return", "mid", "low", "high", "queue", "visited", "=", ":"],
    "algorithm_step": ["step", "first", "second", "then", "finally", "divide", "merge", "enqueue", "dequeue"],
    "final_answer": ["therefore", "thus", "hence", "finally", "aging", "fair", "forever", "mitigation"],
    "concept": ["because", "causes", "condition", "requires", "leads", "happens when", "priority", "fair"],
    "definition": ["is", "means", "defined", "refers"],
}


def _detect_type(region: Region, question_text: str) -> Dict[str, float]:
    text = normalize_text(region.text)
    x1, y1, x2, y2 = region.bbox
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    scores = {
        "definition": 0.15,
        "concept": 0.12,
        "algorithm_step": 0.10,
        "complexity": 0.10,
        "code": 0.10,
        "diagram": 0.08,
        "diagram_gantt": 0.08,
        "final_answer": 0.08,
    }

    for label, keywords in TYPE_KEYWORDS.items():
        if contains_any(text, keywords):
            scores[label] = scores.get(label, 0.0) + 0.45

    if "|" in region.text or "gantt" in text:
        scores["diagram_gantt"] += 0.30
        scores["diagram"] += 0.20

    if any(token in region.text for token in ["=", ":", "[]", "()", "{", "}"]):
        scores["code"] += 0.18

    if text.startswith(("1 ", "1.", "2 ", "2.", "3 ", "step")):
        scores["algorithm_step"] += 0.25

    if width > 350 and height > 90 and len(text.split()) < 20:
        scores["diagram"] += 0.20

    if "complexity" in normalize_text(question_text):
        scores["complexity"] += 0.05

    if region.metadata.get("diagram_segments"):
        scores["diagram_gantt"] += 0.35

    if contains_any(text, ["therefore", "thus", "aging", "hence", "fair", "forever"]):
        scores["final_answer"] += 0.15

    return scores


def extract_typed_evidence(sample: AnswerSample) -> List[EvidenceNode]:
    nodes: List[EvidenceNode] = []
    for region in sample.regions:
        scores = _detect_type(region, sample.question_text)
        predicted_type = max(scores, key=scores.get)
        base_confidence = min(0.99, scores[predicted_type])
        ocr_confidence = region.metadata.get("ocr_confidence")
        if isinstance(ocr_confidence, (int, float)):
            confidence = min(0.99, 0.7 * base_confidence + 0.3 * float(ocr_confidence))
        else:
            confidence = base_confidence
        nodes.append(
            EvidenceNode(
                node_id=region.region_id,
                bbox=region.bbox,
                text=region.text,
                predicted_type=predicted_type,
                confidence=round(confidence, 4),
                source_region_id=region.region_id,
                metadata={"type_scores": scores, **region.metadata},
            )
        )
    return nodes
