from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import yaml

from .graph import build_evidence_graph
from .rubric_engine import execute_rubric
from .schema import AnswerSample, EvidenceNode, GraphEdge, ItemMatch, Region, Rubric, RubricItem, ScoreResult
from .typed_extractor import extract_typed_evidence


def load_yaml(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_answer_sample(path: Path) -> AnswerSample:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return AnswerSample(
        sample_id=payload["sample_id"],
        writer_id=payload["writer_id"],
        qid=payload["qid"],
        question_text=payload["question_text"],
        image_path=payload["image_path"],
        regions=[
            Region(
                region_id=region["id"],
                bbox=tuple(region["bbox"]),
                text=region["text"],
                type_hint=region.get("type_hint"),
                metadata=region.get("metadata", {}),
            )
            for region in payload["regions"]
        ],
        gold_total=payload["gold"]["total_score"],
        gold_item_hits=payload["gold"]["item_hits"],
        gold_evidence=payload["gold"]["evidence"],
    )


def load_rubric(path: Path) -> Rubric:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return Rubric(
        qid=payload["qid"],
        question_type=payload["question_type"],
        question_text=payload["question_text"],
        max_marks=payload["max_marks"],
        items=[
            RubricItem(
                item_id=item["id"],
                item_type=item["type"],
                description=item["description"],
                marks=item["marks"],
                required=item.get("required", False),
                order=item.get("order"),
                prerequisite=item.get("prerequisite"),
                alternatives=item.get("alternatives", []),
            )
            for item in payload["items"]
        ],
    )


def _representation(nodes: List[EvidenceNode], edges: List[GraphEdge], matches: List[ItemMatch]) -> Dict[str, Any]:
    return {
        "nodes": [
            {
                "id": node.node_id,
                "type": node.predicted_type,
                "text": node.text,
                "bbox": list(node.bbox),
                "confidence": node.confidence,
            }
            for node in nodes
        ],
        "edges": [{"src": edge.src, "dst": edge.dst, "relation": edge.relation} for edge in edges],
        "rubric_alignment": [
            {
                "item_id": match.item_id,
                "hit": match.hit,
                "score": match.score,
                "evidence": match.evidence_node_ids,
                "confidence": match.confidence,
            }
            for match in matches
        ],
    }


class ScoreMapPipeline:
    def __init__(
        self,
        use_type_routing: bool = True,
        use_order_constraints: bool = True,
        use_prerequisites: bool = True,
        use_graph_context: bool = True,
    ) -> None:
        self.use_type_routing = use_type_routing
        self.use_order_constraints = use_order_constraints
        self.use_prerequisites = use_prerequisites
        self.use_graph_context = use_graph_context

    @classmethod
    def from_config(cls, config_path: Path, variant: str = "scoremap") -> "ScoreMapPipeline":
        config = load_yaml(config_path)
        variants = config.get("variants", {})
        variant_cfg = variants.get(variant, {})
        return cls(
            use_type_routing=variant_cfg.get("use_type_routing", True),
            use_order_constraints=variant_cfg.get("use_order_constraints", True),
            use_prerequisites=variant_cfg.get("use_prerequisites", True),
            use_graph_context=variant_cfg.get("use_graph_context", True),
        )

    def _attach_graph_context(self, nodes: List[EvidenceNode], edges: List[GraphEdge]) -> None:
        node_lookup = {node.node_id: node for node in nodes}
        neighbor_text: Dict[str, List[str]] = {node.node_id: [] for node in nodes}
        useful_relations = {"reading_order", "adjacent_down", "same_row"}
        for edge in edges:
            if edge.relation not in useful_relations:
                continue
            src = node_lookup.get(edge.src)
            dst = node_lookup.get(edge.dst)
            if not src or not dst:
                continue
            neighbor_text[src.node_id].append(dst.text)
            neighbor_text[dst.node_id].append(src.text)
        for node in nodes:
            node.metadata["context_text"] = " ".join([node.text] + neighbor_text.get(node.node_id, [])[:3])

    def run(self, answer: AnswerSample, rubric: Rubric) -> ScoreResult:
        nodes = extract_typed_evidence(answer)
        edges = build_evidence_graph(nodes)
        if self.use_graph_context:
            self._attach_graph_context(nodes, edges)
        matches = execute_rubric(
            rubric,
            nodes,
            use_type_routing=self.use_type_routing,
            use_order_constraints=self.use_order_constraints,
            use_prerequisites=self.use_prerequisites,
            use_graph_context=self.use_graph_context,
        )
        total_score = round(sum(match.score for match in matches), 2)
        review_flag = any(match.hit and match.confidence < 0.58 for match in matches)
        return ScoreResult(
            sample_id=answer.sample_id,
            qid=answer.qid,
            total_score=total_score,
            max_marks=rubric.max_marks,
            review_flag=review_flag,
            item_results=matches,
            evidence_nodes=nodes,
            edges=edges,
            extracted_representation=_representation(nodes, edges, matches),
        )
