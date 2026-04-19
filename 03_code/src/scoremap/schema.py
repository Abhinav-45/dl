from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


BBox = Tuple[int, int, int, int]


@dataclass
class Region:
    region_id: str
    bbox: BBox
    text: str
    type_hint: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvidenceNode:
    node_id: str
    bbox: BBox
    text: str
    predicted_type: str
    confidence: float
    source_region_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RubricItem:
    item_id: str
    item_type: str
    description: str
    marks: float
    required: bool = False
    order: Optional[int] = None
    prerequisite: Optional[str] = None
    alternatives: List[str] = field(default_factory=list)


@dataclass
class AnswerSample:
    sample_id: str
    writer_id: str
    qid: str
    question_text: str
    image_path: str
    regions: List[Region]
    gold_total: Optional[float] = None
    gold_item_hits: Dict[str, bool] = field(default_factory=dict)
    gold_evidence: Dict[str, List[str]] = field(default_factory=dict)
    source_path: Optional[str] = None


@dataclass
class Rubric:
    qid: str
    question_type: str
    question_text: str
    max_marks: float
    items: List[RubricItem]


@dataclass
class ItemMatch:
    item_id: str
    hit: bool
    score: float
    evidence_node_ids: List[str]
    confidence: float
    rationale: str


@dataclass
class GraphEdge:
    src: str
    dst: str
    relation: str


@dataclass
class ScoreResult:
    sample_id: str
    qid: str
    total_score: float
    max_marks: float
    review_flag: bool
    item_results: List[ItemMatch]
    evidence_nodes: List[EvidenceNode]
    edges: List[GraphEdge]
    extracted_representation: Dict[str, Any]
