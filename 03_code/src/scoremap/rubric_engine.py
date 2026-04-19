from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from .schema import EvidenceNode, ItemMatch, Rubric, RubricItem
from .text_utils import clamp, expand_tokens, extract_complexities, overlap_score, tokenize


TYPE_ROUTING = {
    "definition": {"definition", "concept", "final_answer"},
    "concept": {"concept", "definition", "final_answer"},
    "algorithm_step": {"algorithm_step", "code", "definition"},
    "complexity": {"complexity", "definition", "final_answer"},
    "code": {"code", "algorithm_step"},
    "diagram": {"diagram", "diagram_gantt"},
    "diagram_gantt": {"diagram_gantt", "diagram"},
    "final_answer": {"final_answer", "definition", "concept"},
}


def _candidate_nodes(
    item: RubricItem,
    nodes: List[EvidenceNode],
    use_type_routing: bool,
) -> List[EvidenceNode]:
    if not use_type_routing:
        return nodes
    allowed = TYPE_ROUTING.get(item.item_type, set()) | {item.item_type}
    return [node for node in nodes if node.predicted_type in allowed]


def _coverage_score(description: str, text: str) -> float:
    desc_tokens = [token for token in tokenize(description) if len(token) > 2]
    text_tokens = expand_tokens(tokenize(text))
    if not desc_tokens:
        return 0.0
    matched = 0
    for token in desc_tokens:
        choices = {token} | expand_tokens([token])
        if choices & text_tokens:
            matched += 1
    return matched / len(desc_tokens)


def _best_textual_match(item: RubricItem, nodes: List[EvidenceNode]) -> Tuple[Optional[EvidenceNode], float]:
    best_node = None
    best_score = 0.0
    for node in nodes:
        node_text = node.text
        context_text = node.metadata.get("context_text", "")
        local_score = 0.55 * overlap_score(item.description, node_text) + 0.45 * _coverage_score(item.description, node_text)
        context_score = 0.55 * overlap_score(item.description, context_text) + 0.45 * _coverage_score(item.description, context_text)
        score = max(local_score, 0.70 * local_score + 0.30 * context_score)
        if item.item_type == "code" and node.predicted_type == "code":
            score += 0.10
        if item.item_type in {"diagram", "diagram_gantt"} and node.predicted_type.startswith("diagram"):
            score += 0.15
        if item.item_type == "complexity" and extract_complexities(node.text):
            score += 0.20
        if item.item_type == "final_answer" and any(token in node.text.lower() for token in ["aging", "fair", "forever", "priority"]):
            score += 0.15
        if item.item_type in {"definition", "concept"}:
            score += 0.10 * _coverage_score(item.description, node.text)
        if score > best_score:
            best_node = node
            best_score = score
    return best_node, clamp(best_score)


def _match_complexity(item: RubricItem, nodes: List[EvidenceNode]) -> Tuple[Optional[EvidenceNode], float, str]:
    expected = set(extract_complexities(item.description))
    best_node, base_score = _best_textual_match(item, nodes)
    if best_node is None:
        return None, 0.0, "No candidate region for complexity item."

    seen = set(extract_complexities(best_node.text))
    if expected and seen and expected & seen:
        return best_node, clamp(base_score + 0.35), f"Matched complexity expression {sorted(expected & seen)[0]}."
    if "auxiliary array" in item.description.lower() and "array" in best_node.text.lower():
        return best_node, clamp(base_score + 0.20), "Matched supporting space-complexity phrase."
    return best_node, base_score, "Complexity candidate found but exact asymptotic form is weak."


def _match_diagram(item: RubricItem, nodes: List[EvidenceNode]) -> Tuple[Optional[EvidenceNode], float, str]:
    best_node, base_score = _best_textual_match(item, nodes)
    if best_node is None:
        return None, 0.0, "No diagram candidate found."

    segments = best_node.metadata.get("diagram_segments", [])
    if segments:
        bonus = 0.20 + min(0.20, 0.03 * len(segments))
        return best_node, clamp(base_score + bonus), f"Diagram region includes {len(segments)} parsed segments."
    if "|" in best_node.text and any(token in best_node.text.lower() for token in ["p1", "p2", "p3"]):
        return best_node, clamp(base_score + 0.20), "Detected Gantt-style textual diagram."
    return best_node, base_score, "Diagram candidate found without explicit structured segments."


def _match_algorithm_step(item: RubricItem, nodes: List[EvidenceNode]) -> Tuple[Optional[EvidenceNode], float, str]:
    best_node, base_score = _best_textual_match(item, nodes)
    if best_node is None:
        return None, 0.0, "No algorithm-step candidate found."
    bonus = 0.10 if any(token in best_node.text.lower() for token in ["step", "then", "merge", "divide", "enqueue", "combine", "split"]) else 0.0
    return best_node, clamp(base_score + bonus), "Matched ordered procedural evidence."


def _match_generic(item: RubricItem, nodes: List[EvidenceNode]) -> Tuple[Optional[EvidenceNode], float, str]:
    best_node, best_score = _best_textual_match(item, nodes)
    if best_node is None:
        return None, 0.0, "No textual match found."
    return best_node, best_score, "Matched lexical and semantic overlap."


def execute_rubric(
    rubric: Rubric,
    nodes: List[EvidenceNode],
    use_type_routing: bool = True,
    use_order_constraints: bool = True,
    use_prerequisites: bool = True,
    use_graph_context: bool = True,
) -> List[ItemMatch]:
    results: List[ItemMatch] = []
    last_ordered_index = -1
    node_index = {node.node_id: idx for idx, node in enumerate(sorted(nodes, key=lambda n: (n.bbox[1], n.bbox[0])))}
    awarded: Dict[str, bool] = {}

    for item in sorted(rubric.items, key=lambda entry: (entry.order is None, entry.order or 0, entry.item_id)):
        candidates = _candidate_nodes(item, nodes, use_type_routing)
        if item.item_type == "complexity":
            node, score, rationale = _match_complexity(item, candidates)
        elif item.item_type in {"diagram", "diagram_gantt"}:
            node, score, rationale = _match_diagram(item, candidates)
        elif item.item_type == "algorithm_step":
            node, score, rationale = _match_algorithm_step(item, candidates)
        else:
            node, score, rationale = _match_generic(item, candidates)

        if node and use_order_constraints and item.order is not None:
            candidate_index = node_index.get(node.node_id, -1)
            if candidate_index < last_ordered_index:
                score *= 0.55
                rationale += " Penalized because the matched evidence breaks reading order."
            else:
                last_ordered_index = candidate_index

        if use_prerequisites and item.prerequisite and not awarded.get(item.prerequisite, False):
            score *= 0.30
            rationale += f" Prerequisite {item.prerequisite} was not matched."

        if use_type_routing and use_graph_context:
            threshold = 0.34 if item.item_type in {"definition", "concept", "final_answer"} else 0.38
        elif use_type_routing:
            threshold = 0.39 if item.item_type in {"definition", "concept", "final_answer"} else 0.43
        else:
            threshold = 0.42 if item.item_type in {"definition", "concept", "final_answer"} else 0.48
        hit = score >= threshold
        if node and hit:
            awarded[item.item_id] = True
        confidence = clamp((score + (node.confidence if node else 0.0)) / 2.0)
        results.append(
            ItemMatch(
                item_id=item.item_id,
                hit=hit,
                score=item.marks if hit else 0.0,
                evidence_node_ids=[node.node_id] if node and hit else [],
                confidence=round(confidence, 4),
                rationale=rationale,
            )
        )
    return results
