from __future__ import annotations

from typing import List

from .schema import EvidenceNode, GraphEdge


def _center(node: EvidenceNode) -> tuple[float, float]:
    x1, y1, x2, y2 = node.bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def build_evidence_graph(nodes: List[EvidenceNode]) -> List[GraphEdge]:
    ordered = sorted(nodes, key=lambda node: (node.bbox[1], node.bbox[0]))
    edges: List[GraphEdge] = []

    for idx, node in enumerate(ordered[:-1]):
        nxt = ordered[idx + 1]
        edges.append(GraphEdge(src=node.node_id, dst=nxt.node_id, relation="reading_order"))

    for idx, node in enumerate(ordered):
        cx, cy = _center(node)
        for other in ordered[idx + 1 :]:
            ox, oy = _center(other)
            if abs(cy - oy) < 28:
                edges.append(GraphEdge(src=node.node_id, dst=other.node_id, relation="same_row"))
            if abs(cx - ox) < 40 and abs(cy - oy) < 140:
                edges.append(GraphEdge(src=node.node_id, dst=other.node_id, relation="same_column"))
            if 0 < oy - cy < 120 and abs(cx - ox) < 220:
                edges.append(GraphEdge(src=node.node_id, dst=other.node_id, relation="adjacent_down"))
            if other.bbox[0] - node.bbox[0] > 24 and abs(cy - oy) < 40:
                edges.append(GraphEdge(src=node.node_id, dst=other.node_id, relation="indentation"))
    return edges
