from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional

from pypdf import PdfReader

from .schema import Rubric, RubricItem
from .text_utils import contains_any, normalize_text
from .typed_extractor import TYPE_KEYWORDS


QUESTION_ID_RE = re.compile(r"^question(?:\s+id)?\s*:\s*(.+)$", re.IGNORECASE)
QUESTION_HEADER_RE = re.compile(r"^question\s+(q[\w-]+)\s*[:.-]?\s*(.*)$", re.IGNORECASE)
FIELD_RE = re.compile(r"^([A-Za-z][A-Za-z ]+?)\s*:\s*(.*)$")
MAX_MARKS_RE = re.compile(r"^max(?:imum)?\s*marks?\s*:\s*([0-9]+(?:\.[0-9]+)?)$", re.IGNORECASE)
BULLET_RE = re.compile(r"^(?:[-*\u2022]|\d+\.)\s*(.+)$")
MARKS_INLINE_RE = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*(?:marks?|m)\b", re.IGNORECASE)


def read_answer_key_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        reader = PdfReader(str(path))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages).replace("\r", "\n")
    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8")
    raise ValueError(f"Unsupported answer-key source: {path}")


def parse_answer_key_file(path: Path) -> List[Rubric]:
    text = read_answer_key_text(path)
    return parse_answer_key_text(text)


def parse_answer_key_text(text: str) -> List[Rubric]:
    normalized = text.replace("\r", "\n")
    if "Question ID:" in normalized:
        rubrics = _parse_structured_answer_key(normalized)
        if rubrics:
            return rubrics
    rubrics = _parse_bulleted_answer_key(normalized)
    if rubrics:
        return rubrics
    raise ValueError("Unable to parse the answer key. Use the structured template documented in the README.")


def rubric_to_payload(rubric: Rubric) -> Dict[str, object]:
    return {
        "qid": rubric.qid,
        "question_type": rubric.question_type,
        "question_text": rubric.question_text,
        "max_marks": rubric.max_marks,
        "items": [
            {
                "id": item.item_id,
                "type": item.item_type,
                "description": item.description,
                "marks": item.marks,
                "required": item.required,
                **({"order": item.order} if item.order is not None else {}),
                **({"prerequisite": item.prerequisite} if item.prerequisite else {}),
                **({"alternatives": item.alternatives} if item.alternatives else {}),
            }
            for item in rubric.items
        ],
    }


def write_rubrics(rubrics: List[Rubric], output_dir: Path) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    for rubric in rubrics:
        path = output_dir / f"{rubric.qid}.json"
        path.write_text(json.dumps(rubric_to_payload(rubric), indent=2), encoding="utf-8")
        written.append(path)
    return written


def _parse_structured_answer_key(text: str) -> List[Rubric]:
    rubrics: List[Rubric] = []
    current_question: Dict[str, object] = {}
    current_item: Dict[str, object] = {}
    active_target: Optional[str] = None

    def flush_item() -> None:
        nonlocal current_item, active_target
        if current_item:
            items = current_question.setdefault("items", [])
            items.append(dict(current_item))
            current_item = {}
            active_target = None

    def flush_question() -> None:
        nonlocal current_question, active_target
        flush_item()
        if current_question:
            rubrics.append(_rubric_from_record(current_question))
            current_question = {}
            active_target = None

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if set(line) <= {"=", "-"} and len(line) >= 3:
            if "=" in line:
                flush_question()
            continue

        if QUESTION_ID_RE.match(line):
            flush_question()

        field_match = FIELD_RE.match(line)
        if field_match:
            key = field_match.group(1).strip().lower()
            value = field_match.group(2).strip()
            active_target = key
            if key == "item id":
                flush_item()
                current_item["item id"] = value
            elif key.startswith("item ") and current_item:
                current_item[key] = value
            elif key in {"question id", "question type", "question text", "max marks"}:
                current_question[key] = value
            elif current_item:
                current_item[key] = value
            else:
                current_question[key] = value
            continue

        if active_target and current_item and active_target in current_item:
            current_item[active_target] = f"{current_item[active_target]} {line}".strip()
        elif active_target and active_target in current_question:
            current_question[active_target] = f"{current_question[active_target]} {line}".strip()

    flush_question()
    return rubrics


def _parse_bulleted_answer_key(text: str) -> List[Rubric]:
    rubrics: List[Rubric] = []
    current: Optional[Dict[str, object]] = None

    def flush_current() -> None:
        nonlocal current
        if current is not None:
            rubrics.append(_rubric_from_record(current))
            current = None

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        header_match = QUESTION_HEADER_RE.match(line)
        if header_match:
            flush_current()
            current = {
                "question id": header_match.group(1).upper(),
                "question text": header_match.group(2).strip(),
                "items": [],
            }
            continue

        if current is None:
            continue

        max_match = MAX_MARKS_RE.match(line)
        if max_match:
            current["max marks"] = max_match.group(1)
            continue

        bullet_match = BULLET_RE.match(line)
        if bullet_match:
            current.setdefault("items", []).append(_heuristic_item_record(bullet_match.group(1), current))
            continue

        if "question text" in current and not current.get("items"):
            current["question text"] = f"{current['question text']} {line}".strip()

    flush_current()
    return rubrics


def _heuristic_item_record(line: str, question_record: Dict[str, object]) -> Dict[str, object]:
    marks_match = MARKS_INLINE_RE.search(line)
    marks = float(marks_match.group(1)) if marks_match else 1.0
    description = MARKS_INLINE_RE.sub("", line).strip(" -:;")
    lowered = normalize_text(description)
    required = "optional" not in lowered
    prerequisite = None
    prereq_match = re.search(r"prerequisite\s*[=:]\s*([a-z0-9_]+)", description, re.IGNORECASE)
    if prereq_match:
        prerequisite = prereq_match.group(1)
        description = re.sub(r"prerequisite\s*[=:]\s*[a-z0-9_]+", "", description, flags=re.IGNORECASE).strip(" -:;")

    item_count = len(question_record.get("items", [])) + 1
    return {
        "item id": f"r{item_count}",
        "item type": _infer_item_type(description, str(question_record.get("question text", ""))),
        "description": description,
        "marks": str(marks),
        "required": str(required).lower(),
        "prerequisite": prerequisite or "",
    }


def _rubric_from_record(record: Dict[str, object]) -> Rubric:
    qid = str(record.get("question id", "")).strip() or "QX"
    question_text = str(record.get("question text", "")).strip()
    item_records = [item for item in record.get("items", []) if item]
    items = [_item_from_record(item, question_text) for item in item_records]
    question_type = str(record.get("question type", "")).strip() or _infer_question_type(items)
    max_marks = float(record.get("max marks") or sum(item.marks for item in items))
    return Rubric(
        qid=qid,
        question_type=question_type or "generic",
        question_text=question_text,
        max_marks=max_marks,
        items=items,
    )


def _item_from_record(record: Dict[str, object], question_text: str) -> RubricItem:
    item_id = str(record.get("item id", "")).strip() or "r0"
    description = str(record.get("description", "")).strip()
    item_type = str(record.get("item type", "")).strip() or _infer_item_type(description, question_text)
    marks = float(record.get("marks", 1))
    required = _parse_bool(record.get("required"), default=True)
    order = _parse_int(record.get("order"))
    prerequisite = _clean_optional_text(record.get("prerequisite"))
    alternatives = _parse_list(record.get("alternatives"))
    return RubricItem(
        item_id=item_id,
        item_type=item_type,
        description=description,
        marks=marks,
        required=required,
        order=order,
        prerequisite=prerequisite,
        alternatives=alternatives,
    )


def _infer_item_type(description: str, question_text: str) -> str:
    normalized = normalize_text(f"{question_text} {description}")
    best_label = "concept"
    best_score = -1
    for label, keywords in TYPE_KEYWORDS.items():
        score = sum(1 for keyword in keywords if contains_any(normalized, [keyword]))
        if label == "definition" and contains_any(normalized, ["state", "define", "means"]):
            score += 1
        if label == "diagram_gantt" and contains_any(normalized, ["gantt", "timeline", "round robin"]):
            score += 2
        if label == "code" and contains_any(normalized, ["pseudocode", "algorithm", "enqueue", "mid", "queue"]):
            score += 1
        if label == "complexity" and contains_any(normalized, ["complexity", "o(", "theta(", "omega("]):
            score += 2
        if score > best_score:
            best_label = label
            best_score = score
    return best_label


def _infer_question_type(items: List[RubricItem]) -> str:
    item_types = {item.item_type for item in items}
    if "diagram_gantt" in item_types and "definition" in item_types:
        return "definition_plus_diagram"
    if "algorithm_step" in item_types and "complexity" in item_types:
        return "algorithm_plus_complexity"
    if "code" in item_types and "complexity" in item_types:
        return "code_plus_complexity"
    if "diagram_gantt" in item_types:
        return "diagram_trace"
    if item_types == {"definition"}:
        return "definition_list"
    return "generic"


def _parse_bool(value: object, default: bool = False) -> bool:
    if value is None:
        return default
    normalized = normalize_text(str(value))
    if normalized in {"true", "yes", "1", "required"}:
        return True
    if normalized in {"false", "no", "0", "optional"}:
        return False
    return default


def _parse_int(value: object) -> Optional[int]:
    if value in {None, ""}:
        return None
    try:
        return int(str(value).strip())
    except ValueError:
        return None


def _clean_optional_text(value: object) -> Optional[str]:
    if value in {None, ""}:
        return None
    cleaned = str(value).strip()
    return cleaned or None


def _parse_list(value: object) -> List[str]:
    if value in {None, ""}:
        return []
    raw = str(value).replace(";", ",")
    return [part.strip() for part in raw.split(",") if part.strip()]
