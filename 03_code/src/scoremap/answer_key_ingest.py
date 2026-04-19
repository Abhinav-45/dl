from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import fitz
import numpy as np
from PIL import Image, ImageOps
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
RAW_QID_RE = re.compile(r"\b(Q\s*0*\d[\w-]*)\b", re.IGNORECASE)
RAW_MAX_MARKS_RE = re.compile(r"\[(\d+(?:\.\d+)?)M\]", re.IGNORECASE)

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
_EASYOCR_READER = None


def read_answer_key_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        reader = PdfReader(str(path))
        pages = [page.extract_text() or "" for page in reader.pages]
        text = "\n".join(pages).replace("\r", "\n")
        if text.strip():
            return text
        return ocr_answer_key_text(path)
    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8")
    if suffix in IMAGE_SUFFIXES:
        return ocr_answer_key_text(path)
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
    rubrics = _parse_exam_style_answer_key(normalized)
    if rubrics:
        return rubrics
    raise ValueError(
        "Unable to parse the answer key. Use a text-layer PDF, a structured .txt file, "
        "or an answer-key image/PDF with a visible marking scheme."
    )


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


def ocr_answer_key_text(path: Path) -> str:
    reader = _get_easyocr_reader()
    paragraphs: List[str] = []
    for image in _render_document_images(path):
        normalized = np.asarray(_normalize_image(image))
        blocks = reader.readtext(normalized, detail=0, paragraph=True)
        for block in blocks:
            cleaned = _clean_ocr_paragraph(str(block))
            if cleaned:
                paragraphs.append(cleaned)
    text = "\n".join(paragraphs).strip()
    if not text:
        raise ValueError(f"OCR could not recover readable text from {path.name}.")
    return text


def _get_easyocr_reader():
    global _EASYOCR_READER
    if _EASYOCR_READER is not None:
        return _EASYOCR_READER

    try:
        import easyocr
    except ImportError as exc:
        raise ValueError(
            "EasyOCR is required for image-only answer keys. Install it with "
            "`python -m pip install easyocr`."
        ) from exc

    cache_dir = Path(__file__).resolve().parents[4] / ".ocr_cache" / "easyocr"
    cache_dir.mkdir(parents=True, exist_ok=True)
    _EASYOCR_READER = easyocr.Reader(
        ["en"],
        gpu=False,
        model_storage_directory=str(cache_dir),
        user_network_directory=str(cache_dir),
        verbose=False,
    )
    return _EASYOCR_READER


def _render_document_images(path: Path) -> Iterable[Image.Image]:
    suffix = path.suffix.lower()
    if suffix in IMAGE_SUFFIXES:
        yield ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        return
    if suffix == ".pdf":
        document = fitz.open(path)
        for page in document:
            pixmap = page.get_pixmap(matrix=fitz.Matrix(2.5, 2.5), alpha=False)
            yield Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)
        return
    raise ValueError(f"Unsupported OCR source: {path}")


def _normalize_image(image: Image.Image) -> Image.Image:
    page = ImageOps.exif_transpose(image).convert("L")
    page = ImageOps.autocontrast(page)
    return page.convert("RGB")


def _clean_ocr_paragraph(text: str) -> str:
    cleaned = text.replace("\r", " ").replace("\n", " ")
    replacements = {
        "Qis": "Q is",
        "dequeuel()": "dequeue()",
        "Illf": "If",
        "Ilelse": "else if",
        "linsert": "insert",
        "frm": "from",
        "anda": "and a",
        "Mfor": "M for",
        "inscn": "insert",
        "insent": "insert",
        "deletc": "delete",
    }
    for source, target in replacements.items():
        cleaned = cleaned.replace(source, target)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


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


def _parse_exam_style_answer_key(text: str) -> List[Rubric]:
    paragraphs = [line.strip() for line in text.splitlines() if line.strip()]
    if not paragraphs or "marking scheme" not in text.lower():
        return []

    flat_text = " ".join(paragraphs)
    qid = _infer_raw_qid(paragraphs, flat_text)
    max_marks_match = RAW_MAX_MARKS_RE.search(flat_text)
    max_marks = float(max_marks_match.group(1)) if max_marks_match else 0.0

    question_text = _extract_question_text(paragraphs)
    items = _extract_marking_scheme_items(paragraphs)
    if not items:
        return []

    if max_marks <= 0:
        max_marks = sum(item.marks for item in items)

    rubric = Rubric(
        qid=qid,
        question_type=_infer_question_type(items),
        question_text=question_text,
        max_marks=max_marks,
        items=items,
    )
    return [rubric]


def _infer_raw_qid(paragraphs: Sequence[str], flat_text: str) -> str:
    explicit_match = RAW_QID_RE.search(flat_text)
    if explicit_match:
        return re.sub(r"\s+", "", explicit_match.group(1).upper())

    header = paragraphs[0] if paragraphs else flat_text
    header_match = re.match(r"^\s*0*(\d{1,3})\b", header)
    if header_match:
        return f"Q{int(header_match.group(1))}"

    bracketed_match = re.search(r"\b0*(\d{1,3})\s*\[\d+(?:\.\d+)?M\]", flat_text, flags=re.IGNORECASE)
    if bracketed_match:
        return f"Q{int(bracketed_match.group(1))}"

    return "QX"


def _extract_question_text(paragraphs: Sequence[str]) -> str:
    question_parts: List[str] = []
    for paragraph in paragraphs:
        lowered = paragraph.lower()
        if "marking scheme" in lowered:
            continue
        if "complete the following functions" in lowered:
            prefix = paragraph.split("Complete the following functions", 1)[0].strip(" :")
            if prefix:
                question_parts.append(prefix)
            break
        if "enqueue_ordeque" in lowered or "dequeue_ordeque" in lowered or "isfull" in lowered:
            break
        question_parts.append(paragraph)
    question_text = " ".join(question_parts).strip()
    question_text = re.sub(r"\bfront\b\s+\b\(ORDeque\)\b", "(ORDeque)", question_text, flags=re.IGNORECASE)
    question_text = re.sub(r"\s+", " ", question_text).strip()
    return question_text


def _extract_marking_scheme_items(paragraphs: Sequence[str]) -> List[RubricItem]:
    items: List[RubricItem] = []
    next_id = 1

    enqueue_context = " ".join(
        paragraph
        for paragraph in paragraphs
        if contains_any(paragraph, ["Enqueue_ORDeque", "Queue Full", "a = 0", "a = 1", "Q[front]", "Q[rear]"])
    )
    dequeue_context = " ".join(
        paragraph
        for paragraph in paragraphs
        if contains_any(paragraph, ["Dequeue_ORDeque", "Queue is empty", "Q[rear]", "return (x)", "rear pointer"])
    )
    isfull_context = " ".join(
        paragraph
        for paragraph in paragraphs
        if contains_any(paragraph, ["IsFull", "front = rear + 1", "rear = Q.size", "Return (True)", "Return (false)"])
    )

    for paragraph in paragraphs:
        lowered = paragraph.lower()
        if "marking scheme" not in lowered:
            continue

        if "queue full" in lowered and "two scenarios" in lowered:
            match = re.search(r"(\d+(?:\.\d+)?)m each", lowered)
            branch_marks = float(match.group(1)) if match else 3.0
            queue_full_match = re.search(r"(\d+(?:\.\d+)?)m\s+for\s+checking[^:.;]*queue\s+full", lowered)
            queue_full_marks = float(queue_full_match.group(1)) if queue_full_match else 1.0
            items.append(
                _make_item(
                    next_id,
                    "Checks if the queue is full using isFull(Q), prints Queue Full, and returns before enqueueing.",
                    queue_full_marks,
                )
            )
            next_id += 1
            items.append(
                _make_item(
                    next_id,
                    _describe_enqueue_branch(enqueue_context, "a = 0", "rear"),
                    branch_marks,
                )
            )
            next_id += 1
            items.append(
                _make_item(
                    next_id,
                    _describe_enqueue_branch(enqueue_context, "a = 1", "front"),
                    branch_marks,
                )
            )
            next_id += 1
            continue

        if "queue is empty" in lowered or "rear pointer" in lowered:
            clause_items = _parse_point_clauses(paragraph)
            if clause_items:
                for marks, description in clause_items:
                    items.append(_make_item(next_id, description, marks))
                    next_id += 1
            else:
                items.append(
                    _make_item(
                        next_id,
                        "Checks if front = rear and reports that the queue is empty before deleting.",
                        1.0,
                    )
                )
                next_id += 1
            continue

        if "two conditions" in lowered:
            condition_marks_match = re.search(r"(\d+(?:\.\d+)?)m\s*\(", lowered)
            condition_marks = float(condition_marks_match.group(1)) / 2.0 if condition_marks_match else 2.0
            conditions = _extract_conditions(isfull_context)
            if len(conditions) >= 2:
                for condition in conditions[:2]:
                    items.append(_make_item(next_id, condition, condition_marks))
                    next_id += 1
            else:
                items.append(
                    _make_item(
                        next_id,
                        "Checks the first ORDeque full condition and returns true when it holds.",
                        condition_marks,
                    )
                )
                next_id += 1
                items.append(
                    _make_item(
                        next_id,
                        "Checks the second ORDeque full condition and otherwise returns false.",
                        condition_marks,
                    )
                )
                next_id += 1
            continue

    return items


def _parse_point_clauses(paragraph: str) -> List[Tuple[float, str]]:
    cleaned = paragraph.replace("Marking Scheme:", "").strip()
    clause_matches = re.findall(r"(\d+(?:\.\d+)?)M\s*for\s*([^:.;]+)", cleaned, flags=re.IGNORECASE)
    clauses: List[Tuple[float, str]] = []
    for marks_text, description in clause_matches:
        description = description.strip(" .:;")
        if description:
            clauses.append((float(marks_text), _normalize_description(description)))
    return clauses


def _describe_enqueue_branch(context: str, branch_marker: str, side: str) -> str:
    lowered = context.lower()
    if branch_marker not in lowered:
        if side == "rear":
            return "For a = 0, inserts from rear by circularly updating rear and then storing x in Q[rear]."
        return "For a = 1, inserts from front by storing x in Q[front] and circularly updating front."

    if side == "rear":
        return "For a = 0, inserts from rear by circularly updating rear: if rear = Q.size then rear = 1 else rear = rear + 1, then stores x in Q[rear]."
    return "For a = 1, inserts from front by setting Q[front] = x and circularly updating front: if front = 1 then front = Q.size else front = front - 1."


def _extract_conditions(context: str) -> List[str]:
    patterns = [
        r"front\s*=\s*rear\s*\+\s*1",
        r"front\s*=\s*1\s*\)\s*and\s*\(\s*rear\s*=\s*q\.size",
    ]
    conditions: List[str] = []
    for pattern in patterns:
        match = re.search(pattern, context, flags=re.IGNORECASE)
        if not match:
            continue
        raw = match.group(0)
        cleaned = raw.replace("q.size", "Q.size")
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        if "rear + 1" in cleaned:
            conditions.append("In IsFull(Q), checks the condition front = rear + 1 and returns true when it holds.")
        else:
            conditions.append("In IsFull(Q), checks the condition front = 1 and rear = Q.size, otherwise returns false.")
    return conditions


def _normalize_description(description: str) -> str:
    lowered = description.lower()
    if "queue is empty" in lowered:
        return "Checks if front = rear and reports that the queue is empty."
    if "extracting element" in lowered:
        return "Extracts the element before updating the pointer by assigning x = Q[rear]."
    if "rear pointer correctly" in lowered:
        return "Updates rear circularly after dequeue: if rear = 1 then rear = Q.size else rear = rear - 1, then returns x."
    return description[:1].upper() + description[1:]


def _make_item(index: int, description: str, marks: float) -> RubricItem:
    return RubricItem(
        item_id=f"r{index}",
        item_type=_infer_item_type(description, description),
        description=description,
        marks=marks,
        required=True,
    )


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
        if label == "code" and contains_any(normalized, ["pseudocode", "algorithm", "enqueue", "mid", "queue", "front", "rear", "return"]):
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
    if item_types == {"code"}:
        return "code_multi_function"
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
