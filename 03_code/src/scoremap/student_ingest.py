from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import fitz
import numpy as np
from PIL import Image, ImageOps

from .answer_key_ingest import _get_easyocr_reader
from .pipeline import load_answer_sample
from .schema import Region, Rubric
from .text_utils import normalize_text, overlap_score


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
CODE_HINT_KEYWORDS = {
    "enqueue",
    "dequeue",
    "ordeque",
    "isfull",
    "front",
    "rear",
    "queue",
    "return",
    "print",
    "true",
    "false",
    "empty",
    "full",
    "size",
    "insert",
    "delete",
}
COMMON_LEXICON = {
    "enqueue",
    "dequeue",
    "ordeque",
    "isfull",
    "front",
    "rear",
    "queue",
    "print",
    "return",
    "true",
    "false",
    "size",
    "insert",
    "delete",
    "empty",
    "full",
    "student",
    "array",
    "from",
    "rear",
    "front",
    "if",
    "else",
    "and",
    "or",
    "q",
    "x",
}
TOKEN_BLACKLIST = {"if", "else", "and", "or", "q", "x", "a"}


@dataclass
class DetectedLine:
    bbox: Tuple[int, int, int, int]
    text: str
    confidence: float
    column_index: int
    line_index: int


@dataclass
class ReferenceHints:
    snippets: List[str]
    lexicon: List[str]


class TrOCRTranscriber:
    def __init__(self, model_name: str = "microsoft/trocr-base-handwritten", device: Optional[str] = None) -> None:
        import torch
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel

        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        try:
            self.processor = TrOCRProcessor.from_pretrained(model_name, local_files_only=True)
            self.model = VisionEncoderDecoderModel.from_pretrained(model_name, local_files_only=True)
        except Exception:
            self.processor = TrOCRProcessor.from_pretrained(model_name)
            self.model = VisionEncoderDecoderModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def transcribe(self, image: Image.Image) -> Tuple[str, float]:
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.to(self.device)
        with self._torch.no_grad():
            generated = self.model.generate(
                pixel_values,
                num_beams=4,
                max_new_tokens=96,
                return_dict_in_generate=True,
                output_scores=True,
            )
        text = self.processor.batch_decode(generated.sequences, skip_special_tokens=True)[0].strip()
        confidence = 0.75
        sequence_scores = getattr(generated, "sequences_scores", None)
        if sequence_scores is not None and len(sequence_scores) > 0:
            confidence = float(self._torch.exp(sequence_scores[0]).clamp(0.0, 1.0).item())
        return text, max(0.0, min(0.99, confidence))


def ingest_student_document(
    source_path: Path,
    rubric: Rubric,
    output_dir: Path,
    sample_id: Optional[str] = None,
    writer_id: str = "student",
    page_number: int = 0,
    backend: str = "hybrid",
    model_name: str = "microsoft/trocr-base-handwritten",
    sidecar_path: Optional[Path] = None,
    pdf_dpi: int = 180,
    reference_text: Optional[str] = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    page_image = render_document_page(source_path, page_number=page_number, dpi=pdf_dpi)
    normalized_page = normalize_page_image(page_image)
    page_path = output_dir / "page.png"
    normalized_page.save(page_path)

    if backend == "regions_json":
        if sidecar_path is None:
            raise ValueError("The regions_json backend needs --sidecar pointing to an answer JSON or region list JSON.")
        regions = load_regions_from_sidecar(sidecar_path)
    elif backend == "trocr":
        regions = transcribe_regions(normalized_page, rubric, model_name=model_name)
    elif backend in {"hybrid", "auto"}:
        regions = transcribe_regions_hybrid(
            normalized_page,
            rubric,
            model_name=model_name,
            reference_text=reference_text,
        )
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    payload = {
        "sample_id": sample_id or source_path.stem,
        "writer_id": writer_id,
        "qid": rubric.qid,
        "question_text": rubric.question_text,
        "image_path": "page.png",
        "regions": [region_to_payload(region) for region in regions],
        "source_document": str(source_path.resolve()),
        "page_number": page_number,
        "transcription_backend": backend,
    }
    answer_path = output_dir / "answer.json"
    answer_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return answer_path


def render_document_page(source_path: Path, page_number: int = 0, dpi: int = 180) -> Image.Image:
    suffix = source_path.suffix.lower()
    if suffix in IMAGE_SUFFIXES:
        return ImageOps.exif_transpose(Image.open(source_path)).convert("RGB")
    if suffix == ".pdf":
        document = fitz.open(source_path)
        if page_number < 0 or page_number >= len(document):
            raise IndexError(f"Page {page_number} is out of range for {source_path}.")
        page = document[page_number]
        zoom = max(1.0, dpi / 72.0)
        pixmap = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        return Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)
    raise ValueError(f"Unsupported student document type: {source_path}")


def normalize_page_image(image: Image.Image) -> Image.Image:
    page = ImageOps.exif_transpose(image).convert("L")
    page = ImageOps.autocontrast(page)
    return page.convert("RGB")


def transcribe_regions(image: Image.Image, rubric: Rubric, model_name: str) -> List[Region]:
    recognizer = TrOCRTranscriber(model_name=model_name)
    regions: List[Region] = []
    for line_index, bbox in enumerate(segment_page_into_lines(image), start=1):
        crop = crop_box(image, bbox, padding=10)
        text, confidence = recognizer.transcribe(crop)
        cleaned = " ".join(text.split())
        if not cleaned or len(cleaned) < 2:
            continue
        if looks_like_prompt(cleaned, rubric.question_text, bbox[1], image.height):
            continue
        metadata: Dict[str, object] = {
            "line_index": line_index,
            "ocr_confidence": round(confidence, 4),
            "transcription_backend": "trocr",
        }
        diagram_segments = extract_diagram_segments(cleaned)
        if diagram_segments:
            metadata["diagram_segments"] = diagram_segments
        regions.append(
            Region(
                region_id=f"n{len(regions) + 1}",
                bbox=bbox,
                text=cleaned,
                metadata=metadata,
            )
        )
    return regions


def transcribe_regions_hybrid(
    image: Image.Image,
    rubric: Rubric,
    model_name: str,
    reference_text: Optional[str] = None,
) -> List[Region]:
    hints = build_reference_hints(rubric, reference_text or "")
    detected_lines = detect_lines_with_easyocr(image)
    if not detected_lines:
        return transcribe_regions(image, rubric, model_name=model_name)

    recognizer: Optional[TrOCRTranscriber] = None
    regions: List[Region] = []

    for line in detected_lines:
        easy_text = cleanup_transcription_text(line.text)
        easy_text = canonicalize_text(easy_text, hints)
        easy_score = candidate_alignment_score(easy_text, hints)

        trocr_text = ""
        trocr_confidence = 0.0
        if easy_score < 0.66:
            if recognizer is None:
                recognizer = TrOCRTranscriber(model_name=model_name)
            crop = crop_box(image, line.bbox, padding=12)
            trocr_text, trocr_confidence = recognizer.transcribe(crop)
            trocr_text = canonicalize_text(cleanup_transcription_text(trocr_text), hints)

        chosen_text, chosen_confidence, chosen_source = select_best_line_text(
            easy_text=easy_text,
            easy_confidence=line.confidence,
            trocr_text=trocr_text,
            trocr_confidence=trocr_confidence,
            hints=hints,
        )
        if not chosen_text or len(normalize_text(chosen_text)) < 2:
            continue
        if looks_like_prompt(chosen_text, rubric.question_text, line.bbox[1], image.height):
            continue

        metadata: Dict[str, object] = {
            "line_index": line.line_index,
            "column_index": line.column_index,
            "ocr_confidence": round(chosen_confidence, 4),
            "transcription_backend": f"hybrid:{chosen_source}",
            "easyocr_text": easy_text,
        }
        if trocr_text:
            metadata["trocr_text"] = trocr_text
        diagram_segments = extract_diagram_segments(chosen_text)
        if diagram_segments:
            metadata["diagram_segments"] = diagram_segments
        regions.append(
            Region(
                region_id=f"n{len(regions) + 1}",
                bbox=line.bbox,
                text=chosen_text,
                metadata=metadata,
            )
        )

    if not regions:
        return transcribe_regions(image, rubric, model_name=model_name)
    return merge_line_regions(regions)


def select_best_line_text(
    easy_text: str,
    easy_confidence: float,
    trocr_text: str,
    trocr_confidence: float,
    hints: ReferenceHints,
) -> Tuple[str, float, str]:
    best_text = easy_text
    best_confidence = easy_confidence
    best_source = "easyocr"
    best_score = candidate_alignment_score(easy_text, hints) + (0.1 * easy_confidence)

    if trocr_text:
        trocr_score = candidate_alignment_score(trocr_text, hints) + (0.1 * trocr_confidence)
        if trocr_score > best_score + 0.03 or (len(normalize_text(trocr_text)) > len(normalize_text(best_text)) and trocr_score >= best_score):
            best_text = trocr_text
            best_confidence = trocr_confidence
            best_source = "trocr"
            best_score = trocr_score

    return best_text, best_confidence, best_source


def build_reference_hints(rubric: Rubric, reference_text: str) -> ReferenceHints:
    snippets: List[str] = []
    seeds = [rubric.question_text, *[item.description for item in rubric.items], *extract_reference_snippets(reference_text)]
    for seed in seeds:
        cleaned = cleanup_transcription_text(seed)
        if not cleaned:
            continue
        if not contains_code_hint(cleaned):
            continue
        if any(similarity_score(cleaned, existing) >= 0.94 for existing in snippets):
            continue
        snippets.append(cleaned)

    lexicon = set(COMMON_LEXICON)
    for snippet in snippets:
        for token in re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", snippet):
            normalized = token.lower()
            if len(normalized) >= 3:
                lexicon.add(normalized)

    return ReferenceHints(snippets=snippets, lexicon=sorted(lexicon))


def extract_reference_snippets(reference_text: str) -> List[str]:
    if not reference_text:
        return []

    compact = " ".join(reference_text.replace("\r", "\n").split())
    snippets: List[str] = []

    explicit_patterns = [
        r"Enqueue[_ ]ORDeque\s*\([^)]*\)",
        r"Dequeue[_ ]ORDeque\s*\([^)]*\)",
        r"IsFull\s*\([^)]*\)",
        r"If\s*\(isFull\s*\(Q\)\)\s*Print\s*\([^)]*\)\s*;?\s*Return",
        r"If\s*\(front\s*=\s*rear\)",
        r"Print\s*\(\"?Q\s+is\s+empty\"?\)",
        r"X\s*=\s*Q\[rear\]",
        r"Q\[rear\]\s*=\s*X",
        r"Q\[front\]\s*=\s*X",
        r"If\s*\(rear\s*=\s*Q\.size\)",
        r"If\s*\(rear\s*=\s*1\)",
        r"If\s*\(front\s*=\s*1\)",
        r"Return\s*\(True\)",
        r"Else\s*Return\s*\(false\)",
        r"If\s*\(front\s*=\s*rear\s*1\)\s*or\s*\(\(front\s*=\s*1\)\s*and\s*\(rear\s*=\s*Q\.size\)\)",
    ]
    for pattern in explicit_patterns:
        for match in re.findall(pattern, compact, flags=re.IGNORECASE):
            cleaned = cleanup_transcription_text(match)
            if cleaned and all(similarity_score(cleaned, existing) < 0.94 for existing in snippets):
                snippets.append(cleaned)

    parts = re.split(
        r"(?i)(?=enqueue[_ ]ordeque)|(?=dequeue[_ ]ordeque)|(?=isfull\s*\()|(?=if\s*\()|(?=else\b)|(?=return\b)|(?=print\s*\()|(?=q\[[a-z]+\])",
        compact,
    )
    for part in parts:
        cleaned = cleanup_transcription_text(part)
        if not cleaned or len(cleaned) < 4:
            continue
        if "marking scheme" in cleaned.lower():
            continue
        if not contains_code_hint(cleaned):
            continue
        if len(cleaned) > 180:
            cleaned = cleaned[:180].rsplit(" ", 1)[0]
        if any(similarity_score(cleaned, existing) >= 0.94 for existing in snippets):
            continue
        snippets.append(cleaned)
    return snippets


def detect_lines_with_easyocr(image: Image.Image) -> List[DetectedLine]:
    reader = _get_easyocr_reader()
    results = reader.readtext(np.asarray(image), detail=1, paragraph=False)
    words: List[Dict[str, float | int | str]] = []
    for box, text, confidence in results:
        cleaned = cleanup_transcription_text(str(text))
        xs = [point[0] for point in box]
        ys = [point[1] for point in box]
        x1, x2 = int(min(xs)), int(max(xs))
        y1, y2 = int(min(ys)), int(max(ys))
        width = x2 - x1
        height = y2 - y1
        if not cleaned:
            continue
        if width < 15 or height < 10:
            continue
        if confidence < 0.02 and width < 30:
            continue
        words.append(
            {
                "text": cleaned,
                "confidence": float(confidence),
                "x1": x1,
                "x2": x2,
                "y1": y1,
                "y2": y2,
                "xc": (x1 + x2) / 2.0,
                "yc": (y1 + y2) / 2.0,
                "height": height,
            }
        )

    if not words:
        return []

    boundaries = infer_column_boundaries(words, image.width)
    detected_lines: List[DetectedLine] = []
    next_line_index = 1

    for column_index, (col_left, col_right) in enumerate(boundaries):
        column_words = [word for word in words if col_left <= float(word["xc"]) < col_right]
        if not column_words:
            continue
        median_height = float(np.median([float(word["height"]) for word in column_words]))
        line_threshold = max(18.0, median_height * 0.9)
        grouped: List[Dict[str, object]] = []

        for word in sorted(column_words, key=lambda item: (float(item["yc"]), int(item["x1"]))):
            placed = False
            for group in reversed(grouped[-4:]):
                if abs(float(word["yc"]) - float(group["yc"])) <= line_threshold:
                    group["words"].append(word)
                    ys = [float(entry["yc"]) for entry in group["words"]]
                    group["yc"] = sum(ys) / len(ys)
                    placed = True
                    break
            if not placed:
                grouped.append({"yc": float(word["yc"]), "words": [word]})

        for group in grouped:
            line_words = sorted(group["words"], key=lambda item: int(item["x1"]))
            text = " ".join(str(word["text"]) for word in line_words)
            x1 = max(0, min(int(word["x1"]) for word in line_words) - 12)
            y1 = max(0, min(int(word["y1"]) for word in line_words) - 10)
            x2 = min(image.width - 1, max(int(word["x2"]) for word in line_words) + 12)
            y2 = min(image.height - 1, max(int(word["y2"]) for word in line_words) + 10)
            confidence = float(np.mean([float(word["confidence"]) for word in line_words]))
            detected_lines.append(
                DetectedLine(
                    bbox=(x1, y1, x2, y2),
                    text=text,
                    confidence=confidence,
                    column_index=column_index,
                    line_index=next_line_index,
                )
            )
            next_line_index += 1

    return sorted(detected_lines, key=lambda item: (item.column_index, item.bbox[1], item.bbox[0]))


def infer_column_boundaries(words: Sequence[Dict[str, float | int | str]], image_width: int) -> List[Tuple[int, int]]:
    if len(words) < 8:
        return [(0, image_width)]

    centers = sorted(float(word["xc"]) for word in words)
    best_gap = 0.0
    best_boundary: Optional[float] = None
    for index in range(4, len(centers) - 4):
        gap = centers[index] - centers[index - 1]
        boundary = (centers[index] + centers[index - 1]) / 2.0
        if image_width * 0.22 <= boundary <= image_width * 0.78 and gap > best_gap:
            left_count = sum(1 for center in centers if center < boundary)
            right_count = len(centers) - left_count
            if left_count >= 4 and right_count >= 4:
                best_gap = gap
                best_boundary = boundary

    if best_boundary is not None and best_gap >= image_width * 0.12:
        boundary = int(best_boundary)
        return [(0, boundary), (boundary, image_width)]
    return [(0, image_width)]


def cleanup_transcription_text(text: str) -> str:
    cleaned = text.replace("\r", " ").replace("\n", " ")
    replacements = {
        "Q_is": "Q is",
        "Qis": "Q is",
        "Qetuh": "Return",
        "Qexlh": "Return",
        "Elce": "Else",
        "elve": "else",
        "YeAY": "rear",
        "YeY": "rear",
        "Yea": "rear",
        "Renr": "rear",
        "REQUENE": "Dequeue",
        "Dequeld": "Deque",
    }
    for source, target in replacements.items():
        cleaned = cleaned.replace(source, target)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" `\"'")
    return cleaned


def canonicalize_text(text: str, hints: ReferenceHints) -> str:
    if not text:
        return ""
    corrected = correct_tokens(text, hints.lexicon)
    snapped_text, snapped_score = snap_to_reference_snippet(corrected, hints.snippets)
    if snapped_score >= 0.88:
        return snapped_text
    if is_header_text(snapped_text) and snapped_score >= 0.58:
        return snapped_text
    if snapped_score >= 0.8 and len(normalize_text(corrected).split()) >= 3:
        return snapped_text
    return corrected


def correct_tokens(text: str, lexicon: Sequence[str]) -> str:
    if not text:
        return ""
    if not lexicon:
        return text

    def replace(match: re.Match[str]) -> str:
        token = match.group(0)
        lowered = token.lower()
        if len(lowered) < 3 or lowered in TOKEN_BLACKLIST or lowered in lexicon:
            return token
        candidate = max(lexicon, key=lambda word: SequenceMatcher(None, lowered, word).ratio())
        if SequenceMatcher(None, lowered, candidate).ratio() >= 0.79:
            return candidate
        return token

    return re.sub(r"[A-Za-z_][A-Za-z0-9_]*", replace, text)


def snap_to_reference_snippet(text: str, snippets: Sequence[str]) -> Tuple[str, float]:
    if not text or not snippets:
        return text, 0.0
    best = max(snippets, key=lambda snippet: similarity_score(text, snippet))
    return best, similarity_score(text, best)


def candidate_alignment_score(text: str, hints: ReferenceHints) -> float:
    if not text:
        return 0.0
    normalized = normalize_text(text)
    if not normalized:
        return 0.0
    if not hints.snippets:
        return 0.3

    best_similarity = max(similarity_score(text, snippet) for snippet in hints.snippets)
    best_overlap = max(overlap_score(text, snippet) for snippet in hints.snippets)
    token_hits = sum(1 for token in normalized.split() if token in hints.lexicon)
    keyword_score = min(1.0, token_hits / 4.0)
    return (0.4 * best_similarity) + (0.4 * best_overlap) + (0.2 * keyword_score)


def similarity_score(text_a: str, text_b: str) -> float:
    return SequenceMatcher(None, normalize_text(text_a), normalize_text(text_b)).ratio()


def contains_code_hint(text: str) -> bool:
    normalized = normalize_text(text)
    return any(keyword in normalized for keyword in CODE_HINT_KEYWORDS)


def is_header_text(text: str) -> bool:
    normalized = normalize_text(text)
    return any(keyword in normalized for keyword in ["enqueue", "dequeue", "isfull", "ordeque"])


def merge_line_regions(regions: Sequence[Region]) -> List[Region]:
    if not regions:
        return []

    sorted_regions = sorted(
        regions,
        key=lambda region: (
            int(region.metadata.get("column_index", 0)),
            region.bbox[1],
            region.bbox[0],
        ),
    )
    grouped: List[List[Region]] = [[sorted_regions[0]]]

    for region in sorted_regions[1:]:
        current = grouped[-1]
        previous = current[-1]
        previous_column = int(previous.metadata.get("column_index", 0))
        current_column = int(region.metadata.get("column_index", 0))
        vertical_gap = region.bbox[1] - previous.bbox[3]
        if current_column != previous_column or vertical_gap > 70 or is_header_text(region.text):
            grouped.append([region])
        else:
            current.append(region)

    merged: List[Region] = []
    for index, group in enumerate(grouped, start=1):
        if len(group) == 1:
            single = group[0]
            single.region_id = f"n{index}"
            merged.append(single)
            continue

        bbox = (
            min(region.bbox[0] for region in group),
            min(region.bbox[1] for region in group),
            max(region.bbox[2] for region in group),
            max(region.bbox[3] for region in group),
        )
        merged_text = " ".join(region.text for region in group)
        confidences = [float(region.metadata.get("ocr_confidence", 0.0)) for region in group]
        metadata = {
            "line_span": [int(group[0].metadata.get("line_index", 0)), int(group[-1].metadata.get("line_index", 0))],
            "column_index": int(group[0].metadata.get("column_index", 0)),
            "ocr_confidence": round(float(np.mean(confidences)) if confidences else 0.0, 4),
            "transcription_backend": group[0].metadata.get("transcription_backend", "hybrid"),
        }
        merged.append(Region(region_id=f"n{index}", bbox=bbox, text=merged_text, metadata=metadata))

    return sorted(merged, key=lambda region: (region.bbox[1], region.bbox[0]))


def load_regions_from_sidecar(sidecar_path: Path) -> List[Region]:
    payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "regions" in payload:
        region_records = payload["regions"]
    elif isinstance(payload, list):
        region_records = payload
    else:
        raise ValueError("Sidecar JSON must be an answer sample or a list of region records.")

    regions: List[Region] = []
    for index, record in enumerate(region_records, start=1):
        metadata = dict(record.get("metadata", {}))
        metadata.setdefault("line_index", index)
        metadata.setdefault("transcription_backend", "regions_json")
        regions.append(
            Region(
                region_id=record.get("id", f"n{index}"),
                bbox=tuple(record["bbox"]),
                text=record["text"],
                type_hint=record.get("type_hint"),
                metadata=metadata,
            )
        )
    return regions


def region_to_payload(region: Region) -> Dict[str, object]:
    payload: Dict[str, object] = {
        "id": region.region_id,
        "bbox": list(region.bbox),
        "text": region.text,
        "metadata": region.metadata,
    }
    if region.type_hint:
        payload["type_hint"] = region.type_hint
    return payload


def segment_page_into_lines(image: Image.Image) -> List[Tuple[int, int, int, int]]:
    gray = ImageOps.grayscale(image)
    array = np.asarray(gray)
    threshold = otsu_threshold(array)
    ink_mask = array < threshold

    row_projection = ink_mask.sum(axis=1)
    row_threshold = max(8, int(array.shape[1] * 0.004))
    row_runs = merge_runs(find_runs(row_projection > row_threshold), max_gap=14)

    boxes: List[Tuple[int, int, int, int]] = []
    for top, bottom in row_runs:
        if bottom - top < 10:
            continue
        region_mask = ink_mask[top : bottom + 1, :]
        col_projection = region_mask.sum(axis=0)
        active_cols = np.where(col_projection > max(2, int(region_mask.shape[0] * 0.02)))[0]
        if active_cols.size == 0:
            continue
        left = max(0, int(active_cols[0]) - 8)
        right = min(array.shape[1] - 1, int(active_cols[-1]) + 8)
        boxes.append((left, max(0, top - 4), right, min(array.shape[0] - 1, bottom + 4)))

    return sorted(boxes, key=lambda box: (box[1], box[0]))


def crop_box(image: Image.Image, bbox: Tuple[int, int, int, int], padding: int = 0) -> Image.Image:
    x1, y1, x2, y2 = bbox
    return image.crop((max(0, x1 - padding), max(0, y1 - padding), x2 + padding, y2 + padding))


def looks_like_prompt(text: str, question_text: str, top: int, image_height: int) -> bool:
    if not question_text or top > image_height * 0.28:
        return False
    normalized = text.lower().lstrip("\"'`([{:- ")
    return overlap_score(text, question_text) >= 0.72 or normalized.startswith("question")


def extract_diagram_segments(text: str) -> List[str]:
    segments = []
    for token in text.split("|"):
        token = token.strip()
        if token and token[:1].lower() == "p":
            segments.append(token)
    return segments


def find_runs(mask: Sequence[bool]) -> List[Tuple[int, int]]:
    runs: List[Tuple[int, int]] = []
    start: Optional[int] = None
    for index, active in enumerate(mask):
        if active and start is None:
            start = index
        elif not active and start is not None:
            runs.append((start, index - 1))
            start = None
    if start is not None:
        runs.append((start, len(mask) - 1))
    return runs


def merge_runs(runs: Iterable[Tuple[int, int]], max_gap: int) -> List[Tuple[int, int]]:
    merged: List[Tuple[int, int]] = []
    for start, end in runs:
        if not merged:
            merged.append((start, end))
            continue
        prev_start, prev_end = merged[-1]
        if start - prev_end <= max_gap:
            merged[-1] = (prev_start, end)
        else:
            merged.append((start, end))
    return merged


def otsu_threshold(array: np.ndarray) -> int:
    histogram = np.bincount(array.ravel(), minlength=256).astype(np.float64)
    total = array.size
    weighted_sum = float(np.dot(np.arange(256), histogram))
    background_weight = 0.0
    background_sum = 0.0
    best_threshold = 127
    best_variance = -math.inf

    for threshold in range(256):
        background_weight += histogram[threshold]
        if background_weight == 0:
            continue
        foreground_weight = total - background_weight
        if foreground_weight == 0:
            break
        background_sum += threshold * histogram[threshold]
        background_mean = background_sum / background_weight
        foreground_mean = (weighted_sum - background_sum) / foreground_weight
        between_class_variance = background_weight * foreground_weight * (background_mean - foreground_mean) ** 2
        if between_class_variance > best_variance:
            best_variance = between_class_variance
            best_threshold = threshold
    return best_threshold


def load_ingested_sample(path: Path):
    return load_answer_sample(path)
