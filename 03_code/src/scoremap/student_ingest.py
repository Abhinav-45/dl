from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import fitz
import numpy as np
from PIL import Image, ImageOps

from .pipeline import load_answer_sample
from .schema import Region, Rubric
from .text_utils import overlap_score


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


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
    backend: str = "trocr",
    model_name: str = "microsoft/trocr-base-handwritten",
    sidecar_path: Optional[Path] = None,
    pdf_dpi: int = 180,
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
