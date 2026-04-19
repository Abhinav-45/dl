from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "03_code" / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from scoremap.answer_key_ingest import read_answer_key_text
from scoremap.pipeline import load_answer_sample, load_rubric
from scoremap.student_ingest import ingest_student_document


def main() -> None:
    parser = argparse.ArgumentParser(description="Turn a student image/PDF into the answer JSON format SCOREMAP expects.")
    parser.add_argument("--input", required=True, help="Path to a scanned answer image or PDF.")
    parser.add_argument("--rubric", required=True, help="Path to a rubric JSON file.")
    parser.add_argument("--output-dir", required=True, help="Directory for the generated page image and answer JSON.")
    parser.add_argument("--sample-id", default=None, help="Optional sample id for the generated answer JSON.")
    parser.add_argument("--writer-id", default="student", help="Writer/student identifier to embed in the answer JSON.")
    parser.add_argument("--page", type=int, default=0, help="Zero-based page number when the input is a PDF.")
    parser.add_argument("--backend", choices=["hybrid", "auto", "trocr", "regions_json"], default="hybrid")
    parser.add_argument("--model-name", default="microsoft/trocr-base-handwritten", help="TrOCR model name for the trocr backend.")
    parser.add_argument("--sidecar", default=None, help="Sidecar answer/region JSON for the regions_json backend.")
    parser.add_argument("--answer-key", default=None, help="Optional raw answer-key file to guide hybrid OCR.")
    args = parser.parse_args()

    rubric = load_rubric(Path(args.rubric).resolve())
    reference_text = read_answer_key_text(Path(args.answer_key).resolve()) if args.answer_key else None
    answer_path = ingest_student_document(
        source_path=Path(args.input).resolve(),
        rubric=rubric,
        output_dir=Path(args.output_dir).resolve(),
        sample_id=args.sample_id,
        writer_id=args.writer_id,
        page_number=args.page,
        backend=args.backend,
        model_name=args.model_name,
        sidecar_path=Path(args.sidecar).resolve() if args.sidecar else None,
        reference_text=reference_text,
    )
    sample = load_answer_sample(answer_path)
    print(f"Wrote {answer_path}")
    print(f"Extracted {len(sample.regions)} region(s) for {sample.sample_id}")


if __name__ == "__main__":
    main()
