from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "03_code" / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from scoremap.answer_key_ingest import parse_answer_key_file, write_rubrics
from scoremap.pipeline import ScoreMapPipeline, load_answer_sample, load_rubric, resolve_image_path
from scoremap.render import export_prediction_json, render_overlay
from scoremap.schema import Rubric
from scoremap.student_ingest import ingest_student_document


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full SCOREMAP pipeline from a student document and answer key.")
    parser.add_argument("--document", required=True, help="Path to the student answer image/PDF.")
    parser.add_argument("--answer-key", required=True, help="Path to the answer-key PDF/text file or a single rubric JSON.")
    parser.add_argument("--qid", default=None, help="Question id to grade when the answer key contains multiple questions.")
    parser.add_argument("--output-dir", required=True, help="Directory where parsed rubrics, ingested answer JSON, and predictions will be written.")
    parser.add_argument("--variant", choices=["generic", "no_graph", "scoremap"], default="scoremap")
    parser.add_argument("--backend", choices=["trocr", "regions_json"], default="trocr")
    parser.add_argument("--model-name", default="microsoft/trocr-base-handwritten")
    parser.add_argument("--sidecar", default=None, help="Sidecar answer/region JSON for the regions_json backend.")
    parser.add_argument("--sample-id", default=None)
    parser.add_argument("--writer-id", default="student")
    parser.add_argument("--page", type=int, default=0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rubric = select_rubric(Path(args.answer_key).resolve(), output_dir / "parsed_rubrics", args.qid)
    answer_path = ingest_student_document(
        source_path=Path(args.document).resolve(),
        rubric=rubric,
        output_dir=output_dir / "ingested",
        sample_id=args.sample_id,
        writer_id=args.writer_id,
        page_number=args.page,
        backend=args.backend,
        model_name=args.model_name,
        sidecar_path=Path(args.sidecar).resolve() if args.sidecar else None,
    )
    answer = load_answer_sample(answer_path)
    pipeline = ScoreMapPipeline.from_config(PROJECT_ROOT / "03_code" / "configs" / "default.yaml", args.variant)
    result = pipeline.run(answer, rubric)

    image_path = resolve_image_path(answer, project_root=PROJECT_ROOT)
    render_overlay(image_path, result, output_dir / "overlay.png")
    export_prediction_json(result, output_dir / "prediction.json")

    summary = {
        "sample_id": result.sample_id,
        "qid": result.qid,
        "total_score": result.total_score,
        "max_marks": result.max_marks,
        "review_flag": result.review_flag,
        "regions": len(answer.regions),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def select_rubric(answer_key_path: Path, parsed_output_dir: Path, qid: str | None) -> Rubric:
    if answer_key_path.suffix.lower() == ".json":
        return load_rubric(answer_key_path)

    rubrics = parse_answer_key_file(answer_key_path)
    write_rubrics(rubrics, parsed_output_dir)
    if len(rubrics) == 1:
        return rubrics[0]

    if qid:
        for rubric in rubrics:
            if rubric.qid.lower() == qid.lower():
                return rubric
        raise ValueError(f"Question id {qid} was not found in {answer_key_path}.")

    available = ", ".join(rubric.qid for rubric in rubrics)
    raise ValueError(f"{answer_key_path.name} contains multiple questions ({available}). Re-run with --qid.")


if __name__ == "__main__":
    main()
