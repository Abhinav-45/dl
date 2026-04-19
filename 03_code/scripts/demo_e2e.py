from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "03_code" / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from scoremap.answer_key_ingest import parse_answer_key_file, write_rubrics
from scoremap.pipeline import ScoreMapPipeline, load_answer_sample, resolve_image_path
from scoremap.render import export_prediction_json, render_overlay
from scoremap.student_ingest import ingest_student_document


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full ingestion-to-grading demo on the packaged sample assets.")
    parser.add_argument("--sample", default="writer03_q1", help="Packaged demo sample id.")
    parser.add_argument("--backend", choices=["regions_json", "trocr"], default="regions_json")
    parser.add_argument("--model-name", default="microsoft/trocr-base-handwritten")
    args = parser.parse_args()

    answer_key_path = PROJECT_ROOT / "04_data" / "sample_inputs" / "answer_keys" / "scoremap_answer_key.pdf"
    rubrics = parse_answer_key_file(answer_key_path)
    parsed_dir = PROJECT_ROOT / "06_demo" / "e2e_outputs" / args.sample / "parsed_rubrics"
    write_rubrics(rubrics, parsed_dir)

    target_qid = args.sample.split("_")[-1].upper()
    rubric = next(r for r in rubrics if r.qid.lower() == target_qid.lower())
    input_image = PROJECT_ROOT / "04_data" / "sample_inputs" / "images" / f"{args.sample}.png"
    output_dir = PROJECT_ROOT / "06_demo" / "e2e_outputs" / args.sample

    sidecar = PROJECT_ROOT / "04_data" / "sample_inputs" / "answers" / f"{args.sample}.json"
    answer_path = ingest_student_document(
        source_path=input_image,
        rubric=rubric,
        output_dir=output_dir / "ingested",
        sample_id=args.sample,
        writer_id=args.sample.split("_")[0],
        backend=args.backend,
        model_name=args.model_name,
        sidecar_path=sidecar if args.backend == "regions_json" else None,
    )

    answer = load_answer_sample(answer_path)
    pipeline = ScoreMapPipeline.from_config(PROJECT_ROOT / "03_code" / "configs" / "default.yaml", "scoremap")
    result = pipeline.run(answer, rubric)
    image_path = resolve_image_path(answer, project_root=PROJECT_ROOT)
    render_overlay(image_path, result, output_dir / "overlay.png")
    export_prediction_json(result, output_dir / "prediction.json")
    print(f"{args.sample}: {result.total_score}/{result.max_marks} review={result.review_flag} backend={args.backend}")


if __name__ == "__main__":
    main()
