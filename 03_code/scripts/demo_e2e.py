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


DEMO_SAMPLES = {
    "ordeque_demo": {
        "answer_key_path": PROJECT_ROOT / "06_demo" / "ordeque_demo" / "answer_key_structured.pdf",
        "document_path": PROJECT_ROOT / "06_demo" / "ordeque_demo" / "student_answer.jpeg",
        "sidecar_path": PROJECT_ROOT / "06_demo" / "ordeque_demo" / "student_answer_sidecar.json",
        "qid": "ORQ4",
        "sample_id": "ordeque_demo",
        "writer_id": "student_demo",
        "output_dir": PROJECT_ROOT / "06_demo" / "ordeque_demo" / "outputs",
    }
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full ingestion-to-grading demo on the packaged demo assets.")
    parser.add_argument("--sample", default="ordeque_demo", help="Demo sample id. Use ordeque_demo or one of the legacy packaged samples.")
    parser.add_argument("--backend", choices=["regions_json", "trocr"], default="regions_json")
    parser.add_argument("--model-name", default="microsoft/trocr-base-handwritten")
    args = parser.parse_args()

    config = DEMO_SAMPLES.get(args.sample)
    if config is None:
        run_legacy_sample(args.sample, args.backend, args.model_name)
        return

    answer_key_path = config["answer_key_path"]
    rubrics = parse_answer_key_file(answer_key_path)
    parsed_dir = config["output_dir"] / args.backend / "parsed_rubrics"
    write_rubrics(rubrics, parsed_dir)
    rubric = next(r for r in rubrics if r.qid.lower() == str(config["qid"]).lower())

    output_dir = config["output_dir"] / args.backend
    answer_path = ingest_student_document(
        source_path=config["document_path"],
        rubric=rubric,
        output_dir=output_dir / "ingested",
        sample_id=str(config["sample_id"]),
        writer_id=str(config["writer_id"]),
        backend=args.backend,
        model_name=args.model_name,
        sidecar_path=config["sidecar_path"] if args.backend == "regions_json" else None,
    )

    answer = load_answer_sample(answer_path)
    pipeline = ScoreMapPipeline.from_config(PROJECT_ROOT / "03_code" / "configs" / "default.yaml", "scoremap")
    result = pipeline.run(answer, rubric)
    image_path = resolve_image_path(answer, project_root=PROJECT_ROOT)
    render_overlay(image_path, result, output_dir / "overlay.png")
    export_prediction_json(result, output_dir / "prediction.json")
    print(f"{args.sample}: {result.total_score}/{result.max_marks} review={result.review_flag} backend={args.backend}")


def run_legacy_sample(sample: str, backend: str, model_name: str) -> None:
    answer_key_path = PROJECT_ROOT / "04_data" / "sample_inputs" / "answer_keys" / "scoremap_answer_key.pdf"
    rubrics = parse_answer_key_file(answer_key_path)
    output_dir = PROJECT_ROOT / "06_demo" / "e2e_outputs" / sample
    parsed_dir = output_dir / "parsed_rubrics"
    write_rubrics(rubrics, parsed_dir)

    target_qid = sample.split("_")[-1].upper()
    rubric = next(r for r in rubrics if r.qid.lower() == target_qid.lower())
    input_image = PROJECT_ROOT / "04_data" / "sample_inputs" / "images" / f"{sample}.png"
    sidecar = PROJECT_ROOT / "04_data" / "sample_inputs" / "answers" / f"{sample}.json"
    answer_path = ingest_student_document(
        source_path=input_image,
        rubric=rubric,
        output_dir=output_dir / "ingested",
        sample_id=sample,
        writer_id=sample.split("_")[0],
        backend=backend,
        model_name=model_name,
        sidecar_path=sidecar if backend == "regions_json" else None,
    )

    answer = load_answer_sample(answer_path)
    pipeline = ScoreMapPipeline.from_config(PROJECT_ROOT / "03_code" / "configs" / "default.yaml", "scoremap")
    result = pipeline.run(answer, rubric)
    image_path = resolve_image_path(answer, project_root=PROJECT_ROOT)
    render_overlay(image_path, result, output_dir / "overlay.png")
    export_prediction_json(result, output_dir / "prediction.json")
    print(f"{sample}: {result.total_score}/{result.max_marks} review={result.review_flag} backend={backend}")


if __name__ == "__main__":
    main()
