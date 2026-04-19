from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "03_code" / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from scoremap.pipeline import ScoreMapPipeline, load_answer_sample, load_rubric
from scoremap.render import export_prediction_json, render_overlay


def main() -> None:
    demo_dir = PROJECT_ROOT / "06_demo" / "demo_inputs"
    output_dir = PROJECT_ROOT / "06_demo" / "demo_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    pipeline = ScoreMapPipeline.from_config(PROJECT_ROOT / "03_code" / "configs" / "default.yaml", "scoremap")

    for answer_path in sorted(demo_dir.glob("*.json")):
        answer = load_answer_sample(answer_path)
        rubric = load_rubric(PROJECT_ROOT / "04_data" / "sample_inputs" / "rubrics" / f"{answer.qid}.json")
        result = pipeline.run(answer, rubric)
        sample_out = output_dir / answer.sample_id
        image_path = PROJECT_ROOT / "04_data" / "sample_inputs" / answer.image_path
        render_overlay(image_path, result, sample_out / "overlay.png")
        export_prediction_json(result, sample_out / "prediction.json")
        print(f"{answer.sample_id}: {result.total_score}/{result.max_marks} review={result.review_flag}")


if __name__ == "__main__":
    main()
