from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "03_code" / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from scoremap.pipeline import ScoreMapPipeline, load_answer_sample, load_rubric, resolve_image_path
from scoremap.render import export_prediction_json, render_overlay


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SCOREMAP on one answer JSON.")
    parser.add_argument("--answer", required=True, help="Path to answer JSON")
    parser.add_argument("--rubric", required=True, help="Path to rubric JSON")
    parser.add_argument("--variant", default="scoremap", choices=["generic", "no_graph", "scoremap"])
    args = parser.parse_args()

    answer_path = Path(args.answer).resolve()
    rubric_path = Path(args.rubric).resolve()
    answer = load_answer_sample(answer_path)
    rubric = load_rubric(rubric_path)
    pipeline = ScoreMapPipeline.from_config(PROJECT_ROOT / "03_code" / "configs" / "default.yaml", args.variant)
    result = pipeline.run(answer, rubric)

    output_dir = PROJECT_ROOT / "06_demo" / "demo_outputs" / answer.sample_id
    image_path = resolve_image_path(answer, project_root=PROJECT_ROOT)
    render_overlay(image_path, result, output_dir / "overlay.png")
    export_prediction_json(result, output_dir / "prediction.json")

    summary = {
        "sample_id": result.sample_id,
        "qid": result.qid,
        "total_score": result.total_score,
        "max_marks": result.max_marks,
        "review_flag": result.review_flag,
        "hits": {item.item_id: item.hit for item in result.item_results},
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
