from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "03_code" / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from scoremap.evaluation import run_evaluation, write_ablations
from scoremap.pipeline import ScoreMapPipeline
from scoremap.report_assets import create_report_assets


def main() -> None:
    config_path = PROJECT_ROOT / "03_code" / "configs" / "default.yaml"
    variants = {
        "generic": ScoreMapPipeline.from_config(config_path, "generic"),
        "no_graph": ScoreMapPipeline.from_config(config_path, "no_graph"),
        "scoremap": ScoreMapPipeline.from_config(config_path, "scoremap"),
    }
    metrics = run_evaluation(
        data_root=PROJECT_ROOT / "04_data" / "sample_inputs",
        answers_dir=PROJECT_ROOT / "04_data" / "sample_inputs" / "answers",
        rubrics_dir=PROJECT_ROOT / "04_data" / "sample_inputs" / "rubrics",
        split_path=PROJECT_ROOT / "04_data" / "sample_inputs" / "splits" / "test.json",
        output_dir=PROJECT_ROOT / "05_results",
        variants=variants,
    )
    write_ablations(PROJECT_ROOT / "05_results", metrics)
    create_report_assets(PROJECT_ROOT, metrics)
    print("Evaluation complete.")
    for name, values in metrics.items():
        print(name, values)


if __name__ == "__main__":
    main()
