from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "03_code" / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from scoremap.pipeline import load_answer_sample
from scoremap.text_utils import tokenize


def main() -> None:
    train_split = json.loads((PROJECT_ROOT / "04_data" / "sample_inputs" / "splits" / "train.json").read_text(encoding="utf-8"))
    answers_dir = PROJECT_ROOT / "04_data" / "sample_inputs" / "answers"
    token_counter: Counter[str] = Counter()
    for sample_id in train_split:
        sample = load_answer_sample(answers_dir / f"{sample_id}.json")
        for region in sample.regions:
            token_counter.update(tokenize(region.text))

    artifact = {
        "num_train_samples": len(train_split),
        "top_tokens": token_counter.most_common(50),
        "notes": "SCOREMAP uses heuristic typed evidence extraction in this lightweight prototype; training material is stored mainly for auditability.",
    }
    output_path = PROJECT_ROOT / "05_results" / "logs" / "train_artifact.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(f"Wrote training artifact to {output_path}")


if __name__ == "__main__":
    main()
