from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "03_code" / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from scoremap.answer_key_ingest import parse_answer_key_file, write_rubrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse an answer-key PDF/text file into SCOREMAP rubric JSON files.")
    parser.add_argument("--input", required=True, help="Path to the answer-key PDF/text file.")
    parser.add_argument("--output-dir", required=True, help="Directory where parsed rubric JSON files will be written.")
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve()
    rubrics = parse_answer_key_file(input_path)
    written = write_rubrics(rubrics, output_dir)
    print(f"Parsed {len(rubrics)} question(s) from {input_path.name}")
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
