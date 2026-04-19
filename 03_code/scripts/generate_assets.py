from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "03_code" / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from scoremap.dataset_builder import build_dataset
from scoremap.report_assets import create_admin_pdf, write_admin_files, write_claim_files


PRIOR_WORK = """
# prior_work_basis.md

## Donut: Document Understanding Transformer without OCR
Influence on our work:
- motivated an OCR-free view of document understanding
- informed our decision to use image-conditioned parsing rather than depend entirely on separate OCR tooling
- served as the conceptual basis for our generative baseline and page-to-JSON comparison

## LayoutLMv3: Pre-training for Document AI with Unified Text and Image Masking
Influence on our work:
- highlighted the importance of jointly using text, image, and spatial/layout information
- inspired our region-level fusion of transcription, geometry, and answer-type prediction
- served as the conceptual basis for a layout-aware rubric alignment baseline

## Pix2Struct: Screenshot Parsing as Pretraining for Visual Language Understanding
Influence on our work:
- motivated question-conditioned structured prediction from page images
- inspired the extracted JSON representation used in our demo outputs
- served as the conceptual basis for a question-aware baseline scorer

## How our project differs from the base papers
The base papers are general document-understanding models. SCOREMAP targets handwritten CS answer grading with partial credit, pseudocode/code fragments, complexity terms, and diagrams. Our novelty is not a new backbone, but a rubric-grounded scoring formulation that builds a typed evidence graph and executes typed rubric items to produce grounded scores and evidence overlays.
"""


CLAIMS = """
# claimed_contribution.md

## What we reproduced
We reproduced lightweight baselines inspired by Donut, LayoutLMv3, and Pix2Struct at the level of document parsing and question-conditioned structured prediction. We also implemented a generic rubric scorer to serve as a fair non-structured baseline.

## What we modified
Instead of directly predicting marks from a page embedding, we reformulated grading as typed evidence extraction followed by executable rubric matching. The pipeline explicitly distinguishes definition spans, algorithm steps, complexity expressions, code-like lines, and diagram regions.

## What did not work
A whole-page black-box scorer was difficult to interpret and could not justify partial credit. We also found that removing order and prerequisite constraints weakened scoring quality on algorithm traces and Gantt-chart questions.

## What we believe is our contribution
Our main contribution is a structure-aware grading prototype for handwritten CS answers that combines:
1. typed evidence extraction,
2. typed evidence graphs,
3. executable rubrics with type-specific scoring heads, and
4. grounded evidence overlays for every awarded rubric item.

We additionally provide a small pilot benchmark with writer-wise splits, rubric hit labels, and evidence annotations to make the scoring pipeline auditable during the course demo.
"""


def main() -> None:
    build_dataset(PROJECT_ROOT)
    write_admin_files(PROJECT_ROOT)
    create_admin_pdf(PROJECT_ROOT)
    write_claim_files(PROJECT_ROOT, PRIOR_WORK, CLAIMS)
    print("Generated dataset, admin files, and claims.")


if __name__ == "__main__":
    main()
