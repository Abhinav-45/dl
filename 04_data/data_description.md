# Data Description

## Included Benchmark

This submission package contains a lightweight pilot benchmark for SCOREMAP under `04_data/sample_inputs/`.

- Total question-level samples: `18`
- Writers: `3`
- Question archetypes: `6`
- Split policy: writer-wise

## Question Types

1. Definition plus Gantt-chart explanation
2. Algorithm steps plus complexity
3. Pseudocode plus complexity
4. Definition/list question
5. Graph algorithm explanation plus code plus complexity
6. Round Robin trace plus fairness comment

## Files

- `sample_inputs/answers/*.json`: structured answer samples with regions, texts, and gold rubric alignment
- `sample_inputs/images/*.png`: handwritten-style rendered answer pages aligned to the answer JSON
- `sample_inputs/rubrics/*.json`: executable rubric files used by the scorer
- `sample_inputs/splits/train.json`: `writer01`
- `sample_inputs/splits/val.json`: `writer02`
- `sample_inputs/splits/test.json`: `writer03`

## Annotation Format

Each answer JSON contains:

- `sample_id`, `writer_id`, `qid`, and `question_text`
- `image_path`
- region list with `bbox`, `text`, `type_hint`, and metadata
- gold supervision:
  - total score
  - rubric item hits/misses
  - evidence region ids per rubric item

## Preprocessing

The included pages are synthetic handwritten-style renders produced from the aligned region annotations. For the packaged benchmark:

- margins are normalized during page rendering
- each answer is stored as a per-question crop/page
- diagram regions carry simple structured metadata for Gantt-like traces

## Scope Reduction

This pilot benchmark is intentionally small and writer-wise split because the course guide explicitly allows scope reduction when compute, time, or dataset access is limited. The current dataset is designed to demonstrate:

- typed evidence extraction
- rubric-grounded scoring
- evidence localization
- ablation-ready evaluation

without claiming general coverage of all handwritten CS answer formats.
