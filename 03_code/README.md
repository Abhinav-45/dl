# SCOREMAP Code Guide

This folder now contains a full SCOREMAP pipeline with two layers:

- the original rubric-aligned scoring backend
- a new ingestion layer that turns student documents and answer-key PDFs into the JSON formats the scorer expects

## Environment

Tested with:
- Python `3.13`
- `Pillow`, `numpy`, `matplotlib`, `scikit-learn`, `PyYAML`
- `pypdf`, `PyMuPDF`

Install the base dependencies:

```powershell
python -m pip install -r 03_code\requirements.txt
```

For handwriting transcription with TrOCR, also install:

```powershell
python -m pip install -r 03_code\requirements-ingestion.txt
```

The first `trocr` run will download the configured Hugging Face checkpoint.

## Project Layout

- `src/scoremap/`: scorer, evaluation, rendering, answer-key parsing, student-document ingestion
- `scripts/generate_assets.py`: creates sample data, demo assets, answer-key PDF/text, admin assets, and claim files
- `scripts/train.py`: writes the lightweight audit artifact from the writer-wise train split
- `scripts/eval.py`: runs benchmark evaluation and regenerates report assets
- `scripts/infer.py`: grades one existing answer JSON and exports overlays
- `scripts/parse_answer_key.py`: parses an answer-key PDF/text file into rubric JSON files
- `scripts/ingest_student.py`: converts a student image/PDF into an `answer.json`
- `scripts/grade_document.py`: runs the full document -> ingest -> score -> overlay flow
- `scripts/demo.py`: runs the original packaged scorer demo
- `scripts/demo_e2e.py`: runs the new end-to-end ingestion demo on packaged assets
- `configs/default.yaml`: model-variant flags for the scorer and ablations

## End-to-End Data Flow

The repo now supports this path:

1. `answer key PDF/text` -> parsed rubric JSON
2. `student image/PDF` -> `answer.json` with regions/bboxes/text
3. `answer.json + rubric.json` -> SCOREMAP scoring pipeline
4. `prediction.json + overlay.png` for auditability

The scoring core is still the same typed-evidence/rubric-alignment model. The new modules only fill the ingestion gap.

## Exact Commands

Generate the packaged benchmark and legacy synthetic assets:

```powershell
python 03_code\scripts\generate_assets.py
```

Build the lightweight audit artifact:

```powershell
python 03_code\scripts\train.py
```

Run evaluation on the writer-wise test split:

```powershell
python 03_code\scripts\eval.py
```

Run one inference example from the legacy packaged answer JSON:

```powershell
python 03_code\scripts\infer.py --answer 04_data\sample_inputs\answers\writer03_q1.json --rubric 04_data\sample_inputs\rubrics\Q1.json --variant scoremap
```

Parse the real ORDeque demo answer key into rubric JSON:

```powershell
python 03_code\scripts\parse_answer_key.py --input 06_demo\ordeque_demo\answer_key_structured.pdf --output-dir 06_demo\ordeque_demo\parsed_rubrics
```

Run the primary stable end-to-end demo on the real ORDeque materials:

```powershell
python 03_code\scripts\demo_e2e.py
```

This uses:

- `06_demo\ordeque_demo\question_reference.png`
- `06_demo\ordeque_demo\answer_key_reference.png`
- `06_demo\ordeque_demo\answer_key_structured.pdf`
- `06_demo\ordeque_demo\student_answer.jpeg`
- `06_demo\ordeque_demo\student_answer_sidecar.json`

Run the same ORDeque example through the generic CLI instead of the demo wrapper:

```powershell
python 03_code\scripts\grade_document.py --document 06_demo\ordeque_demo\student_answer.jpeg --answer-key 06_demo\ordeque_demo\answer_key_structured.pdf --qid ORQ4 --output-dir 06_demo\ordeque_demo\manual_run --backend regions_json --sidecar 06_demo\ordeque_demo\student_answer_sidecar.json
```

If you want to test live handwriting transcription instead of the stable sidecar path:

```powershell
python 03_code\scripts\demo_e2e.py --backend trocr
```

The old synthetic scorer-only demo batch is still available:

```powershell
python 03_code\scripts\demo.py
```

## Primary Demo Assets

The main showcase example in this repo is now the ORDeque question under `06_demo/ordeque_demo/`.

Files:

- original question screenshot: `question_reference.png`
- original answer-key screenshot: `answer_key_reference.png`
- parser-friendly transcription of the answer key: `answer_key_structured.txt` and `answer_key_structured.pdf`
- handwritten student answer: `student_answer.jpeg`
- deterministic region sidecar: `student_answer_sidecar.json`

The structured PDF is intentionally included because the original answer-key screenshot is image-only, while the parser expects a text-layer PDF or text file.

## Answer-Key Format

The most reliable path is a text-layer PDF or `.txt` file that follows the structured template emitted under `04_data/sample_inputs/answer_keys/scoremap_answer_key.pdf`.

Each question block looks like:

```text
Question ID: Q1
Question Type: definition_plus_diagram
Question Text: Explain starvation in OS scheduling and illustrate with a Gantt chart.
Max Marks: 5
Item ID: r1
Item Type: definition
Description: States that starvation means indefinite postponement or waiting forever.
Marks: 1
Required: true
Order:
Prerequisite:
Alternatives:
================================================
```

The parser also has a fallback bullet-mode parser, but the structured format is what the repo verifies end to end.

## Student-Document Ingestion

Two ingestion backends are supported:

- `trocr`: real handwriting recognition using TrOCR plus simple line segmentation
- `regions_json`: deterministic fallback for demos/regression tests that reuses an existing answer/region JSON as the transcription sidecar

For the ORDeque demo, `regions_json` is the recommended path because it is stable and auditable on the real attached handwritten page. The live `trocr` path remains available, but it is currently experimental on that sample.

`trocr` works best when:
- the input contains one question answer per page/image
- the scan is reasonably straight and high contrast
- the answer text is line-based rather than free-form diagrams only

If the input is a PDF, SCOREMAP uses `PyMuPDF` to render the selected page before segmentation.

## Notes

- The packaged dataset is still a pilot benchmark with synthetic handwritten-style pages and gold labels.
- The new ingestion code makes the repo usable on real inputs, but the current scorer is still question-level; multi-question exam scripts should be split question by question before grading.
- Every grading run still produces explainable outputs rather than a black-box final score.
