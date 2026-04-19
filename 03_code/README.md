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

Generate the packaged benchmark, demo assets, and sample answer-key PDF:

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

Run one inference example from an existing answer JSON:

```powershell
python 03_code\scripts\infer.py --answer 04_data\sample_inputs\answers\writer03_q1.json --rubric 04_data\sample_inputs\rubrics\Q1.json --variant scoremap
```

Parse the packaged answer-key PDF into rubric JSON files:

```powershell
python 03_code\scripts\parse_answer_key.py --input 04_data\sample_inputs\answer_keys\scoremap_answer_key.pdf --output-dir 04_data\parsed_rubrics
```

Ingest a student document into SCOREMAP answer JSON with TrOCR:

```powershell
python 03_code\scripts\ingest_student.py --input 04_data\sample_inputs\images\writer03_q1.png --rubric 04_data\sample_inputs\rubrics\Q1.json --output-dir 06_demo\ingested_writer03_q1 --backend trocr
```

Run the full document-to-score pipeline:

```powershell
python 03_code\scripts\grade_document.py --document 04_data\sample_inputs\images\writer03_q1.png --answer-key 04_data\sample_inputs\answer_keys\scoremap_answer_key.pdf --qid Q1 --output-dir 06_demo\graded_writer03_q1 --backend trocr
```

Run the packaged regression-safe end-to-end demo without live OCR by reusing the provided region sidecar:

```powershell
python 03_code\scripts\demo_e2e.py --sample writer03_q1 --backend regions_json
```

Run the original scorer-only demo batch:

```powershell
python 03_code\scripts\demo.py
```

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

`trocr` works best when:
- the input contains one question answer per page/image
- the scan is reasonably straight and high contrast
- the answer text is line-based rather than free-form diagrams only

If the input is a PDF, SCOREMAP uses `PyMuPDF` to render the selected page before segmentation.

## Notes

- The packaged dataset is still a pilot benchmark with synthetic handwritten-style pages and gold labels.
- The new ingestion code makes the repo usable on real inputs, but the current scorer is still question-level; multi-question exam scripts should be split question by question before grading.
- Every grading run still produces explainable outputs rather than a black-box final score.
