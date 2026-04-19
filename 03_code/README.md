# SCOREMAP Code Guide

This repo now supports the intended end-to-end flow from only two raw inputs:

1. the teacher's answer key with the question
2. the student's handwritten answer sheet

From those two files, the code now:

- parses the answer key into rubric JSON
- transcribes the handwritten answer into `answer.json`
- runs the SCOREMAP scoring backend
- writes `prediction.json` and `overlay.png`

## Environment

Tested with:

- Python `3.13`
- `Pillow`, `numpy`, `matplotlib`, `scikit-learn`, `PyYAML`
- `pypdf`, `PyMuPDF`
- `easyocr`, `transformers`, `torch`

Install the base dependencies:

```powershell
python -m pip install -r 03_code\requirements.txt
```

Install the ingestion stack:

```powershell
python -m pip install -r 03_code\requirements-ingestion.txt
```

The first raw-image run will download the OCR model files used by EasyOCR and TrOCR.

## Main Scripts

- `scripts/grade_document.py`
  Full two-file pipeline. This is the primary entry point.
- `scripts/parse_answer_key.py`
  Parses an answer-key image/PDF/text file into rubric JSON.
- `scripts/ingest_student.py`
  Converts a handwritten page into the `answer.json` format.
- `scripts/demo_e2e.py`
  Runs the packaged ORDeque demo on the real attached example.
- `scripts/infer.py`
  Scores an existing `answer.json` against a rubric JSON.
- `scripts/eval.py`
  Runs the original benchmark evaluation.

## Primary Workflow

The normal workflow is now:

1. raw answer-key image/PDF/text
2. raw handwritten answer image/PDF
3. parsed rubric JSON + ingested student JSON
4. scored output with evidence overlay

The scorer itself is still the same typed-evidence and rubric-alignment backend. The new work is the ingestion layer in front of it.

## Two-File Command

Use this for real runs:

```powershell
python 03_code\scripts\grade_document.py --document <student_answer_image_or_pdf> --answer-key <answer_key_image_pdf_or_text> --output-dir <run_output_dir> --backend hybrid
```

If the answer key contains only one question, no `--qid` is needed. If it contains multiple questions, add `--qid`.

The default `hybrid` backend does this:

- OCRs image-only answer keys automatically
- detects handwritten line regions with EasyOCR
- runs TrOCR on difficult handwritten crops
- uses answer-key-guided canonicalization before scoring

On CPU, the first `hybrid` run can take a few minutes because both OCR models may need to initialize.

## Packaged Demo

The main showcase example is under `06_demo/ordeque_demo/`:

- `question_reference.png`
- `answer_key_reference.png`
- `student_answer.jpeg`

Run it with:

```powershell
python 03_code\scripts\demo_e2e.py
```

That command now uses the raw answer-key screenshot and raw handwritten answer directly.

The current packaged raw-input run also works through the generic CLI:

```powershell
python 03_code\scripts\grade_document.py --document 06_demo\ordeque_demo\student_answer.jpeg --answer-key 06_demo\ordeque_demo\answer_key_reference.png --output-dir 06_demo\ordeque_demo\raw_two_file_run_v2 --backend hybrid
```

Its outputs are:

- `parsed_rubrics\Q4.json`
- `ingested\answer.json`
- `prediction.json`
- `overlay.png`

## Deterministic Regression Path

The older deterministic path is still available:

```powershell
python 03_code\scripts\demo_e2e.py --backend regions_json
```

That uses `student_answer_sidecar.json` and is useful for regression testing, but it is no longer the primary workflow.

## Legacy Benchmark Commands

Generate packaged benchmark assets:

```powershell
python 03_code\scripts\generate_assets.py
```

Build the lightweight audit artifact:

```powershell
python 03_code\scripts\train.py
```

Run evaluation:

```powershell
python 03_code\scripts\eval.py
```

Run scorer-only inference on an existing answer JSON:

```powershell
python 03_code\scripts\infer.py --answer 04_data\sample_inputs\answers\writer03_q1.json --rubric 04_data\sample_inputs\rubrics\Q1.json --variant scoremap
```

## Notes

- The repo is now usable from the two raw input files alone, but handwriting quality still affects accuracy.
- The current scorer is still question-level; multi-question exam scripts should be split question by question before grading.
- Every run remains explainable: rubric JSON, ingested answer JSON, overlay, and item-wise prediction are all saved.
