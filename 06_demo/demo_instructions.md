# Demo Instructions

## Primary Workflow

The repo now supports the intended real workflow:

- input 1: the teacher's answer key with the question
- input 2: the student's handwritten answer sheet
- output: parsed rubric JSON, ingested student `answer.json`, `prediction.json`, and `overlay.png`

No structured `.txt` file or sidecar JSON is required for the normal run.

## Packaged Demo Example

The main demo lives in `06_demo/ordeque_demo/`:

- `question_reference.png`
- `answer_key_reference.png`
- `student_answer.jpeg`

These are the real attached demo materials.

## One-Command Demo

Run:

```powershell
python 03_code\scripts\demo_e2e.py
```

This now uses:

- raw answer-key image: `06_demo\ordeque_demo\answer_key_reference.png`
- raw handwritten answer: `06_demo\ordeque_demo\student_answer.jpeg`
- backend: `hybrid`

The `hybrid` backend uses:

- EasyOCR to detect and read the answer-key image
- EasyOCR to detect handwritten line regions
- TrOCR on difficult handwritten crops
- answer-key-guided canonicalization before scoring

On a CPU-only machine, the first `hybrid` run can take a few minutes.

## Generic Two-File Command

You can run the same flow directly with:

```powershell
python 03_code\scripts\grade_document.py --document 06_demo\ordeque_demo\student_answer.jpeg --answer-key 06_demo\ordeque_demo\answer_key_reference.png --output-dir 06_demo\ordeque_demo\raw_two_file_run_v2 --backend hybrid
```

No `--qid` is needed here because the answer key contains only one question.

## What Gets Written

If you run `demo_e2e.py`, the main outputs are written under:

- `06_demo\ordeque_demo\outputs\hybrid\parsed_rubrics\Q4.json`
- `06_demo\ordeque_demo\outputs\hybrid\ingested\answer.json`
- `06_demo\ordeque_demo\outputs\hybrid\prediction.json`
- `06_demo\ordeque_demo\outputs\hybrid\overlay.png`

If you run the generic two-file CLI above, the main outputs are:

- `06_demo\ordeque_demo\raw_two_file_run_v2\parsed_rubrics\Q4.json`
- `06_demo\ordeque_demo\raw_two_file_run_v2\ingested\answer.json`
- `06_demo\ordeque_demo\raw_two_file_run_v2\prediction.json`
- `06_demo\ordeque_demo\raw_two_file_run_v2\overlay.png`

The current packaged raw-input run scores `8/14` on the handwritten sample.

## What To Show In A Demo

1. Open `question_reference.png`.
2. Open `answer_key_reference.png`.
3. Open `student_answer.jpeg`.
4. Run `python 03_code\scripts\demo_e2e.py`.
5. Open `parsed_rubrics\Q4.json` to show the answer key was parsed automatically.
6. Open `ingested\answer.json` to show the handwritten page was converted into machine-readable regions.
7. Open `overlay.png` to show evidence localization.
8. Open `prediction.json` to show marks per rubric item.

## Deterministic Regression Path

The old deterministic demo still exists for comparison:

```powershell
python 03_code\scripts\demo_e2e.py --backend regions_json
```

That path uses `student_answer_sidecar.json` and remains useful for regression testing, but it is no longer the primary workflow.
