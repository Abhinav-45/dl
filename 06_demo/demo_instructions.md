# Demo Instructions

## Goal

Show both SCOREMAP layers:

1. the original scorer on packaged answer JSON
2. the new ingestion-to-grading pipeline that starts from an answer-key PDF and a student document

## Fast Scorer Demo

Run:

```powershell
python 03_code\scripts\demo.py
```

This uses:

- demo inputs: `06_demo/demo_inputs/`
- packaged rubric JSON: `04_data/sample_inputs/rubrics/`
- outputs: `06_demo/demo_outputs/`

## Full Ingestion Demo

Run:

```powershell
python 03_code\scripts\demo_e2e.py --sample writer03_q1 --backend regions_json
```

This uses:

- answer-key PDF: `04_data/sample_inputs/answer_keys/scoremap_answer_key.pdf`
- student document image: `04_data/sample_inputs/images/writer03_q1.png`
- region sidecar for deterministic transcription: `04_data/sample_inputs/answers/writer03_q1.json`
- outputs: `06_demo/e2e_outputs/writer03_q1/`

If TrOCR is installed and the checkpoint is available, you can switch to live handwriting transcription:

```powershell
python 03_code\scripts\demo_e2e.py --sample writer03_q1 --backend trocr
```

## Suggested Viva Flow

1. Open `04_data/sample_inputs/answer_keys/scoremap_answer_key.pdf`.
2. Explain that the parser converts this into rubric JSON automatically.
3. Open `04_data/sample_inputs/images/writer03_q1.png`.
4. Run the end-to-end demo command.
5. Show `06_demo/e2e_outputs/writer03_q1/parsed_rubrics/Q1.json`.
6. Show `06_demo/e2e_outputs/writer03_q1/ingested/answer.json`.
7. Open `06_demo/e2e_outputs/writer03_q1/overlay.png`.
8. Open `06_demo/e2e_outputs/writer03_q1/prediction.json`.

## Meaningful Behaviors to Highlight

- the answer key is no longer hand-entered as rubric JSON
- the student document is converted into region JSON before scoring
- the scorer remains rubric-aligned and evidence-grounded
- overlays and `prediction.json` make the score auditable
- `trocr` is the live transcription backend, while `regions_json` is the regression-safe fallback
