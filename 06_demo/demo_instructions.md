# Demo Instructions

## Primary Demo Example

The repo now uses the real ORDeque example as the main demo:

- question paper reference: `06_demo/ordeque_demo/question_reference.png`
- answer-key screenshot reference: `06_demo/ordeque_demo/answer_key_reference.png`
- structured answer key for automated parsing: `06_demo/ordeque_demo/answer_key_structured.pdf`
- handwritten student answer sheet: `06_demo/ordeque_demo/student_answer.jpeg`
- deterministic transcription sidecar used for the live demo: `06_demo/ordeque_demo/student_answer_sidecar.json`

The structured PDF is a faithful transcription of the attached answer-key screenshot. It exists because the parser needs a text-layer PDF or text file, while the original screenshot is image-only.

## Stable Demo Command

Run:

```powershell
python 03_code\scripts\demo_e2e.py
```

This defaults to:

- sample: `ordeque_demo`
- backend: `regions_json`

That path is the recommended viva/demo path because it uses the real attached question/answer materials while keeping the transcription deterministic and reproducible.

## What the Stable Demo Produces

Outputs are written to:

- `06_demo/ordeque_demo/outputs/regions_json/overlay.png`
- `06_demo/ordeque_demo/outputs/regions_json/prediction.json`
- `06_demo/ordeque_demo/outputs/regions_json/ingested/answer.json`
- `06_demo/ordeque_demo/outputs/regions_json/parsed_rubrics/ORQ4.json`

On the current packaged sidecar, the score is `13/14` with a review flag because one rubric item remains borderline.

## Suggested Viva Flow

1. Open `06_demo/ordeque_demo/question_reference.png` to show the original problem statement.
2. Open `06_demo/ordeque_demo/answer_key_reference.png` to show the teacher's answer key and marking scheme.
3. Open `06_demo/ordeque_demo/student_answer.jpeg` to show the handwritten student submission.
4. Run `python 03_code\scripts\demo_e2e.py`.
5. Open `06_demo/ordeque_demo/answer_key_structured.pdf` and explain that this is the parser-friendly transcription of the answer-key screenshot.
6. Open `06_demo/ordeque_demo/outputs/regions_json/parsed_rubrics/ORQ4.json` to show the rubric items and marks.
7. Open `06_demo/ordeque_demo/outputs/regions_json/ingested/answer.json` to show the regionized student answer.
8. Open `06_demo/ordeque_demo/outputs/regions_json/overlay.png` to show localized evidence.
9. Open `06_demo/ordeque_demo/outputs/regions_json/prediction.json` to show item-wise mark decisions and rationales.

## Important Note About TrOCR

You can still try:

```powershell
python 03_code\scripts\demo_e2e.py --backend trocr
```

But for this real handwritten ORDeque page, TrOCR is currently not reliable enough to be the primary demo path. The `regions_json` backend is the recommended demo because it keeps the focus on answer-key parsing, structured ingestion, rubric execution, and auditable scoring.

## Legacy Material

The older synthetic demo samples under `06_demo/demo_inputs/` and `04_data/sample_inputs/` are still in the repo for regression testing and baseline comparisons, but they are no longer the primary showcase example.
