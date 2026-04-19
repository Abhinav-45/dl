# Demo Instructions

## Goal

Show SCOREMAP grading a handwritten-style CS answer end-to-end in under 5 minutes.

## Command

Run:

```powershell
python 03_code\scripts\demo.py
```

## What the Demo Uses

- Demo inputs: `06_demo/demo_inputs/`
- Rubrics: `04_data/sample_inputs/rubrics/`
- Generated outputs: `06_demo/demo_outputs/`

## Suggested Viva Flow

1. Open one sample page from `06_demo/demo_inputs/`, for example `writer03_q1.png`.
2. Run the demo command.
3. Show the console summary with predicted score and review flag.
4. Open the corresponding overlay image in `06_demo/demo_outputs/<sample_id>/overlay.png`.
5. Open `prediction.json` in the same folder to show:
   - typed evidence graph nodes
   - rubric item hit/miss decisions
   - evidence node ids used for each awarded item
6. Repeat for one algorithm-heavy sample such as `writer03_q2`.

## Meaningful Behaviors to Highlight

- region typing differs across prose, algorithm steps, complexity statements, and diagrams
- grading is rubric-aligned instead of end-to-end black-box scoring
- every awarded item is tied to localized evidence
- uncertain cases raise a `review_flag`

## Backup Material

Backup qualitative outputs are already available under:

- `06_demo/demo_outputs/`
- `05_results/figures/`

These are backup aids only and do not replace the live run required in the course guide.
