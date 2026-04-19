# SCOREMAP Code Guide

This folder contains the runnable SCOREMAP prototype used in the submission package.

## Environment

Tested with:
- Python `3.13`
- `Pillow`, `numpy`, `matplotlib`, `scikit-learn`, `PyYAML`

Install dependencies:

```powershell
python -m pip install -r 03_code\requirements.txt
```

## Project Layout

- `src/scoremap/`: core pipeline, evaluation, rendering, synthetic benchmark generation
- `scripts/generate_assets.py`: creates sample data, admin assets, and claim files
- `scripts/train.py`: writes a lightweight training artifact from the writer-wise train split
- `scripts/eval.py`: runs evaluation, writes `05_results/*.csv`, and generates report assets
- `scripts/infer.py`: grades one sample answer JSON and exports evidence overlays
- `scripts/demo.py`: runs the demo set end-to-end
- `configs/default.yaml`: model-variant flags for baselines and ablations

## Exact Commands

Generate the benchmark and required non-code assets:

```powershell
python 03_code\scripts\generate_assets.py
```

Build the lightweight training artifact:

```powershell
python 03_code\scripts\train.py
```

Run evaluation on the writer-wise test split and regenerate results/report assets:

```powershell
python 03_code\scripts\eval.py
```

Run one inference example:

```powershell
python 03_code\scripts\infer.py --answer 04_data\sample_inputs\answers\writer03_q1.json --rubric 04_data\sample_inputs\rubrics\Q1.json --variant scoremap
```

Launch the packaged demo batch:

```powershell
python 03_code\scripts\demo.py
```

## Hardware

The included prototype runs on CPU. No GPU is required for the packaged benchmark because the typed evidence extraction and rubric execution are lightweight and deterministic.

## Notes

- The included sample data is a pilot benchmark with handwritten-style rendered pages plus aligned region annotations.
- The code is structured so that a stronger transcription module can later replace the lightweight region input stage without changing the scoring engine.
