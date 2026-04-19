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
