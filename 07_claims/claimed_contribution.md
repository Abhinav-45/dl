# claimed_contribution.md

## What we reproduced
We reproduced lightweight baselines inspired by Donut, LayoutLMv3, and Pix2Struct at the level of document parsing and question-conditioned structured prediction. We also implemented a generic rubric scorer to serve as a fair non-structured baseline.

## What we modified
Instead of directly predicting marks from a page embedding, we reformulated grading as typed evidence extraction followed by executable rubric matching. The pipeline explicitly distinguishes definition spans, algorithm steps, complexity expressions, code-like lines, and diagram regions.

## What did not work
A whole-page black-box scorer was difficult to interpret and could not justify partial credit. We also found that removing order and prerequisite constraints weakened scoring quality on algorithm traces and Gantt-chart questions.

## What we believe is our contribution
Our main contribution is a structure-aware grading prototype for handwritten CS answers that combines:
1. typed evidence extraction,
2. typed evidence graphs,
3. executable rubrics with type-specific scoring heads, and
4. grounded evidence overlays for every awarded rubric item.

We additionally provide a small pilot benchmark with writer-wise splits, rubric hit labels, and evidence annotations to make the scoring pipeline auditable during the course demo.
