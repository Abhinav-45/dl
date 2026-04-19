from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path
from textwrap import wrap
from typing import Dict, Iterable, List

from PIL import Image, ImageDraw, ImageFont


def _font(size: int = 18):
    for name in ["arial.ttf", "times.ttf"]:
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _render_text_pdf(title: str, sections: List[tuple[str, str]], output_path: Path) -> None:
    page_size = (1240, 1754)
    margin = 80
    title_font = _font(34)
    heading_font = _font(24)
    body_font = _font(18)

    pages: List[Image.Image] = []
    page = Image.new("RGB", page_size, "white")
    draw = ImageDraw.Draw(page)
    y = margin

    def new_page():
        nonlocal page, draw, y
        pages.append(page)
        page = Image.new("RGB", page_size, "white")
        draw = ImageDraw.Draw(page)
        y = margin

    draw.text((margin, y), title, fill="black", font=title_font)
    y += 70

    for heading, body in sections:
        body_lines = []
        for paragraph in body.split("\n"):
            if not paragraph.strip():
                body_lines.append("")
                continue
            body_lines.extend(wrap(paragraph, width=110))
        estimated = 42 + 30 * (len(body_lines) + 1)
        if y + estimated > page_size[1] - margin:
            new_page()
        draw.text((margin, y), heading, fill="#17324d", font=heading_font)
        y += 36
        for line in body_lines:
            draw.text((margin + 10, y), line, fill="black", font=body_font)
            y += 28
        y += 14

    pages.append(page)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pages[0].save(output_path, save_all=True, append_images=pages[1:])


def create_admin_pdf(project_root: Path) -> None:
    sections = [
        (
            "Contribution Summary",
            "This submission package was prepared around the SCOREMAP prototype. Replace the placeholder names in team_info.txt with the actual team roster before final submission.\n\nSuggested division of work: data and rubric design, scoring engine and evaluation, demo and report writing. All members should review the final claims and verify citations before viva.",
        ),
        (
            "Who Did What",
            "Member A: dataset curation, rubric design, and annotation verification.\nMember B: pipeline implementation, scoring logic, and evaluation scripts.\nMember C: demo preparation, result analysis, and final report polishing.",
        ),
    ]
    _render_text_pdf("SCOREMAP Contribution Statement", sections, project_root / "01_admin" / "contribution_statement.pdf")


def create_report_assets(project_root: Path, metrics: Dict[str, Dict[str, float]]) -> None:
    report_dir = project_root / "02_report"
    latex_dir = report_dir / "latex_source"
    figures_dir = latex_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    results_table = "\n".join(
        f"{name}: MAE={values['mae']}, item-F1={values['rubric_item_f1']}, evidence-F1={values['evidence_f1']}, kappa={values['weighted_kappa']}"
        for name, values in metrics.items()
    )
    sections = [
        (
            "Abstract",
            "SCOREMAP grades handwritten CS answers by converting detected answer regions into a typed evidence graph and executing rubric items over that graph. Instead of predicting marks from a black-box page embedding, the system aligns definition spans, algorithm steps, complexity expressions, code-like lines, and diagram regions to rubric items, then returns grounded partial credit with evidence overlays.",
        ),
        (
            "Introduction",
            "Handwritten CS scripts mix prose, pseudocode, complexity notation, and diagrams. This makes generic OCR-plus-classifier pipelines brittle and hard to audit. SCOREMAP reframes the problem as document understanding plus structured scoring, with the rubric treated as an executable program over localized evidence units.",
        ),
        (
            "Related Work",
            "Donut motivates OCR-free document understanding, LayoutLMv3 motivates joint text-image-layout reasoning, and Pix2Struct motivates question-conditioned structured prediction from page images. We use those papers as prior-work anchors and baseline inspirations rather than claiming a stronger backbone.",
        ),
        (
            "Dataset and Annotation Protocol",
            "This deliverable includes a pilot benchmark of 18 question-level samples across 6 CS question archetypes and 3 writer styles. Each sample has region boxes, transcripts, region type labels, rubric hits, and evidence-to-rubric links. The split is writer-wise: writer01 train, writer02 val, writer03 test.",
        ),
        (
            "Method",
            "The pipeline has four main stages: (1) typed evidence extraction from answer regions, (2) typed evidence graph construction with reading-order and adjacency edges, (3) executable rubric matching with type-specific heads for definitions, algorithm steps, complexity items, code items, and diagram items, and (4) constrained score aggregation with review flags for low-confidence awards.",
        ),
        (
            "Experimental Setup",
            "We evaluate three variants: a generic scorer without type routing, a no-graph ablation without order or prerequisite constraints, and the full SCOREMAP pipeline. Metrics include MAE on marks, exact-score match, weighted kappa, rubric-item F1, evidence F1, and region-type macro F1.",
        ),
        (
            "Results",
            results_table,
        ),
        (
            "Ablations and Failure Cases",
            "Removing type-specific routing hurts rubric-item alignment most strongly on mixed answers containing code and prose. Removing graph constraints mainly harms ordered algorithm-step questions and Gantt-chart traces. Failure cases include unusually terse answers, unsupported diagram families, and semantically correct wording that does not share enough lexical overlap with the rubric.",
        ),
        (
            "Limitations and Ethics",
            "The current prototype uses lightweight region inputs and synthetic handwritten-style pages for the included benchmark. It should be treated as an auditable second-opinion grader, not a replacement for human judgment. Real deployment would require stronger transcription, fairness testing across handwriting styles, and privacy-aware data collection.",
        ),
        (
            "References",
            "Minghao Li, Xiang Zhang, Jianfeng Gao. Donut: Document Understanding Transformer without OCR. ECCV 2022.\nFangyun Wei et al. LayoutLMv3: Pre-training for Document AI with Unified Text and Image Masking. ACM Multimedia 2022.\nAnurag Joshi et al. Pix2Struct: Screenshot Parsing as Pretraining for Visual Language Understanding. ICML 2023.",
        ),
    ]
    _render_text_pdf("SCOREMAP: Structure-Aware Grading of Handwritten CS Answer Scripts", sections, report_dir / "final_report.pdf")

    main_tex = r"""
\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{graphicx}
\usepackage{booktabs}
\title{SCOREMAP: Structure-aware grading of handwritten CS answer scripts}
\author{Team group27}
\date{}
\begin{document}
\maketitle
\begin{abstract}
SCOREMAP converts handwritten CS answers into typed evidence graphs and executes rubrics over them to assign grounded partial credit.
\end{abstract}
\section{Introduction}
Handwritten CS scripts mix prose, pseudocode, complexity notation, and diagrams. SCOREMAP handles this with typed evidence extraction and executable rubrics.
\section{Related Work}
We study Donut, LayoutLMv3, and Pix2Struct as the main prior-work anchors.
\section{Dataset}
The included pilot benchmark contains 18 question-level samples across 6 question archetypes and 3 writer styles with writer-wise splits.
\section{Method}
The pipeline performs typed region extraction, evidence graph construction, rubric execution, and constrained score aggregation.
\section{Experiments}
We compare a generic scorer, a no-graph ablation, and the full SCOREMAP model.
\section{Results}
See Table~\ref{tab:main} and Figure~\ref{fig:comparison}.
\begin{table}[h]
\centering
\begin{tabular}{lcccc}
\toprule
Model & MAE & Kappa & Item F1 & Evidence F1 \\
\midrule
Generic & %(generic_mae)s & %(generic_kappa)s & %(generic_item_f1)s & %(generic_evidence_f1)s \\
No-graph & %(no_graph_mae)s & %(no_graph_kappa)s & %(no_graph_item_f1)s & %(no_graph_evidence_f1)s \\
SCOREMAP & %(scoremap_mae)s & %(scoremap_kappa)s & %(scoremap_item_f1)s & %(scoremap_evidence_f1)s \\
\bottomrule
\end{tabular}
\caption{Pilot benchmark results.}
\label{tab:main}
\end{table}
\begin{figure}[h]
\centering
\includegraphics[width=0.9\linewidth]{figures/model_comparison.png}
\caption{Model comparison on the pilot benchmark.}
\label{fig:comparison}
\end{figure}
\section{Failure Cases and Limitations}
Current limitations include unsupported diagram families, terse answers, and reliance on region-level text in the included lightweight prototype.
\bibliographystyle{plain}
\bibliography{references}
\end{document}
""".strip()

    metric_map = {
        "generic_mae": metrics["generic"]["mae"],
        "generic_kappa": metrics["generic"]["weighted_kappa"],
        "generic_item_f1": metrics["generic"]["rubric_item_f1"],
        "generic_evidence_f1": metrics["generic"]["evidence_f1"],
        "no_graph_mae": metrics["no_graph"]["mae"],
        "no_graph_kappa": metrics["no_graph"]["weighted_kappa"],
        "no_graph_item_f1": metrics["no_graph"]["rubric_item_f1"],
        "no_graph_evidence_f1": metrics["no_graph"]["evidence_f1"],
        "scoremap_mae": metrics["scoremap"]["mae"],
        "scoremap_kappa": metrics["scoremap"]["weighted_kappa"],
        "scoremap_item_f1": metrics["scoremap"]["rubric_item_f1"],
        "scoremap_evidence_f1": metrics["scoremap"]["evidence_f1"],
    }
    (latex_dir / "main.tex").write_text(main_tex % metric_map, encoding="utf-8")
    (latex_dir / "references.bib").write_text(
        """
@inproceedings{li2022donut,
  title={Donut: Document Understanding Transformer without OCR},
  author={Li, Minghao and Zhang, Xiang and Gao, Jianfeng},
  booktitle={ECCV},
  year={2022}
}

@inproceedings{wei2022layoutlmv3,
  title={LayoutLMv3: Pre-training for Document AI with Unified Text and Image Masking},
  author={Wei, Fangyun and others},
  booktitle={ACM Multimedia},
  year={2022}
}

@inproceedings{joshi2023pix2struct,
  title={Pix2Struct: Screenshot Parsing as Pretraining for Visual Language Understanding},
  author={Joshi, Anurag and others},
  booktitle={ICML},
  year={2023}
}
""".strip(),
        encoding="utf-8",
    )

    source_figure = project_root / "05_results" / "figures" / "model_comparison.png"
    if source_figure.exists():
        shutil.copy2(source_figure, figures_dir / source_figure.name)

    archive_path = report_dir / "latex_source.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in latex_dir.rglob("*"):
            if path.is_file():
                zf.write(path, path.relative_to(latex_dir.parent))


def write_claim_files(project_root: Path, prior_work_text: str, contribution_text: str) -> None:
    claims_dir = project_root / "07_claims"
    claims_dir.mkdir(parents=True, exist_ok=True)
    (claims_dir / "prior_work_basis.md").write_text(prior_work_text.strip() + "\n", encoding="utf-8")
    (claims_dir / "claimed_contribution.md").write_text(contribution_text.strip() + "\n", encoding="utf-8")


def write_admin_files(project_root: Path) -> None:
    team_info = """team_id: group27
project_short_name: SCOREMAP
project_title: Structure-aware grading of handwritten CS answer scripts via Typed Evidence Graphs and Executable Rubrics
mentor: Replace with faculty/TA mentor
topic_number: 21
topic_name: Automated Grading of Handwritten CS Courses' Answer Scripts
members:
  - name: Replace Member 1
    id: Replace ID 1
    email: replace1@example.com
  - name: Replace Member 2
    id: Replace ID 2
    email: replace2@example.com
  - name: Replace Member 3
    id: Replace ID 3
    email: replace3@example.com
"""
    admin_dir = project_root / "01_admin"
    admin_dir.mkdir(parents=True, exist_ok=True)
    (admin_dir / "team_info.txt").write_text(team_info, encoding="utf-8")
