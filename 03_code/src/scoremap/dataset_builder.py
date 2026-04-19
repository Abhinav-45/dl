from __future__ import annotations

import json
import random
import shutil
from pathlib import Path
from textwrap import wrap
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFilter, ImageFont


def _font(size: int = 18):
    for name in ["seguisbi.ttf", "segoepr.ttf", "arial.ttf"]:
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


RUBRICS = [
    {
        "qid": "Q1",
        "question_type": "definition_plus_diagram",
        "question_text": "Explain starvation in OS scheduling and illustrate with a Gantt chart.",
        "max_marks": 5,
        "items": [
            {"id": "r1", "type": "definition", "description": "States that starvation means indefinite postponement or waiting forever.", "marks": 1, "required": True},
            {"id": "r2", "type": "concept", "description": "Links starvation to priority scheduling or unfair scheduling.", "marks": 1, "required": True, "prerequisite": "r1"},
            {"id": "r3", "type": "diagram_gantt", "description": "Draws a valid Gantt chart with labeled execution intervals.", "marks": 2, "required": True},
            {"id": "r4", "type": "final_answer", "description": "Mentions aging or increasing priority as mitigation.", "marks": 1, "required": False, "prerequisite": "r1"},
        ],
    },
    {
        "qid": "Q2",
        "question_type": "algorithm_plus_complexity",
        "question_text": "Write the steps of merge sort and state its time complexity.",
        "max_marks": 6,
        "items": [
            {"id": "r1", "type": "algorithm_step", "description": "Divide the array into halves recursively.", "marks": 2, "required": True, "order": 1},
            {"id": "r2", "type": "algorithm_step", "description": "Merge the sorted halves in order.", "marks": 2, "required": True, "order": 2},
            {"id": "r3", "type": "complexity", "description": "States time complexity O(n log n).", "marks": 1, "required": True},
            {"id": "r4", "type": "complexity", "description": "Mentions extra array or space complexity O(n).", "marks": 1, "required": False},
        ],
    },
    {
        "qid": "Q3",
        "question_type": "code_plus_complexity",
        "question_text": "Write pseudocode for binary search and give its time complexity.",
        "max_marks": 6,
        "items": [
            {"id": "r1", "type": "code", "description": "Initializes low and high indices.", "marks": 2, "required": True},
            {"id": "r2", "type": "code", "description": "Computes mid and compares key with a[mid].", "marks": 2, "required": True},
            {"id": "r3", "type": "code", "description": "Updates low or high depending on comparison.", "marks": 1, "required": True},
            {"id": "r4", "type": "complexity", "description": "States time complexity O(log n).", "marks": 1, "required": True},
        ],
    },
    {
        "qid": "Q4",
        "question_type": "definition_list",
        "question_text": "List Coffman conditions for deadlock and briefly explain any one.",
        "max_marks": 5,
        "items": [
            {"id": "r1", "type": "definition", "description": "Mentions mutual exclusion.", "marks": 1, "required": True},
            {"id": "r2", "type": "definition", "description": "Mentions hold and wait.", "marks": 1, "required": True},
            {"id": "r3", "type": "definition", "description": "Mentions no preemption.", "marks": 1, "required": True},
            {"id": "r4", "type": "definition", "description": "Mentions circular wait.", "marks": 1, "required": True},
            {"id": "r5", "type": "concept", "description": "Explains at least one condition in plain language.", "marks": 1, "required": False},
        ],
    },
    {
        "qid": "Q5",
        "question_type": "graph_algorithm",
        "question_text": "Explain BFS traversal, write queue-based pseudocode, and state the complexity.",
        "max_marks": 6,
        "items": [
            {"id": "r1", "type": "definition", "description": "States that BFS visits vertices level by level.", "marks": 1, "required": True},
            {"id": "r2", "type": "code", "description": "Uses a queue to process vertices.", "marks": 2, "required": True},
            {"id": "r3", "type": "code", "description": "Marks vertices as visited before enqueueing neighbours.", "marks": 2, "required": True},
            {"id": "r4", "type": "complexity", "description": "States complexity O(V + E).", "marks": 1, "required": True},
        ],
    },
    {
        "qid": "Q6",
        "question_type": "diagram_trace",
        "question_text": "Draw a Round Robin schedule with quantum 2 for processes P1, P2, P3 and comment on fairness.",
        "max_marks": 5,
        "items": [
            {"id": "r1", "type": "diagram_gantt", "description": "Shows a Round Robin Gantt chart with repeated process slices.", "marks": 3, "required": True},
            {"id": "r2", "type": "concept", "description": "Mentions equal time quantum or cyclic service.", "marks": 1, "required": True},
            {"id": "r3", "type": "final_answer", "description": "Comments that fairness is improved because no process waits forever.", "marks": 1, "required": False, "prerequisite": "r2"},
        ],
    },
]


SAMPLE_SPECS = [
    {
        "writer_id": "writer01",
        "style": {"jitter": 2, "blur": 0.1, "ink": "#1b1b1b"},
        "answers": {
            "Q1": [
                ("n1", "Starvation means a process can wait forever or be postponed indefinitely.", "definition", ["r1"]),
                ("n2", "It happens in priority scheduling when low priority jobs keep getting delayed.", "concept", ["r2"]),
                ("n3", "Gantt: |P2 0-2|P2 2-4|P3 4-6|P1 6-8|", "diagram_gantt", ["r3"]),
                ("n4", "Aging can raise the priority over time and reduce starvation.", "final_answer", ["r4"]),
            ],
            "Q2": [
                ("n1", "Step 1: divide the array into two halves until single elements remain.", "algorithm_step", ["r1"]),
                ("n2", "Step 2: merge the sorted halves back in order.", "algorithm_step", ["r2"]),
                ("n3", "Time complexity is O(n log n).", "complexity", ["r3"]),
                ("n4", "Extra auxiliary array gives O(n) space.", "complexity", ["r4"]),
            ],
            "Q3": [
                ("n1", "low = 0, high = n - 1", "code", ["r1"]),
                ("n2", "while low <= high: mid = (low + high) // 2", "code", ["r2"]),
                ("n3", "if key < a[mid] high = mid - 1 else low = mid + 1", "code", ["r3"]),
                ("n4", "Time complexity is O(log n).", "complexity", ["r4"]),
            ],
            "Q4": [
                ("n1", "Mutual exclusion and hold and wait are needed.", "definition", ["r1", "r2"]),
                ("n2", "No preemption means resources cannot be taken away.", "definition", ["r3", "r5"]),
                ("n3", "Circular wait means P1 waits for P2 and so on.", "definition", ["r4"]),
            ],
            "Q5": [
                ("n1", "BFS visits nodes level by level.", "definition", ["r1"]),
                ("n2", "Use a queue, enqueue source, then dequeue and process neighbours.", "code", ["r2"]),
                ("n3", "Mark visited before enqueueing each unvisited neighbour.", "code", ["r3"]),
                ("n4", "Complexity is O(V + E).", "complexity", ["r4"]),
            ],
            "Q6": [
                ("n1", "Gantt: |P1 0-2|P2 2-4|P3 4-6|P1 6-8|P2 8-9|", "diagram_gantt", ["r1"]),
                ("n2", "Round robin gives each process a time quantum in cyclic order.", "concept", ["r2"]),
                ("n3", "It is fairer because no process waits forever.", "final_answer", ["r3"]),
            ],
        },
    },
    {
        "writer_id": "writer02",
        "style": {"jitter": 4, "blur": 0.25, "ink": "#232b55"},
        "answers": {
            "Q1": [
                ("n1", "Starvation is indefinite waiting.", "definition", ["r1"]),
                ("n2", "Low priority process may not get CPU in unfair priority scheduling.", "concept", ["r2"]),
                ("n3", "Gantt: |P2 0-2|P2 2-4|P2 4-6|P1 6-8|", "diagram_gantt", ["r3"]),
            ],
            "Q2": [
                ("n1", "Split list into halves recursively.", "algorithm_step", ["r1"]),
                ("n2", "Merge sorted sublists.", "algorithm_step", ["r2"]),
                ("n3", "O(n log n)", "complexity", ["r3"]),
            ],
            "Q3": [
                ("n1", "start with low and high pointers", "code", ["r1"]),
                ("n2", "mid = (low+high)/2 and compare target with middle element", "code", ["r2"]),
                ("n3", "update bounds and continue", "code", ["r3"]),
            ],
            "Q4": [
                ("n1", "Conditions are mutual exclusion, hold and wait, no preemption, circular wait.", "definition", ["r1", "r2", "r3", "r4"]),
            ],
            "Q5": [
                ("n1", "BFS uses queue and visits level wise.", "definition", ["r1"]),
                ("n2", "enqueue source and then neighbours", "code", ["r2"]),
                ("n3", "O(V + E)", "complexity", ["r4"]),
            ],
            "Q6": [
                ("n1", "Gantt: |P1 0-2|P2 2-4|P3 4-6|P1 6-8|", "diagram_gantt", ["r1"]),
                ("n2", "Each process gets equal quantum.", "concept", ["r2"]),
            ],
        },
    },
    {
        "writer_id": "writer03",
        "style": {"jitter": 6, "blur": 0.45, "ink": "#4a3b2f"},
        "answers": {
            "Q1": [
                ("n1", "Starvation means the job keeps waiting.", "definition", ["r1"]),
                ("n2", "aging helps by increasing priority slowly", "final_answer", ["r4"]),
            ],
            "Q2": [
                ("n1", "merge sort first divides, later combines arrays", "algorithm_step", ["r1", "r2"]),
                ("n2", "time is O(n log n) and extra array O(n)", "complexity", ["r3", "r4"]),
            ],
            "Q3": [
                ("n1", "mid is checked each time in binary search", "code", ["r2"]),
                ("n2", "if smaller move left else move right", "code", ["r3"]),
                ("n3", "O(log n)", "complexity", ["r4"]),
            ],
            "Q4": [
                ("n1", "Deadlock needs mutual exclusion and hold wait.", "definition", ["r1", "r2"]),
                ("n2", "Resources are not preempted.", "definition", ["r3", "r5"]),
            ],
            "Q5": [
                ("n1", "BFS visits level by level.", "definition", ["r1"]),
                ("n2", "Use queue. visited array avoids repetition.", "code", ["r2", "r3"]),
            ],
            "Q6": [
                ("n1", "Gantt: |P1|P2|P3|P1|", "diagram_gantt", ["r1"]),
                ("n2", "cyclic service is fair", "concept", ["r2"]),
                ("n3", "no one waits forever", "final_answer", ["r3"]),
            ],
        },
    },
]


def _build_gold(item_ids: List[str], rubric_items: List[Dict[str, object]]) -> Tuple[Dict[str, bool], float]:
    item_hits = {item["id"]: False for item in rubric_items}
    total = 0.0
    for item in rubric_items:
        if item["id"] in item_ids:
            item_hits[item["id"]] = True
            total += float(item["marks"])
    return item_hits, total


def _draw_wrapped(draw: ImageDraw.ImageDraw, xy: Tuple[int, int], text: str, width: int, font, fill: str) -> int:
    x, y = xy
    lines = wrap(text, width=width)
    line_height = 25
    for idx, line in enumerate(lines):
        draw.text((x, y + idx * line_height), line, font=font, fill=fill)
    return y + max(1, len(lines)) * line_height


def _render_sample_image(output_path: Path, question_text: str, regions: List[Dict[str, object]], style: Dict[str, object]) -> None:
    random.seed(output_path.stem)
    image = Image.new("RGB", (900, 1200), color="#f7f2ea")
    draw = ImageDraw.Draw(image)
    question_font = _font(24)
    answer_font = _font(20)
    draw.rectangle([20, 20, 880, 110], outline="#3b5973", width=3)
    question_y = _draw_wrapped(draw, (35, 35), f"Question: {question_text}", width=62, font=question_font, fill="#233443")

    for region in regions:
        x1, y1, x2, y2 = region["bbox"]
        jitter = int(style["jitter"])
        x1 += random.randint(-jitter, jitter)
        y1 += random.randint(-jitter, jitter)
        draw.text((x1, y1), region["text"], font=answer_font, fill=style["ink"])
        if region["type_hint"] == "diagram_gantt":
            draw.rounded_rectangle([x1, y1 + 28, min(x2, 820), y1 + 95], outline="#00897b", width=2, radius=8)
            segments = []
            for token in region["text"].split("|"):
                token = token.strip()
                if token and token.lower().startswith("p"):
                    segments.append(token)
            start_x = x1 + 12
            top = y1 + 42
            height = 32
            colors = ["#90caf9", "#ffcc80", "#a5d6a7", "#ef9a9a"]
            for idx, seg in enumerate(segments):
                end_x = start_x + 85
                draw.rectangle([start_x, top, end_x, top + height], outline="#00695c", fill=colors[idx % len(colors)], width=2)
                draw.text((start_x + 10, top + 8), seg.split()[0], fill="#102027", font=_font(16))
                start_x = end_x

    image = image.filter(ImageFilter.GaussianBlur(radius=float(style["blur"])))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def _answer_key_text(rubrics: List[Dict[str, object]]) -> str:
    lines: List[str] = ["SCOREMAP Structured Answer Key", ""]
    for rubric in rubrics:
        lines.extend(
            [
                f"Question ID: {rubric['qid']}",
                f"Question Type: {rubric['question_type']}",
                f"Question Text: {rubric['question_text']}",
                f"Max Marks: {rubric['max_marks']}",
            ]
        )
        for item in rubric["items"]:
            lines.extend(
                [
                    f"Item ID: {item['id']}",
                    f"Item Type: {item['type']}",
                    f"Description: {item['description']}",
                    f"Marks: {item['marks']}",
                    f"Required: {str(item.get('required', False)).lower()}",
                    f"Order: {item.get('order', '')}",
                    f"Prerequisite: {item.get('prerequisite', '')}",
                    f"Alternatives: {', '.join(item.get('alternatives', []))}",
                ]
            )
        lines.append("=" * 48)
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def _write_answer_key_pdf(text: str, output_path: Path) -> None:
    lines = text.splitlines()
    lines_per_page = 40
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"font.family": "monospace"}):
        pages = [lines[idx : idx + lines_per_page] for idx in range(0, len(lines), lines_per_page)]
        if not pages:
            pages = [[]]
        from matplotlib.backends.backend_pdf import PdfPages

        with PdfPages(output_path) as pdf:
            for page_lines in pages:
                fig = plt.figure(figsize=(8.27, 11.69))
                fig.text(0.04, 0.97, "\n".join(page_lines), va="top", ha="left", fontsize=10, family="monospace")
                fig.patch.set_facecolor("white")
                pdf.savefig(fig)
                plt.close(fig)


def build_dataset(project_root: Path) -> Dict[str, List[str]]:
    data_root = project_root / "04_data" / "sample_inputs"
    answers_dir = data_root / "answers"
    images_dir = data_root / "images"
    rubrics_dir = data_root / "rubrics"
    answer_keys_dir = data_root / "answer_keys"
    splits_dir = data_root / "splits"
    demo_dir = project_root / "06_demo" / "demo_inputs"
    for directory in [answers_dir, images_dir, rubrics_dir, answer_keys_dir, splits_dir, demo_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    for rubric in RUBRICS:
        (rubrics_dir / f"{rubric['qid']}.json").write_text(json.dumps(rubric, indent=2), encoding="utf-8")

    answer_key_text = _answer_key_text(RUBRICS)
    (answer_keys_dir / "scoremap_answer_key.txt").write_text(answer_key_text, encoding="utf-8")
    _write_answer_key_pdf(answer_key_text, answer_keys_dir / "scoremap_answer_key.pdf")

    split_map = {"train": [], "val": [], "test": []}
    writer_split = {"writer01": "train", "writer02": "val", "writer03": "test"}

    for spec in SAMPLE_SPECS:
        writer_id = spec["writer_id"]
        split = writer_split[writer_id]
        for rubric in RUBRICS:
            qid = rubric["qid"]
            sample_id = f"{writer_id.lower()}_{qid.lower()}"
            answer_regions = []
            gold_item_ids: List[str] = []
            gold_evidence: Dict[str, List[str]] = {item["id"]: [] for item in rubric["items"]}
            y_cursor = 145
            for line_index, (region_id, text, type_hint, item_ids) in enumerate(spec["answers"][qid], start=1):
                bbox = [58, y_cursor, 810, y_cursor + 90]
                metadata = {"line_index": line_index}
                if type_hint == "diagram_gantt":
                    metadata["diagram_segments"] = [token.strip() for token in text.split("|") if token.strip().startswith("P")]
                    bbox = [58, y_cursor, 840, y_cursor + 135]
                answer_regions.append(
                    {
                        "id": region_id,
                        "bbox": bbox,
                        "text": text,
                        "type_hint": type_hint,
                        "metadata": metadata,
                    }
                )
                gold_item_ids.extend(item_ids)
                for item_id in item_ids:
                    gold_evidence[item_id].append(region_id)
                y_cursor += 145 if type_hint == "diagram_gantt" else 95

            item_hits, total = _build_gold(gold_item_ids, rubric["items"])
            image_relative = f"images/{sample_id}.png"
            payload = {
                "sample_id": sample_id,
                "writer_id": writer_id,
                "qid": qid,
                "question_text": rubric["question_text"],
                "image_path": image_relative,
                "regions": answer_regions,
                "gold": {
                    "total_score": total,
                    "item_hits": item_hits,
                    "evidence": gold_evidence,
                },
            }
            (answers_dir / f"{sample_id}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
            _render_sample_image(images_dir / f"{sample_id}.png", rubric["question_text"], answer_regions, spec["style"])
            split_map[split].append(sample_id)

    for split, sample_ids in split_map.items():
        (splits_dir / f"{split}.json").write_text(json.dumps(sample_ids, indent=2), encoding="utf-8")

    demo_samples = split_map["test"][:2]
    for sample_id in demo_samples:
        shutil.copy2(answers_dir / f"{sample_id}.json", demo_dir / f"{sample_id}.json")
        shutil.copy2(images_dir / f"{sample_id}.png", demo_dir / f"{sample_id}.png")
    return split_map
