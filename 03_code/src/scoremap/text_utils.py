from __future__ import annotations

import math
import re
from difflib import SequenceMatcher
from typing import Iterable, List, Sequence, Set


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "then",
    "this",
    "to",
    "using",
    "with",
}

SYNONYMS = {
    "forever": {"indefinite", "postponement", "starvation", "wait"},
    "indefinite": {"forever", "postponement", "wait"},
    "waits": {"wait", "forever"},
    "wait": {"waits", "forever", "indefinite"},
    "priority": {"scheduling", "unfair"},
    "aging": {"increase", "priority", "mitigation"},
    "divide": {"split", "halves", "partition"},
    "split": {"divide", "partition", "halves"},
    "merge": {"combine", "sorted", "halves"},
    "combine": {"merge", "sorted"},
    "queue": {"fifo", "enqueue", "dequeue"},
    "visited": {"mark", "seen"},
    "deadlock": {"blocked", "waiting"},
    "complexity": {"time", "space", "bigo"},
    "gantt": {"timeline", "chart", "segments"},
    "mutual": {"exclusive", "exclusion"},
    "hold": {"wait"},
}

ASYMPTOTIC_PATTERN = re.compile(r"(?:O|Theta|Omega)\s*\([^)]*\)", re.IGNORECASE)


def normalize_text(text: str) -> str:
    text = text.lower()
    text = text.replace("theta", "theta")
    text = re.sub(r"[^a-z0-9()+\-*/\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> List[str]:
    normalized = normalize_text(text)
    return [tok for tok in normalized.split() if tok and tok not in STOPWORDS]


def expand_tokens(tokens: Sequence[str]) -> Set[str]:
    expanded = set(tokens)
    for token in list(tokens):
        for synonym in SYNONYMS.get(token, set()):
            expanded.add(synonym)
    return expanded


def overlap_score(text_a: str, text_b: str) -> float:
    tokens_a = expand_tokens(tokenize(text_a))
    tokens_b = expand_tokens(tokenize(text_b))
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = len(tokens_a & tokens_b)
    dice = (2.0 * intersection) / (len(tokens_a) + len(tokens_b))
    seq = SequenceMatcher(None, normalize_text(text_a), normalize_text(text_b)).ratio()
    return min(1.0, 0.75 * dice + 0.25 * seq)


def contains_any(text: str, keywords: Iterable[str]) -> bool:
    normalized = normalize_text(text)
    tokens = set(normalized.split())
    for keyword in keywords:
        normalized_keyword = normalize_text(keyword)
        if not normalized_keyword:
            continue
        if " " in normalized_keyword or any(not char.isalnum() for char in normalized_keyword):
            if normalized_keyword in normalized:
                return True
        elif normalized_keyword in tokens:
            return True
    return False


def extract_complexities(text: str) -> List[str]:
    values = []
    for match in ASYMPTOTIC_PATTERN.finditer(text.replace(" ", "")):
        values.append(match.group(0).lower())
    if values:
        return values

    normalized = normalize_text(text)
    compact = normalized.replace(" ", "")
    fallbacks = re.findall(r"o\([^)]*\)|theta\([^)]*\)|omega\([^)]*\)", compact)
    return fallbacks


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def logit_confidence(score: float) -> float:
    score = clamp(score, 1e-6, 1.0 - 1e-6)
    odds = score / (1.0 - score)
    return clamp((math.log(odds) + 5.0) / 10.0)
