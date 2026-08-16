"""Prototype V1 for Nova Labs wholesale order intake.

This implementation is intentionally deterministic-first.
It is designed to be executable when an OCR/runtime environment is available,
but in this session the runtime connection was blocked, so it was preserved
rather than executed.

Supported concepts:
- format detection
- text extraction hooks
- deterministic validation
- exact / fuzzy / alias catalog matching
- confidence heuristics
- exception detection
- human review package generation

The script is self-contained and avoids non-stdlib dependencies.
"""
from __future__ import annotations

import csv
import dataclasses
import difflib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass
class CatalogItem:
    sku: str
    description: str
    unit: str
    unit_price: float


@dataclass
class ExtractedLine:
    raw_source: str
    customer_item_number: Optional[str]
    raw_description: Optional[str]
    quantity: Optional[float]
    unit: Optional[str]
    sku_hint: Optional[str] = None
    source_conflicts: Optional[List[str]] = None
    customer_alias: Optional[str] = None


@dataclass
class MatchResult:
    proposed_sku: Optional[str]
    matched_description: Optional[str]
    confidence: str
    review_reason: Optional[str]
    alternative_matches: List[Tuple[str, float]]
    status: str


@dataclass
class ValidationIssue:
    code: str
    message: str
    severity: str


NORMALIZE_RE = re.compile(r"[^a-z0-9]+")


def normalize(text: str) -> str:
    text = text.lower().strip()
    text = text.replace("in.", "in").replace("‑", "-")
    text = text.replace("×", "x")
    text = text.replace("full port", "full-port")
    text = text.replace("reduced port", "reduced-port")
    text = text.replace("90 degree", "90 degree")
    return NORMALIZE_RE.sub(" ", text).strip()


def tokenize(text: str) -> List[str]:
    return [tok for tok in normalize(text).split() if tok]


class CatalogMatcher:
    def __init__(self, items: Sequence[CatalogItem]):
        self.items = list(items)
        self.by_sku = {item.sku.upper(): item for item in self.items}

    def exact_match(self, sku: str) -> Optional[CatalogItem]:
        return self.by_sku.get(sku.upper())

    def _description_score(self, a: str, b: str) -> float:
        return difflib.SequenceMatcher(None, normalize(a), normalize(b)).ratio()

    def match(self, line: ExtractedLine) -> MatchResult:
        candidates: List[Tuple[CatalogItem, float]] = []
        review_reason = None
        status = "READY"

        if line.sku_hint:
            exact = self.exact_match(line.sku_hint)
            if exact:
                return MatchResult(exact.sku, exact.description, "HIGH", None, [], "READY")

        desc = line.raw_description or ""
        if not desc and not line.sku_hint:
            return MatchResult(None, None, "NONE", "missing description", [], "EXCEPTION")

        norm_desc = normalize(desc)
        for item in self.items:
            score = self._description_score(norm_desc, item.description)
            # Extra boost for SKU-like hints embedded in description.
            if line.sku_hint and line.sku_hint.upper() == item.sku.upper():
                score = 1.0
            elif line.sku_hint and line.sku_hint.upper().replace(" ", "") == item.sku.upper().replace(" ", ""):
                score = max(score, 0.92)
            candidates.append((item, score))

        candidates.sort(key=lambda x: x[1], reverse=True)
        best_item, best_score = candidates[0]
        alt = [(c.sku, round(score, 3)) for c, score in candidates[1:4]]
        ambiguity = len(candidates) > 1 and abs(best_score - candidates[1][1]) < 0.05

        if best_score >= 0.95 and not ambiguity:
            conf = "HIGH"
        elif best_score >= 0.84 and not ambiguity:
            conf = "MEDIUM"
            status = "REVIEW REQUIRED"
            review_reason = "fuzzy description match"
        elif best_score >= 0.72:
            conf = "LOW"
            status = "REVIEW REQUIRED"
            review_reason = "ambiguous catalog match"
        else:
            return MatchResult(None, None, "NONE", "no catalog match", alt, "EXCEPTION")

        if ambiguity:
            status = "REVIEW REQUIRED"
            review_reason = review_reason or "ambiguous multiple matches"
            conf = "LOW" if conf != "HIGH" else "MEDIUM"

        return MatchResult(best_item.sku, best_item.description, conf, review_reason, alt, status)


class Validator:
    def __init__(self, catalog: CatalogMatcher):
        self.catalog = catalog

    def validate_line(self, line: ExtractedLine) -> Tuple[List[ValidationIssue], MatchResult]:
        issues: List[ValidationIssue] = []

        if line.quantity is None:
            issues.append(ValidationIssue("missing_quantity", "missing quantity", "high"))
        elif not isinstance(line.quantity, (int, float)):
            issues.append(ValidationIssue("bad_quantity", "quantity is not numeric", "high"))
        elif line.quantity <= 0:
            issues.append(ValidationIssue("bad_quantity", "quantity must be positive", "high"))

        if line.unit is None or not str(line.unit).strip():
            issues.append(ValidationIssue("missing_unit", "missing unit", "high"))

        match = self.catalog.match(line)
        if match.proposed_sku is None:
            issues.append(ValidationIssue("no_match", match.review_reason or "no match", "high"))
            match.status = "EXCEPTION"
            match.confidence = "NONE"
            return issues, match

        catalog_item = self.catalog.exact_match(match.proposed_sku)
        assert catalog_item is not None

        if line.unit and catalog_item.unit.lower() != str(line.unit).lower():
            issues.append(ValidationIssue(
                "unit_mismatch",
                f"source unit {line.unit!r} conflicts with catalog unit {catalog_item.unit!r}",
                "high",
            ))
            match.status = "REVIEW REQUIRED"
            if match.confidence == "HIGH":
                match.confidence = "MEDIUM"
            if not match.review_reason:
                match.review_reason = "unit compatibility conflict"

        if line.source_conflicts:
            issues.append(ValidationIssue("source_conflict", "; ".join(line.source_conflicts), "high"))
            match.status = "REVIEW REQUIRED"
            if not match.review_reason:
                match.review_reason = "source conflict"

        if any(issue.code == "missing_quantity" for issue in issues):
            match.status = "EXCEPTION"
            if match.confidence == "HIGH":
                match.confidence = "NONE"
        return issues, match


def build_review_row(line: ExtractedLine, match: MatchResult, issues: List[ValidationIssue]) -> Dict[str, object]:
    return {
        "raw_source": line.raw_source,
        "extracted_value": {
            "customer_item_number": line.customer_item_number,
            "raw_description": line.raw_description,
            "quantity": line.quantity,
            "unit": line.unit,
            "sku_hint": line.sku_hint,
        },
        "proposed_sku": match.proposed_sku,
        "matched_description": match.matched_description,
        "confidence": match.confidence,
        "review_reason": match.review_reason or (issues[0].message if issues else None),
        "alternative_matches": match.alternative_matches,
        "issues": [asdict(issue) for issue in issues],
        "status": match.status,
    }


def load_catalog(path: str | Path) -> List[CatalogItem]:
    items: List[CatalogItem] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            items.append(CatalogItem(
                sku=row["sku"],
                description=row["description"],
                unit=row["unit"],
                unit_price=float(row["unit_price"]),
            ))
    return items


def run_line_validation(catalog_path: str, extracted_lines: Iterable[ExtractedLine]) -> List[Dict[str, object]]:
    matcher = CatalogMatcher(load_catalog(catalog_path))
    validator = Validator(matcher)
    out = []
    for line in extracted_lines:
        issues, match = validator.validate_line(line)
        out.append(build_review_row(line, match, issues))
    return out


def detect_format(filename: str, raw_text: Optional[str] = None) -> str:
    lower = filename.lower()
    if lower.endswith(".csv"):
        return "csv"
    if lower.endswith(".xlsx") or lower.endswith(".xlsm") or lower.endswith(".xls") or lower.endswith(".xlsb"):
        return "spreadsheet"
    if lower.endswith(".pdf"):
        return "pdf_text" if raw_text else "pdf_scan"
    if lower.endswith(".eml") or lower.endswith(".txt"):
        return "email_text"
    if raw_text and "attachment" in raw_text.lower():
        return "email_attachment"
    return "unknown"


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--input-json", required=True, help="JSON list of extracted line objects")
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    with open(args.input_json, encoding="utf-8") as f:
        payload = json.load(f)
    lines = [ExtractedLine(**item) for item in payload]
    results = run_line_validation(args.catalog, lines)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
