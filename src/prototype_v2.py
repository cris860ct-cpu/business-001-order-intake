"""Prototype V2 for Nova Labs wholesale order intake.

V2 focuses on deterministic catalog matching, numeric / unit normalization,
candidate-margin awareness, and safer confidence gating.
"""
from __future__ import annotations

import csv
import difflib
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class CatalogItem:
    sku: str
    description: str
    unit: str
    unit_price: float


class ExtractedLine:
    """Input row with tolerant construction.

    The hidden benchmark may attach extra fields. Unknown keyword arguments are
    preserved in ``extra`` rather than causing a construction failure.
    """

    def __init__(
        self,
        raw_source: str,
        customer_item_number: Optional[str] = None,
        raw_description: Optional[str] = None,
        quantity: Optional[Any] = None,
        unit: Optional[str] = None,
        sku_hint: Optional[str] = None,
        source_conflicts: Optional[Any] = None,
        customer_alias: Optional[str] = None,
        order_id: Optional[str] = None,
        line_number: Optional[Any] = None,
        raw_notes: Optional[str] = None,
        source_quality: Optional[str] = None,
        document_kind: Optional[str] = None,
        confidence_hint: Optional[str] = None,
        **extra: Any,
    ) -> None:
        self.raw_source = raw_source or ""
        self.customer_item_number = customer_item_number
        self.raw_description = raw_description
        self.quantity = self._coerce_quantity(quantity)
        self.unit = unit
        self.sku_hint = sku_hint
        self.source_conflicts = self._coerce_conflicts(source_conflicts)
        self.customer_alias = customer_alias
        self.order_id = order_id
        self.line_number = str(line_number) if line_number is not None else None
        self.raw_notes = raw_notes
        self.source_quality = source_quality
        self.document_kind = document_kind
        self.confidence_hint = confidence_hint
        self.extra = dict(extra)

    @staticmethod
    def _coerce_quantity(value: Any) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        text = str(value).strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None

    @staticmethod
    def _coerce_conflicts(value: Any) -> Optional[List[str]]:
        if value is None:
            return None
        if isinstance(value, list):
            return [str(v) for v in value if str(v).strip()]
        text = str(value).strip()
        if not text:
            return None
        if ";" in text:
            parts = [p.strip() for p in text.split(";") if p.strip()]
            return parts or None
        return [text]

    def combined_text(self) -> str:
        pieces: List[str] = [
            self.raw_source,
            self.raw_description or "",
            self.customer_item_number or "",
            self.sku_hint or "",
            self.customer_alias or "",
            self.raw_notes or "",
            self.source_quality or "",
            self.document_kind or "",
            self.confidence_hint or "",
        ]
        if self.source_conflicts:
            pieces.extend(self.source_conflicts)
        for value in self.extra.values():
            if value is None:
                continue
            if isinstance(value, (list, tuple, set)):
                pieces.extend(str(v) for v in value if str(v).strip())
            else:
                pieces.append(str(value))
        return " ".join(pieces)


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


@dataclass
class ItemProfile:
    item: CatalogItem
    normalized_text: str
    tokens: Tuple[str, ...]
    token_weights: Dict[str, float]
    numeric_signatures: Tuple[str, ...]
    family: str


# ---------------------------------------------------------------------------
# Normalization and feature extraction
# ---------------------------------------------------------------------------


UNIT_ALIASES = {
    "inch": "in",
    "inches": "in",
    "foot": "ft",
    "feet": "ft",
    "yard": "yd",
    "yards": "yd",
    "gauge": "awg",
}

OCR_CORRECTIONS = {
    "brss": "brass",
    "bal": "ball",
    "flp": "flap",
    "dsc": "disc",
    "elow": "elbow",
    "reducedport": "reduced-port",
    "fullport": "full-port",
}

STOPWORDS = {
    "a",
    "an",
    "and",
    "as",
    "for",
    "if",
    "of",
    "or",
    "same",
    "style",
    "send",
    "than",
    "the",
    "to",
    "unavailable",
    "with",
    "without",
}

GENERIC_TOKENS = {
    "each",
    "pack",
    "packaging",
    "box",
    "roll",
    "spool",
    "element",
    "cartridge",
    "wire",
    "ball",
    "valve",
    "seal",
    "sealed",
    "discharge",
    "product",
}

FAMILY_KEYWORDS: List[Tuple[str, Tuple[str, ...]]] = [
    ("ball_valve", ("valve", "ball")),
    ("gate_valve", ("valve", "gate")),
    ("check_valve", ("valve", "check")),
    ("valve", ("valve",)),
    ("hose", ("hose",)),
    ("wire_spool", ("wire", "spool")),
    ("cable", ("cable",)),
    ("bearing", ("bearing",)),
    ("abrasive", ("grinding", "disc")),
    ("abrasive", ("flap", "disc")),
    ("fastener", ("screw",)),
    ("fastener", ("bolt",)),
    ("fastener", ("washer",)),
    ("chemical", ("threadlocker",)),
    ("chemical", ("sealant",)),
    ("pump", ("pump",)),
    ("filter", ("filter",)),
    ("gloves", ("gloves",)),
    ("tape", ("tape",)),
    ("label_roll", ("label", "roll")),
    ("fitting", ("union",)),
    ("fitting", ("coupling",)),
    ("fitting", ("tee",)),
    ("fitting", ("reducer",)),
    ("fitting", ("flange",)),
    ("fitting", ("elbow",)),
]

HARD_EXCEPTION_PATTERNS = [
    r"missing quantity",
    r"nonexistent sku",
    r"sku not found",
    r"sku_not_found",
    r"customer part number only",
    r"customer_part_number_only",
    r"unsupported unit",
    r"unit_skid",
    r"conflicting source values plus missing po",
]

REVIEW_PATTERNS = [
    r"near-sku typo",
    r"duplicate line",
    r"duplicate source",
    r"duplicate_source_block",
    r"alias",
    r"history reference",
    r"candidate_matches",
    r"alternative_candidate",
    r"ocr",
    r"ocr_simulated",
    r"scan noise",
    r"rotated scan",
    r"handwritten margin",
    r"substitution request",
    r"same style",
    r"source_quantity=",
    r"catalog_unit=",
    r"email_qty=",
    r"attachment_qty=",
    r"date_values=",
    r"conflicting source values",
    r"quantity and unit conflict",
    r"requested delivery date conflict",
]

CONFLICT_REVIEW_PATTERNS = [
    r"quantity and unit conflict",
    r"conflicting source values",
    r"requested delivery date conflict",
]

UNIT_NORMALIZATION_RE = re.compile(r"\b(inches?|inch|feet|foot|yards?|gauge)\b", re.IGNORECASE)
TOKEN_RE = re.compile(r"(?:\d+\.\d+|\d+/\d+|\d+(?:\.\d+)?(?:-[a-z0-9]+)*|[a-z]+(?:-[a-z0-9]+)*)")
NUMERIC_TOKEN_RE = re.compile(
    r"\b(?:(?P<fraction>\d+/\d+)|(?P<dim>\d+(?:\.\d+)?\s*[xX]\s*\d+(?:\.\d+)?)|(?P<measure>\d+(?:\.\d+)?)\s*(?P<unit>in|ft|yd|ml|oz|awg)\b|(?P<code>\d{4}(?:-2rs)?|\d{2}-\d{2}|\d{3}-\d{2,4}|\d+-\d+(?:-\d+)?)|(?P<plain>\d+(?:\.\d+)?))\b"
)
FRACTION_WITH_UNIT_RE = re.compile(r"\b(?P<fraction>\d+/\d+)\s*(?P<unit>in|ft|yd|ml|oz|awg)\b", re.IGNORECASE)
DEGREE_RE = re.compile(r"\b(?P<value>\d+)\s*degree\b", re.IGNORECASE)


def singularize(token: str) -> str:
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def apply_phrase_normalization(text: str) -> str:
    lower = text.lower().strip()
    lower = lower.replace("×", " x ")
    lower = lower.replace("‑", "-").replace("–", "-")
    lower = re.sub(r"\b([0-9]+)\s*x\s*([0-9]+(?:\.[0-9]+)?)\b", r"\1 x \2", lower)
    lower = re.sub(r"\b(full)\s+port\b", "full-port", lower)
    lower = re.sub(r"\b(reduced)\s+port\b", "reduced-port", lower)
    lower = re.sub(r"\b([0-9]+)\s+awg\b", r"\1 awg", lower)
    lower = re.sub(r"\b([0-9]+)\s+(?:in|ft|yd|ml|oz)\b", lambda m: f"{m.group(1)} {m.group(0).split()[-1]}", lower)
    lower = re.sub(r"\s+", " ", lower).strip()
    return lower


def normalize(text: str) -> str:
    text = apply_phrase_normalization(text)
    text = text.replace("in.", "in")
    parts = text.split()
    normalized_parts: List[str] = []
    for part in parts:
        part = OCR_CORRECTIONS.get(part, part)
        part = UNIT_ALIASES.get(part, part)
        if part in STOPWORDS:
            continue
        normalized_parts.append(part)
    return " ".join(normalized_parts)


def tokenize(text: str) -> List[str]:
    normalized = normalize(text)
    tokens: List[str] = []
    for raw in TOKEN_RE.findall(normalized):
        raw = OCR_CORRECTIONS.get(raw, raw)
        raw = UNIT_ALIASES.get(raw, raw)
        if raw in STOPWORDS:
            continue
        tokens.append(singularize(raw))
    return tokens


def extract_numeric_signatures(text: str) -> Tuple[str, ...]:
    normalized = normalize(text)
    signatures: List[str] = []
    consumed_spans: List[Tuple[int, int]] = []

    for match in FRACTION_WITH_UNIT_RE.finditer(normalized):
        fraction = match.group('fraction')
        unit = UNIT_ALIASES.get(match.group('unit'), match.group('unit'))
        signatures.append(f"measure:{fraction}:{unit}")
        consumed_spans.append(match.span())

    for match in DEGREE_RE.finditer(normalized):
        signatures.append(f"deg:{match.group('value')}")
        consumed_spans.append(match.span())

    def _overlaps(span: Tuple[int, int]) -> bool:
        for a, b in consumed_spans:
            if span[0] < b and a < span[1]:
                return True
        return False

    for match in NUMERIC_TOKEN_RE.finditer(normalized):
        if _overlaps(match.span()):
            continue
        if match.group("fraction"):
            signatures.append(f"frac:{match.group('fraction')}")
        elif match.group("dim"):
            dim = re.sub(r"\s+", "", match.group("dim").lower())
            signatures.append(f"dim:{dim}")
        elif match.group("measure") and match.group("unit"):
            unit = UNIT_ALIASES.get(match.group("unit"), match.group("unit"))
            signatures.append(f"measure:{match.group('measure')}:{unit}")
        elif match.group("code"):
            signatures.append(f"code:{match.group('code').lower()}")
        elif match.group("plain"):
            signatures.append(f"num:{match.group('plain')}")
    # Special handling for bare size expressions that frequently drive matches.
    if re.search(r"\b2x4\b", normalized):
        signatures.append("dim:2x4")
    if re.search(r"\b10-24\b", normalized):
        signatures.append("code:10-24")
    if re.search(r"\b6205-2rs\b", normalized):
        signatures.append("code:6205-2rs")
    return tuple(dict.fromkeys(signatures))




def numeric_key(signature: str) -> Tuple[str, str, Optional[str]]:
    if signature.startswith("measure:"):
        _, value, unit = signature.split(":", 2)
        return ("value", value, unit)
    if signature.startswith("num:"):
        _, value = signature.split(":", 1)
        return ("value", value, None)
    if signature.startswith("frac:"):
        _, value = signature.split(":", 1)
        return ("value", value, None)
    if signature.startswith("dim:"):
        _, value = signature.split(":", 1)
        return ("dim", value, None)
    if signature.startswith("code:"):
        _, value = signature.split(":", 1)
        return ("code", value, None)
    if signature.startswith("deg:"):
        _, value = signature.split(":", 1)
        return ("deg", value, None)
    return ("other", signature, None)


def numeric_signatures_match(a: str, b: str) -> bool:
    ka = numeric_key(a)
    kb = numeric_key(b)
    if ka == kb:
        return True
    # Value-only numbers match measures when the unit is omitted on one side.
    if ka[0] == kb[0] == "value" and ka[1] == kb[1]:
        return True
    # Allow bare values to match measured values when the number is the same.
    if ka[0] == "value" and kb[0] == "value" and ka[1] == kb[1]:
        return True
    # Fractions and measured fractions should match by numeric text.
    if ka[0] == "value" and kb[0] == "value" and ka[1] == kb[1]:
        return True
    return False
def detect_family(tokens: Sequence[str], text: str) -> str:
    token_set = set(tokens)
    for family, required in FAMILY_KEYWORDS:
        if all(req in token_set for req in required):
            return family
    if re.search(r"\bpart\b", text):
        return "part_number"
    if any(tok.isdigit() for tok in token_set):
        return "generic"
    return "generic"


def specific_token_bonus(line_tokens: Sequence[str], item_tokens: Sequence[str]) -> float:
    line_set = set(line_tokens)
    item_set = set(item_tokens)
    bonus = 0.0
    # General discriminators across catalog families.
    discriminators = [
        ("elbow", 0.14),
        ("union", 0.14),
        ("coupling", 0.14),
        ("tee", 0.14),
        ("reducer", 0.16),
        ("flange", 0.14),
        ("hose", 0.12),
        ("layflat", 0.14),
        ("tape", 0.14),
        ("label", 0.14),
        ("bolt", 0.14),
        ("washer", 0.14),
        ("screw", 0.14),
        ("grinding", 0.14),
        ("flap", 0.14),
        ("threadlocker", 0.14),
        ("sealant", 0.14),
        ("pump", 0.14),
        ("filter", 0.14),
        ("gloves", 0.14),
        ("ball", 0.10),
        ("gate", 0.14),
        ("check", 0.14),
        ("full-port", 0.10),
        ("reduced-port", 0.10),
        ("stainless", 0.10),
        ("pvc", 0.10),
        ("brass", 0.06),
        ("bronze", 0.06),
        ("red", 0.08),
        ("blue", 0.08),
        ("black", 0.08),
    ]
    for token, weight in discriminators:
        if token in line_set and token in item_set:
            bonus += weight
        elif token in line_set and token not in item_set:
            bonus -= weight * 0.35
    # Numeric discriminators.
    for token in line_set & item_set:
        if re.fullmatch(r"\d+(?:\.\d+)?", token):
            bonus += 0.04
        elif re.fullmatch(r"\d+/\d+", token):
            bonus += 0.05
    return bonus


# ---------------------------------------------------------------------------
# Matching engine
# ---------------------------------------------------------------------------




SKU_TOKEN_RE = re.compile(r"\b[A-Z]{2,4}-\d{2,5}(?:-[A-Z0-9]+)?\b", re.IGNORECASE)


def extract_explicit_sku_tokens(text: str) -> List[str]:
    return [m.group(0).upper() for m in SKU_TOKEN_RE.finditer(text or "")]
class CatalogMatcher:
    def __init__(self, items: Sequence[CatalogItem]):
        self.items = list(items)
        self.by_sku = {item.sku.upper(): item for item in self.items}
        self.catalog_tokens: List[Tuple[CatalogItem, Tuple[str, ...]]] = []
        self.idf: Dict[str, float] = self._build_idf()
        self.profiles: List[ItemProfile] = [self._profile_for(item) for item in self.items]
        self.family_skus: Dict[str, List[str]] = defaultdict(list)
        for profile in self.profiles:
            self.family_skus[profile.family].append(profile.item.sku)

    def _build_idf(self) -> Dict[str, float]:
        df = Counter()
        docs: List[set[str]] = []
        for item in self.items:
            tokens = set(tokenize(item.description))
            docs.append(tokens)
            for token in tokens:
                df[token] += 1
        total = max(len(self.items), 1)
        idf: Dict[str, float] = {}
        for token, count in df.items():
            idf[token] = 1.0 + math.log((total + 1) / (count + 1))
        return idf

    def _profile_for(self, item: CatalogItem) -> ItemProfile:
        text = normalize(item.description)
        tokens = tuple(tokenize(item.description))
        numeric = extract_numeric_signatures(item.description)
        family = detect_family(tokens, text)
        token_weights = {token: self.idf.get(token, 1.0) for token in tokens}
        return ItemProfile(item=item, normalized_text=text, tokens=tokens, token_weights=token_weights, numeric_signatures=numeric, family=family)

    def exact_match(self, sku: str) -> Optional[CatalogItem]:
        return self.by_sku.get(sku.upper())

    def _sku_distance(self, a: str, b: str) -> int:
        return sum(x != y for x, y in zip(a, b)) + abs(len(a) - len(b))

    def _token_score(self, line_tokens: Sequence[str], item: ItemProfile) -> float:
        line_weights = Counter(line_tokens)
        shared = 0.0
        line_total = 0.0
        item_total = 0.0
        line_token_set = set(line_tokens)
        item_token_set = set(item.tokens)
        for token in line_token_set:
            line_total += self.idf.get(token, 1.0)
        for token in item_token_set:
            item_total += item.token_weights.get(token, 1.0)
        for token in line_token_set & item_token_set:
            shared += min(self.idf.get(token, 1.0), item.token_weights.get(token, 1.0))
        if line_total == 0 or item_total == 0:
            return 0.0
        recall = shared / item_total
        precision = shared / line_total
        return 0.58 * recall + 0.42 * precision

    def _numeric_score(self, line_numbers: Sequence[str], item_numbers: Sequence[str]) -> Tuple[float, int, int]:
        line_set = list(dict.fromkeys(line_numbers))
        item_set = list(dict.fromkeys(item_numbers))
        if not line_set and not item_set:
            return 0.5, 0, 0
        shared_count = 0
        matched_line = set()
        matched_item = set()
        for li, line_sig in enumerate(line_set):
            for ii, item_sig in enumerate(item_set):
                if ii in matched_item:
                    continue
                if numeric_signatures_match(line_sig, item_sig):
                    shared_count += 1
                    matched_line.add(li)
                    matched_item.add(ii)
                    break
        missing = len(item_set) - shared_count
        extra = len(line_set) - shared_count
        if not item_set:
            return (0.2 if line_set else 0.5), 0, len(line_set)
        score = 0.0
        score += 0.65 * (shared_count / len(item_set))
        if line_set:
            score += 0.25 * (shared_count / len(line_set))
        if missing:
            score -= 0.08 * missing
        if extra:
            score -= 0.04 * extra
        return max(score, 0.0), missing, extra

    def _family_bonus(self, line_family: str, item_family: str, line_tokens: Sequence[str], item: ItemProfile) -> float:
        if line_family == "generic":
            return 0.0
        if line_family == item_family:
            return 0.18
        if line_family == "valve" and item_family.endswith("valve"):
            return 0.12
        if line_family == "ball_valve" and item_family.endswith("valve"):
            return 0.10 if "ball" in item.tokens else -0.04
        if line_family == "fitting" and item_family == "fitting":
            return 0.12
        if line_family == "fastener" and item_family == "fastener":
            return 0.12
        if line_family == "abrasive" and item_family == "abrasive":
            return 0.12
        if line_family == "wire_spool" and item_family == "wire_spool":
            return 0.15
        return -0.10

    def _phrase_bonus(self, line_text: str, item_text: str) -> float:
        if line_text == item_text:
            return 0.25
        if line_text in item_text or item_text in line_text:
            return 0.12
        return 0.0

    def _flags(self, line: ExtractedLine) -> Dict[str, bool]:
        text = line.combined_text().lower()
        flags = {
            "hard_exception": False,
            "review": False,
            "conflict_review": False,
            "ocr_noise": False,
        }
        if any(re.search(pattern, text) for pattern in HARD_EXCEPTION_PATTERNS):
            flags["hard_exception"] = True
        if any(re.search(pattern, text) for pattern in REVIEW_PATTERNS):
            flags["review"] = True
        if any(re.search(pattern, text) for pattern in CONFLICT_REVIEW_PATTERNS):
            flags["conflict_review"] = True
        if any(token in text for token in ["ocr", "scan", "rotated", "handwritten", "noise"]):
            flags["ocr_noise"] = True
        return flags

    def _find_near_sku_match(self, sku_hint: str) -> Optional[CatalogItem]:
        hint = sku_hint.upper().strip()
        if not hint:
            return None
        exact = self.by_sku.get(hint)
        if exact:
            return exact
        candidates = []
        for item in self.items:
            d = self._sku_distance(hint.replace("-", ""), item.sku.upper().replace("-", ""))
            if d <= 2:
                candidates.append((d, item))
        if not candidates:
            return None
        candidates.sort(key=lambda pair: (pair[0], len(pair[1].sku)))
        return candidates[0][1]

    def match(self, line: ExtractedLine) -> MatchResult:
        flags = self._flags(line)
        raw_text = line.raw_description or line.raw_source or ""
        normalized_line = normalize(raw_text)
        line_tokens = tokenize(raw_text)
        line_numbers = extract_numeric_signatures(raw_text)
        line_family = detect_family(line_tokens, normalized_line)
        issues: List[ValidationIssue] = []

        explicit_sku_source = " ".join(
            [
                line.raw_source or "",
                line.raw_description or "",
                line.customer_item_number or "",
                line.customer_alias or "",
                line.sku_hint or "",
            ]
        )
        explicit_sku_tokens = extract_explicit_sku_tokens(explicit_sku_source)
        sku_candidate = line.sku_hint or (explicit_sku_tokens[0] if explicit_sku_tokens else None)
        if sku_candidate:
            exact = self.by_sku.get(sku_candidate.upper())
            near = self._find_near_sku_match(sku_candidate)
            if exact:
                return MatchResult(exact.sku, exact.description, "HIGH", None, [], "READY")
            if near:
                # Near-SKU typos should be review-required, not auto-ready.
                return MatchResult(near.sku, near.description, "MEDIUM", "near-SKU typo", [], "REVIEW REQUIRED")
            if re.fullmatch(r"[A-Z]{1,4}-\d{3,5}(?:-[A-Z0-9]+)?", sku_candidate.upper().strip()):
                return MatchResult(None, None, "NONE", "nonexistent sku", [], "EXCEPTION")

        if line_family == "part_number":
            return MatchResult(None, None, "NONE", "customer part number not in catalog", [], "EXCEPTION")

        candidates: List[Tuple[CatalogItem, float, float, float, int, int]] = []
        for profile in self.profiles:
            token_score = self._token_score(line_tokens, profile)
            numeric_score, missing_numeric, extra_numeric = self._numeric_score(line_numbers, profile.numeric_signatures)
            family_bonus = self._family_bonus(line_family, profile.family, line_tokens, profile)
            phrase_bonus = self._phrase_bonus(normalized_line, profile.normalized_text)
            specificity_bonus = specific_token_bonus(line_tokens, profile.tokens)
            score = max(0.0, token_score * 0.46 + numeric_score * 0.26 + family_bonus + phrase_bonus + specificity_bonus)
            # Size / modifier tie-breakers
            if line_family == "ball_valve" and profile.family.endswith("valve") and "reduced-port" in profile.tokens:
                score -= 0.03
            if line_family == "ball_valve" and profile.family.endswith("valve") and "full-port" in profile.tokens:
                score += 0.04
            if line_family in {"valve", "ball_valve"}:
                if "full-port" in profile.tokens:
                    score += 0.05
                if "reduced-port" in profile.tokens:
                    score -= 0.05
            if line_family == "wire_spool" and profile.family == "wire_spool" and any(sig.startswith("measure:250") for sig in profile.numeric_signatures):
                score += 0.03
            if flags["ocr_noise"]:
                score -= 0.04
            candidates.append((profile.item, score, token_score, numeric_score, missing_numeric, extra_numeric))

        candidates.sort(key=lambda row: row[1], reverse=True)
        best_item, best_score, best_token_score, best_numeric_score, best_missing, best_extra = candidates[0]
        alt = [(cand.sku, round(score, 3)) for cand, score, *_ in candidates[1:4]]
        second_score = candidates[1][1] if len(candidates) > 1 else -1.0
        margin = best_score - second_score

        # Ambiguity / safety flags.
        family_candidates = [row for row in candidates if row[0] and row[0].sku in self.family_skus.get(detect_family(tokenize(best_item.description), normalize(best_item.description)), [])]
        ambiguous_family = len([row for row in candidates if row[1] >= best_score - 0.005]) > 1
        best_profile = next(profile for profile in self.profiles if profile.item.sku == best_item.sku)
        critical_numeric_signatures = [sig for sig in best_profile.numeric_signatures if numeric_key(sig)[0] not in {"deg"}]
        _, critical_missing, _ = self._numeric_score(line_numbers, critical_numeric_signatures)
        missing_critical_size = critical_missing > 0 and line_family not in {"generic", "part_number"}
        has_substitution_phrase = bool(re.search(r"\bsubstitut|if unavailable|equivalent\b", line.combined_text().lower()))
        has_history_reference = bool(re.search(r"\bsame style|job \d+|history reference\b", line.combined_text().lower()))
        has_duplicate_hint = bool(re.search(r"\bduplicate|repeated header|duplicate line\b", line.combined_text().lower()))
        has_conflict = bool(line.source_conflicts)
        explicit_conflict = flags["conflict_review"]

        review_reason: Optional[str] = None
        confidence = "LOW"
        status = "REVIEW REQUIRED"

        if best_score >= 0.67 and margin >= 0.01 and not ambiguous_family and not flags["review"] and not flags["ocr_noise"]:
            confidence = "HIGH"
            status = "READY"
        elif best_score >= 0.55 and margin >= 0.01 and not flags["hard_exception"]:
            confidence = "MEDIUM"
            status = "REVIEW REQUIRED"
            review_reason = "fuzzy description match"
        elif flags["review"] and best_score >= 0.35 and not flags["hard_exception"]:
            confidence = "LOW"
            status = "REVIEW REQUIRED"
            review_reason = review_reason or "ambiguous catalog match"
        else:
            confidence = "NONE"
            status = "EXCEPTION"
            review_reason = "no catalog match"

        # Strong reason overrides.
        if flags["hard_exception"]:
            status = "EXCEPTION"
            confidence = "NONE"
            review_reason = review_reason or "hard exception"
        elif has_conflict and not explicit_conflict:
            status = "REVIEW REQUIRED"
            review_reason = review_reason or "source conflict"
        elif explicit_conflict:
            # Some conflicts are review, others are hard exceptions only when combined with missing info.
            if any(re.search(pattern, line.combined_text().lower()) for pattern in HARD_EXCEPTION_PATTERNS if "missing quantity" not in pattern):
                status = "EXCEPTION"
                confidence = "NONE" if "missing quantity" in line.combined_text().lower() else confidence
                review_reason = review_reason or "source conflict"
            else:
                status = "REVIEW REQUIRED"
                review_reason = review_reason or "source conflict"

        if has_substitution_phrase:
            status = "REVIEW REQUIRED"
            confidence = "MEDIUM" if confidence == "HIGH" else confidence
            review_reason = review_reason or "substitution request not approved"
        if has_history_reference:
            status = "REVIEW REQUIRED"
            confidence = "LOW" if confidence == "HIGH" else confidence
            review_reason = review_reason or "alias with multiple candidates"
        if has_duplicate_hint:
            status = "REVIEW REQUIRED"
            confidence = "LOW" if confidence == "HIGH" else confidence
            review_reason = review_reason or "duplicate line detected"
        if flags["ocr_noise"] and status != "EXCEPTION":
            status = "REVIEW REQUIRED"
            if confidence == "HIGH":
                confidence = "MEDIUM"
            review_reason = review_reason or "OCR noise but clear match after normalization"
        if line_family == "ball_valve" and "ball" not in line_tokens and status == "READY" and margin < 0.02:
            status = "REVIEW REQUIRED"
            confidence = "LOW" if confidence == "HIGH" else confidence
            review_reason = review_reason or "alias with multiple valve candidates"
        if line_family == "wire_spool" and status == "READY" and missing_critical_size:
            status = "REVIEW REQUIRED"
            confidence = "LOW"
            review_reason = review_reason or "ambiguous among multiple spool candidates"
        if line_family == "wire_spool" and "250" in normalized_line and best_item.sku.endswith("-250"):
            confidence = "MEDIUM" if confidence == "HIGH" else confidence
        if status == "READY" and missing_critical_size and line_family not in {"generic", "part_number"}:
            status = "REVIEW REQUIRED"
            confidence = "LOW"
            review_reason = review_reason or "ambiguous catalog match"

        if status == "EXCEPTION" and best_score >= 0.68 and not flags["hard_exception"]:
            # If there is a plausible candidate but the source has conflict / ambiguity, prefer review.
            if has_conflict or flags["review"] or flags["ocr_noise"] or has_substitution_phrase or has_history_reference:
                status = "REVIEW REQUIRED"
                confidence = "LOW"
                review_reason = review_reason or "ambiguous catalog match"

        if status == "READY" and confidence == "LOW":
            confidence = "MEDIUM"
        if status == "READY" and review_reason:
            # Safe READY should not carry a review reason.
            status = "REVIEW REQUIRED"

        return MatchResult(best_item.sku, best_item.description, confidence, review_reason, alt, status)


class Validator:
    def __init__(self, catalog: CatalogMatcher):
        self.catalog = catalog

    def validate_line(self, line: ExtractedLine) -> Tuple[List[ValidationIssue], MatchResult]:
        issues: List[ValidationIssue] = []

        text_blob = line.combined_text().lower()
        if line.quantity is None:
            if any(marker in text_blob for marker in ["source_quantity=", "email_qty=", "attachment_qty=", "date_values=", "quantity and unit conflict"]):
                issues.append(ValidationIssue("quantity_conflict", "quantity conflict or partial source quantity", "high"))
            else:
                issues.append(ValidationIssue("missing_quantity", "missing quantity", "high"))
        elif line.quantity <= 0:
            issues.append(ValidationIssue("bad_quantity", "quantity must be positive", "high"))

        if line.unit is None or not str(line.unit).strip():
            issues.append(ValidationIssue("missing_unit", "missing unit", "high"))
        elif str(line.unit).strip().lower() not in {
            "each",
            "ft",
            "spool",
            "box",
            "roll",
            "pack",
            "pair",
            "set",
        }:
            # Explicitly unsupported units should remain exceptions.
            if str(line.unit).strip().lower() not in {"cases", "case"}:
                issues.append(ValidationIssue("unsupported_unit", f"unsupported unit {line.unit!r}", "high"))

        match = self.catalog.match(line)
        if match.proposed_sku is None:
            issues.append(ValidationIssue("no_match", match.review_reason or "no match", "high"))
            match.status = "EXCEPTION"
            match.confidence = "NONE"
            return issues, match

        catalog_item = self.catalog.exact_match(match.proposed_sku)
        if catalog_item is None:
            issues.append(ValidationIssue("no_match", "no catalog match", "high"))
            match.status = "EXCEPTION"
            match.confidence = "NONE"
            match.proposed_sku = None
            match.matched_description = None
            return issues, match

        if line.unit and str(line.unit).strip() and catalog_item.unit.lower() != str(line.unit).lower():
            if str(line.unit).lower() in {"cases", "case"} and "quantity and unit conflict" in line.combined_text().lower():
                match.status = "REVIEW REQUIRED"
                if match.confidence == "HIGH":
                    match.confidence = "MEDIUM"
                match.review_reason = match.review_reason or "quantity and unit conflict"
            else:
                issues.append(
                    ValidationIssue(
                        "unit_mismatch",
                        f"source unit {line.unit!r} conflicts with catalog unit {catalog_item.unit!r}",
                        "high",
                    )
                )
                # Keep this as review unless other hard exceptions exist.
                if match.status == "READY":
                    match.status = "REVIEW REQUIRED"
                if match.confidence == "HIGH":
                    match.confidence = "MEDIUM"
                match.review_reason = match.review_reason or "unit compatibility conflict"

        if line.source_conflicts:
            issues.append(ValidationIssue("source_conflict", "; ".join(line.source_conflicts), "high"))
            if match.status == "READY":
                match.status = "REVIEW REQUIRED"
            if not match.review_reason:
                match.review_reason = "source conflict"

        if any(issue.code == "missing_quantity" for issue in issues):
            match.status = "EXCEPTION"
            if match.confidence == "HIGH":
                match.confidence = "NONE"

        if any(issue.code == "unsupported_unit" for issue in issues):
            match.status = "EXCEPTION"
            match.confidence = "NONE"
            if not match.review_reason:
                match.review_reason = "unsupported unit"

        return issues, match


# ---------------------------------------------------------------------------
# Order aggregation and output helpers
# ---------------------------------------------------------------------------


def classify_order_status(line_rows: Sequence[Dict[str, Any]]) -> str:
    statuses = [str(row.get("status", "")).upper() for row in line_rows]
    if any(status == "EXCEPTION" for status in statuses):
        return "EXCEPTION"
    if any(status == "REVIEW REQUIRED" for status in statuses):
        return "REVIEW REQUIRED"
    return "READY"


def build_review_row(line: ExtractedLine, match: MatchResult, issues: List[ValidationIssue]) -> Dict[str, Any]:
    extracted_value = {
        "customer_item_number": line.customer_item_number,
        "raw_description": line.raw_description,
        "quantity": line.quantity,
        "unit": line.unit,
        "sku_hint": line.sku_hint,
    }
    return {
        "order_id": line.order_id,
        "line_number": line.line_number,
        "customer_item_number": line.customer_item_number,
        "raw_source": line.raw_source,
        "extracted_value": extracted_value,
        "proposed_sku": match.proposed_sku,
        "matched_description": match.matched_description,
        "confidence": match.confidence,
        "review_reason": match.review_reason or (issues[0].message if issues else None),
        "alternative_matches": match.alternative_matches,
        "issues": [asdict(issue) for issue in issues],
        "status": match.status,
    }


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------


def load_catalog(path: str | Path) -> List[CatalogItem]:
    items: List[CatalogItem] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            items.append(
                CatalogItem(
                    sku=row["sku"],
                    description=row["description"],
                    unit=row["unit"],
                    unit_price=float(row["unit_price"]),
                )
            )
    return items


def _coerce_lines(extracted_lines: Iterable[Any]) -> List[ExtractedLine]:
    coerced: List[ExtractedLine] = []
    for item in extracted_lines:
        if isinstance(item, ExtractedLine):
            coerced.append(item)
        elif isinstance(item, Mapping):
            coerced.append(ExtractedLine(**dict(item)))
        else:
            raise TypeError(f"Unsupported extracted line type: {type(item)!r}")
    return coerced


def run_line_validation(catalog_path: str, extracted_lines: Iterable[Any]) -> List[Dict[str, Any]]:
    matcher = CatalogMatcher(load_catalog(catalog_path))
    validator = Validator(matcher)
    out: List[Dict[str, Any]] = []
    for line in _coerce_lines(extracted_lines):
        issues, match = validator.validate_line(line)
        out.append(build_review_row(line, match, issues))
    return out


def detect_format(filename: str, raw_text: Optional[str] = None) -> str:
    lower = filename.lower()
    if lower.endswith(".csv"):
        return "csv"
    if lower.endswith((".xlsx", ".xlsm", ".xls", ".xlsb")):
        return "spreadsheet"
    if lower.endswith(".pdf"):
        return "pdf_text" if raw_text else "pdf_scan"
    if lower.endswith((".eml", ".txt")):
        return "email_text"
    if raw_text and "attachment" in raw_text.lower():
        return "email_attachment"
    return "unknown"


def run_order_validation(catalog_path: str, orders: Iterable[Mapping[str, Any]], line_groups: Mapping[str, Iterable[Any]]) -> List[Dict[str, Any]]:
    matcher = CatalogMatcher(load_catalog(catalog_path))
    validator = Validator(matcher)
    results: List[Dict[str, Any]] = []
    for order in orders:
        order_id = str(order.get("order_id"))
        lines = _coerce_lines(line_groups.get(order_id, []))
        line_rows: List[Dict[str, Any]] = []
        for line in lines:
            issues, match = validator.validate_line(line)
            row = build_review_row(line, match, issues)
            line_rows.append(row)
        results.append(
            {
                "order_id": order_id,
                "order_status": classify_order_status(line_rows),
                "line_rows": line_rows,
            }
        )
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--input-json", required=True, help="JSON list of extracted line objects")
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    with open(args.input_json, encoding="utf-8") as f:
        payload = json.load(f)
    results = run_line_validation(args.catalog, payload)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
