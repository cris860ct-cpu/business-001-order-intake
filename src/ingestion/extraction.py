from __future__ import annotations

import dataclasses
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


FIELD_PATTERNS = {
    'customer': re.compile(r'^(?:customer|bill to)\s*[:#-]\s*(.+)$', re.IGNORECASE),
    'po_number': re.compile(r'^(?:po(?: number)?|purchase order|po\s*#|po#)\s*[:#-]\s*(.+)$', re.IGNORECASE),
    'order_date': re.compile(r'^(?:order date|date)\s*[:#-]\s*(.+)$', re.IGNORECASE),
    'requested_delivery_date': re.compile(r'^(?:requested delivery(?: date)?|delivery date|ship date)\s*[:#-]\s*(.+)$', re.IGNORECASE),
}

SKU_RE = re.compile(r'\b[A-Z]{1,4}-\d{2,5}(?:-[A-Z0-9]+)?\b')
OCR_TRANSLATION = str.maketrans({'｜': '|', '丨': '|', '∣': '|', '⎮': '|', '│': '|', '：': ':', '；': ';', '＃': '#'})


def normalize_ocr_symbols(text: str) -> str:
    return text.translate(OCR_TRANSLATION)


@dataclass
class SourceBlock:
    source_ref: str
    text: str
    source_type: str


@dataclass
class ParsedLine:
    line_number: Optional[str]
    customer_item_number: Optional[str]
    sku_hint: Optional[str]
    raw_description: Optional[str]
    quantity: Optional[float]
    unit: Optional[str]
    price: Optional[str]
    source_reference: str
    raw_text: str
    raw_notes: str = ""
    source_conflicts: List[str] = field(default_factory=list)
    source_references: List[str] = field(default_factory=list)


@dataclass
class ParsedOrder:
    customer: Optional[str] = None
    po_number: Optional[str] = None
    order_date: Optional[str] = None
    requested_delivery_date: Optional[str] = None
    source_type: Optional[str] = None
    source_path: Optional[str] = None
    notes: str = ""
    line_items: List[ParsedLine] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    parse_status: str = "OK"
    parse_method: str = "deterministic"
    ocr_status: str = "NOT_REQUIRED"
    parse_time_ms: float = 0.0
    ocr_time_ms: float = 0.0
    source_text: str = ""


@dataclass
class ParsedDocument:
    source_path: str
    source_type: str
    blocks: List[SourceBlock]
    raw_text: str = ""
    attachments: List[SourceBlock] = field(default_factory=list)
    parse_status: str = "OK"
    ocr_status: str = "NOT_REQUIRED"
    warnings: List[str] = field(default_factory=list)


UNIT_SET = {'each', 'ft', 'spool', 'box', 'roll', 'pack', 'pair', 'set'}


def normalize_whitespace(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip()


def parse_date(value: str) -> Optional[str]:
    value = normalize_whitespace(value)
    if not value:
        return None
    fmts = [
        '%Y-%m-%d', '%m/%d/%Y', '%m/%d/%y', '%d-%b-%Y', '%b %d, %Y', '%m-%d-%Y',
        '%Y/%m/%d', '%d/%m/%Y',
    ]
    for fmt in fmts:
        try:
            return datetime.strptime(value, fmt).date().isoformat()
        except ValueError:
            pass
    return value


def parse_quantity(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = normalize_whitespace(str(value))
    if not text:
        return None
    text = text.replace(',', '')
    m = re.search(r'[-+]?\d+(?:\.\d+)?', text)
    if not m:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None


def extract_sku_hint(text: str) -> Optional[str]:
    m = SKU_RE.search(text or '')
    return m.group(0).upper() if m else None


def is_note_line(text: str) -> bool:
    lower = text.lower().strip()
    return lower.startswith(('note:', 'notes:', 'source note:', 'footer note:', 'conflict:', 'comment:')) or 'candidate_matches=' in lower or 'duplicate_source_block' in lower or 'ocr_simulated' in lower or 'quantity and unit conflict' in lower or 'same style' in lower or 'history reference' in lower or 'unverifiable' in lower or 'email_qty=' in lower or 'attachment_qty=' in lower or 'source_quantity=' in lower or 'catalog_unit=' in lower or 'date_values=' in lower or 'alternative_candidate=' in lower or 'unit_skid' in lower or 'sku_not_found=' in lower or 'customer_part_number_only' in lower


def extract_header_fields(text: str) -> Dict[str, Optional[str]]:
    fields: Dict[str, Optional[str]] = {'customer': None, 'po_number': None, 'order_date': None, 'requested_delivery_date': None}
    for line in text.splitlines():
        stripped = normalize_ocr_symbols(line.strip())
        if not stripped:
            continue
        for key, pattern in FIELD_PATTERNS.items():
            if fields[key] is None:
                m = pattern.match(stripped)
                if m:
                    val = normalize_whitespace(m.group(1))
                    fields[key] = parse_date(val) if 'date' in key else val
    return fields


def _split_row(line: str) -> List[str]:
    line = normalize_ocr_symbols(line)
    if '|' in line:
        return [part.strip() for part in line.split('|')]
    if '\t' in line:
        return [part.strip() for part in line.split('\t')]
    return [part.strip() for part in re.split(r'\s{2,}', line) if part.strip()]


def parse_text_to_lines(text: str, source_ref: str) -> Tuple[List[ParsedLine], List[str]]:
    notes: List[str] = []
    lines: List[ParsedLine] = []
    current: Optional[ParsedLine] = None

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped:
            current = None
            continue

        row = _split_row(stripped)
        if row and re.fullmatch(r'\d+', row[0]):
            # Support formats like: line | item | description | qty | unit | price | note
            customer_item = row[1] if len(row) > 1 else None
            raw_description = row[2] if len(row) > 2 else None
            quantity = parse_quantity(row[3]) if len(row) > 3 else None
            unit = row[4] if len(row) > 4 else None
            price = row[5] if len(row) > 5 else None
            if len(row) > 6:
                notes.append(' '.join(row[6:]).strip())
            current = ParsedLine(
                line_number=row[0],
                customer_item_number=customer_item,
                sku_hint=extract_sku_hint(str(raw_description or '')),
                raw_description=normalize_whitespace(raw_description or ''),
                quantity=quantity,
                unit=normalize_whitespace(unit or '') or None,
                price=normalize_whitespace(price or '') or None,
                source_reference=source_ref,
                raw_text=stripped,
                source_references=[source_ref],
            )
            lines.append(current)
            continue

            continue

        if is_note_line(stripped):
            notes.append(stripped)
            if current is not None:
                current.raw_notes = (current.raw_notes + ' ' + stripped).strip()
            continue

        # Continuation line: append to most recent item description / notes.
        if current is not None and not re.match(r'^(?:customer|po|order date|requested delivery|notes?|ship to|bill to)\s*[:#-]', stripped, re.I):
            current.raw_description = normalize_whitespace((current.raw_description or '') + ' ' + stripped)
            current.raw_text = current.raw_text + '\n' + stripped
            continue

        # Otherwise ignore generic headers / footer noise.
        pass
    return lines, notes


def merge_lines(lines: Iterable[ParsedLine]) -> List[ParsedLine]:
    merged: Dict[Tuple[str, str], ParsedLine] = {}
    for line in lines:
        key = (line.line_number or '', line.customer_item_number or line.raw_description or '')
        if key not in merged:
            merged[key] = dataclasses.replace(line, source_references=list(line.source_references))
            continue
        existing = merged[key]
        for field in ('sku_hint', 'raw_description', 'quantity', 'unit', 'price'):
            a = getattr(existing, field)
            b = getattr(line, field)
            if a is None and b is not None:
                setattr(existing, field, b)
            elif a is not None and b is not None and a != b:
                existing.source_conflicts.append(f'{field}={a};{field}={b}')
                setattr(existing, field, None if field != 'raw_description' else f'{a} / {b}')
        if line.raw_notes:
            existing.raw_notes = (existing.raw_notes + ' ' + line.raw_notes).strip()
        existing.source_references.extend(x for x in line.source_references if x not in existing.source_references)
        existing.raw_text = existing.raw_text + '\n' + line.raw_text
    return list(merged.values())


def parse_order_from_text_blocks(blocks: Sequence[SourceBlock], source_path: str, source_type: str) -> ParsedOrder:
    combined_text = '\n'.join(block.text for block in blocks)
    headers = extract_header_fields(combined_text)
    all_lines: List[ParsedLine] = []
    notes: List[str] = []
    for block in blocks:
        block_lines, block_notes = parse_text_to_lines(block.text, block.source_ref)
        all_lines.extend(block_lines)
        notes.extend(block_notes)
    merged_lines = merge_lines(all_lines)
    order = ParsedOrder(
        customer=headers.get('customer'),
        po_number=headers.get('po_number'),
        order_date=headers.get('order_date'),
        requested_delivery_date=headers.get('requested_delivery_date'),
        source_type=source_type,
        source_path=source_path,
        notes=' '.join(dict.fromkeys(n for n in notes if n)),
        line_items=merged_lines,
        source_text=combined_text,
    )
    return order


def parsed_order_to_v2_inputs(order: ParsedOrder, document_kind: str) -> List[Dict[str, Any]]:
    inputs: List[Dict[str, Any]] = []
    for line in order.line_items:
        inputs.append({
            'order_id': order.po_number or order.source_path,
            'line_number': line.line_number,
            'customer_item_number': line.customer_item_number,
            'raw_source': line.raw_text,
            'raw_description': line.raw_description,
            'quantity': line.quantity,
            'unit': line.unit,
            'sku_hint': line.sku_hint,
            'source_conflicts': line.source_conflicts or None,
            'raw_notes': ' '.join(x for x in [order.notes, line.raw_notes] if x),
            'document_kind': document_kind,
            'source_quality': document_kind,
            'customer_alias': order.customer,
            'confidence_hint': None,
        })
    return inputs
