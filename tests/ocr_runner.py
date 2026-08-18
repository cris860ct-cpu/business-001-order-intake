from __future__ import annotations

import csv
import json
import math
import shutil
import textwrap
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

from src.ingestion import extract_order_from_document, parsed_order_to_v2_inputs
from src.ingestion.extraction import ParsedDocument, SourceBlock, parse_order_from_text_blocks
from src.ingestion.ocr_parser import ocr_image
from src.prototype_v2 import run_order_validation


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_ORDERS = ROOT / 'benchmark' / 'mission_014c2_orders.csv'
GROUND_TRUTH_LINES = ROOT / 'ground_truth' / 'mission_014c2_line_items.csv'
CATALOG = ROOT / 'catalog' / 'mission_014c2_catalog.csv'
RAW_DIR = ROOT / 'raw_sources' / '014c6'
RESULTS_DIR = ROOT / 'results' / '014c6'

SCAN_SET = [
    # existing scan cases first
    {'order_id': 'O035', 'source_path': ROOT / 'raw_sources' / '014c5' / 'O035.png', 'base_order_id': 'O035', 'style': 'existing_hard_scan', 'kind': 'existing'},
    {'order_id': 'O036', 'source_path': ROOT / 'raw_sources' / '014c5' / 'O036.png', 'base_order_id': 'O036', 'style': 'existing_hard_scan', 'kind': 'existing'},
    {'order_id': 'O039', 'source_path': ROOT / 'raw_sources' / '014c5' / 'O039.png', 'base_order_id': 'O039', 'style': 'existing_hard_scan', 'kind': 'existing'},
    # seven new synthetic scan variants
    {'order_id': 'O001', 'base_order_id': 'O001', 'style': 'clean_footer_noise', 'kind': 'synthetic', 'use_sku_hint': False, 'degrade': {'rotate': 1.5, 'contrast': 1.2, 'blur': 0.25}},
    {'order_id': 'O004', 'base_order_id': 'O004', 'style': 'skew_shadow', 'kind': 'synthetic', 'use_sku_hint': False, 'degrade': {'rotate': -2.0, 'contrast': 1.05, 'blur': 0.35}},
    {'order_id': 'O005', 'base_order_id': 'O005', 'style': 'grid_table', 'kind': 'synthetic', 'use_sku_hint': False, 'degrade': {'rotate': 0.0, 'contrast': 1.35, 'blur': 0.15}},
    {'order_id': 'O009', 'base_order_id': 'O009', 'style': 'fax_like_hint', 'kind': 'synthetic', 'use_sku_hint': True, 'degrade': {'rotate': 1.0, 'contrast': 1.05, 'blur': 0.4}},
    {'order_id': 'O011', 'base_order_id': 'O011', 'style': 'spreadsheet_photo', 'kind': 'synthetic', 'use_sku_hint': False, 'degrade': {'rotate': -1.0, 'contrast': 1.15, 'blur': 0.2}},
    {'order_id': 'O013', 'base_order_id': 'O013', 'style': 'annotation_hint', 'kind': 'synthetic', 'use_sku_hint': True, 'degrade': {'rotate': 2.0, 'contrast': 0.95, 'blur': 0.3}},
    {'order_id': 'O015', 'base_order_id': 'O015', 'style': 'shadow_compression', 'kind': 'synthetic', 'use_sku_hint': True, 'degrade': {'rotate': -1.75, 'contrast': 1.1, 'blur': 0.2}},
]

# Human-speed sample: 6 normal, 3 messy, 1 true exception.
HUMAN_SPEED_SAMPLE = ['O001', 'O002', 'O003', 'O006', 'O007', 'O011', 'O004', 'O005', 'O009', 'O035']


@dataclass
class OrderFixture:
    order_id: str
    customer: str
    po_number: str
    order_date: str
    requested_delivery_date: str
    order_status: str
    lines: List[Dict[str, str]]



def load_orders() -> Dict[str, Dict[str, str]]:
    with BENCHMARK_ORDERS.open(encoding='utf-8', newline='') as f:
        return {row['order_id']: row for row in csv.DictReader(f)}



def load_truth() -> Dict[str, List[Dict[str, str]]]:
    out: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    with GROUND_TRUTH_LINES.open(encoding='utf-8', newline='') as f:
        for row in csv.DictReader(f):
            out[row['order_id']].append(row)
    for order_id in out:
        out[order_id].sort(key=lambda row: int(row['line_number']))
    return out



def ensure_dirs() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)



def normalize_text(text: str) -> str:
    return ' '.join(text.lower().split())



def token_counts(text: str) -> Counter:
    return Counter(token for token in normalize_text(text).split() if token)



def char_accuracy(reference: str, observed: str) -> float:
    if not reference and not observed:
        return 1.0
    return round(SequenceMatcher(None, normalize_text(reference), normalize_text(observed)).ratio(), 4)



def token_f1(reference: str, observed: str) -> float:
    ref = token_counts(reference)
    obs = token_counts(observed)
    if not ref and not obs:
        return 1.0
    if not ref or not obs:
        return 0.0
    overlap = sum(min(ref[t], obs[t]) for t in ref.keys() | obs.keys())
    precision = overlap / max(sum(obs.values()), 1)
    recall = overlap / max(sum(ref.values()), 1)
    if precision + recall == 0:
        return 0.0
    return round(2 * precision * recall / (precision + recall), 4)



def make_fixture(order_id: str, orders: Dict[str, Dict[str, str]], truth: Dict[str, List[Dict[str, str]]]) -> OrderFixture:
    order = orders[order_id]
    return OrderFixture(
        order_id=order_id,
        customer=order['customer'],
        po_number=order['po_number'],
        order_date=order['order_date'],
        requested_delivery_date=order['requested_delivery_date'],
        order_status=order['order_status'],
        lines=truth[order_id],
    )



def compose_reference_text(fixture: OrderFixture, use_sku_hint: bool = False) -> Tuple[str, Dict[str, Optional[str]]]:
    lines = [
        'Nova Labs Wholesale Order',
        f'Customer: {fixture.customer}',
        f'PO #: {fixture.po_number}',
        f'Order Date: {fixture.order_date}',
        f'Requested Delivery: {fixture.requested_delivery_date}',
        '',
    ]
    expected_hint_lines: Dict[str, Optional[str]] = {}
    for line in fixture.lines:
        desc = line['raw_customer_description']
        expected_hint_lines[line['line_number']] = None
        if use_sku_hint and line['line_number'] == '1':
            sku = line['distributor_sku']
            desc = f'{sku} {desc}'
            expected_hint_lines[line['line_number']] = sku
        lines.append(f"{line['line_number']} | {line['customer_item_number']} | {desc} | {line['quantity']} | {line['unit']} | {line['unit_price']}")
    return '\n'.join(lines), expected_hint_lines



def render_scan_image(path: Path, fixture: OrderFixture, style: str, degrade: Dict[str, float], use_sku_hint: bool = False) -> Dict[str, Any]:
    reference_text, expected_hints = compose_reference_text(fixture, use_sku_hint=use_sku_hint)
    img = Image.new('RGB', (1650, 2200), 'white')
    draw = ImageDraw.Draw(img)
    font = None
    try:
        # Default font is intentionally used for compatibility and OCR readability.
        from PIL import ImageFont
        font = ImageFont.load_default()
    except Exception:
        font = None

    y = 70
    for idx, line in enumerate(reference_text.splitlines()):
        if style == 'grid_table' and idx >= 6:
            draw.line((70, y - 16, 1580, y - 16), fill=(215, 215, 215), width=1)
        if style == 'annotation_hint' and idx == len(reference_text.splitlines()) - 1:
            draw.text((1120, y - 14), 'expedite if possible', fill=(110, 110, 110), font=font)
        draw.text((80, y), line, fill=(20, 20, 20), font=font)
        y += 95

    if style == 'clean_footer_noise':
        draw.text((80, 2020), 'Printed from order desk', fill=(120, 120, 120), font=font)
        draw.text((1180, 2020), 'Page 1 of 1', fill=(120, 120, 120), font=font)
    elif style == 'skew_shadow':
        shadow = Image.new('RGBA', img.size, (0, 0, 0, 0))
        sd = ImageDraw.Draw(shadow)
        sd.rectangle((40, 40, 1600, 2140), fill=(0, 0, 0, 22))
        img = Image.alpha_composite(img.convert('RGBA'), shadow).convert('RGB')
    elif style == 'fax_like_hint':
        draw.rectangle((60, 60, 1590, 2140), outline=(180, 180, 180), width=2)
        draw.text((80, 2050), 'fax transmission 08/15 09:22', fill=(140, 140, 140), font=font)
    elif style == 'spreadsheet_photo':
        for row_y in range(650, 2050, 95):
            draw.line((70, row_y, 1580, row_y), fill=(210, 210, 210), width=1)
        for col_x in [70, 180, 470, 880, 1060, 1180, 1330, 1580]:
            draw.line((col_x, 650, col_x, 2050), fill=(220, 220, 220), width=1)
    elif style == 'shadow_compression':
        shadow = Image.new('RGBA', img.size, (0, 0, 0, 0))
        sd = ImageDraw.Draw(shadow)
        sd.ellipse((900, 1100, 1650, 2000), fill=(0, 0, 0, 18))
        img = Image.alpha_composite(img.convert('RGBA'), shadow).convert('RGB')

    angle = degrade.get('rotate', 0.0)
    contrast = degrade.get('contrast', 1.0)
    blur = degrade.get('blur', 0.0)
    if angle:
        img = img.rotate(angle, expand=True, fillcolor='white')
    if contrast and contrast != 1.0:
        img = ImageEnhance.Contrast(img).enhance(contrast)
    if blur and blur > 0:
        img = img.filter(ImageFilter.GaussianBlur(radius=blur))

    img.save(path)
    return {'reference_text': reference_text, 'expected_hint_lines': expected_hints}



def load_scan_document(order_id: str, path: Path) -> ParsedDocument:
    text = path.read_bytes().decode('utf-8', errors='replace') if path.suffix.lower() == '.txt' else ''
    if text:
        return ParsedDocument(source_path=str(path), source_type='text', blocks=[SourceBlock(source_ref=f'{path.name}:text', text=text, source_type='text')], raw_text=text)
    return ParsedDocument(source_path=str(path), source_type='image', blocks=[], raw_text='')



def generate_scan_set(orders: Dict[str, Dict[str, str]], truth: Dict[str, List[Dict[str, str]]]) -> Dict[str, Any]:
    ensure_dirs()
    manifest: Dict[str, Any] = {'orders': []}
    for spec in SCAN_SET:
        order_id = spec['order_id']
        if spec['kind'] == 'existing':
            source_path = spec['source_path']
            target_path = RAW_DIR / source_path.name
            shutil.copy2(source_path, target_path)
            fixture = make_fixture(order_id, orders, truth)
            reference_text, expected_hints = compose_reference_text(fixture)
            manifest['orders'].append({
                'order_id': order_id,
                'base_order_id': order_id,
                'source_path': str(target_path.relative_to(ROOT)),
                'style': spec['style'],
                'kind': 'existing',
                'expected_status': fixture.order_status,
                'reference_text': reference_text,
                'expected_hint_lines': expected_hints,
                'use_sku_hint': False,
            })
            continue
        fixture = make_fixture(spec['base_order_id'], orders, truth)
        target_path = RAW_DIR / f"{order_id}_{spec['style']}.png"
        data = render_scan_image(target_path, fixture, spec['style'], spec['degrade'], use_sku_hint=spec.get('use_sku_hint', False))
        manifest['orders'].append({
            'order_id': order_id,
            'base_order_id': spec['base_order_id'],
            'source_path': str(target_path.relative_to(ROOT)),
            'style': spec['style'],
            'kind': 'synthetic',
            'expected_status': fixture.order_status,
            'reference_text': data['reference_text'],
            'expected_hint_lines': data['expected_hint_lines'],
            'use_sku_hint': spec.get('use_sku_hint', False),
        })
    (RAW_DIR / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest



def _order_from_ocr_text(order_id: str, source_path: str, ocr_text: str, ocr_status: str, confidence: Optional[float]) -> Tuple[Any, Any]:
    doc = ParsedDocument(
        source_path=source_path,
        source_type='image',
        blocks=[SourceBlock(source_ref=f'{Path(source_path).name}:ocr', text=ocr_text, source_type='image')],
        raw_text=ocr_text,
        parse_status=ocr_status,
        ocr_status=ocr_status,
    )
    order = parse_order_from_text_blocks(doc.blocks, source_path, 'image')
    order.parse_status = ocr_status
    order.ocr_status = ocr_status
    order.source_text = ocr_text
    if confidence is not None:
        order.notes = (order.notes + f' OCR_CONFIDENCE={confidence:.4f}').strip()
    return doc, order



def evaluate_scan_set(manifest: Dict[str, Any], orders: Dict[str, Dict[str, str]], truth: Dict[str, List[Dict[str, str]]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    failure_table: List[Dict[str, Any]] = []
    agg: Dict[str, Any] = {
        'ocr_text_char_accuracy': [],
        'ocr_text_token_f1': [],
        'header_accuracy': [],
        'line_recall': [],
        'quantity_accuracy': [],
        'sku_hint_accuracy': [],
        'catalog_match_accuracy': [],
        'exception_recall': [],
        'false_confidence_count': 0,
        'wrong_sku_ready': 0,
        'wrong_quantity_ready': 0,
        'parser_or_ocr_failure_ready': 0,
        'auto_ready': 0,
        'review_required': 0,
        'exception': 0,
        'orders': 0,
        'lines_total': 0,
        'lines_parsed': 0,
        'ready_lines': 0,
        'review_lines': 0,
        'exception_lines': 0,
        'actual_exception_orders': 0,
        'pred_exception_orders': 0,
        'pred_ready_orders': 0,
        'pred_review_orders': 0,
    }

    for entry in manifest['orders']:
        order_id = entry['order_id']
        fixture = make_fixture(entry['base_order_id'], orders, truth)
        source_path = ROOT / entry['source_path']
        ocr = ocr_image(source_path)
        reference_text = entry['reference_text']
        char_acc = char_accuracy(reference_text, ocr.text)
        token_acc = token_f1(reference_text, ocr.text)
        agg['ocr_text_char_accuracy'].append(char_acc)
        agg['ocr_text_token_f1'].append(token_acc)

        if fixture.order_status == 'EXCEPTION':
            agg['actual_exception_orders'] += 1

        if ocr.status == 'OCR_BLOCKED':
            doc, order = _order_from_ocr_text(order_id, str(source_path), '', 'OCR_BLOCKED', ocr.confidence)
            final_status = 'EXCEPTION'
            ai_result = {
                'order_id': order_id,
                'order_status': final_status,
                'ocr_status': ocr.status,
                'ocr_confidence': ocr.confidence,
                'ocr_quality_score': ocr.quality_score,
                'ocr_preprocessing': ocr.preprocessing,
                'ocr_text': ocr.text,
                'parse_status': order.parse_status,
                'parse_time_ms': None,
                'line_rows': [],
                'blocked_reason': ocr.warning or 'OCR blocked',
            }
            results.append(ai_result)
            failure_table.append({
                'order_id': order_id,
                'source_type': 'scan',
                'expected': fixture.order_status,
                'actual': final_status,
                'failure_class': 'F11 OCR FAILURE',
                'severity': 'high',
                'root_cause': ocr.warning or 'OCR dependency unavailable',
                'fix_candidate': 'install or configure a working OCR engine',
            })
            agg['exception'] += 1
            continue

        if not ocr.text.strip() or (ocr.confidence is not None and ocr.confidence < 0.45) or ocr.status != 'OK':
            # Fail closed on low-confidence OCR but still parse the text when any text is available so metrics can measure the extraction gap.
            if ocr.text.strip():
                _, order = _order_from_ocr_text(order_id, str(source_path), ocr.text, ocr.status, ocr.confidence)
            else:
                order = None
            ai_result = {
                'order_id': order_id,
                'order_status': 'EXCEPTION',
                'ocr_status': ocr.status,
                'ocr_confidence': ocr.confidence,
                'ocr_quality_score': ocr.quality_score,
                'ocr_preprocessing': ocr.preprocessing,
                'ocr_text': ocr.text,
                'parse_status': ocr.status,
                'parse_time_ms': None,
                'line_rows': [],
                'blocked_reason': ocr.warning or 'OCR low confidence',
            }
            if ocr.text.strip():
                line_inputs = parsed_order_to_v2_inputs(order, 'scan')
                for row in line_inputs:
                    row['order_id'] = order_id
                    row['source_quality'] = f'scan:{ocr.preprocessing}'
                    row['confidence_hint'] = f'{ocr.confidence:.4f}' if ocr.confidence is not None else None
                if line_inputs:
                    validated = run_order_validation(str(CATALOG), [{'order_id': order_id}], {order_id: line_inputs})[0]
                else:
                    validated = {
                        'order_id': order_id,
                        'order_status': 'EXCEPTION',
                        'line_rows': [],
                    }
                validated['order_status'] = 'EXCEPTION'
                validated['ocr_status'] = ocr.status
                validated['ocr_confidence'] = ocr.confidence
                validated['ocr_quality_score'] = ocr.quality_score
                validated['ocr_preprocessing'] = ocr.preprocessing
                validated['ocr_text'] = ocr.text
                validated['reference_text'] = reference_text
                validated['expected_status'] = fixture.order_status
                results.append(validated)
            else:
                results.append(ai_result)
            failure_table.append({
                'order_id': order_id,
                'source_type': 'scan',
                'expected': fixture.order_status,
                'actual': 'EXCEPTION',
                'failure_class': 'F11 OCR FAILURE',
                'severity': 'medium',
                'root_cause': ocr.warning or 'OCR low confidence',
                'fix_candidate': 'improve preprocessing or require human review',
            })
            agg['exception'] += 1
            continue

        _, order = _order_from_ocr_text(order_id, str(source_path), ocr.text, ocr.status, ocr.confidence)
        line_inputs = parsed_order_to_v2_inputs(order, 'scan')
        for row in line_inputs:
            row['order_id'] = order_id
            row['source_quality'] = f'scan:{ocr.preprocessing}'
            row['confidence_hint'] = f'{ocr.confidence:.4f}' if ocr.confidence is not None else None
        if line_inputs:
            validated = run_order_validation(str(CATALOG), [{'order_id': order_id}], {order_id: line_inputs})[0]
            final_status = validated['order_status']
            if ocr.status != 'OK':
                final_status = 'EXCEPTION'
        else:
            validated = {'order_id': order_id, 'order_status': 'EXCEPTION', 'line_rows': []}
            final_status = 'EXCEPTION'
        validated['order_status'] = final_status
        validated['ocr_status'] = ocr.status
        validated['ocr_confidence'] = ocr.confidence
        validated['ocr_quality_score'] = ocr.quality_score
        validated['ocr_preprocessing'] = ocr.preprocessing
        validated['ocr_text'] = ocr.text
        validated['reference_text'] = reference_text
        validated['expected_status'] = fixture.order_status
        results.append(validated)

        order_truth = truth[entry['base_order_id']]
        header_hits = sum(1 for a, b in [(order.customer, fixture.customer), (order.po_number, fixture.po_number), (order.order_date, fixture.order_date), (order.requested_delivery_date, fixture.requested_delivery_date)] if a == b)
        header_accuracy = round(header_hits / 4.0, 4)
        agg['header_accuracy'].append(header_accuracy)

        parsed_lines = validated.get('line_rows', [])
        gt_lines = order_truth
        agg['lines_total'] += len(gt_lines)
        agg['lines_parsed'] += len(parsed_lines)
        line_matches = 0
        quantity_hits = 0
        sku_hint_hits = 0
        sku_hint_total = 0
        catalog_hits = 0
        ready_wrong_sku = 0
        ready_wrong_qty = 0
        false_confidence = 0
        parser_or_ocr_failure_ready = 1 if final_status == 'READY' and ocr.status != 'OK' else 0
        agg['parser_or_ocr_failure_ready'] += parser_or_ocr_failure_ready

        by_line = {row['line_number']: row for row in parsed_lines}
        for gt in gt_lines:
            pred = by_line.get(gt['line_number'])
            if pred:
                line_matches += 1
                if str(pred.get('extracted_value', {}).get('quantity')) == str(float(gt['quantity'])) or str(pred.get('extracted_value', {}).get('quantity')) == str(gt['quantity']):
                    quantity_hits += 1
                if gt['distributor_sku']:
                    sku_hint_total += 1 if entry['use_sku_hint'] and gt['line_number'] == '1' else 0
                    if entry['use_sku_hint'] and gt['line_number'] == '1':
                        if pred.get('extracted_value', {}).get('sku_hint') == gt['distributor_sku'] or pred.get('proposed_sku') == gt['distributor_sku']:
                            sku_hint_hits += 1
                if pred.get('proposed_sku') == gt['distributor_sku']:
                    catalog_hits += 1
                if pred.get('status', '').upper() == 'READY' and pred.get('proposed_sku') != gt['distributor_sku']:
                    ready_wrong_sku += 1
                if pred.get('status', '').upper() == 'READY' and str(pred.get('extracted_value', {}).get('quantity')) != str(gt['quantity']):
                    ready_wrong_qty += 1
                if pred.get('confidence') == 'HIGH' and pred.get('proposed_sku') != gt['distributor_sku']:
                    false_confidence += 1
            else:
                failure_table.append({
                    'order_id': order_id,
                    'source_type': 'scan',
                    'expected': f"line {gt['line_number']} present",
                    'actual': 'line missing',
                    'failure_class': 'F2 EXTRACTION FAILURE',
                    'severity': 'high',
                    'root_cause': 'OCR missed the line or parser could not segment it',
                    'fix_candidate': 'improve image preprocessing or OCR segmentation',
                })

        if fixture.order_status == 'EXCEPTION':
            observed_exception = 1 if final_status == 'EXCEPTION' else 0
            agg['exception_recall'].append(observed_exception)

        agg['line_recall'].append(round(line_matches / max(len(gt_lines), 1), 4))
        agg['quantity_accuracy'].append(round(quantity_hits / max(len(gt_lines), 1), 4))
        agg['sku_hint_accuracy'].append(round(sku_hint_hits / max(sku_hint_total, 1), 4) if sku_hint_total else None)
        agg['catalog_match_accuracy'].append(round(catalog_hits / max(len(gt_lines), 1), 4))
        agg['false_confidence_count'] += false_confidence
        agg['wrong_sku_ready'] += ready_wrong_sku
        agg['wrong_quantity_ready'] += ready_wrong_qty

        if final_status == 'READY':
            agg['auto_ready'] += 1
            agg['pred_ready_orders'] += 1
        elif final_status == 'REVIEW REQUIRED':
            agg['review_required'] += 1
            agg['pred_review_orders'] += 1
        else:
            agg['exception'] += 1
            agg['pred_exception_orders'] += 1

    # normalize exception recall list so existing hard scans count as success if blocked closed.
    exception_recall_values = [1 if item == 1 else 0 for item in agg['exception_recall']]

    agg['orders'] = len(manifest['orders'])

    metrics = {
        'ocr_character_accuracy': round(sum(agg['ocr_text_char_accuracy']) / max(len(agg['ocr_text_char_accuracy']), 1), 4),
        'ocr_token_f1': round(sum(agg['ocr_text_token_f1']) / max(len(agg['ocr_text_token_f1']), 1), 4),
        'header_extraction_accuracy': round(sum(agg['header_accuracy']) / max(len(agg['header_accuracy']), 1), 4),
        'line_item_extraction_recall': round(sum(agg['line_recall']) / max(len(agg['line_recall']), 1), 4),
        'quantity_extraction_accuracy': round(sum(agg['quantity_accuracy']) / max(len(agg['quantity_accuracy']), 1), 4),
        'sku_hint_extraction_accuracy': round(sum(v for v in agg['sku_hint_accuracy'] if v is not None) / max(sum(1 for v in agg['sku_hint_accuracy'] if v is not None), 1), 4) if any(v is not None for v in agg['sku_hint_accuracy']) else None,
        'catalog_match_accuracy': round(sum(agg['catalog_match_accuracy']) / max(len(agg['catalog_match_accuracy']), 1), 4),
        'exception_recall': round(sum(exception_recall_values) / max(len(exception_recall_values), 1), 4),
        'false_confidence_rate': round(agg['false_confidence_count'] / max(agg['lines_parsed'], 1), 4),
        'wrong_sku_ready': int(agg['wrong_sku_ready']),
        'wrong_quantity_ready': int(agg['wrong_quantity_ready']),
        'parser_or_ocr_failure_presented_as_ready': int(agg['parser_or_ocr_failure_ready']),
        'percent_auto_ready': round(agg['auto_ready'] / max(agg['orders'], 1), 4),
        'percent_review_required': round(agg['review_required'] / max(len(manifest['orders']), 1), 4),
        'percent_exception': round(agg['exception'] / max(len(manifest['orders']), 1), 4),
        'orders_tested': len(manifest['orders']),
        'lines_total': agg['lines_total'],
        'lines_parsed': agg['lines_parsed'],
        'parser_or_ocr_failure_presented_as_ready_rate': round(agg['parser_or_ocr_failure_ready'] / max(len(manifest['orders']), 1), 4),
    }
    safety = {
        'wrong_sku_ready': metrics['wrong_sku_ready'],
        'wrong_quantity_ready': metrics['wrong_quantity_ready'],
        'parser_or_ocr_failure_presented_as_ready': metrics['parser_or_ocr_failure_presented_as_ready'],
        'false_confidence_rate': metrics['false_confidence_rate'],
        'exception_recall': metrics['exception_recall'],
    }
    return results, metrics, {'failure_table': failure_table, 'safety': safety}



def build_human_speed_package(orders: Dict[str, Dict[str, str]], truth: Dict[str, List[Dict[str, str]]], ai_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    package_root = RESULTS_DIR / 'human_speed_test_package'
    if package_root.exists():
        shutil.rmtree(package_root)
    (package_root / 'source_docs').mkdir(parents=True, exist_ok=True)
    (package_root / 'manual_sheets').mkdir(parents=True, exist_ok=True)
    (package_root / 'ai_outputs').mkdir(parents=True, exist_ok=True)

    ai_by_order = {row['order_id']: row for row in ai_results}
    selected = []
    for order_id in HUMAN_SPEED_SAMPLE:
        order = orders[order_id]
        src = ROOT / 'raw_sources' / '014c5' / f"{order_id}.{ 'xlsx' if order['input_type'] == 'xlsx' else 'pdf' if order['input_type'] == 'typed_pdf' else 'eml' if order['input_type'] == 'email' else 'png'}"
        # Fallback for order-specific assets if needed.
        if not src.exists():
            # Use any matching raw source available.
            candidates = list((ROOT / 'raw_sources' / '014c5').glob(f'{order_id}.*'))
            if candidates:
                src = candidates[0]
        if src.exists():
            shutil.copy2(src, package_root / 'source_docs' / src.name)
        manual_md = [
            f'# Manual entry sheet: {order_id}',
            '',
            f'Customer: {order["customer"]}',
            f'PO #: {order["po_number"]}',
            f'Order date: {order["order_date"]}',
            f'Requested delivery: {order["requested_delivery_date"]}',
            '',
            '## Instructions',
            '- Start timing when you begin reading the source document.',
            '- Stop timing when the structured order record is complete and ready for downstream review.',
            '- Record any errors, corrections, and uncertainty questions.',
            '',
            '## Blank manual entry fields',
            '- Manual time (seconds):',
            '- Errors / corrections:',
            '- Uncertainty questions:',
            '',
            '## Raw line items',
        ]
        for line in truth[order_id]:
            manual_md.append(f"- {line['line_number']} | {line['customer_item_number']} | {line['raw_customer_description']} | {line['quantity']} | {line['unit']} | {line['unit_price']}")
        (package_root / 'manual_sheets' / f'{order_id}.md').write_text('\n'.join(manual_md), encoding='utf-8')
        (package_root / 'ai_outputs' / f'{order_id}.json').write_text(json.dumps(ai_by_order.get(order_id, {}), indent=2), encoding='utf-8')
        selected.append(order_id)

    instructions = textwrap.dedent('''
    # Human speed test protocol

    Selected sample: 6 normal, 3 messy, 1 true exception.

    Workflow A — manual entry
    1. Open the source document.
    2. Start timing when you begin reading.
    3. Enter the structured order record manually.
    4. Stop timing when the record is complete and ready for downstream entry/review.
    5. Record corrections and uncertainty questions.

    Workflow B — AI-assisted review
    1. Open the AI-assisted output.
    2. Start timing when the prepared result is opened.
    3. Approve or correct the structured record.
    4. Stop timing when the final record is ready.
    5. Record edits, false flags, and missed errors.

    Timing must be recorded by Cristian; do not estimate.
    ''').strip()
    (package_root / 'timing_instructions.md').write_text(instructions, encoding='utf-8')

    scoring = ['order_id,manual_time_seconds,ai_review_time_seconds,manual_errors,ai_edits,false_flags,missed_errors,notes']
    for order_id in HUMAN_SPEED_SAMPLE:
        scoring.append(f'{order_id},,,,,,,' )
    (package_root / 'scoring_sheet.csv').write_text('\n'.join(scoring), encoding='utf-8')

    package_manifest = {
        'sample': HUMAN_SPEED_SAMPLE,
        'package_root': str(package_root.relative_to(ROOT)),
        'source_docs': [str(p.relative_to(ROOT)) for p in sorted((package_root / 'source_docs').glob('*'))],
        'manual_sheets': [str(p.relative_to(ROOT)) for p in sorted((package_root / 'manual_sheets').glob('*'))],
        'ai_outputs': [str(p.relative_to(ROOT)) for p in sorted((package_root / 'ai_outputs').glob('*'))],
    }
    (RESULTS_DIR / 'human_speed_test_package_manifest.json').write_text(json.dumps(package_manifest, indent=2), encoding='utf-8')
    return package_manifest



def main() -> None:
    ensure_dirs()
    orders = load_orders()
    truth = load_truth()
    manifest = generate_scan_set(orders, truth)
    results, metrics, extra = evaluate_scan_set(manifest, orders, truth)
    package_manifest = build_human_speed_package(orders, truth, json.loads((ROOT / 'results' / '014c5' / '014c5_raw_results.json').read_text()))

    (RESULTS_DIR / 'ocr_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    (RESULTS_DIR / 'ocr_results.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    (RESULTS_DIR / 'ocr_metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    (RESULTS_DIR / 'ocr_failure_table.json').write_text(json.dumps(extra['failure_table'], indent=2), encoding='utf-8')
    (RESULTS_DIR / 'ocr_safety_metrics.json').write_text(json.dumps(extra['safety'], indent=2), encoding='utf-8')
    (RESULTS_DIR / 'ocr_package_manifest.json').write_text(json.dumps(package_manifest, indent=2), encoding='utf-8')

    summary = {
        'orders_tested': len(manifest['orders']),
        'ocr_character_accuracy': metrics['ocr_character_accuracy'],
        'ocr_token_f1': metrics['ocr_token_f1'],
        'header_extraction_accuracy': metrics['header_extraction_accuracy'],
        'line_item_extraction_recall': metrics['line_item_extraction_recall'],
        'catalog_match_accuracy': metrics['catalog_match_accuracy'],
        'exception_recall': metrics['exception_recall'],
        'percent_auto_ready': metrics['percent_auto_ready'],
        'percent_review_required': metrics['percent_review_required'],
        'percent_exception': metrics['percent_exception'],
    }
    (RESULTS_DIR / 'ocr_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
