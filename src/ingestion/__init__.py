from __future__ import annotations

from pathlib import Path

from .email_parser import parse_email
from .extraction import ParsedDocument, ParsedOrder, SourceBlock, parse_order_from_text_blocks, parsed_order_to_v2_inputs
from .ocr_parser import ocr_image
from .pdf_parser import parse_pdf, parse_pdf_bytes
from .spreadsheet_parser import parse_workbook


IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}
EMAIL_EXTS = {'.eml', '.msg'}
SPREADSHEET_EXTS = {'.xlsx', '.xlsm', '.xlsb', '.xls'}
PDF_EXTS = {'.pdf'}
TEXT_EXTS = {'.txt', '.md'}


def detect_document_type(path: str | Path) -> str:
    suffix = Path(path).suffix.lower()
    if suffix in PDF_EXTS:
        return 'pdf'
    if suffix in SPREADSHEET_EXTS:
        return 'xlsx'
    if suffix in EMAIL_EXTS:
        return 'email'
    if suffix in IMAGE_EXTS:
        return 'image'
    if suffix in TEXT_EXTS:
        return 'text'
    return 'unknown'


def parse_document(path: str | Path) -> ParsedDocument:
    p = Path(path)
    kind = detect_document_type(p)
    if kind == 'pdf':
        return parse_pdf(p)
    if kind == 'xlsx':
        return parse_workbook(p)
    if kind == 'email':
        return parse_email(p)
    if kind == 'image':
        result = ocr_image(p)
        text = result.text or ''
        return ParsedDocument(source_path=str(p), source_type='image', blocks=[SourceBlock(source_ref=f'{p.name}:ocr', text=text, source_type='image')], raw_text=text, parse_status=result.status, ocr_status=result.status)
    text = p.read_text(encoding='utf-8', errors='replace')
    return ParsedDocument(source_path=str(p), source_type=kind, blocks=[SourceBlock(source_ref=f'{p.name}:text', text=text, source_type=kind)], raw_text=text)


def extract_order_from_document(path: str | Path) -> ParsedOrder:
    doc = parse_document(path)
    order = parse_order_from_text_blocks(doc.blocks + doc.attachments, str(path), doc.source_type)
    order.parse_status = doc.parse_status
    order.ocr_status = doc.ocr_status
    order.source_text = doc.raw_text
    if doc.parse_status == 'OCR_BLOCKED' and not order.notes:
        order.notes = 'OCR BLOCKED'
    return order


__all__ = [
    'ParsedDocument', 'ParsedOrder', 'SourceBlock', 'parse_document', 'extract_order_from_document',
    'parsed_order_to_v2_inputs', 'parse_order_from_text_blocks', 'parse_pdf_bytes',
]
