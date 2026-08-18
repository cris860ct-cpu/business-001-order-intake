from __future__ import annotations

from email import policy
from email.parser import BytesParser
from io import BytesIO
from pathlib import Path
from typing import List

from .extraction import ParsedDocument, SourceBlock
from .pdf_parser import parse_pdf_bytes
from .spreadsheet_parser import parse_workbook_bytes


TEXT_TYPES = {'text/plain', 'text/html'}


def _body_text_from_message(msg) -> str:
    if msg.is_multipart():
        parts: List[str] = []
        for part in msg.walk():
            content_type = part.get_content_type()
            if part.get_content_disposition() == 'attachment':
                continue
            if content_type == 'text/plain':
                parts.append(part.get_content())
            elif content_type == 'text/html' and not parts:
                parts.append(part.get_content())
        return '\n'.join(parts)
    return msg.get_content()


def parse_email_bytes(data: bytes, source_name: str) -> ParsedDocument:
    msg = BytesParser(policy=policy.default).parsebytes(data)
    blocks: List[SourceBlock] = []
    body = _body_text_from_message(msg)
    if body:
        blocks.append(SourceBlock(source_ref=f'{source_name}:body', text=body, source_type='email'))
    attachments: List[SourceBlock] = []

    for part_idx, part in enumerate(msg.iter_attachments(), start=1):
        filename = part.get_filename() or f'attachment-{part_idx}'
        payload = part.get_payload(decode=True) or b''
        ctype = part.get_content_type().lower()
        ref_base = f'{source_name}:{filename}'
        if filename.lower().endswith(('.xlsx', '.xlsm', '.xlsb')) or 'spreadsheet' in ctype or 'excel' in ctype:
            nested = parse_workbook_bytes(payload, filename)
            attachments.extend(nested.blocks)
        elif filename.lower().endswith('.pdf') or ctype == 'application/pdf':
            nested = parse_pdf_bytes(payload, filename)
            attachments.extend(nested.blocks)
        else:
            text = payload.decode('utf-8', errors='replace')
            attachments.append(SourceBlock(source_ref=f'{ref_base}:text', text=text, source_type='email-attachment'))

    raw_text = '\n'.join(block.text for block in blocks + attachments)
    return ParsedDocument(source_path=source_name, source_type='email', blocks=blocks, raw_text=raw_text, attachments=attachments)


def parse_email(path: str | Path) -> ParsedDocument:
    p = Path(path)
    return parse_email_bytes(p.read_bytes(), p.name)
