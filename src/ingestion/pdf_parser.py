from __future__ import annotations

import re
from pathlib import Path
from typing import List

from .extraction import ParsedDocument, SourceBlock


STRING_RE = re.compile(rb'\((?:\\.|[^\\()])*\)\s*Tj')


def _unescape_pdf_string(data: bytes) -> str:
    out = bytearray()
    i = 0
    while i < len(data):
        ch = data[i:i+1]
        if ch == b'\\' and i + 1 < len(data):
            nxt = data[i + 1:i + 2]
            if nxt in {b'\\', b'(', b')'}:
                out.extend(nxt)
                i += 2
                continue
            if nxt == b'n':
                out.extend(b'\n')
                i += 2
                continue
            if nxt == b'r':
                out.extend(b'\r')
                i += 2
                continue
            if nxt == b't':
                out.extend(b'\t')
                i += 2
                continue
        out.extend(ch)
        i += 1
    return out.decode('utf-8', errors='replace')


def _extract_blocks(data: bytes, source_name: str) -> List[SourceBlock]:
    blocks: List[SourceBlock] = []
    for idx, stream_match in enumerate(re.finditer(rb'stream\r?\n(.*?)\r?\nendstream', data, re.S), start=1):
        stream_data = stream_match.group(1)
        lines: List[str] = []
        for s in STRING_RE.findall(stream_data):
            end = s.rfind(b')')
            literal = s[1:end] if end > 0 else s[1:]
            lines.append(_unescape_pdf_string(literal))
        text = '\n'.join(lines)
        blocks.append(SourceBlock(source_ref=f'{source_name}:page{idx}', text=text, source_type='pdf'))
    return blocks


def parse_pdf_bytes(data: bytes, source_name: str) -> ParsedDocument:
    if not data.startswith(b'%PDF'):
        text = data.decode('utf-8', errors='replace')
        return ParsedDocument(source_path=source_name, source_type='pdf', blocks=[SourceBlock(source_ref=f'{source_name}:text', text=text, source_type='pdf')], raw_text=text, parse_status='NON_PDF_FALLBACK')
    blocks = _extract_blocks(data, source_name)
    raw_text = '\n'.join(block.text for block in blocks)
    return ParsedDocument(source_path=source_name, source_type='pdf', blocks=blocks, raw_text=raw_text)


def parse_pdf(path: str | Path) -> ParsedDocument:
    p = Path(path)
    return parse_pdf_bytes(p.read_bytes(), p.name)
