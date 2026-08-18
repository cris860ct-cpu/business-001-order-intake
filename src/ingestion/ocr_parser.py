from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

from PIL import Image, ImageOps

from .extraction import ParsedDocument


@dataclass
class OCRResult:
    status: str
    text: str
    engine: Optional[str] = None
    confidence: Optional[float] = None
    quality_score: Optional[float] = None
    line_count: int = 0
    token_count: int = 0
    preprocessing: Optional[str] = None
    warning: Optional[str] = None


@lru_cache(maxsize=1)
def _load_rapidocr():
    try:
        from rapidocr_onnxruntime import RapidOCR  # type: ignore
    except Exception as exc:  # pragma: no cover - import failure depends on env
        raise RuntimeError(f'rapidocr_onnxruntime unavailable: {exc}') from exc
    return RapidOCR()


def _image_to_array(image: Image.Image):
    import numpy as np  # type: ignore

    return np.array(image.convert('RGB'))


def _text_from_result(result: Sequence[Sequence[object]] | None) -> Tuple[str, float, int]:
    if not result:
        return '', 0.0, 0
    ordered = sorted(result, key=lambda item: (float(item[0][0][1]), float(item[0][0][0])) if item and item[0] else (0.0, 0.0))
    parts = []
    confidences = []
    for item in ordered:
        if len(item) < 3:
            continue
        text = str(item[1]).strip()
        if not text:
            continue
        parts.append(text)
        try:
            confidences.append(float(item[2]))
        except Exception:
            pass
    text = '\n'.join(parts)
    confidence = sum(confidences) / len(confidences) if confidences else 0.0
    return text, confidence, len(parts)


def _variant_images(image: Image.Image):
    gray = ImageOps.grayscale(image)
    gray_up2 = gray.resize((image.width * 2, image.height * 2), Image.Resampling.LANCZOS)
    return [
        ('original', image.convert('RGB')),
        ('gray_upscaled_2x', gray_up2.convert('RGB')),
    ]


def _score_variant(text: str, confidence: float, line_count: int) -> float:
    token_count = len(text.split())
    return confidence * 0.8 + min(0.2, token_count / 200.0) + min(0.1, line_count / 20.0)


def ocr_image(path: str | Path) -> OCRResult:
    p = Path(path)
    image = Image.open(p)
    try:
        engine = _load_rapidocr()
    except Exception as exc:
        return OCRResult(status='OCR_BLOCKED', text='', engine=None, warning=str(exc))

    best: OCRResult | None = None
    for preprocessing, variant in _variant_images(image):
        try:
            result, _elapsed = engine(_image_to_array(variant))
        except Exception as exc:
            if best is None:
                best = OCRResult(status='OCR_BLOCKED', text='', engine='rapidocr', warning=str(exc), preprocessing=preprocessing)
            continue
        text, confidence, line_count = _text_from_result(result)
        token_count = len(text.split())
        status = 'OK' if text and confidence >= 0.45 else 'OCR_LOW_CONFIDENCE'
        candidate = OCRResult(
            status=status,
            text=text,
            engine='rapidocr',
            confidence=round(confidence, 4),
            quality_score=round(_score_variant(text, confidence, line_count), 4),
            line_count=line_count,
            token_count=token_count,
            preprocessing=preprocessing,
            warning=None if text else 'no text detected',
        )
        if best is None or (candidate.quality_score or 0.0) > (best.quality_score or 0.0):
            best = candidate

    if best is None:
        return OCRResult(status='OCR_LOW_CONFIDENCE', text='', engine='rapidocr', warning='no OCR result produced')
    return best
