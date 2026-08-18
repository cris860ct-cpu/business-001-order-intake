from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import re

from src.ingestion import extract_order_from_document, parse_document
from src.ingestion.extraction import ParsedDocument, ParsedOrder, parsed_order_to_v2_inputs
from src.prototype_v2 import CatalogMatcher, load_catalog, run_order_validation


HEADER_PATTERNS: Dict[str, re.Pattern[str]] = {
    "customer": re.compile(r"^(?:customer|customer name|bill to)\s*[:#-]\s*(.+)$", re.IGNORECASE),
    "customer_account_id": re.compile(
        r"^(?:customer account(?: id| number)?|account(?: id| number)?|acct(?: id| number)?|account)\s*[:#-]\s*(.+)$",
        re.IGNORECASE,
    ),
    "bill_to": re.compile(r"^bill to\s*[:#-]\s*(.+)$", re.IGNORECASE),
    "ship_to": re.compile(r"^ship to\s*[:#-]\s*(.+)$", re.IGNORECASE),
    "po_number": re.compile(r"^(?:po(?: number)?|purchase order|po\s*#|po#)\s*[:#-]\s*(.+)$", re.IGNORECASE),
    "order_date": re.compile(r"^(?:order date|date)\s*[:#-]\s*(.+)$", re.IGNORECASE),
    "requested_delivery_date": re.compile(
        r"^(?:requested delivery(?: date)?|delivery date|ship date)\s*[:#-]\s*(.+)$",
        re.IGNORECASE,
    ),
}


@dataclass
class FieldObservation:
    value: Optional[str]
    source_ref: Optional[str]
    confidence: float
    validation_status: str
    raw_text: Optional[str] = None


@dataclass
class OrderEnvelope:
    customer: FieldObservation
    customer_account_id: FieldObservation
    bill_to: FieldObservation
    ship_to: FieldObservation
    po_number: FieldObservation
    order_date: FieldObservation
    requested_delivery_date: FieldObservation
    unresolved_fields: List[str]
    completeness_status: str
    fulfillment_status: str
    review_flags: List[str]


@dataclass
class ReviewPackage:
    order_id: str
    original_source: Dict[str, Any]
    ai_prepared_order: Dict[str, Any]
    order_status: str
    completeness_status: str
    review_flags: List[str]
    line_rows: List[Dict[str, Any]]
    envelope: Dict[str, Any]



def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()



def _document_lines(document: ParsedDocument) -> List[Tuple[str, str]]:
    lines: List[Tuple[str, str]] = []
    for block in document.blocks + document.attachments:
        for raw_line in block.text.splitlines():
            stripped = raw_line.strip()
            if stripped:
                lines.append((block.source_ref, stripped))
    return lines



def _extract_observation(
    field_name: str,
    document: ParsedDocument,
    fallback_value: Optional[str],
    allow_fallback: bool = True,
) -> FieldObservation:
    for source_ref, line in _document_lines(document):
        normalized = _normalize_text(line)
        match = HEADER_PATTERNS[field_name].match(normalized)
        if match:
            value = _normalize_text(match.group(1)) or None
            if value:
                return FieldObservation(
                    value=value,
                    source_ref=source_ref,
                    confidence=1.0,
                    validation_status="VALID",
                    raw_text=line,
                )

    if allow_fallback and fallback_value:
        return FieldObservation(
            value=_normalize_text(fallback_value),
            source_ref=f"{document.source_path}:parsed",
            confidence=0.9,
            validation_status="VALID",
            raw_text=fallback_value,
        )

    return FieldObservation(value=None, source_ref=None, confidence=0.0, validation_status="NOT_PROVIDED", raw_text=None)



def extract_order_envelope(order: ParsedOrder, document: ParsedDocument) -> OrderEnvelope:
    customer = _extract_observation("customer", document, order.customer)
    customer_account_id = _extract_observation("customer_account_id", document, None, allow_fallback=False)
    bill_to = _extract_observation("bill_to", document, None, allow_fallback=False)
    ship_to = _extract_observation("ship_to", document, None, allow_fallback=False)
    po_number = _extract_observation("po_number", document, order.po_number)
    order_date = _extract_observation("order_date", document, order.order_date)
    requested_delivery_date = _extract_observation("requested_delivery_date", document, order.requested_delivery_date)

    unresolved_fields: List[str] = []
    review_flags: List[str] = []
    for field_name, obs in [
        ("customer", customer),
        ("po_number", po_number),
        ("order_date", order_date),
        ("requested_delivery_date", requested_delivery_date),
    ]:
        if not obs.value:
            unresolved_fields.append(field_name)
            review_flags.append(f"MISSING_{field_name.upper()}")

    fulfillment_resolved = bool(ship_to.value or bill_to.value or customer_account_id.value)
    fulfillment_status = "RESOLVED" if fulfillment_resolved else "NOT_PROVIDED"
    if not fulfillment_resolved:
        review_flags.append("FULFILLMENT_NOT_PROVIDED")

    completeness_status = "COMPLETE" if not unresolved_fields else "INCOMPLETE"

    return OrderEnvelope(
        customer=customer,
        customer_account_id=customer_account_id,
        bill_to=bill_to,
        ship_to=ship_to,
        po_number=po_number,
        order_date=order_date,
        requested_delivery_date=requested_delivery_date,
        unresolved_fields=unresolved_fields,
        completeness_status=completeness_status,
        fulfillment_status=fulfillment_status,
        review_flags=review_flags,
    )



def _parse_price(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace("$", "")
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None



def _price_status(source_price: Optional[float], catalog_price: Optional[float]) -> Tuple[str, Optional[float], Optional[float]]:
    if source_price is None and catalog_price is None:
        return "CATALOG PRICE UNAVAILABLE", None, None
    if source_price is None:
        return "SOURCE PRICE NOT PROVIDED", None, catalog_price
    if catalog_price is None:
        return "CATALOG PRICE UNAVAILABLE", source_price, None
    if abs(source_price - catalog_price) < 0.005:
        return "PRICE MATCH", source_price, catalog_price
    return "PRICE MISMATCH", source_price, catalog_price



def enrich_line_rows_with_pricing(line_rows: List[Dict[str, Any]], catalog: CatalogMatcher) -> List[Dict[str, Any]]:
    enriched: List[Dict[str, Any]] = []
    for row in line_rows:
        sku = row.get("proposed_sku")
        catalog_item = catalog.exact_match(str(sku)) if sku else None
        extracted = row.get("extracted_value", {}) or {}
        source_price = _parse_price(extracted.get("source_unit_price") or extracted.get("unit_price") or row.get("source_unit_price"))
        catalog_price = catalog_item.unit_price if catalog_item else None
        price_validation, source_price, catalog_price = _price_status(source_price, catalog_price)
        price_difference = None
        if source_price is not None and catalog_price is not None:
            price_difference = round(source_price - catalog_price, 2)
        review_flags = list(row.get("review_flags", []))
        if price_validation == "PRICE MISMATCH":
            review_flags.append("PRICE_MISMATCH")
        enriched_row = dict(row)
        enriched_row.update(
            {
                "source_unit_price": source_price,
                "catalog_unit_price": catalog_price,
                "price_difference": price_difference,
                "price_validation": price_validation,
                "review_flags": review_flags,
            }
        )
        enriched.append(enriched_row)
    return enriched



def evaluate_order_completion(order: ParsedOrder, envelope: OrderEnvelope, line_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    unresolved_fields = list(envelope.unresolved_fields)
    review_flags = list(envelope.review_flags)
    line_statuses = [str(row.get("status", "")).upper() for row in line_rows]
    price_statuses = [str(row.get("price_validation", "")).upper() for row in line_rows]

    if any(status == "PRICE_MISMATCH" for status in price_statuses):
        review_flags.append("PRICE_MISMATCH")
    if any(status == "CATALOG PRICE UNAVAILABLE" for status in price_statuses):
        review_flags.append("CATALOG_PRICE_UNAVAILABLE")
    if any(status == "SOURCE PRICE NOT PROVIDED" for status in price_statuses):
        review_flags.append("SOURCE_PRICE_NOT_PROVIDED")

    if order.parse_status == "OCR_BLOCKED" or not order.line_items:
        order_status = "EXCEPTION"
        review_flags.append("OCR_BLOCKED")
    elif any(status == "EXCEPTION" for status in line_statuses):
        order_status = "EXCEPTION"
        review_flags.append("LINE_EXCEPTION")
    elif any(status == "REVIEW REQUIRED" for status in line_statuses):
        order_status = "REVIEW REQUIRED"
    elif unresolved_fields:
        order_status = "REVIEW REQUIRED"
    else:
        order_status = "READY"

    completeness_status = "COMPLETE" if order_status == "READY" else "INCOMPLETE"
    return {
        "order_status": order_status,
        "completeness_status": completeness_status,
        "unresolved_fields": unresolved_fields,
        "review_flags": list(dict.fromkeys(review_flags)),
        "line_statuses": line_statuses,
        "price_statuses": price_statuses,
    }



def build_clean_review_output(source_path: str | Path, catalog_path: str | Path) -> ReviewPackage:
    document = parse_document(source_path)
    order = extract_order_from_document(source_path)
    envelope = extract_order_envelope(order, document)

    line_inputs = parsed_order_to_v2_inputs(order, order.source_type)
    price_by_line = {str(line.line_number): _parse_price(line.price) for line in order.line_items}
    for entry in line_inputs:
        entry["source_unit_price"] = price_by_line.get(str(entry.get("line_number")))

    validation = run_order_validation(
        str(catalog_path),
        [{"order_id": Path(str(source_path)).stem}],
        {Path(str(source_path)).stem: line_inputs},
    )[0]
    matcher = CatalogMatcher(load_catalog(catalog_path))
    for row in validation["line_rows"]:
        row["source_unit_price"] = price_by_line.get(str(row.get("line_number")))
    enriched_lines = enrich_line_rows_with_pricing(validation["line_rows"], matcher)
    completion = evaluate_order_completion(order, envelope, enriched_lines)

    ai_prepared_order = {
        "order_information": {
            "customer": asdict(envelope.customer),
            "customer_account_id": asdict(envelope.customer_account_id),
            "bill_to": asdict(envelope.bill_to),
            "ship_to": asdict(envelope.ship_to),
            "po_number": asdict(envelope.po_number),
            "order_date": asdict(envelope.order_date),
            "requested_delivery_date": asdict(envelope.requested_delivery_date),
        },
        "line_items": [
            {
                "customer_item_number": row.get("customer_item_number"),
                "raw_description": row.get("extracted_value", {}).get("raw_description"),
                "matched_sku": row.get("proposed_sku"),
                "matched_catalog_description": row.get("matched_description"),
                "quantity": row.get("extracted_value", {}).get("quantity"),
                "unit": row.get("extracted_value", {}).get("unit"),
                "source_price": row.get("source_unit_price"),
                "catalog_price": row.get("catalog_unit_price"),
                "price_difference": row.get("price_difference"),
                "price_validation": row.get("price_validation"),
                "confidence": row.get("confidence"),
                "review_reason": row.get("review_reason"),
                "alternative_matches": row.get("alternative_matches") if row.get("confidence") != "HIGH" or row.get("review_reason") else [],
                "status": row.get("status"),
            }
            for row in enriched_lines
        ],
        "order_validation": {
            "completeness_status": completion["completeness_status"],
            "unresolved_fields": completion["unresolved_fields"],
            "conflicts": [row.get("review_reason") for row in enriched_lines if row.get("review_reason")],
            "review_flags": completion["review_flags"],
            "final_ai_status": completion["order_status"],
            "fulfillment_status": envelope.fulfillment_status,
        },
    }

    return ReviewPackage(
        order_id=Path(str(source_path)).stem,
        original_source={
            "source_type": document.source_type,
            "source_path": str(source_path),
            "text": document.raw_text,
            "integrity_check": document.raw_text == parse_document(source_path).raw_text,
        },
        ai_prepared_order=ai_prepared_order,
        order_status=completion["order_status"],
        completeness_status=completion["completeness_status"],
        review_flags=completion["review_flags"],
        line_rows=enriched_lines,
        envelope={
            "customer": asdict(envelope.customer),
            "customer_account_id": asdict(envelope.customer_account_id),
            "bill_to": asdict(envelope.bill_to),
            "ship_to": asdict(envelope.ship_to),
            "po_number": asdict(envelope.po_number),
            "order_date": asdict(envelope.order_date),
            "requested_delivery_date": asdict(envelope.requested_delivery_date),
            "unresolved_fields": completion["unresolved_fields"],
            "fulfillment_status": envelope.fulfillment_status,
        },
    )


__all__ = [
    "FieldObservation",
    "OrderEnvelope",
    "ReviewPackage",
    "build_clean_review_output",
    "enrich_line_rows_with_pricing",
    "evaluate_order_completion",
    "extract_order_envelope",
]
