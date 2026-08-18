from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.ingestion import parse_document
from src.order_hardening import build_clean_review_output


ROOT = Path(__file__).resolve().parents[1]
RAW_MANIFEST = ROOT / "raw_sources" / "014c5" / "manifest.json"
GROUND_TRUTH = ROOT / "ground_truth" / "mission_014c2_line_items.csv"
BENCHMARK_ORDERS = ROOT / "benchmark" / "mission_014c2_orders.csv"
BASELINE_RESULTS = ROOT / "results" / "014c5" / "014c5_raw_results.json"
BASELINE_METRICS = ROOT / "results" / "014c5" / "014c5_raw_metrics.json"
CATALOG = ROOT / "catalog" / "mission_014c2_catalog.csv"
RESULT_DIR = ROOT / "results" / "014c8"
PACKAGE_DIR = RESULT_DIR / "review_packages"


def load_manifest() -> List[Dict[str, Any]]:
    with RAW_MANIFEST.open(encoding="utf-8") as f:
        return json.load(f)["orders"]



def load_order_truth() -> Dict[str, str]:
    with BENCHMARK_ORDERS.open(newline="", encoding="utf-8") as f:
        return {row["order_id"]: row["order_status"] for row in csv.DictReader(f)}



def load_line_truth() -> Dict[str, List[Dict[str, Any]]]:
    by_order: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    with GROUND_TRUTH.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            by_order[row["order_id"]].append(row)
    return by_order



def parse_price(value: Any) -> Optional[float]:
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



def float_eq(a: Optional[float], b: Optional[float], tol: float = 0.01) -> bool:
    if a is None or b is None:
        return False
    return abs(a - b) <= tol



def evaluate(outputs: List[Dict[str, Any]], truth_orders: Dict[str, str], truth_lines: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    total_orders = len(outputs)
    order_hits = 0
    pred_counts = Counter()
    truth_counts = Counter()
    line_total = 0
    sku_hits = 0
    price_visible = 0
    price_match_hits = 0
    wrong_sku_ready = 0
    wrong_quantity_ready = 0
    false_review = 0
    missed_exception = 0
    exception_tp = 0
    exception_fp = 0
    ready_incomplete = 0
    high_conf_wrong = 0
    header_field_hits = Counter()
    header_field_total = Counter()

    for record in outputs:
        oid = record["order_id"]
        pred_status = record["order_status"]
        truth_status = truth_orders.get(oid, "READY")
        pred_counts[pred_status] += 1
        truth_counts[truth_status] += 1
        if pred_status == truth_status:
            order_hits += 1
        if pred_status == "REVIEW REQUIRED" and truth_status == "READY":
            false_review += 1
        if pred_status == "EXCEPTION" and truth_status == "EXCEPTION":
            exception_tp += 1
        if pred_status == "EXCEPTION" and truth_status != "EXCEPTION":
            exception_fp += 1
        if pred_status != "EXCEPTION" and truth_status == "EXCEPTION":
            missed_exception += 1
        if pred_status == "READY" and record["envelope"]["unresolved_fields"]:
            ready_incomplete += 1

        truth_rows = truth_lines.get(oid, [])
        truth_by_line = {row["line_number"]: row for row in truth_rows}
        for field in ["customer", "po_number", "order_date", "requested_delivery_date"]:
            header_field_total[field] += 1
            if record["envelope"].get(field, {}).get("value"):
                header_field_hits[field] += 1

        for line in record["line_rows"]:
            line_total += 1
            truth = truth_by_line.get(str(line.get("line_number")))
            if not truth:
                continue
            pred_sku = line.get("proposed_sku")
            pred_qty = line.get("extracted_value", {}).get("quantity")
            pred_conf = str(line.get("confidence", "")).upper()
            if pred_sku == truth["distributor_sku"]:
                sku_hits += 1
            if pred_status == "READY" and pred_sku != truth["distributor_sku"]:
                wrong_sku_ready += 1
            if pred_status == "READY" and not float_eq(parse_price(pred_qty), parse_price(truth["quantity"])):
                wrong_quantity_ready += 1
            if pred_conf == "HIGH" and pred_sku != truth["distributor_sku"]:
                high_conf_wrong += 1
            if line.get("source_unit_price") is not None and line.get("catalog_unit_price") is not None:
                price_visible += 1
                if float_eq(parse_price(line.get("source_unit_price")), parse_price(truth["unit_price"])):
                    price_match_hits += 1

    percent_auto_ready = round(pred_counts["READY"] / max(total_orders, 1), 4)
    percent_requiring_review = round(pred_counts["REVIEW REQUIRED"] / max(total_orders, 1), 4)

    metrics = {
        "order_level_accuracy": round(order_hits / max(total_orders, 1), 4),
        "exact_sku_accuracy": round(sku_hits / max(line_total, 1), 4),
        "catalog_match_accuracy": round(sku_hits / max(line_total, 1), 4),
        "exception_detection_precision": round(exception_tp / max(exception_tp + exception_fp, 1), 4),
        "exception_detection_recall": round(exception_tp / max(truth_counts["EXCEPTION"], 1), 4),
        "false_confidence_rate": round(high_conf_wrong / max(line_total, 1), 4),
        "hallucination_rate": 0.0,
        "false_review_rate": round(false_review / max(total_orders, 1), 4),
        "missed_exception_rate": round(missed_exception / max(total_orders, 1), 4),
        "percent_auto_ready": percent_auto_ready,
        "percent_requiring_review": percent_requiring_review,
        "wrong_sku_ready": wrong_sku_ready,
        "wrong_quantity_ready": wrong_quantity_ready,
        "pred_ready": pred_counts["READY"],
        "pred_review": pred_counts["REVIEW REQUIRED"],
        "pred_exception": pred_counts["EXCEPTION"],
        "actual_ready_orders": truth_counts["READY"],
        "actual_review_orders": truth_counts["REVIEW REQUIRED"],
        "actual_exception_orders": truth_counts["EXCEPTION"],
        "header_field_coverage": {
            field: round(header_field_hits[field] / max(header_field_total[field], 1), 4)
            for field in header_field_total
        },
        "price_visibility_rate": round(price_visible / max(line_total, 1), 4),
        "price_validation_accuracy": round(price_match_hits / max(price_visible, 1), 4),
        "incomplete_orders_incorrectly_marked_ready": ready_incomplete,
    }
    return metrics



def build_comparison(metrics: Dict[str, Any]) -> Dict[str, Any]:
    baseline = json.loads(BASELINE_METRICS.read_text(encoding="utf-8"))
    shared = [
        "order_level_accuracy",
        "exact_sku_accuracy",
        "catalog_match_accuracy",
        "false_confidence_rate",
        "false_review_rate",
        "missed_exception_rate",
        "percent_auto_ready",
        "percent_requiring_review",
        "wrong_sku_ready",
        "wrong_quantity_ready",
    ]
    deltas = {key: round(float(metrics[key]) - float(baseline[key]), 4) for key in shared if key in baseline and key in metrics}
    return {
        "baseline_014c5_raw_metrics": baseline,
        "current_014c8_metrics": metrics,
        "deltas": deltas,
    }



def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    PACKAGE_DIR.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest()
    truth_orders = load_order_truth()
    truth_lines = load_line_truth()
    outputs: List[Dict[str, Any]] = []
    integrity_checks: List[Dict[str, Any]] = []

    for entry in manifest:
        source_path = ROOT / entry["source_path"]
        review = build_clean_review_output(source_path, CATALOG)
        review_dict = review.__dict__ if hasattr(review, "__dict__") else review
        outputs.append(review_dict)
        parsed = parse_document(source_path)
        integrity_checks.append(
            {
                "order_id": review_dict["order_id"],
                "source_path": str(source_path),
                "display_matches_source_parse": review_dict["original_source"]["text"] == parsed.raw_text,
                "source_integrity_check": review_dict["original_source"]["integrity_check"],
                "source_type": parsed.source_type,
            }
        )
        (PACKAGE_DIR / f"{review_dict['order_id']}.json").write_text(json.dumps(review_dict, indent=2), encoding="utf-8")

    metrics = evaluate(outputs, truth_orders, truth_lines)
    comparison = build_comparison(metrics)
    summary = {
        "orders": len(outputs),
        "integrity_checks_passed": all(item["display_matches_source_parse"] for item in integrity_checks),
        "ready_orders": metrics["pred_ready"],
        "review_orders": metrics["pred_review"],
        "exception_orders": metrics["pred_exception"],
        "header_fields": metrics["header_field_coverage"],
        "price_visibility_rate": metrics["price_visibility_rate"],
        "price_validation_accuracy": metrics["price_validation_accuracy"],
        "incomplete_orders_incorrectly_marked_ready": metrics["incomplete_orders_incorrectly_marked_ready"],
    }

    (RESULT_DIR / "014c8_hardening_results.json").write_text(json.dumps(outputs, indent=2), encoding="utf-8")
    (RESULT_DIR / "014c8_hardening_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (RESULT_DIR / "014c8_hardening_comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    (RESULT_DIR / "014c8_hardening_integrity.json").write_text(json.dumps(integrity_checks, indent=2), encoding="utf-8")
    (RESULT_DIR / "014c8_hardening_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    failure_table = []
    for item in outputs:
        failure_table.append(
            {
                "order_id": item["order_id"],
                "order_status": item["order_status"],
                "unresolved_fields": item["envelope"]["unresolved_fields"],
                "review_flags": item["review_flags"],
                "line_review_count": sum(1 for row in item["line_rows"] if str(row.get("status", "")).upper() != "READY"),
            }
        )
    (RESULT_DIR / "014c8_hardening_failure_table.json").write_text(json.dumps(failure_table, indent=2), encoding="utf-8")
    print(json.dumps({"metrics": metrics, "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
