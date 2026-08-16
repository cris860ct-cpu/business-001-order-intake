"""Benchmark harness placeholder for Nova Labs Mission 014C.3.

It compares prototype output against locked ground truth.
This file is preserved for later execution in a connected runtime.
"""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List


def read_csv_rows(path: str | Path) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def summarize_orders(orders: List[Dict[str, str]]) -> Dict[str, int]:
    counts = Counter(order["order_status"] for order in orders)
    return dict(counts)


def compare_predictions(predicted: List[Dict[str, object]], ground_truth: List[Dict[str, str]]) -> Dict[str, object]:
    # Minimal alignment by order_id and line_number.
    gt_index = {
        (row["order_id"], row["line_number"]): row
        for row in ground_truth
    }
    correct_sku = 0
    total = 0
    review_required = 0
    for pred in predicted:
        key = (str(pred.get("order_id")), str(pred.get("line_number")))
        if key not in gt_index:
            continue
        total += 1
        gt = gt_index[key]
        if pred.get("proposed_sku") == gt.get("distributor_sku"):
            correct_sku += 1
        if pred.get("status") != "READY":
            review_required += 1
    return {
        "total_compared": total,
        "exact_sku_accuracy": correct_sku / total if total else None,
        "percent_requiring_review": review_required / total if total else None,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--orders", required=True)
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    orders = read_csv_rows(args.orders)
    ground_truth = read_csv_rows(args.ground_truth)
    with open(args.predictions, encoding="utf-8") as f:
        predictions = json.load(f)

    report = {
        "order_summary": summarize_orders(orders),
        "comparison": compare_predictions(predictions, ground_truth),
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
