"""Compare extracted findings with manually annotated gold findings."""

import argparse
import csv
import json
import re
import math
from collections import Counter, defaultdict
from pathlib import Path


CATEGORIES = (
    "assumption",
    "alternative",
    "constraint",
    "forecast",
    "evaluation_criterion",
    "decision_rationale",
    "project_objective",
)
GRADES = {"supported", "partially supported", "unsupported", "contradicted"}


def normalize(text):
    return re.sub(r"\s+", " ", text.casefold()).strip()


def quotes_overlap(left, right):
    """Match quotes when either normalized quote contains the other."""
    left, right = normalize(left), normalize(right)
    return bool(left and right and (left in right or right in left))


def evaluate(extracted, gold):
    matched_gold = set()
    rows = []
    for item in extracted:
        match_index = next(
            (i for i, manual in enumerate(gold)
             if i not in matched_gold
             and quotes_overlap(item.get("evidence_quote", ""), manual.get("evidence_quote", ""))),
            None,
        )
        manual = gold[match_index] if match_index is not None else {}
        if match_index is not None:
            matched_gold.add(match_index)
        rows.append({
            "category": item.get("category", ""),
            "claim": item.get("claim", ""),
            "evidence_quote": item.get("evidence_quote", ""),
            "classification_reason": item.get("classification_reason", ""),
            "document": item.get("document", ""),
            "page": item.get("page", ""),
            "quote_verified": item.get("quote_verified", ""),
            "parse_mode": item.get("parse_mode", ""),
            "finding_id": item.get("finding_id", ""),
            "printed_page_label": item.get("printed_page_label", ""),
            "epistemic_status": item.get("epistemic_status", ""),
            "attributed_to": item.get("attributed_to", ""),
            "refers_to_alternative": item.get("refers_to_alternative", ""),
            "gold_category": manual.get("category", ""),
            "matched_gold": match_index is not None,
            "support_grade": item.get("support_grade", ""),
            "gold_epistemic_status": manual.get("epistemic_status", ""),
            "gold_attributed_to": manual.get("attributed_to", ""),
            "gold_refers_to_alternative": manual.get("refers_to_alternative", ""),
        })

    counts = defaultdict(Counter)
    for row in rows:
        predicted = row["category"]
        gold_category = row["gold_category"]
        if row["matched_gold"] and gold_category in CATEGORIES and predicted in CATEGORIES:
            counts[gold_category]["tp"] += 1
            counts[predicted]["predicted"] += 1
        elif predicted in CATEGORIES:
            counts[predicted]["predicted"] += 1
    for i, item in enumerate(gold):
        if i not in matched_gold and item.get("category") in CATEGORIES:
            counts[item["category"]]["gold"] += 1
    # Derive denominators using all predicted and gold items, including matches.
    predicted_counts = Counter(r["category"] for r in rows)
    gold_counts = Counter(item.get("category") for item in gold)
    metrics = {}
    for category in CATEGORIES:
        tp = sum(1 for r in rows if r["matched_gold"] and r["category"] == category and r["gold_category"] == category)
        precision = tp / predicted_counts[category] if predicted_counts[category] else 0.0
        recall = tp / gold_counts[category] if gold_counts[category] else 0.0
        metrics[category] = {"precision": precision, "recall": recall,
                             "true_positives": tp, "predicted": predicted_counts[category], "gold": gold_counts[category]}
    matrix = {gold_cat: {pred_cat: 0 for pred_cat in CATEGORIES} for gold_cat in CATEGORIES}
    for row in rows:
        if row["matched_gold"] and row["gold_category"] in CATEGORIES and row["category"] in CATEGORIES:
            matrix[row["gold_category"]][row["category"]] += 1
    for field, gold_field in (("epistemic_status", "gold_epistemic_status"),
                              ("attributed_to", "gold_attributed_to"),
                              ("refers_to_alternative", "gold_refers_to_alternative")):
        correct = sum(1 for r in rows if r["matched_gold"] and normalize(str(r[field])) == normalize(str(r[gold_field])))
        metrics[field] = {"accuracy": correct / len(rows) if rows else 0.0, "correct": correct, "total": len(rows)}
    correct = sum(1 for r in rows if r["matched_gold"] and r["category"] == r["gold_category"])
    metrics["overall"] = {"precision": correct / len(rows) if rows else 0.0,
                           "recall": correct / len(gold) if gold else 0.0,
                           "true_positives": correct, "predicted": len(rows), "gold": len(gold),
                           "quote_verification_failure_rate": sum(r["quote_verified"] is False for r in rows) / len(rows) if rows else 0.0}
    return rows, metrics, matrix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extracted", type=Path, default=Path("outputs/findings_single_20261030T022825Z.json"))
    parser.add_argument("--gold", type=Path, default=Path("gold/page91.json"))
    parser.add_argument("--csv", type=Path, default=Path("outputs/evaluation_page91.csv"))
    parser.add_argument("--metrics", type=Path, default=Path("outputs/metrics_page91.json"))
    args = parser.parse_args()
    extracted = json.loads(args.extracted.read_text(encoding="utf-8"))
    if isinstance(extracted, dict):
        extracted = extracted.get("findings", [])
    gold = json.loads(args.gold.read_text(encoding="utf-8"))
    if isinstance(gold, dict):
        gold = gold.get("findings", [])
    # A graded gold file can also provide human grades for matching extracted
    # findings. Copy the gold grade onto the corresponding output row.
    rows, metrics, matrix = evaluate(extracted, gold)
    gold_by_quote = {normalize(x.get("evidence_quote", "")): x for x in gold}
    for row in rows:
        match = next((x for x in gold if quotes_overlap(row["evidence_quote"], x.get("evidence_quote", ""))), None)
        if match and match.get("support_grade") in GRADES:
            row["support_grade"] = match["support_grade"]
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    fields = ["finding_id", "category", "claim", "evidence_quote", "classification_reason", "epistemic_status", "attributed_to", "refers_to_alternative", "document", "page", "printed_page_label", "quote_verified", "parse_mode", "gold_category", "matched_gold", "support_grade", "gold_epistemic_status", "gold_attributed_to", "gold_refers_to_alternative"]
    with args.csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print("Category metrics (exact category match among quote-overlap matches):")
    for category in CATEGORIES:
        values=metrics[category]
        precision=f"{values['precision']:.3f}" if values['predicted'] else "n/a"
        print(f"{category}: precision={precision} recall={values['recall']:.3f} ({values['true_positives']}/{values['predicted']} predicted, {values['gold']} gold)")
    for field in ("epistemic_status", "attributed_to", "refers_to_alternative"):
        value=metrics[field]; print(f"{field} accuracy: {value['accuracy']:.3f} ({value['correct']}/{value['total']})")
    print(f"Overall category precision={metrics['overall']['precision']:.3f} recall={metrics['overall']['recall']:.3f}; quote verification failure rate={metrics['overall']['quote_verification_failure_rate']:.3f}")
    print("\nConfusion matrix (rows=gold, columns=extracted):")
    print("gold\\extracted," + ",".join(CATEGORIES))
    for category in CATEGORIES:
        print(category + "," + ",".join(str(matrix[category][pred]) for pred in CATEGORIES))
    print(f"\nEditable finding grades: {args.csv} (use supported, partially supported, unsupported, or contradicted)")
    args.metrics.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Metrics saved to: {args.metrics}")


if __name__ == "__main__":
    main()
