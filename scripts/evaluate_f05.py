#!/usr/bin/env python3
"""
Macro F0.5 evaluator for the ER challenge.

Scores every Source 1 entity (the union of ground truth and Source 1 ids, so
singletons are always included) and averages per-entity F0.5:

    F0.5 = 1.25 * P * R / (0.25 * P + R)

Singleton rules: no true matches and no predictions -> 1.0; no true matches
but any prediction -> 0.0. An S1 id missing from the predictions file counts
as an empty prediction.
"""

import argparse
import os
from collections import defaultdict
from typing import Dict, Optional, Set, Tuple

NULL_VALUES = {"", "nan", "none", "null", "<null>"}


def load_matches(path: str, value_col: str = "matched_entity_ids") -> Tuple[Dict[str, Set[str]], int]:
    """Load 'source1_entity_id<TAB>comma-separated ids' into {s1_id: set(ids)}.

    Returns the mapping and the number of duplicate S1 rows (merged by union).
    """
    matches: Dict[str, Set[str]] = {}
    duplicates = 0
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\r\n").split("\t")
        if header[0] != "source1_entity_id" or value_col not in header:
            raise ValueError(f"{path}: unexpected header {header}")
        col = header.index(value_col)
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            s1_id = parts[0].strip()
            if not s1_id:
                continue
            raw = parts[col].strip() if len(parts) > col else ""
            ids = set() if raw.lower() in NULL_VALUES else {x.strip() for x in raw.split(",") if x.strip()}
            if s1_id in matches:
                duplicates += 1
                matches[s1_id] |= ids
            else:
                matches[s1_id] = ids
    return matches, duplicates


def load_s1_countries(path: Optional[str]) -> Dict[str, str]:
    """Map S1 entity_id -> country (for the per-country breakdown)."""
    countries: Dict[str, str] = {}
    if not path or not os.path.isfile(path):
        return countries
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\r\n").split("\t")
        id_col, c_col = header.index("entity_id"), header.index("country")
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) > max(id_col, c_col):
                countries[parts[id_col].strip()] = parts[c_col].strip()
    return countries


def entity_scores(true_ids: Set[str], pred_ids: Set[str]) -> Tuple[float, float, float]:
    """Return (precision, recall, F0.5) for one S1 entity."""
    if not true_ids:
        # Singleton: perfect only if we also predicted nothing
        return (1.0, 1.0, 1.0) if not pred_ids else (0.0, 0.0, 0.0)
    if not pred_ids:
        return 0.0, 0.0, 0.0
    tp = len(true_ids & pred_ids)
    if tp == 0:
        return 0.0, 0.0, 0.0
    p = tp / len(pred_ids)
    r = tp / len(true_ids)
    return p, r, (1.25 * p * r) / (0.25 * p + r)


def evaluate(pred_path: str, truth_path: str, s1_path: Optional[str]) -> None:
    truth, truth_dups = load_matches(truth_path)
    preds, pred_dups = load_matches(pred_path)
    countries = load_s1_countries(s1_path)

    universe = set(truth) | set(countries)
    unknown_pred_ids = set(preds) - universe
    missing_pred_ids = universe - set(preds)

    # bucket -> [count, sum_p, sum_r, sum_f]
    buckets: Dict[str, list] = defaultdict(lambda: [0, 0.0, 0.0, 0.0])
    tp_total = pred_total = true_total = 0
    perfect = 0
    singleton_fp = 0      # true singletons we predicted matches for
    missed_entities = 0   # non-singletons we predicted empty for

    for s1_id in universe:
        true_ids = truth.get(s1_id, set())
        pred_ids = preds.get(s1_id, set())
        p, r, f = entity_scores(true_ids, pred_ids)

        kind = "singleton" if not true_ids else "non-singleton"
        country = countries.get(s1_id, "unknown")
        for key in ("ALL", f"type:{kind}", f"country:{country}"):
            b = buckets[key]
            b[0] += 1
            b[1] += p
            b[2] += r
            b[3] += f

        tp_total += len(true_ids & pred_ids)
        pred_total += len(pred_ids)
        true_total += len(true_ids)
        perfect += f == 1.0
        if not true_ids and pred_ids:
            singleton_fp += 1
        if true_ids and not pred_ids:
            missed_entities += 1

    n = buckets["ALL"][0]
    micro_p = tp_total / pred_total if pred_total else 0.0
    micro_r = tp_total / true_total if true_total else 0.0
    micro_f = (1.25 * micro_p * micro_r) / (0.25 * micro_p + micro_r) if (micro_p + micro_r) else 0.0

    line = "=" * 80
    print(line)
    print("MACRO F0.5 EVALUATION REPORT")
    print(line)
    print(f"  Predictions file      : {pred_path}")
    print(f"  Ground truth file     : {truth_path}")
    print(f"  S1 entities scored    : {n:,}")
    print(f"  Predicted S1 rows     : {len(preds):,}  (duplicates merged: {pred_dups:,})")
    print(f"  Ground-truth S1 rows  : {len(truth):,}  (duplicates merged: {truth_dups:,})")
    if missing_pred_ids:
        print(f"  [WARN] {len(missing_pred_ids):,} S1 entities missing from predictions (scored as empty)")
    if unknown_pred_ids:
        print(f"  [WARN] {len(unknown_pred_ids):,} predicted S1 ids not in ground truth / Source 1 (ignored)")
    print(line)
    all_b = buckets["ALL"]
    print(f"  >>> MACRO F0.5        : {all_b[3] / n:.6f}")
    print(f"      Macro Precision   : {all_b[1] / n:.6f}")
    print(f"      Macro Recall      : {all_b[2] / n:.6f}")
    print(f"      Micro P / R / F0.5: {micro_p:.4f} / {micro_r:.4f} / {micro_f:.4f}")
    print(f"      Perfect entities  : {perfect:,} ({perfect / n * 100:.2f}%)")
    print(line)
    print(f"  {'Segment':<26}{'Count':>12}{'Share':>9}{'Macro P':>10}{'Macro R':>10}{'F0.5':>10}")
    for key in sorted(k for k in buckets if k != "ALL"):
        c, sp, sr, sf = buckets[key]
        print(f"  {key:<26}{c:>12,}{c / n * 100:>8.2f}%{sp / c:>10.4f}{sr / c:>10.4f}{sf / c:>10.4f}")
    print(line)
    n_single = buckets["type:singleton"][0]
    n_multi = buckets["type:non-singleton"][0]
    print("  Error breakdown")
    print(f"    Singletons with false matches : {singleton_fp:,} / {n_single:,}"
          f"  (costs {singleton_fp / n:.4f} macro F0.5)")
    print(f"    Non-singletons predicted empty: {missed_entities:,} / {n_multi:,}"
          f"  (costs {missed_entities / n:.4f} macro F0.5)")
    print(f"    True links / predicted links / correct: {true_total:,} / {pred_total:,} / {tp_total:,}")
    print(line)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Macro F0.5 evaluator for ER predictions")
    parser.add_argument("--pred", default="output/matching_results.tsv",
                        help="Predicted matching_results.tsv")
    parser.add_argument("--truth", default="dataset/train/train_ground_truth.tsv",
                        help="Ground truth TSV")
    parser.add_argument("--s1", default="dataset/train/train_source1.tsv",
                        help="Source 1 TSV (adds singletons absent from truth + per-country breakdown)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    evaluate(args.pred, args.truth, args.s1)
