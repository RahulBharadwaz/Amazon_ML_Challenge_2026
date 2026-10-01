#!/usr/bin/env python3
"""Why do held-out true links fall below the stage-1 top-M? Separable errors vs exact ties."""
import argparse
import glob
import os
from collections import Counter

import sys

import lightgbm as lgb
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import stage1_pruner as s1p  # noqa: E402


def main(a):
    model = lgb.Booster(model_file=os.path.join(a.out_dir, "stage1_lgbm.txt"))
    col = {f: i for i, f in enumerate(s1p.ALL_FEATURES)}
    stats = Counter()
    tie_hist = Counter()
    examples = []
    for f in sorted(glob.glob(os.path.join(a.out_dir, "pairs", "eval_*.npz"))):
        z = np.load(f)
        X, Q, C, y = z["X"], z["Q"], z["C"], z["y"]
        p = model.predict(X)
        rk = s1p.topm_ranks(Q, p)
        starts = np.flatnonzero(np.r_[True, Q[1:] != Q[:-1]])
        ends = np.r_[starts[1:], len(Q)]
        pos = np.flatnonzero(y == 1)
        stats["true_in_union"] += len(pos)
        for i in pos[rk[pos] >= a.m]:
            g = np.searchsorted(starts, i, side="right") - 1
            s, e = starts[g], ends[g]
            xs = X[s:e]
            same = ((np.abs(xs[:, col["cos_name"]] - X[i, col["cos_name"]]) < 1e-4) &
                    (np.abs(xs[:, col["cos_addr"]] - X[i, col["cos_addr"]]) < 1e-4) &
                    (np.abs(xs[:, col["cos_raw"]] - X[i, col["cos_raw"]]) < 1e-4))
            n_tie = int(same.sum())                     # candidates with identical name+address evidence
            n_better = int((p[s:e] > p[i]).sum())
            stats["miss"] += 1
            stats["miss_q_addr_empty"] += int(X[i, col["q_addr_empty"]] > 0)
            stats["miss_name_identical"] += int(X[i, col["cos_name"]] > 0.999)
            stats["miss_tie_ge_m"] += int(n_tie >= a.m)
            stats["expected_recoverable"] += min(1.0, a.m / max(n_tie, 1)) if n_tie >= a.m else 1.0
            tie_hist[min(n_tie, 100) // 10 * 10] += 1
            if len(examples) < 25:
                top = np.argsort(-p[s:e])[:5] + s
                examples.append((int(C[i]), float(p[i]), int(rk[i]), n_tie, n_better,
                                 X[i, [col["cos_name"], col["cos_addr"], col["cos_enc"], col["q_addr_empty"]]].round(3).tolist(),
                                 [int(c) for c in C[top]]))
    names = {}
    wanted = {e[0] for e in examples} | {c for e in examples for c in e[6]}
    with open(os.path.join(a.data_dir, "train_source1.tsv"), encoding="utf-8") as fh:
        fh.readline()
        for k, line in enumerate(fh):
            if k in wanted:
                parts = line.rstrip("\n").split("\t")
                names[k] = f"{parts[1]} | {parts[2]}"
    m = stats["miss"]
    print(f"true links in union {stats['true_in_union']:,} | ranked >= {a.m}: {m:,}")
    print(f"  ceiling: an ideal ranker could still recover ~{stats['expected_recoverable']:,.0f} of these "
          f"(exact-tie groups larger than {a.m} are a coin flip)")
    for k in ("miss_q_addr_empty", "miss_name_identical", "miss_tie_ge_m"):
        print(f"  {k:24s} {stats[k]:>7,}  ({stats[k] / max(1, m):.1%})")
    print("  identical-evidence group size histogram:", sorted(tie_hist.items()))
    for c, p, r, nt, nb, feats, top in examples:
        print(f"\nTRUE {names.get(c)}  p={p:.4f} rank={r} ties={nt} better={nb} [name,addr,enc,q_addr_empty]={feats}")
        for t in top:
            print(f"    top: {names.get(t)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="results/v6/g1b_stage1_pruner")
    ap.add_argument("--data-dir", default="dataset/train")
    ap.add_argument("--m", type=int, default=20)
    main(ap.parse_args())
