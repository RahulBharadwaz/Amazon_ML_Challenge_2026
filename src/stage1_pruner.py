#!/usr/bin/env python3
"""
ML Challenge 2026 - V6 Stage-1 candidate pruner (Gate G1b).

Retrieval (v6_retrieval) returns a deep, noisy union: up to ~300 Source 1
candidates per target from three channels, ~78 on average. RRF fusion ranks
them badly because it only sees ranks, and each channel's score is missing
for candidates the channel did not retrieve.

Stage 1 re-scores every (target, candidate) pair with cheap, dense evidence:
  - exact cosines on all three representations for EVERY pair (sparse row dot
    products + one 128-d dot), not just for the channel that found it
  - per-channel retrieval rank (absent = RANK_ABSENT)
  - target-level context: gap to the target's best score and rank by each
    cosine, union size, channel agreement
  - missing-field flags on both sides
A LightGBM binary model (trained on training-fold targets only) scores pairs;
each target keeps its top-M (M <= 20), optionally cut further by a probability
floor tuned on training data.

Pairs are processed in query blocks and written to disk as float32 blocks, so
memory stays bounded regardless of candidate volume.
"""

import glob
import os
import time
from typing import Dict, List, Sequence, Tuple

import numpy as np
import scipy.sparse as sp

CHANNELS = ("name", "addr", "enc")
RANK_ABSENT = 128

FEATURES = [
    "cos_name", "cos_addr", "cos_enc",
    "rank_name", "rank_addr", "rank_enc", "n_channels",
    "gap_name", "gap_addr", "gap_enc",
    "qrank_name", "qrank_addr", "qrank_enc",
    "max_name", "max_addr", "max_enc",
    "second_enc_gap",
    "union_size",
    "q_name_empty", "q_addr_empty", "s_addr_empty",
    "cos_sum", "cos_min",
]
RAW_FEATURES = ["cos_raw", "gap_raw", "qrank_raw", "raw_exact"]
ALL_FEATURES = FEATURES + RAW_FEATURES


# ------------------------------------------------------------------------------
# Pair construction
# ------------------------------------------------------------------------------

def block_pairs(entries: Dict[str, List[Tuple[np.ndarray, np.ndarray]]], lo: int, hi: int, n_s1: int):
    """Union of channel candidates for queries in [lo, hi).

    entries[ch] is a list of (query idx, S1 idx matrix) - one row per query.
    Returns sorted unique (Q, C) and per-channel rank matrix (RANK_ABSENT if absent).
    """
    qs, cs, chs, rks = [], [], [], []
    for ch_id, ch in enumerate(CHANNELS):
        for qidx, I in entries[ch]:
            sel = (qidx >= lo) & (qidx < hi)
            if not sel.any():
                continue
            Ib = I[sel]
            k = Ib.shape[1]
            q = np.repeat(qidx[sel].astype(np.int64), k)
            c = Ib.reshape(-1).astype(np.int64)
            r = np.tile(np.arange(k, dtype=np.int16), len(Ib))
            keep = c >= 0
            qs.append(q[keep]); cs.append(c[keep]); rks.append(r[keep])
            chs.append(np.full(int(keep.sum()), ch_id, dtype=np.int8))
    if not qs:
        return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros((0, 3), np.int16)
    keys = np.concatenate(qs) * n_s1 + np.concatenate(cs)
    uniq, inv = np.unique(keys, return_inverse=True)
    ranks = np.full((len(uniq), 3), RANK_ABSENT, dtype=np.int16)
    ranks[inv, np.concatenate(chs)] = np.concatenate(rks)
    return uniq // n_s1, uniq % n_s1, ranks


def rowdot(A: sp.csr_matrix, B: sp.csr_matrix, ia: np.ndarray, ib: np.ndarray, chunk: int = 500_000):
    """Cosine of row pairs (rows are L2-normalized)."""
    out = np.empty(len(ia), dtype=np.float32)
    for s in range(0, len(ia), chunk):
        out[s:s + chunk] = np.asarray(A[ia[s:s + chunk]].multiply(B[ib[s:s + chunk]]).sum(1)).ravel()
    return out


def embdot(QE: np.ndarray, SE: np.ndarray, ia: np.ndarray, ib: np.ndarray, chunk: int = 500_000):
    out = np.empty(len(ia), dtype=np.float32)
    for s in range(0, len(ia), chunk):
        out[s:s + chunk] = np.einsum("ij,ij->i", QE[ia[s:s + chunk]].astype(np.float32),
                                     SE[ib[s:s + chunk]].astype(np.float32))
    return out


def _group_stats(Q: np.ndarray, x: np.ndarray):
    """Per-pair: group max of x, and rank of x within its query group (0 = best)."""
    starts = np.flatnonzero(np.r_[True, Q[1:] != Q[:-1]])
    sizes = np.diff(np.r_[starts, len(Q)])
    gmax = np.repeat(np.maximum.reduceat(x, starts), sizes)
    order = np.lexsort((-x, Q))
    pos = np.empty(len(Q), dtype=np.int64)
    pos[order] = np.arange(len(Q))
    rank = pos - np.repeat(starts, sizes)
    return gmax, rank.astype(np.float32), starts, sizes


def pair_features(Q, C, ranks, q_local, c_local, qn, qa, sn, sa, q_emb, s_emb,
                  q_name_empty, q_addr_empty, s_addr_empty) -> np.ndarray:
    """Feature matrix (float32, columns = FEATURES) for pairs sorted by query."""
    ql, cl = q_local, c_local
    cos = {
        "name": rowdot(qn, sn, ql, cl),
        "addr": rowdot(qa, sa, ql, cl),
        "enc": embdot(q_emb, s_emb, ql, cl),
    }
    X = np.empty((len(Q), len(FEATURES)), dtype=np.float32)
    col = {f: i for i, f in enumerate(FEATURES)}
    for ch in CHANNELS:
        X[:, col["cos_" + ch]] = cos[ch]
    for i, ch in enumerate(CHANNELS):
        X[:, col["rank_" + ch]] = ranks[:, i]
    X[:, col["n_channels"]] = (ranks < RANK_ABSENT).sum(1)
    starts = sizes = None
    for ch in CHANNELS:
        gmax, rk, starts, sizes = _group_stats(Q, cos[ch])
        X[:, col["gap_" + ch]] = cos[ch] - gmax
        X[:, col["qrank_" + ch]] = rk
        X[:, col["max_" + ch]] = gmax
    # margin between the best and second-best encoder score of the target
    enc = cos["enc"]
    order = np.lexsort((-enc, Q))
    sorted_enc = enc[order]
    first = sorted_enc[starts]
    second = np.where(sizes > 1, sorted_enc[np.minimum(starts + 1, len(Q) - 1)], 0.0)
    X[:, col["second_enc_gap"]] = np.repeat(first - second, sizes)
    X[:, col["union_size"]] = np.repeat(sizes, sizes)
    X[:, col["q_name_empty"]] = q_name_empty[ql]
    X[:, col["q_addr_empty"]] = q_addr_empty[ql]
    X[:, col["s_addr_empty"]] = s_addr_empty[cl]
    X[:, col["cos_sum"]] = cos["name"] + cos["addr"] + cos["enc"]
    X[:, col["cos_min"]] = np.minimum(np.minimum(cos["name"], cos["addr"]), cos["enc"])
    return X


def raw_extra(Q: np.ndarray, ql: np.ndarray, cl: np.ndarray, qr: sp.csr_matrix, sr: sp.csr_matrix) -> np.ndarray:
    """Surface-form name evidence columns (RAW_FEATURES) for pairs sorted by query."""
    cos = rowdot(qr, sr, ql, cl)
    gmax, rk, _, _ = _group_stats(Q, cos)
    return np.stack([cos, cos - gmax, rk, (cos > 0.999).astype(np.float32)], axis=1).astype(np.float32)


def add_raw_features(path: str, qi: np.ndarray, s_rows: np.ndarray, qr: sp.csr_matrix, sr: sp.csr_matrix):
    """Second pass over a written block: append surface-form name evidence (FEATURES -> ALL_FEATURES)."""
    with np.load(path) as z:
        X, Q, C, y = z["X"], z["Q"], z["C"], z["y"]
    if X.shape[1] == len(ALL_FEATURES):
        return
    extra = raw_extra(Q, np.searchsorted(qi, Q), np.searchsorted(s_rows, C), qr, sr)
    write_block(path, np.hstack([X, extra]), Q, C, y)


def write_block(path: str, X, Q, C, y):
    np.savez(path, X=X, Q=Q.astype(np.int32), C=C.astype(np.int32), y=y.astype(np.int8))


# ------------------------------------------------------------------------------
# Training / selection
# ------------------------------------------------------------------------------

def load_blocks(pattern: str):
    Xs, Qs, Cs, ys = [], [], [], []
    for f in sorted(glob.glob(pattern)):
        z = np.load(f)
        Xs.append(z["X"]); Qs.append(z["Q"]); Cs.append(z["C"]); ys.append(z["y"])
    return np.concatenate(Xs), np.concatenate(Qs), np.concatenate(Cs), np.concatenate(ys)


def train(pattern: str, n_threads: int, seed: int = 0, val_share: float = 0.1, log=print):
    """Binary LightGBM on training-fold pairs; early stopping on a query-level split."""
    import lightgbm as lgb
    X, Q, _, y = load_blocks(pattern)
    uq = np.unique(Q)
    rng = np.random.default_rng(seed)
    val_q = rng.choice(uq, size=int(len(uq) * val_share), replace=False)
    is_val = np.isin(Q, val_q)
    log(f"  [stage1] training pairs {len(y):,} ({y.sum():,} positive) over {len(uq):,} targets; "
        f"feature matrix {X.nbytes / 2**20:,.0f} MB")
    params = dict(objective="binary", learning_rate=0.08, num_leaves=63, min_data_in_leaf=200,
                  feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  num_threads=n_threads, seed=seed, verbose=-1, max_bin=127)
    names = ALL_FEATURES if X.shape[1] == len(ALL_FEATURES) else FEATURES
    dtr = lgb.Dataset(X[~is_val], y[~is_val], feature_name=names, free_raw_data=True)
    dva = lgb.Dataset(X[is_val], y[is_val], reference=dtr, free_raw_data=True)
    model = lgb.train(params, dtr, num_boost_round=600, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(40, verbose=False)])
    log(f"  [stage1] {model.best_iteration} trees, val logloss {model.best_score['valid_0']['binary_logloss']:.5f}")
    # validation split scores, for tuning the probability floor on training data only
    p_val = model.predict(X[is_val], num_iteration=model.best_iteration)
    val = (Q[is_val], y[is_val], p_val)
    del X, dtr, dva
    return model, val


def topm_ranks(Q: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Rank of each pair within its query by descending probability (Q sorted)."""
    starts = np.flatnonzero(np.r_[True, Q[1:] != Q[:-1]])
    sizes = np.diff(np.r_[starts, len(Q)])
    order = np.lexsort((-p, Q))
    pos = np.empty(len(Q), dtype=np.int64)
    pos[order] = np.arange(len(Q))
    return pos - np.repeat(starts, sizes)


def tune_floor(val, M: int, max_recall_loss: float) -> float:
    """Largest probability floor that loses <= max_recall_loss recall@M on the training-fold split."""
    Q, y, p = val
    rk = topm_ranks(Q, p)
    pos = y == 1
    base = (rk[pos] < M).mean()
    best = 0.0
    for f in np.concatenate([np.geomspace(1e-6, 0.05, 60)]):
        r = ((rk[pos] < M) & (p[pos] >= f)).mean()
        if base - r <= max_recall_loss:
            best = f
        else:
            break
    return float(best)


def evaluate(model, pattern: str, truth_eval: np.ndarray, is_random: np.ndarray,
             Ms: Sequence[int], floor: float, M_floor: int):
    """Recall@M over held-out links + candidates kept per random target."""
    hit_rank = np.full(len(truth_eval), -1, dtype=np.int64)     # rank of the true S1 per query
    hit_p = np.zeros(len(truth_eval), dtype=np.float32)
    kept_m = {m: 0 for m in Ms}
    kept_floor = 0
    n_rand = int(is_random.sum())
    for f in sorted(glob.glob(pattern)):
        z = np.load(f)
        X, Q, y = z["X"], z["Q"], z["y"]
        p = model.predict(X, num_iteration=model.best_iteration).astype(np.float32)
        rk = topm_ranks(Q, p)
        pos = y == 1
        hit_rank[Q[pos]] = rk[pos]
        hit_p[Q[pos]] = p[pos]
        rsel = is_random[Q]
        for m in Ms:
            kept_m[m] += int((rsel & (rk < m)).sum())
        kept_floor += int((rsel & (rk < M_floor) & (p >= floor)).sum())
    ev = truth_eval >= 0
    rec = {m: float(((hit_rank[ev] >= 0) & (hit_rank[ev] < m)).mean()) for m in Ms}
    rec_floor = float(((hit_rank[ev] >= 0) & (hit_rank[ev] < M_floor) & (hit_p[ev] >= floor)).mean())
    vol = {m: kept_m[m] / max(1, n_rand) for m in Ms}
    return rec, vol, rec_floor, kept_floor / max(1, n_rand), hit_rank, hit_p
