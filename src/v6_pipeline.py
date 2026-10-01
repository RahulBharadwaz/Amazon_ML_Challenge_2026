#!/usr/bin/env python3
"""
ML Challenge 2026 - V6 end-to-end pipeline (retrieval -> stage-1 pruning -> stage-2 matching).

  fit         Learn every artefact from TRAINING data only (train fold of Source 1):
              transliteration, phonetics, lexicons, noise tokens, IDF, encoder, stage-1 pruner.
  candidates  Stream EVERY target of a corpus (train or test) through retrieval + stage 1,
              in bounded chunks against a resident per-country Source 1 index, and write
              only the kept pairs (top-M above the stage-1 probability floor).
  stage2      Pair features (stage-1 evidence, fuzzy string scores, address numbers,
              target- and Source 1-side competition), LightGBM trained on the train fold,
              one parent per target, acceptance threshold tuned on the tune fold, and
              macro F0.5 on the held-out fold (Gate G2).

Source 1 folds (by entity, seed 42): train 80% | tune 10% | holdout 10%.
Targets used to train the encoder or the stage-1 model are excluded from stage-2
training rows (their upstream scores are in-sample), but still compete everywhere.
"""

import argparse
import glob
import json
import os
import pickle
import sys
import time
from collections import Counter

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

import stage1_pruner as s1p
import v6_retrieval as v6
from diagnose_v6_retrieval import MemMonitor, id_codes, lookup, read_tsv, stream_tsv

COLS = ["entity_id", "business_name", "business_address", "country"]
T0 = time.time()
MEM = None


def log(msg):
    rss = f"{MEM.current():,.0f} MB (peak {MEM.peak:,.0f})" if MEM else ""
    print(f"[{time.time() - T0:7.0f}s | {rss}] {msg}", flush=True)


def keep_hash(codes: np.ndarray, frac: float) -> np.ndarray:
    """Deterministic pseudo-random subset of ids (identical across passes)."""
    h = (codes.astype(np.uint64) * np.uint64(2654435761)) % np.uint64(1 << 32)
    return h.astype(np.float64) / float(1 << 32) < frac


# ------------------------------------------------------------------------------
# Corpus loading
# ------------------------------------------------------------------------------

class Corpus:
    """Source 1 records (+ ground truth links when available) and the target-inclusion rule."""

    def __init__(self, data_dir: str, prefix: str, frac: float = 1.0, with_gt: bool = True):
        self.data_dir, self.prefix, self.frac = data_dir, prefix, frac
        tb = read_tsv(os.path.join(data_dir, f"{prefix}_source1.tsv"), COLS)
        codes = id_codes(tb["entity_id"])
        active = keep_hash(codes, frac) if frac < 1 else np.ones(len(codes), bool)
        tb = tb.filter(pa.array(active))
        self.codes = codes[active]
        self.name = tb["business_name"].to_pylist()
        self.addr = tb["business_address"].to_pylist()
        self.country = np.array(tb["country"].to_pylist(), dtype=object)
        del tb
        self.order = np.argsort(self.codes)
        self.sorted = self.codes[self.order]
        self.n = len(self.codes)
        self.link_s1 = self.link_tgt = None
        if with_gt:
            gt = read_tsv(os.path.join(data_dir, f"{prefix}_ground_truth.tsv"),
                          ["source1_entity_id", "matched_entity_ids"])
            gt_s1 = id_codes(gt["source1_entity_id"])
            lists = pc.split_pattern(gt["matched_entity_ids"], ",")
            parents = pc.list_parent_indices(lists).to_numpy()
            flat = pc.list_flatten(lists)
            nonempty = pc.greater(pc.utf8_length(flat), 0).to_numpy(zero_copy_only=False)
            link_tgt = id_codes(pc.filter(flat, pa.array(nonempty)))
            link_s1 = lookup(self.sorted, self.order, gt_s1[parents[nonempty]])
            ok = link_s1 >= 0
            self.link_s1, self.link_tgt = link_s1[ok], link_tgt[ok]
            o = np.argsort(self.link_tgt)
            self.gt_tgt_sorted, self.gt_s1_by_tgt = self.link_tgt[o], self.link_s1[o]

    def true_s1(self, tgt_codes: np.ndarray) -> np.ndarray:
        """Ground-truth Source 1 row of each target code (-1 = no match)."""
        pos = np.clip(np.searchsorted(self.gt_tgt_sorted, tgt_codes), 0, len(self.gt_tgt_sorted) - 1)
        return np.where(self.gt_tgt_sorted[pos] == tgt_codes, self.gt_s1_by_tgt[pos], -1)

    def include_targets(self, codes: np.ndarray) -> np.ndarray:
        """Full corpus when frac == 1; otherwise targets of active S1 plus a hashed share of the rest."""
        if self.frac >= 1:
            return np.ones(len(codes), bool)
        linked = self.true_s1(codes) >= 0
        return linked | keep_hash(codes, self.frac)

    def stream_targets(self, country=None, wanted: np.ndarray = None, batch_rows: int = 50_000,
                       slice_k: int = 0, n_slices: int = 1):
        """Yield (codes, names, addrs, countries) chunks of target records, optionally filtered."""
        buf = [[], [], [], []]
        n_buf = 0
        for fn in (f"{self.prefix}_source2.tsv", f"{self.prefix}_source3.tsv"):
            for batch in stream_tsv(os.path.join(self.data_dir, fn), COLS):
                codes = id_codes(batch.column(0))
                m = self.include_targets(codes)
                if country is not None:
                    m &= pc.equal(batch.column(3), country).to_numpy(zero_copy_only=False)
                if wanted is not None:
                    m &= np.isin(codes, wanted)
                if n_slices > 1:
                    h = (codes.astype(np.uint64) * np.uint64(0x9E3779B1)) % np.uint64(1 << 31)
                    m &= (h % np.uint64(n_slices)).astype(np.int64) == slice_k
                if not m.any():
                    continue
                sub = batch.filter(pa.array(m))
                buf[0].append(codes[m])
                buf[1] += sub.column(1).to_pylist()
                buf[2] += sub.column(2).to_pylist()
                buf[3] += sub.column(3).to_pylist()
                n_buf += int(m.sum())
                while n_buf >= batch_rows:
                    codes_all = np.concatenate(buf[0])
                    yield codes_all[:batch_rows], buf[1][:batch_rows], buf[2][:batch_rows], buf[3][:batch_rows]
                    buf = [[codes_all[batch_rows:]], buf[1][batch_rows:], buf[2][batch_rows:], buf[3][batch_rows:]]
                    n_buf -= batch_rows
        if n_buf:
            yield np.concatenate(buf[0]), buf[1], buf[2], buf[3]


def s1_folds(n: int, seed: int) -> np.ndarray:
    """0 = train (80%), 1 = tune (10%), 2 = holdout (10%)."""
    u = np.random.default_rng(seed).random(n)
    return np.where(u < 0.1, 2, np.where(u < 0.2, 1, 0)).astype(np.int8)


# ------------------------------------------------------------------------------
# Per-country Source 1 index + chunk scoring (shared by fit and candidates)
# ------------------------------------------------------------------------------

class Artifacts:
    def __init__(self, canon, idf_name, idf_addr, idf_raw, model, cfg, stage1=None):
        self.canon, self.idf_name, self.idf_addr, self.idf_raw = canon, idf_name, idf_addr, idf_raw
        self.model, self.cfg, self.stage1 = model, cfg, stage1

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        self.canon._cache = {}
        with open(os.path.join(path, "text.pkl"), "wb") as f:
            pickle.dump({"canon": self.canon, "idf_name": self.idf_name, "idf_addr": self.idf_addr,
                         "idf_raw": self.idf_raw}, f)
        import torch
        torch.save(self.model.state_dict(), os.path.join(path, "encoder.pt"))
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(self.cfg, f, indent=2)
        if self.stage1 is not None:
            self.stage1.save_model(os.path.join(path, "stage1_lgbm.txt"), num_iteration=self.stage1.best_iteration)

    @classmethod
    def load(cls, path, n_threads):
        import lightgbm as lgb
        import torch
        torch.set_num_threads(n_threads)
        with open(os.path.join(path, "text.pkl"), "rb") as f:
            t = pickle.load(f)
        model = v6.build_encoder()
        model.load_state_dict(torch.load(os.path.join(path, "encoder.pt")))
        model.eval()
        with open(os.path.join(path, "config.json")) as f:
            cfg = json.load(f)
        s1 = os.path.join(path, "stage1_lgbm.txt")
        stage1 = lgb.Booster(model_file=s1) if os.path.exists(s1) else None
        return cls(t["canon"], t["idf_name"], t["idf_addr"], t["idf_raw"], model, cfg, stage1)


class CountryIndex:
    """Resident Source 1 structures of one country: sparse name/address indices, HNSW-SQ8 over
    encoder embeddings, and the normalized matrices needed for exact stage-1 cosines."""

    def __init__(self, art: Artifacts, corpus: Corpus, s_rows: np.ndarray, procs: int, seed: int):
        names = [corpus.name[i] for i in s_rows]
        addrs = [corpus.addr[i] for i in s_rows]
        sn, sa = v6.featurize(art.canon, names, addrs, procs)
        self.sn, self.sa = art.idf_name.transform(sn), art.idf_addr.transform(sa)
        self.sr = art.idf_raw.transform(v6.featurize_raw(art.canon, names, procs))
        del names, addrs
        self.s_emb = v6.encode(art.model, v6.encoder_inputs(self.sn, self.sa))
        self.name_idx = v6.SparseIndex(self.sn, art.idf_name.prune, procs)
        self.addr_idx = v6.SparseIndex(self.sa, art.idf_addr.prune, procs)
        self.hnsw = v6.HnswIndex(self.s_emb, procs, seed=seed)
        self.s_rows = s_rows
        self.s_addr_empty = np.diff(self.sa.indptr) == 0
        v6.trim_memory()


def score_chunk(art: Artifacts, ci: CountryIndex, names, addrs, n_s1: int, procs: int):
    """Retrieval + stage-1 features for a chunk of targets. Returns (Q local, C global S1 row, X)."""
    cfg = art.cfg
    K, KD, tie = cfg["k"], cfg["k_deep"], cfg["tie_ratio"]
    qn, qa = v6.featurize(art.canon, names, addrs, procs)
    qn, qa = art.idf_name.transform(qn), art.idf_addr.transform(qa)
    qr = art.idf_raw.transform(v6.featurize_raw(art.canon, names, procs))
    q_emb = v6.encode(art.model, v6.encoder_inputs(qn, qa))
    qn_empty, qa_empty = np.diff(qn.indptr) == 0, np.diff(qa.indptr) == 0
    n = len(names)
    qi = np.arange(n)
    s_rows = ci.s_rows
    E = {c: [] for c in s1p.CHANNELS}

    def to_global(I):
        return np.where(I >= 0, s_rows[np.maximum(I, 0)], -1).astype(np.int32)

    def record(ch, I, deep, Id):
        E[ch].append((qi[~deep], to_global(I[~deep])))
        if deep.any():
            E[ch].append((qi[deep], to_global(Id)))

    I, D = ci.name_idx.search(qn, K)
    deep_n = qa_empty | v6.unresolved(D, tie)
    record("name", I, deep_n, ci.name_idx.search(qn[np.flatnonzero(deep_n)], KD)[0] if deep_n.any() else None)
    I, D = ci.addr_idx.search(qa, K)
    deep_a = qn_empty | v6.unresolved(D, tie)
    record("addr", I, deep_a, ci.addr_idx.search(qa[np.flatnonzero(deep_a)], KD)[0] if deep_a.any() else None)
    deep_e = deep_n | deep_a
    I, _ = ci.hnsw.search(q_emb, K)
    record("enc", I, deep_e, ci.hnsw.search(q_emb[deep_e], KD)[0] if deep_e.any() else None)

    Q, C, R = s1p.block_pairs(E, 0, n, n_s1)
    cl = np.searchsorted(s_rows, C)
    X = s1p.pair_features(Q, C, R, Q, cl, qn, qa, ci.sn, ci.sa, q_emb, ci.s_emb,
                          qn_empty, qa_empty, ci.s_addr_empty)
    X = np.hstack([X, s1p.raw_extra(Q, Q, cl, qr, ci.sr)])
    return Q, C, X


# ------------------------------------------------------------------------------
# fit
# ------------------------------------------------------------------------------

def cmd_fit(a):
    rng = np.random.default_rng(a.seed)
    ws = a.work_dir
    os.makedirs(ws, exist_ok=True)
    corpus = Corpus(a.data_dir, "train", a.frac)
    fold = s1_folds(corpus.n, a.seed)
    np.savez(os.path.join(ws, "s1_meta.npz"), codes=corpus.codes, fold=fold)
    log(f"Source 1 {corpus.n:,} | folds train {np.sum(fold == 0):,} tune {np.sum(fold == 1):,} "
        f"holdout {np.sum(fold == 2):,} | links {len(corpus.link_s1):,}")

    train_links = np.flatnonzero(fold[corpus.link_s1] == 0)
    enc_idx = rng.choice(train_links, size=min(a.n_pairs, len(train_links)), replace=False)
    rest = np.setdiff1d(train_links, enc_idx, assume_unique=True)
    s1t_idx = rng.choice(rest, size=min(a.n_stage1_train, len(rest)), replace=False)
    biased = np.unique(np.concatenate([corpus.link_tgt[enc_idx], corpus.link_tgt[s1t_idx]]))
    np.save(os.path.join(ws, "biased_targets.npy"), biased)

    # target texts for the learning samples
    t_codes, t_name, t_addr, t_country = [], [], [], []
    for codes, names, addrs, ctry in corpus.stream_targets(wanted=biased, batch_rows=200_000):
        t_codes.append(codes); t_name += names; t_addr += addrs; t_country += ctry
    t_codes = np.concatenate(t_codes)
    t_country = np.array(t_country, dtype=object)
    t_order = np.argsort(t_codes)
    t_sorted = t_codes[t_order]
    log(f"Loaded {len(t_codes):,} learning targets (encoder pairs {len(enc_idx):,}, stage-1 {len(s1t_idx):,})")

    # --- text artefacts (train fold only) ---
    p_s1 = corpus.link_s1[enc_idx]
    p_t = lookup(t_sorted, t_order, corpus.link_tgt[enc_idx])
    ok = p_t >= 0
    p_s1, p_t = p_s1[ok], p_t[ok]
    word_pairs = v6.collect_translit_pairs((corpus.name[i], t_name[j]) for i, j in zip(p_s1, p_t))
    theta, word_dict, unit_mass = v6.learn_transliteration(word_pairs, iters=a.em_iters, log=log)
    vowels, letter_class, drop_h = v6.learn_phonetics(theta, unit_mass)
    canon = v6.Canonicalizer(theta, word_dict, vowels, letter_class, drop_h)
    name_pairs = [(canon.tokens(corpus.name[i], {}), canon.tokens(t_name[j], {})) for i, j in zip(p_s1, p_t)]
    addr_pairs = [(canon.tokens(corpus.addr[i], {}), canon.tokens(t_addr[j], {})) for i, j in zip(p_s1, p_t)]
    canon.lex_name = v6.learn_lexicon(name_pairs)
    canon.lex_addr = v6.learn_lexicon(addr_pairs)
    name_pairs = [([canon.lex_name.get(w, w) for w in s], [canon.lex_name.get(w, w) for w in t])
                  for s, t in name_pairs]
    canon.noise = v6.learn_noise_tokens(name_pairs)
    del name_pairs, addr_pairs, word_pairs
    log(f"  text: {len(word_dict)} translit words | lexicon name {len(canon.lex_name)} addr "
        f"{len(canon.lex_addr)} | noise {len(canon.noise)}")

    train_rows = np.flatnonzero(fold == 0)
    idf_rows = rng.choice(train_rows, size=min(a.n_idf, len(train_rows)), replace=False)
    rn, ra = v6.featurize(canon, [corpus.name[i] for i in idf_rows], [corpus.addr[i] for i in idf_rows], a.procs)
    idf_name, idf_addr = v6.IdfModel(rn, a.max_df_frac), v6.IdfModel(ra, a.max_df_frac)
    idf_raw = v6.IdfModel(v6.featurize_raw(canon, [corpus.name[i] for i in idf_rows], a.procs), a.max_df_frac)
    del rn, ra

    # --- encoder ---
    rn, ra = v6.featurize(canon, [corpus.name[i] for i in p_s1], [corpus.addr[i] for i in p_s1], a.procs)
    s_in = v6.encoder_inputs(idf_name.transform(rn), idf_addr.transform(ra))
    rn, ra = v6.featurize(canon, [t_name[j] for j in p_t], [t_addr[j] for j in p_t], a.procs)
    t_in = v6.encoder_inputs(idf_name.transform(rn), idf_addr.transform(ra))
    del rn, ra
    grp = np.unique(corpus.country[p_s1], return_inverse=True)[1]
    model = v6.train_encoder(s_in, t_in, grp, epochs=a.epochs, n_threads=a.procs, seed=a.seed, log=log)
    del s_in, t_in
    v6.trim_memory()
    cfg = {"k": a.k, "k_deep": a.k_deep, "tie_ratio": a.tie_ratio, "m_max": a.m_max, "seed": a.seed}
    art = Artifacts(canon, idf_name, idf_addr, idf_raw, model, cfg)

    # --- stage-1 pruner on held-apart train-fold targets ---
    s1_dir = os.path.join(ws, "stage1_train")
    os.makedirs(s1_dir, exist_ok=True)
    for f in glob.glob(os.path.join(s1_dir, "*.npz")):
        os.remove(f)
    s1t_codes = corpus.link_tgt[s1t_idx]
    s1t_rows = lookup(t_sorted, t_order, s1t_codes)
    s1t_true = corpus.link_s1[s1t_idx]
    q_base = 0
    for ctry in sorted(set(corpus.country.tolist())):
        s_rows = np.flatnonzero(corpus.country == ctry)
        sel = (s1t_rows >= 0) & (t_country[np.maximum(s1t_rows, 0)] == ctry)
        if not sel.any():
            continue
        ci = CountryIndex(art, corpus, s_rows, a.procs, a.seed)
        rows, truth = s1t_rows[sel], s1t_true[sel]
        for b, s in enumerate(range(0, len(rows), a.chunk)):
            r, tr = rows[s:s + a.chunk], truth[s:s + a.chunk]
            Q, C, X = score_chunk(art, ci, [t_name[i] for i in r], [t_addr[i] for i in r], corpus.n, a.procs)
            s1p.write_block(os.path.join(s1_dir, f"{ctry}_{b:04d}.npz"), X, Q + q_base + s, C, tr[Q] == C)
        q_base += len(rows)
        del ci
        v6.trim_memory()
        log(f"  stage-1 training pairs written for {ctry} ({sel.sum():,} targets)")
    stage1, val = s1p.train(os.path.join(s1_dir, "*.npz"), a.procs, seed=a.seed, log=log)
    cfg["floor"] = s1p.tune_floor(val, a.m_max, a.floor_loss)
    art.stage1 = stage1
    art.save(os.path.join(ws, "artifacts"))
    log(f"fit done: stage-1 {stage1.best_iteration} trees, probability floor {cfg['floor']:.2e}")


# ------------------------------------------------------------------------------
# candidates
# ------------------------------------------------------------------------------

def cmd_candidates(a):
    ws = a.work_dir
    art = Artifacts.load(os.path.join(ws, "artifacts"), a.procs)
    corpus = Corpus(a.data_dir, a.prefix, a.frac, with_gt=(a.prefix == "train"))
    out = os.path.join(ws, f"cand_{a.prefix}")
    os.makedirs(out, exist_ok=True)
    tag = f"s{a.slice}"
    countries = [a.country] if a.country else sorted(set(corpus.country.tolist()))
    for ctry in countries:
        for f in glob.glob(os.path.join(out, f"{ctry}_{tag}_*")):
            os.remove(f)
    M, floor = art.cfg["m_max"], art.cfg["floor"]
    stats = Counter()
    for ctry in countries:
        t = time.time()
        s_rows = np.flatnonzero(corpus.country == ctry)
        ci = CountryIndex(art, corpus, s_rows, a.procs, art.cfg["seed"])
        log(f"[{ctry}] index over {len(s_rows):,} Source 1 records built in {time.time() - t:.0f}s")
        for b, (codes, names, addrs, _) in enumerate(corpus.stream_targets(
                country=ctry, batch_rows=a.chunk, slice_k=a.slice, n_slices=a.n_slices)):
            Q, C, X = score_chunk(art, ci, names, addrs, corpus.n, a.procs)
            p1 = art.stage1.predict(X).astype(np.float32)
            rk = s1p.topm_ranks(Q, p1)
            keep = (rk < M) & (p1 >= floor)
            np.savez(os.path.join(out, f"{ctry}_{tag}_{b:05d}.npz"), tgt=codes[Q[keep]], s1=C[keep].astype(np.int32),
                     p1=p1[keep], X1=X[keep])
            tq = np.unique(Q[keep])
            pq.write_table(pa.table({"code": codes[tq], "name": [names[i] for i in tq],
                                     "addr": [addrs[i] for i in tq]}),
                           os.path.join(out, f"{ctry}_{tag}_{b:05d}_t.parquet"))
            stats["targets"] += len(codes)
            stats["pairs_scored"] += len(Q)
            stats["pairs_kept"] += int(keep.sum())
            if corpus.link_s1 is not None:
                tru = corpus.true_s1(codes)
                stats["linked_targets"] += int((tru >= 0).sum())
                stats["linked_kept"] += int((tru[Q[keep]] == C[keep]).sum())
            if b % 20 == 0:
                log(f"[{ctry}] chunk {b}: {stats['targets']:,} targets | kept {stats['pairs_kept']:,} of "
                    f"{stats['pairs_scored']:,} pairs")
        del ci
        v6.trim_memory()
    stats["runtime_s"] = time.time() - T0
    stats["peak_mem_mb"] = MEM.peak
    if stats["linked_targets"]:
        stats["candidate_recall"] = stats["linked_kept"] / stats["linked_targets"]
    with open(os.path.join(ws, f"cand_{a.prefix}_{a.country or 'all'}_{tag}_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    log(f"candidates done: {json.dumps(stats)}")


# ------------------------------------------------------------------------------
# stage 2
# ------------------------------------------------------------------------------

S2_EXTRA = ["p1", "p1_rank_t", "p1_margin_t", "n_kept_t",
            "s1_n_cand", "s1_n_top1", "s1_sum_p1", "s1_max_p1", "p1_gap_s1", "rank_in_s1",
            "fz_ratio", "fz_tsort", "fz_tset", "fz_partial", "jw", "core_ratio", "core_tset",
            "core_equal", "raw_equal", "addr_tset", "addr_partial", "addr_canon_tset",
            "t_addr_empty", "num_shared", "num_conflict", "num_fuzzy", "num_t", "num_s",
            "len_t", "len_s"]
S2_FEATURES = s1p.ALL_FEATURES + S2_EXTRA


def _numbers(tokens):
    nums = set()
    for t in tokens:
        if not t.isalpha():
            for run in "".join(ch if ch.isdigit() else " " for ch in t).split():
                nums.add(run.lstrip("0") or "0")
    return nums


def _deletes(nums):
    out = set(nums)
    for n in nums:
        if len(n) >= 3:
            out.update(n[:j] + n[j + 1:] for j in range(len(n)))
    return out


def _record_view(canon, name, addr):
    """Strings and number sets used by stage-2 string features."""
    name, addr = name or "", addr or ""
    atoks = canon.addr_tokens(addr)
    nums = _numbers(atoks)
    return (v6.fold(name), " ".join(canon.name_core(name)), v6.fold(addr), " ".join(atoks), nums, _deletes(nums))


def group_rank(keys: np.ndarray, vals: np.ndarray):
    """Per element: rank (0 = best) of vals within its key group, group max, group size."""
    order = np.lexsort((-vals, keys))
    ks = keys[order]
    starts = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1]])
    sizes = np.diff(np.r_[starts, len(ks)])
    rank = np.empty(len(keys), np.int64)
    rank[order] = np.arange(len(keys)) - np.repeat(starts, sizes)
    gmax = np.empty(len(keys), vals.dtype)
    gmax[order] = np.repeat(vals[order][starts], sizes)
    return rank, gmax


def s2_features(canon, corpus, cdir, fdir, procs, pattern="*.npz"):
    """Stage-2 feature matrices (one .npy per candidate chunk). Returns tgt, s1, p1, offsets, files."""
    from rapidfuzz import fuzz, process
    from rapidfuzz.distance import JaroWinkler
    files = sorted(glob.glob(os.path.join(cdir, pattern)))

    # --- pass A: global competition statistics ---
    tgt, s1, p1, sizes = [], [], [], []
    for f in files:
        with np.load(f) as z:
            tgt.append(z["tgt"]); s1.append(z["s1"]); p1.append(z["p1"]); sizes.append(len(z["p1"]))
    tgt, s1, p1 = np.concatenate(tgt), np.concatenate(s1).astype(np.int64), np.concatenate(p1)
    offsets = np.r_[0, np.cumsum(sizes)]
    t_rank, t_max = group_rank(tgt, p1)
    inv = np.unique(tgt, return_inverse=True)[1]
    t_n = np.bincount(inv)[inv].astype(np.float32)
    del inv
    # margin: best candidate vs second best of its target, others vs best
    order = np.lexsort((-p1, tgt))
    ts = tgt[order]
    starts = np.flatnonzero(np.r_[True, ts[1:] != ts[:-1]])
    gsz = np.diff(np.r_[starts, len(ts)])
    p_sorted = p1[order]
    sec = np.where(gsz > 1, p_sorted[np.minimum(starts + 1, len(ts) - 1)], 0.0)
    sec_full = np.empty(len(p1), np.float32)
    sec_full[order] = np.repeat(sec, gsz)
    t_margin = np.where(t_rank == 0, p1 - sec_full, p1 - t_max).astype(np.float32)
    del order, ts, p_sorted, sec_full
    s_rank, s_max = group_rank(s1, p1)
    s_ncand = np.bincount(s1, minlength=corpus.n)
    s_top1 = np.bincount(s1[t_rank == 0], minlength=corpus.n)
    s_sum = np.bincount(s1, weights=p1, minlength=corpus.n)
    log(f"stage-2 pairs {len(p1):,} over {len(files)} chunks | targets {len(np.unique(tgt)):,}")

    # --- pass B: features per chunk ---
    os.makedirs(fdir, exist_ok=True)
    t = time.time()
    for k, f in enumerate(files):
        lo, hi = offsets[k], offsets[k + 1]
        if hi == lo:
            continue
        with np.load(f) as z:
            X1 = z["X1"]
        tb = pq.read_table(f[:-4] + "_t.parquet")
        tcode = tb["code"].to_numpy()
        tview = {c: _record_view(canon, n, ad) for c, n, ad in
                 zip(tcode, tb["name"].to_pylist(), tb["addr"].to_pylist())}
        cs = s1[lo:hi]
        sview = {c: _record_view(canon, corpus.name[c], corpus.addr[c]) for c in np.unique(cs)}
        tv = [tview[c] for c in tgt[lo:hi]]
        sv = [sview[c] for c in cs]
        w = procs

        def cp(scorer, i):
            return process.cpdist([x[i] for x in tv], [x[i] for x in sv], scorer=scorer, workers=w).astype(np.float32)

        n_t = np.array([len(x[4]) for x in tv], np.float32)
        n_s = np.array([len(x[4]) for x in sv], np.float32)
        shared = np.array([len(x[4] & y[4]) for x, y in zip(tv, sv)], np.float32)
        fuzzy = np.array([bool(x[5] & y[5]) for x, y in zip(tv, sv)], np.float32)
        extra = np.column_stack([
            p1[lo:hi], t_rank[lo:hi], t_margin[lo:hi], t_n[lo:hi],
            s_ncand[cs], s_top1[cs], s_sum[cs], s_max[lo:hi], p1[lo:hi] - s_max[lo:hi], s_rank[lo:hi],
            cp(fuzz.ratio, 0), cp(fuzz.token_sort_ratio, 0), cp(fuzz.token_set_ratio, 0),
            cp(fuzz.partial_ratio, 0), cp(JaroWinkler.normalized_similarity, 0),
            cp(fuzz.ratio, 1), cp(fuzz.token_set_ratio, 1),
            np.array([x[1] == y[1] for x, y in zip(tv, sv)], np.float32),
            np.array([x[0] == y[0] for x, y in zip(tv, sv)], np.float32),
            cp(fuzz.token_set_ratio, 2), cp(fuzz.partial_ratio, 2), cp(fuzz.token_set_ratio, 3),
            np.array([not x[2].strip() for x in tv], np.float32),
            shared, ((n_t > 0) & (n_s > 0) & (shared == 0)).astype(np.float32), fuzzy, n_t, n_s,
            np.array([len(x[0]) for x in tv], np.float32), np.array([len(x[0]) for x in sv], np.float32),
        ]).astype(np.float32)
        np.save(os.path.join(fdir, f"{k:05d}.npy"), np.hstack([X1, extra]))
        if k % 25 == 0:
            log(f"  stage-2 features: chunk {k + 1}/{len(files)} ({time.time() - t:.0f}s)")
    return tgt, s1, p1, offsets, files


def cmd_stage2(a):
    import lightgbm as lgb
    ws = a.work_dir
    art = Artifacts.load(os.path.join(ws, "artifacts"), a.procs)
    corpus = Corpus(a.data_dir, "train", a.frac)
    meta = np.load(os.path.join(ws, "s1_meta.npz"))
    assert np.array_equal(meta["codes"], corpus.codes), "Source 1 order differs from fit"
    fold = meta["fold"]
    biased = np.load(os.path.join(ws, "biased_targets.npy"))
    fdir = os.path.join(ws, "s2_feats")
    tgt, s1, p1, offsets, files = s2_features(art.canon, corpus, os.path.join(ws, "cand_train"), fdir, a.procs)
    y_all = corpus.true_s1(tgt) == s1

    # --- training rows: train fold, unbiased targets; tune fold for early stopping ---
    pair_fold = fold[s1]
    is_biased = np.isin(tgt, biased)
    tr_idx = np.flatnonzero((pair_fold == 0) & ~is_biased)
    rng = np.random.default_rng(a.seed)
    if len(tr_idx) > a.max_train_rows:
        tr_idx = np.sort(rng.choice(tr_idx, a.max_train_rows, replace=False))
    va_idx = np.flatnonzero(pair_fold == 1)

    def gather(idx):
        out = np.empty((len(idx), len(S2_FEATURES)), np.float32)
        pos = 0
        for k in range(len(files)):
            lo, hi = offsets[k], offsets[k + 1]
            sel = idx[(idx >= lo) & (idx < hi)]
            if len(sel):
                out[pos:pos + len(sel)] = np.load(os.path.join(fdir, f"{k:05d}.npy"), mmap_mode="r")[sel - lo]
                pos += len(sel)
        return out

    Xtr, Xva = gather(tr_idx), gather(va_idx)
    log(f"stage-2 training rows {len(tr_idx):,} ({y_all[tr_idx].sum():,} pos) | tune rows {len(va_idx):,}")
    params = dict(objective="binary", learning_rate=0.05, num_leaves=127, min_data_in_leaf=100,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  num_threads=a.procs, seed=a.seed, verbose=-1)
    dtr = lgb.Dataset(Xtr, y_all[tr_idx], feature_name=S2_FEATURES, free_raw_data=True)
    dva = lgb.Dataset(Xva, y_all[va_idx], reference=dtr, free_raw_data=True)
    booster = lgb.train(params, dtr, num_boost_round=2000, valid_sets=[dva],
                        callbacks=[lgb.early_stopping(60, verbose=False)])
    del Xtr, Xva, dtr, dva
    booster.save_model(os.path.join(ws, "stage2_lgbm.txt"), num_iteration=booster.best_iteration)
    log(f"stage-2 model: {booster.best_iteration} trees, tune logloss "
        f"{booster.best_score['valid_0']['binary_logloss']:.5f}")

    # --- predict all pairs ---
    p2 = np.empty(len(p1), np.float32)
    for k in range(len(files)):
        lo, hi = offsets[k], offsets[k + 1]
        if hi > lo:
            p2[lo:hi] = booster.predict(np.load(os.path.join(fdir, f"{k:05d}.npy")),
                                        num_iteration=booster.best_iteration)

    # --- decision: one parent per target, threshold tuned on the tune fold ---
    rank2, _ = group_rank(tgt, p2)
    best = rank2 == 0
    b_tgt, b_s1, b_p = tgt[best], s1[best], p2[best]
    b_ok = corpus.true_s1(b_tgt) == b_s1
    n_true = np.bincount(corpus.link_s1, minlength=corpus.n)

    def macro(tau, which):
        acc = b_p >= tau
        n_pred = np.bincount(b_s1[acc], minlength=corpus.n)
        tp = np.bincount(b_s1[acc & b_ok], minlength=corpus.n)
        rows = fold == which
        nt, npd, t_p = n_true[rows], n_pred[rows], tp[rows]
        P = np.where(npd > 0, t_p / np.maximum(npd, 1), 0.0)
        R = np.where(nt > 0, t_p / np.maximum(nt, 1), 0.0)
        F = np.where(nt == 0, (npd == 0).astype(float),
                     np.where(t_p > 0, 1.25 * P * R / np.maximum(0.25 * P + R, 1e-12), 0.0))
        return F, P, R, nt, npd

    grid = np.round(np.arange(0.05, 0.96, 0.01), 2)
    tune_scores = [macro(tau, 1)[0].mean() for tau in grid]
    tau = float(grid[int(np.argmax(tune_scores))])
    F, P, R, nt, npd = macro(tau, 2)
    ho = fold == 2
    ctry = corpus.country[ho]
    single = nt == 0

    # retrieval + pruning ceiling on the same holdout entities (perfect decisions)
    cand_true = np.bincount(s1[y_all], minlength=corpus.n)[ho]
    r_ceiling = np.where(nt > 0, cand_true / np.maximum(nt, 1), 1.0)
    ceiling = np.where(nt > 0, np.where(r_ceiling > 0, 1.25 * r_ceiling / (0.25 + r_ceiling), 0.0), 1.0).mean()

    summary = {
        "tau": tau, "tune_macro_f05": float(max(tune_scores)),
        "holdout_macro_f05": float(F.mean()),
        "holdout_entities": int(ho.sum()),
        "singletons": {"n": int(single.sum()), "correct_empty": float((npd[single] == 0).mean())},
        "non_singletons": {"n": int((~single).sum()), "macro_f05": float(F[~single].mean()),
                           "precision": float(P[~single].mean()), "recall": float(R[~single].mean()),
                           "predicted_empty": float((npd[~single] == 0).mean())},
        "by_country": {c: float(F[ctry == c].mean()) for c in sorted(set(ctry.tolist()))},
        "ceiling_macro_f05_perfect_decisions": float(ceiling),
        "feature_gain_top15": dict(sorted(zip(S2_FEATURES, booster.feature_importance("gain").round().astype(int).tolist()),
                                          key=lambda kv: -kv[1])[:15]),
        "runtime_s": time.time() - T0, "peak_mem_mb": MEM.peak,
    }
    with open(os.path.join(ws, "g2_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(ws, "stage2_meta.json"), "w") as f:
        json.dump({"tau": tau}, f)
    print("=" * 88)
    print("  GATE G2 - V6 stage-2 matching, macro F0.5 on held-out Source 1 entities")
    print("=" * 88)
    print(f"  Held-out entities {summary['holdout_entities']:,} | threshold tau {tau} (tuned on tune fold, "
          f"tune F0.5 {summary['tune_macro_f05']:.4f})")
    print(f"  HOLDOUT MACRO F0.5                      : {summary['holdout_macro_f05']:.4f}")
    print(f"  Non-singletons: F0.5 {summary['non_singletons']['macro_f05']:.4f} | P {summary['non_singletons']['precision']:.4f} "
          f"| R {summary['non_singletons']['recall']:.4f} | predicted empty {summary['non_singletons']['predicted_empty']:.2%}")
    print(f"  Singletons: {summary['singletons']['n']:,} | correctly empty {summary['singletons']['correct_empty']:.4f}")
    print("  By country: " + " | ".join(f"{c} {v:.4f}" for c, v in summary["by_country"].items()))
    print(f"  Ceiling with perfect decisions on these candidates: {summary['ceiling_macro_f05_perfect_decisions']:.4f}")
    print(f"  Top features: {list(summary['feature_gain_top15'])[:10]}")
    print(f"  Runtime {summary['runtime_s']:.0f}s | peak process-tree memory {MEM.peak:,.0f} MB")
    print("=" * 88)


def cmd_prep(a):
    """Folds + empty biased list for a (sampled) training corpus whose artefacts come from elsewhere."""
    ws = a.work_dir
    os.makedirs(ws, exist_ok=True)
    corpus = Corpus(a.data_dir, "train", a.frac)
    np.savez(os.path.join(ws, "s1_meta.npz"), codes=corpus.codes, fold=s1_folds(corpus.n, a.seed))
    np.save(os.path.join(ws, "biased_targets.npy"), np.zeros(0, np.int64))
    log(f"prep: {corpus.n:,} Source 1 records (frac {a.frac})")


def code_to_id(c) -> str:
    return f"S{int(c) // 10**10}-{int(c) % 10**10}"


def cmd_predict(a):
    """Stage 2 on the test candidates -> matching_results.tsv + candidate_pairs.tsv."""
    import lightgbm as lgb
    ws = a.work_dir
    art = Artifacts.load(os.path.join(ws, "artifacts"), a.procs)
    corpus = Corpus(a.data_dir, "test", 1.0, with_gt=False)
    cdir = os.path.join(ws, "cand_test")
    pattern = f"{a.country}_*.npz" if a.country else "*.npz"
    fdir = os.path.join(ws, f"s2_feats_test_{a.country or 'all'}")
    if a.stage1_only:
        # fallback: stage-1 probability as the match score
        files = sorted(glob.glob(os.path.join(cdir, pattern)))
        tgt, s1, p1 = [], [], []
        for f in files:
            with np.load(f) as z:
                tgt.append(z["tgt"]); s1.append(z["s1"]); p1.append(z["p1"])
        tgt, s1, p1 = np.concatenate(tgt), np.concatenate(s1).astype(np.int64), np.concatenate(p1)
        p2 = p1
        tau = 0.5 if a.tau is None else a.tau
    else:
        booster = lgb.Booster(model_file=os.path.join(ws, "stage2_lgbm.txt"))
        tau = json.load(open(os.path.join(ws, "stage2_meta.json")))["tau"] if a.tau is None else a.tau
        tgt, s1, p1, offsets, files = s2_features(art.canon, corpus, cdir, fdir, a.procs, pattern)
        p2 = np.empty(len(p1), np.float32)
        for k in range(len(files)):
            lo, hi = offsets[k], offsets[k + 1]
            if hi > lo:
                p2[lo:hi] = booster.predict(np.load(os.path.join(fdir, f"{k:05d}.npy")))
    rank2, _ = group_rank(tgt, p2)
    acc = (rank2 == 0) & (p2 >= tau)
    log(f"predict: {len(p1):,} candidate pairs | accepted {acc.sum():,} links (tau {tau}) | "
        f"S1 with matches {len(np.unique(s1[acc])):,} / {corpus.n:,}")

    def grouped(mask):
        o = np.lexsort((-p2[mask], s1[mask]))
        ss, tt = s1[mask][o], tgt[mask][o]
        out = {}
        starts = np.flatnonzero(np.r_[True, ss[1:] != ss[:-1]]) if len(ss) else []
        ends = np.r_[starts[1:], len(ss)] if len(ss) else []
        for st, en in zip(starts, ends):
            out[int(ss[st])] = ",".join(code_to_id(c) for c in tt[st:en])
        return out

    matches, cands = grouped(acc), grouped(np.ones(len(p1), bool))
    od = a.output_dir
    os.makedirs(od, exist_ok=True)
    sfx = f"_{a.country}" if a.country else ""
    with open(os.path.join(od, f"matching_results{sfx}.tsv"), "w", encoding="utf-8", newline="\n") as fm, \
            open(os.path.join(od, f"candidate_pairs{sfx}.tsv"), "w", encoding="utf-8", newline="\n") as fc:
        fm.write("source1_entity_id\tmatched_entity_ids\n")
        fc.write("source1_entity_id\tcandidate_entity_ids\n")
        for i, c in enumerate(corpus.codes):
            if a.country and corpus.country[i] != a.country:
                continue
            sid = code_to_id(c)
            fm.write(f"{sid}\t{matches.get(i, '')}\n")
            fc.write(f"{sid}\t{cands.get(i, '')}\n")
    log(f"wrote {od}/matching_results.tsv and candidate_pairs.tsv ({corpus.n:,} rows each)")


def main():
    global MEM
    p = argparse.ArgumentParser(description="V6 ER pipeline")
    p.add_argument("command", choices=["fit", "candidates", "stage2", "all", "prep", "predict"])
    p.add_argument("--country", default=None, help="candidates: one country only")
    p.add_argument("--slice", type=int, default=0)
    p.add_argument("--n-slices", type=int, default=1)
    p.add_argument("--tau", type=float, default=None)
    p.add_argument("--output-dir", default="output")
    p.add_argument("--stage1-only", action="store_true", help="predict: fallback without the stage-2 model")
    p.add_argument("--data-dir", default="dataset/train")
    p.add_argument("--prefix", default="train", help="corpus file prefix for 'candidates' (train/test)")
    p.add_argument("--work-dir", default="work/v6")
    p.add_argument("--frac", type=float, default=1.0)
    p.add_argument("--procs", type=int, default=os.cpu_count() or 1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-pairs", type=int, default=1_000_000)
    p.add_argument("--n-stage1-train", type=int, default=120_000)
    p.add_argument("--n-idf", type=int, default=600_000)
    p.add_argument("--max-df-frac", type=float, default=0.003)
    p.add_argument("--em-iters", type=int, default=6)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--k-deep", type=int, default=100)
    p.add_argument("--tie-ratio", type=float, default=0.9)
    p.add_argument("--m-max", type=int, default=20)
    p.add_argument("--floor-loss", type=float, default=0.0005)
    p.add_argument("--chunk", type=int, default=50_000)
    p.add_argument("--max-train-rows", type=int, default=6_000_000)
    a = p.parse_args()
    MEM = MemMonitor()
    MEM.start()
    import diagnose_v6_retrieval as d
    d.MEM = MEM
    if a.command in ("fit", "all"):
        cmd_fit(a)
    if a.command in ("candidates", "all"):
        cmd_candidates(a)
    if a.command in ("stage2", "all"):
        cmd_stage2(a)
    if a.command == "prep":
        cmd_prep(a)
    if a.command == "predict":
        cmd_predict(a)


if __name__ == "__main__":
    main()
