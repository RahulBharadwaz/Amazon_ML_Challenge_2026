#!/usr/bin/env python3
"""
Gate G1 diagnostic for V6 Target -> Source 1 retrieval.

Protocol:
  - Hold out 10% of training Source 1 records (seed 42), singletons included.
  - Learn every artefact (transliteration, phonetics, lexicons, noise tokens, IDF,
    encoder) from the remaining 90% ("fit fold") only.
  - Index ALL training Source 1 records (holdout + fit fold act as competitors), per country.
  - Query with every true target of the holdout anchors. Retrieval is per target, so
    this is exactly the recall a full 10.3M-target run would give for these links.
  - A uniform random sample of all targets is also queried to measure candidate volume.

Reports recall per channel, union recall, recall of the RRF-fused top-M (the input a
stage-1 ranker would prune), per country / source, the oracle macro-F0.5 ceiling and
peak memory of the whole process tree.
"""

import argparse
import glob
import json
import os
import sys
import threading
import time
from collections import Counter

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pv

import stage1_pruner as s1p
import v6_retrieval as v6

T0 = time.time()


# ------------------------------------------------------------------------------
# Memory monitor (whole process tree, so pool workers are included)
# ------------------------------------------------------------------------------

class MemMonitor(threading.Thread):
    def __init__(self, interval=0.5):
        super().__init__(daemon=True)
        import psutil
        self.proc = psutil.Process(os.getpid())
        self.peak = 0.0
        self.interval = interval

    def current(self):
        """Process-tree memory: PSS on Linux (shared fork pages counted once), else RSS."""
        try:
            procs = [self.proc] + self.proc.children(recursive=True)
            if sys.platform.startswith("linux"):
                return sum(p.memory_full_info().pss for p in procs) / 2**20
            return sum(p.memory_info().rss for p in procs) / 2**20
        except Exception:
            return 0.0

    def run(self):
        while True:
            self.peak = max(self.peak, self.current())
            time.sleep(self.interval)


MEM = None


def log(msg):
    rss = f"{MEM.current():,.0f} MB (peak {MEM.peak:,.0f})" if MEM else ""
    print(f"[{time.time() - T0:7.0f}s | {rss}] {msg}", flush=True)


# ------------------------------------------------------------------------------
# Loading
# ------------------------------------------------------------------------------

def read_tsv(path, cols):
    return pv.read_csv(
        path,
        read_options=pv.ReadOptions(block_size=1 << 26),
        parse_options=pv.ParseOptions(delimiter="\t", quote_char=False,
                                      invalid_row_handler=lambda row: "skip"),
        convert_options=pv.ConvertOptions(column_types={c: pa.string() for c in cols},
                                          strings_can_be_null=False, include_columns=cols))


def stream_tsv(path, cols, block_size=1 << 24):
    """Record batches of a TSV (bounded memory, unlike reading the whole file)."""
    reader = pv.open_csv(
        path,
        read_options=pv.ReadOptions(block_size=block_size),
        parse_options=pv.ParseOptions(delimiter="	", quote_char=False,
                                      invalid_row_handler=lambda row: "skip"),
        convert_options=pv.ConvertOptions(column_types={c: pa.string() for c in cols},
                                          strings_can_be_null=False, include_columns=cols))
    for batch in reader:
        yield batch


def id_codes(arr) -> np.ndarray:
    """'S2-681193310' -> 2 * 10**10 + 681193310 (int64)."""
    arr = pc.utf8_trim_whitespace(arr)
    src = pc.cast(pc.utf8_slice_codeunits(arr, 1, 2), pa.int64())
    num = pc.cast(pc.utf8_slice_codeunits(arr, 3, 64), pa.int64())
    return (pc.add(pc.multiply(src, 10**10), num)).to_numpy(zero_copy_only=False)


def code_to_id(c: int) -> str:
    return f"S{c // 10**10}-{c % 10**10}"


def lookup(sorted_codes, order, codes):
    """Row index of each code (or -1)."""
    pos = np.searchsorted(sorted_codes, codes)
    pos = np.clip(pos, 0, len(sorted_codes) - 1)
    ok = sorted_codes[pos] == codes
    return np.where(ok, order[pos], -1)


# ------------------------------------------------------------------------------
# Evaluation helpers
# ------------------------------------------------------------------------------

def channel_hit(entries, truth, nq):
    """Per query: is the true S1 among this channel's retrieved candidates?"""
    hit = np.zeros(nq, bool)
    for qidx, I in entries:
        t = truth[qidx]
        hit[qidx] |= (I == t[:, None]).any(1) & (t >= 0)
    return hit


def fused_rank_blocked(entries, truth, nq, n_s1, block=100_000, rrf_k=60.0):
    """RRF over all channel entries, in query blocks (bounded memory).

    Returns the rank of the true S1 in the fused list (-1 when absent) and the
    union size per query."""
    rank = np.full(nq, -1, dtype=np.int64)
    union = np.zeros(nq, dtype=np.int64)
    for lo in range(0, nq, block):
        hi = min(nq, lo + block)
        qs, cs, ws = [], [], []
        for qidx, I in entries:
            sel = (qidx >= lo) & (qidx < hi)
            if not sel.any():
                continue
            Ib = I[sel]
            k = Ib.shape[1]
            q = np.repeat(qidx[sel].astype(np.int64) - lo, k)
            c = Ib.reshape(-1).astype(np.int64)
            r = np.tile(np.arange(k), len(Ib))
            keep = c >= 0
            qs.append(q[keep]); cs.append(c[keep]); ws.append(1.0 / (rrf_k + r[keep]))
        if not qs:
            continue
        keys = np.concatenate(qs) * n_s1 + np.concatenate(cs)
        uniq, inv = np.unique(keys, return_inverse=True)
        score = np.bincount(inv, weights=np.concatenate(ws))
        uq, uc = uniq // n_s1, uniq % n_s1
        nb = hi - lo
        union[lo:hi] = np.bincount(uq, minlength=nb)
        tb = truth[lo:hi]
        true_score = np.full(nb, -1.0)
        is_true = tb[uq] == uc
        true_score[uq[is_true]] = score[is_true]
        higher = (score > true_score[uq]) & (true_score[uq] >= 0)
        rb = np.bincount(uq, weights=higher, minlength=nb).astype(np.int64)
        rb[true_score < 0] = -1
        rank[lo:hi] = rb
    return rank, union


def f05(r):
    return np.where(r > 0, 1.25 * r / (0.25 + r), 0.0)


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------

def run(a):
    global MEM
    MEM = MemMonitor()
    MEM.start()
    rng = np.random.default_rng(a.seed)
    d = a.data_dir
    os.makedirs(a.out_dir, exist_ok=True)
    timings = {}

    # --- Source 1 + ground truth ---------------------------------------------
    t = time.time()
    s1 = read_tsv(os.path.join(d, "train_source1.tsv"), ["entity_id", "business_name", "business_address", "country"])
    s1_codes = id_codes(s1["entity_id"])
    n_all = len(s1_codes)
    active = rng.random(n_all) < a.frac
    s1 = s1.filter(pa.array(active))
    s1_codes = s1_codes[active]
    n_s1 = len(s1_codes)
    s1_name = s1["business_name"].to_pylist()
    s1_addr = s1["business_address"].to_pylist()
    s1_country = np.array(s1["country"].to_pylist(), dtype=object)
    del s1
    s1_order = np.argsort(s1_codes)
    s1_sorted = s1_codes[s1_order]
    holdout = rng.random(n_s1) < a.holdout_frac
    log(f"Source 1: {n_s1:,} records indexed (frac {a.frac}); holdout {holdout.sum():,}")

    gt = read_tsv(os.path.join(d, "train_ground_truth.tsv"), ["source1_entity_id", "matched_entity_ids"])
    gt_s1 = id_codes(gt["source1_entity_id"])
    lists = pc.split_pattern(gt["matched_entity_ids"], ",")
    parents = pc.list_parent_indices(lists).to_numpy()
    flat = pc.list_flatten(lists)
    nonempty = pc.greater(pc.utf8_length(flat), 0).to_numpy(zero_copy_only=False)
    link_tgt = id_codes(pc.filter(flat, pa.array(nonempty)))
    link_s1 = lookup(s1_sorted, s1_order, gt_s1[parents[nonempty]])
    del gt, lists, flat
    keep = link_s1 >= 0
    link_s1, link_tgt = link_s1[keep], link_tgt[keep]
    ho_link = holdout[link_s1]
    fit_idx = np.flatnonzero(~ho_link)
    pair_idx = rng.choice(fit_idx, size=min(a.n_pairs, len(fit_idx)), replace=False)
    rest = np.setdiff1d(fit_idx, pair_idx, assume_unique=True)
    s1q_idx = rng.choice(rest, size=min(a.n_stage1_train, len(rest)), replace=False)
    log(f"Links: {len(link_s1):,} ({ho_link.sum():,} holdout, {len(fit_idx):,} fit); "
        f"{len(pair_idx):,} fit pairs sampled for learning")

    # --- Targets: holdout links, learning pairs, random sample ---------------
    need = np.unique(np.concatenate([link_tgt[ho_link], link_tgt[pair_idx], link_tgt[s1q_idx]]))
    parts = []
    n_targets_total = 0
    cols = ["entity_id", "business_name", "business_address", "country"]
    for fn in ("train_source2.tsv", "train_source3.tsv"):
        kept = 0
        for batch in stream_tsv(os.path.join(d, fn), cols):
            codes = id_codes(batch.column(0))
            n_targets_total += len(codes)
            rnd = rng.random(len(codes)) < a.random_frac * a.frac
            m = np.isin(codes, need) | rnd
            if m.any():
                sub = batch.filter(pa.array(m))
                parts.append((codes[m], rnd[m], sub.column(1).to_pylist(),
                              sub.column(2).to_pylist(), sub.column(3).to_pylist()))
                kept += int(m.sum())
        log(f"  {fn}: kept {kept:,} target records (streamed)")
    t_codes = np.concatenate([p[0] for p in parts])
    t_rand = np.concatenate([p[1] for p in parts])
    t_name = [x for p in parts for x in p[2]]
    t_addr = [x for p in parts for x in p[3]]
    t_country = np.array([x for p in parts for x in p[4]], dtype=object)
    del parts
    t_order = np.argsort(t_codes)
    t_sorted = t_codes[t_order]
    timings["load"] = time.time() - t

    # --- Learn artefacts on the fit fold -------------------------------------
    t = time.time()
    p_s1 = link_s1[pair_idx]
    p_t = lookup(t_sorted, t_order, link_tgt[pair_idx])
    ok = p_t >= 0
    p_s1, p_t = p_s1[ok], p_t[ok]
    word_pairs = v6.collect_translit_pairs((s1_name[i], t_name[j]) for i, j in zip(p_s1, p_t))
    theta, word_dict, unit_mass = v6.learn_transliteration(word_pairs, iters=a.em_iters, log=log)
    vowels, letter_class, drop_h = v6.learn_phonetics(theta, unit_mass)
    classes = Counter(letter_class.values())
    log(f"  [phonetics] vowels={''.join(sorted(vowels))} | merged classes: "
        f"{sorted(''.join(sorted(k for k, v in letter_class.items() if v == c)) for c in classes if classes[c] > 1)}"
        f" | drop aspiration-h={drop_h}")
    canon = v6.Canonicalizer(theta, word_dict, vowels, letter_class, drop_h)
    # Char-model generalization: phonetic-key agreement on all aligned words (dictionary bypassed)
    agree = tot = 0
    for (iw, lw), c in word_pairs.items():
        tot += c
        agree += c * (canon.phon(canon.translit_chars(iw)) == canon.phon(lw))
    rare = [(iw, lw) for (iw, lw), c in word_pairs.items() if c == 1][:10]
    log(f"  [translit] char-model phonetic-key agreement {agree / max(tot, 1):.3f} over {tot:,} word tokens; "
        f"samples: {[(lw, canon.translit_chars(iw)) for iw, lw in rare]}")

    name_pairs = [(canon.tokens(s1_name[i], {}), canon.tokens(t_name[j], {})) for i, j in zip(p_s1, p_t)]
    addr_pairs = [(canon.tokens(s1_addr[i], {}), canon.tokens(t_addr[j], {})) for i, j in zip(p_s1, p_t)]
    canon.lex_name = v6.learn_lexicon(name_pairs)
    canon.lex_addr = v6.learn_lexicon(addr_pairs)
    name_pairs = [([canon.lex_name.get(w, w) for w in s], [canon.lex_name.get(w, w) for w in tt])
                  for s, tt in name_pairs]
    canon.noise = v6.learn_noise_tokens(name_pairs)
    del name_pairs, addr_pairs
    log(f"  [lexicon] name {len(canon.lex_name)} entries e.g. {list(canon.lex_name.items())[:12]}")
    log(f"  [lexicon] addr {len(canon.lex_addr)} entries e.g. {list(canon.lex_addr.items())[:12]}")
    log(f"  [noise] {len(canon.noise)} name tokens e.g. {sorted(canon.noise)[:30]}")
    demo = [s1_name[i] + "  <=>  " + t_name[j] + "  ->  " + " ".join(canon.name_core(t_name[j]))
            for i, j in zip(p_s1[:4000], p_t[:4000]) if any(v6.is_indic(w) for w in v6.raw_tokens(t_name[j]))][:8]
    for s in demo:
        log(f"  [translit demo] {s}")

    fit_rows = np.flatnonzero(~holdout)
    idf_rows = rng.choice(fit_rows, size=min(a.n_idf, len(fit_rows)), replace=False)
    raw_n, raw_a = v6.featurize(canon, [s1_name[i] for i in idf_rows], [s1_addr[i] for i in idf_rows], a.procs)
    idf_name = v6.IdfModel(raw_n, a.max_df_frac)
    idf_addr = v6.IdfModel(raw_a, a.max_df_frac)
    del raw_n, raw_a
    idf_raw = v6.IdfModel(v6.featurize_raw(canon, [s1_name[i] for i in idf_rows], a.procs), a.max_df_frac)
    log(f"  [idf] fitted on {len(idf_rows):,} fit-fold S1 records; pruned from index: "
        f"name {idf_name.prune.sum():,} / addr {idf_addr.prune.sum():,} hashed features")
    timings["learn_text"] = time.time() - t

    # --- Encoder --------------------------------------------------------------
    t = time.time()
    rn, ra = v6.featurize(canon, [s1_name[i] for i in p_s1], [s1_addr[i] for i in p_s1], a.procs)
    s_in = v6.encoder_inputs(idf_name.transform(rn), idf_addr.transform(ra))
    rn, ra = v6.featurize(canon, [t_name[j] for j in p_t], [t_addr[j] for j in p_t], a.procs)
    t_in = v6.encoder_inputs(idf_name.transform(rn), idf_addr.transform(ra))
    del rn, ra
    grp = np.unique(s1_country[p_s1], return_inverse=True)[1]
    model = v6.train_encoder(s_in, t_in, grp, epochs=a.epochs, n_threads=a.procs, seed=a.seed, log=log)
    del s_in, t_in
    v6.trim_memory()
    v6.save_artifacts(os.path.join(a.out_dir, "artifacts_fitfold"), canon, idf_name, idf_addr, model)
    timings["encoder"] = time.time() - t

    # --- Queries (targets not needed for querying are dropped to free memory) --
    ho_codes = link_tgt[ho_link]
    ho_row = lookup(t_sorted, t_order, ho_codes)
    fq_row = lookup(t_sorted, t_order, link_tgt[s1q_idx])
    q_rows = np.unique(np.concatenate([ho_row[ho_row >= 0], fq_row[fq_row >= 0], np.flatnonzero(t_rand)]))
    nq = len(q_rows)
    remap = np.full(len(t_codes), -1, dtype=np.int64)
    remap[q_rows] = np.arange(nq)
    truth = np.full(nq, -1, dtype=np.int64)
    ho_s1 = link_s1[ho_link]
    okr = ho_row >= 0
    truth[remap[ho_row[okr]]] = ho_s1[okr]
    truth_fit = np.full(nq, -1, dtype=np.int64)          # labels of stage-1 training targets
    okf = fq_row >= 0
    truth_fit[remap[fq_row[okf]]] = link_s1[s1q_idx][okf]
    t_codes, t_rand, t_country = t_codes[q_rows], t_rand[q_rows], t_country[q_rows]
    t_name = [t_name[r] for r in q_rows]
    t_addr = [t_addr[r] for r in q_rows]
    del t_sorted, t_order, remap, p_s1, p_t, link_tgt
    v6.trim_memory()
    role_train = (truth_fit >= 0) & ~t_rand
    truth_all = np.where(truth >= 0, truth, truth_fit)
    pair_dir = os.path.join(a.out_dir, "pairs")
    os.makedirs(pair_dir, exist_ok=True)
    for f in os.listdir(pair_dir):
        os.remove(os.path.join(pair_dir, f))
    log(f"Queries: {nq:,} ({(truth >= 0).sum():,} holdout links, {t_rand.sum():,} random targets, "
        f"{role_train.sum():,} stage-1 training targets)")

    K, KD = a.k, a.k_deep
    E = {"name": [], "addr": [], "enc": []}          # per channel: list of (query idx, S1 idx matrix)
    top1 = {"name": np.full(nq, -1, np.int64), "addr": np.full(nq, -1, np.int64)}
    deep_flags = {c: np.zeros(nq, bool) for c in E}
    timings["channels"] = Counter()
    countries = sorted(set(s1_country.tolist()))
    for ctry in countries:
        s_rows = np.flatnonzero(s1_country == ctry)
        qi = np.flatnonzero(t_country == ctry)
        if len(qi) == 0:
            continue

        def to_global(I):
            return np.where(I >= 0, s_rows[np.maximum(I, 0)], -1).astype(np.int32)

        Ec = {c: [] for c in E}

        def record(ch, I, deep, Id):
            Ec[ch].append((qi[~deep], to_global(I[~deep])))
            if deep.any():
                Ec[ch].append((qi[deep], to_global(Id)))
            deep_flags[ch][qi] = deep

        t = time.time()
        sn, sa = v6.featurize(canon, [s1_name[i] for i in s_rows], [s1_addr[i] for i in s_rows], a.procs)
        sn, sa = idf_name.transform(sn), idf_addr.transform(sa)
        qn, qa = v6.featurize(canon, [t_name[i] for i in qi], [t_addr[i] for i in qi], a.procs)
        qn, qa = idf_name.transform(qn), idf_addr.transform(qa)
        qn_empty, qa_empty = np.diff(qn.indptr) == 0, np.diff(qa.indptr) == 0
        s_emb = v6.encode(model, v6.encoder_inputs(sn, sa))
        q_emb = v6.encode(model, v6.encoder_inputs(qn, qa))
        timings["channels"]["featurize+encode"] += time.time() - t
        log(f"[{ctry}] {len(s_rows):,} S1 | {len(qi):,} queries featurized and encoded")

        # Name: go deep when the target has no address, or the top-K names are unresolved ties
        t = time.time()
        index = v6.SparseIndex(sn, idf_name.prune, a.procs)
        I, D = index.search(qn, K)
        deep_n = qa_empty | v6.unresolved(D, a.tie_ratio)
        Id = index.search(qn[np.flatnonzero(deep_n)], KD)[0] if deep_n.any() else None
        record("name", I, deep_n, Id)
        top1["name"][qi] = to_global(I[:, :1])[:, 0]
        del index, I, D, Id
        v6.trim_memory()
        timings["channels"]["name"] += time.time() - t
        log(f"[{ctry}] name channel done (deep {deep_n.mean():.1%})")

        # Address: go deep when the target has no usable name, or the top-K addresses tie
        t = time.time()
        index = v6.SparseIndex(sa, idf_addr.prune, a.procs)
        I, D = index.search(qa, K)
        deep_a = qn_empty | v6.unresolved(D, a.tie_ratio)
        Id = index.search(qa[np.flatnonzero(deep_a)], KD)[0] if deep_a.any() else None
        record("addr", I, deep_a, Id)
        top1["addr"][qi] = to_global(I[:, :1])[:, 0]
        del index, I, D, Id
        v6.trim_memory()
        timings["channels"]["addr"] += time.time() - t
        log(f"[{ctry}] address channel done (deep {deep_a.mean():.1%})")

        # Encoder: deep wherever either sparse channel went deep
        t = time.time()
        index = v6.HnswIndex(s_emb, a.procs, seed=a.seed)
        deep_e = deep_n | deep_a
        I, _ = index.search(q_emb, K)
        Id = index.search(q_emb[deep_e], KD)[0] if deep_e.any() else None
        record("enc", I, deep_e, Id)
        del index, I, Id
        v6.trim_memory()
        timings["channels"]["hnsw"] += time.time() - t
        log(f"[{ctry}] HNSW-SQ8 encoder channel done (deep {deep_e.mean():.1%})")

        # Stage-1 pair features (exact cosines on all three representations for every pair)
        t = time.time()
        s_addr_empty = np.diff(sa.indptr) == 0
        n_pairs_c = 0
        for nb, b0 in enumerate(range(0, len(qi), a.s1_block)):
            qb = qi[b0:b0 + a.s1_block]
            Q, C, R = s1p.block_pairs(Ec, int(qb[0]), int(qb[-1]) + 1, n_s1)
            if len(Q) == 0:
                continue
            X = s1p.pair_features(Q, C, R, np.searchsorted(qi, Q), np.searchsorted(s_rows, C),
                                  qn, qa, sn, sa, q_emb, s_emb, qn_empty, qa_empty, s_addr_empty)
            lab = truth_all[Q] == C
            trm = role_train[Q]
            for m, tag in ((trm, "train"), (~trm, "eval")):
                if m.any():
                    s1p.write_block(os.path.join(pair_dir, f"{tag}_{ctry}_{nb:04d}.npz"), X[m], Q[m], C[m], lab[m])
            n_pairs_c += len(Q)
            del X, Q, C, R, lab, trm
        for c in E:
            E[c].extend(Ec[c])
        del sn, sa, s_emb, qn, qa, q_emb, Ec
        v6.trim_memory()
        # second pass: surface-form name evidence, built after the retrieval matrices are freed
        sr = idf_raw.transform(v6.featurize_raw(canon, [s1_name[i] for i in s_rows], a.procs))
        qr = idf_raw.transform(v6.featurize_raw(canon, [t_name[i] for i in qi], a.procs))
        for fpath in sorted(glob.glob(os.path.join(pair_dir, f"*_{ctry}_*.npz"))):
            s1p.add_raw_features(fpath, qi, s_rows, qr, sr)
        del sr, qr
        v6.trim_memory()
        timings["channels"]["stage1_features"] += time.time() - t
        log(f"[{ctry}] stage-1 features for {n_pairs_c:,} pairs written")

    # --- Metrics ----------------------------------------------------------------
    ev = truth >= 0
    tr = truth[ev]
    hit_ch = {c: channel_hit(E[c], truth, nq)[ev] for c in E}
    frank, union = fused_rank_blocked([e for c in E for e in E[c]], truth, nq, n_s1)
    fr = frank[ev]
    hit_union = fr >= 0

    ho_country = s1_country[tr]
    ho_src = np.array([code_to_id(c)[:2] for c in t_codes[ev]])
    cross_country = (t_country[ev] != ho_country)

    def rec(mask):
        return float(mask.mean()) if len(mask) else float("nan")

    summary = {
        "holdout_anchors": int(holdout.sum()),
        "holdout_links": int(ev.sum()),
        "k_per_channel": K, "k_deep": KD, "tie_ratio": a.tie_ratio,
        "deep_share_all_queries": {c: float(v.mean()) for c, v in deep_flags.items()},
        "recall_name": rec(hit_ch["name"]),
        "recall_addr": rec(hit_ch["addr"]),
        "recall_encoder": rec(hit_ch["enc"]),
        "recall_name_or_addr": rec(hit_ch["name"] | hit_ch["addr"]),
        "recall_union": rec(hit_union),
        "recall_fused_top": {m: rec((fr >= 0) & (fr < m)) for m in (1, 3, 5, 10, 20, 50)},
        "cross_country_links": int(cross_country.sum()),
    }
    by_c = {}
    for c in sorted(set(ho_country.tolist())):
        m = ho_country == c
        by_c[c] = {"links": int(m.sum()), "union": rec(hit_union[m]),
                   "fused_top5": rec((fr[m] >= 0) & (fr[m] < 5)),
                   "name": rec(hit_ch["name"][m]), "addr": rec(hit_ch["addr"][m]), "enc": rec(hit_ch["enc"][m])}
    summary["by_country"] = by_c
    summary["by_source"] = {s: {"links": int((ho_src == s).sum()), "union": rec(hit_union[ho_src == s])}
                            for s in ("S2", "S3")}

    # Oracle macro F0.5 over holdout anchors (singletons score 1 with a perfect classifier)
    ho_anchor_ids = np.flatnonzero(holdout)
    n_true = np.bincount(ho_s1, minlength=n_s1)[ho_anchor_ids]
    oracle = {}
    for label, hit in [("union", hit_union)] + [(f"fused_top{m}", (fr >= 0) & (fr < m)) for m in (3, 5, 10, 20)]:
        hits = np.bincount(tr, weights=hit, minlength=n_s1)[ho_anchor_ids]
        r = np.where(n_true > 0, hits / np.maximum(n_true, 1), 1.0)
        oracle[label] = float(np.where(n_true > 0, f05(r), 1.0).mean())
    summary["oracle_macro_f05"] = oracle

    # Identical-name ambiguity among misses: a target without address whose anchor's core
    # name is shared by >= 2 Source 1 records of the same country cannot be told apart.
    t = time.time()
    core_count = Counter((s1_country[i], " ".join(canon.name_core(s1_name[i]))) for i in range(n_s1))
    miss_all = np.flatnonzero(~hit_union)
    ev_rows = np.flatnonzero(ev)
    amb = np.array([not str(t_addr[ev_rows[k]]).strip() and
                    core_count[(s1_country[tr[k]], " ".join(canon.name_core(s1_name[tr[k]])))] >= 2
                    for k in miss_all], dtype=bool)
    summary["ambiguity"] = {
        "misses": int(len(miss_all)),
        "ambiguous_identical_name_no_address": int(amb.sum()),
        "recall_union_excluding_ambiguous": float(hit_union.sum() / max(1, len(hit_union) - amb.sum())),
    }
    del core_count
    timings["ambiguity"] = time.time() - t

    rnd_q = t_rand
    vol = {"mean_union_per_target": float(union[rnd_q].mean()) if rnd_q.any() else float("nan"),
           "deep_share_random_targets": {c: float(v[rnd_q].mean()) for c, v in deep_flags.items()}}
    for m in (5, 10, 20):
        vol[f"mean_fused_top{m}_per_target"] = float(np.minimum(union[rnd_q], m).mean()) if rnd_q.any() else float("nan")
    vol["est_test_pairs_union_M"] = vol["mean_union_per_target"] * 9_969_589 / 1e6
    vol["est_test_pairs_top10_M"] = vol["mean_fused_top10_per_target"] * 9_969_589 / 1e6
    summary["volume"] = vol
    summary["timings_s"] = {k: (dict(v) if isinstance(v, Counter) else v) for k, v in timings.items()}
    summary["runtime_s"] = time.time() - T0
    summary["peak_tree_mem_mb"] = MEM.peak

    # Missed links for error analysis
    miss = miss_all[rng.permutation(len(miss_all))[:3000]]
    with open(os.path.join(a.out_dir, "missed_links.tsv"), "w", encoding="utf-8") as f:
        f.write("anchor_id\ttarget_id\tcountry\tanchor_name\tanchor_addr\ttarget_name\ttarget_addr\t"
                "anchor_core\ttarget_core\tname_top1\taddr_top1\n")
        for k in miss:
            s, row = tr[k], ev_rows[k]
            n1, a1 = top1["name"][row], top1["addr"][row]
            f.write("\t".join([
                code_to_id(s1_codes[s]), code_to_id(t_codes[row]), str(s1_country[s]),
                s1_name[s], s1_addr[s], t_name[row], t_addr[row],
                " ".join(canon.name_core(s1_name[s])), " ".join(canon.name_core(t_name[row])),
                s1_name[n1] if n1 >= 0 else "", s1_addr[a1] if a1 >= 0 else "",
            ]).replace("\r", " ").replace("\n", " ") + "\n")
    with open(os.path.join(a.out_dir, "g1_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # --- Gate G1b: stage-1 pruning ---------------------------------------------
    del E, top1
    v6.trim_memory()
    t = time.time()
    s1_model, val = s1p.train(os.path.join(pair_dir, "train_*.npz"), a.procs, seed=a.seed, log=log)
    floor = s1p.tune_floor(val, a.m_max, a.floor_loss)
    del val
    Ms = (5, 10, 15, 20)
    rec_m, vol_m, rec_floor, vol_floor, hit_rank, _ = s1p.evaluate(
        s1_model, os.path.join(pair_dir, "eval_*.npz"), truth, t_rand, Ms, floor, a.m_max)
    s1_model.save_model(os.path.join(a.out_dir, "stage1_lgbm.txt"))
    gain = dict(zip(s1p.ALL_FEATURES, s1_model.feature_importance("gain").round().astype(int).tolist()))
    kept = (hit_rank[ev] >= 0) & (hit_rank[ev] < a.m_max)
    hits = np.bincount(tr, weights=kept, minlength=n_s1)[ho_anchor_ids]
    r = np.where(n_true > 0, hits / np.maximum(n_true, 1), 1.0)
    summary["stage1"] = {
        "recall_at_m": {int(m): v for m, v in rec_m.items()},
        "kept_per_target_at_m": {int(m): v for m, v in vol_m.items()},
        "prob_floor": floor, "recall_top20_floor": rec_floor, "kept_per_target_top20_floor": vol_floor,
        "est_test_pairs_top20_M": vol_m[a.m_max] * 9_969_589 / 1e6,
        "est_test_pairs_top20_floor_M": vol_floor * 9_969_589 / 1e6,
        "oracle_macro_f05_top20": float(np.where(n_true > 0, f05(r), 1.0).mean()),
        "feature_gain": dict(sorted(gain.items(), key=lambda kv: -kv[1])),
        "runtime_s": time.time() - t,
    }
    summary["runtime_s"] = time.time() - T0
    summary["peak_tree_mem_mb"] = MEM.peak
    with open(os.path.join(a.out_dir, "g1_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    amb_s = summary["ambiguity"]
    print("=" * 88)
    print("  GATE G1 - V6 Target->S1 retrieval, 10% held-out training anchors")
    print("=" * 88)
    print(f"  Holdout anchors {summary['holdout_anchors']:,} | true links {summary['holdout_links']:,} | "
          f"S1 index {n_s1:,} | K={K}, deep K={KD} (tie ratio {a.tie_ratio})")
    print("  Deep-search share of random targets     : " +
          " | ".join(f"{c} {v:.1%}" for c, v in vol["deep_share_random_targets"].items()))
    print(f"  Recall  name {summary['recall_name']:.4f} | address {summary['recall_addr']:.4f} | "
          f"encoder {summary['recall_encoder']:.4f} | name+addr {summary['recall_name_or_addr']:.4f}")
    print(f"  UNION RECALL (candidate ceiling)        : {summary['recall_union']:.4f}")
    print("  RRF-fused top-M recall                  : " +
          " | ".join(f"@{m} {v:.4f}" for m, v in summary["recall_fused_top"].items()))
    for c, v in by_c.items():
        print(f"    {c:8s} links {v['links']:>8,} | union {v['union']:.4f} | top5 {v['fused_top5']:.4f} | "
              f"name {v['name']:.4f} addr {v['addr']:.4f} enc {v['enc']:.4f}")
    print(f"  By source: " + " | ".join(f"{s} {v['union']:.4f}" for s, v in summary["by_source"].items()))
    print(f"  Cross-country links (unreachable by partition): {summary['cross_country_links']:,}")
    print(f"  Misses {amb_s['misses']:,}, of which identical-name + no-address (ambiguous): "
          f"{amb_s['ambiguous_identical_name_no_address']:,} -> union recall excluding them "
          f"{amb_s['recall_union_excluding_ambiguous']:.4f}")
    print("  Oracle macro F0.5 ceiling               : " +
          " | ".join(f"{k} {v:.4f}" for k, v in oracle.items()))
    print(f"  Volume per target: union {vol['mean_union_per_target']:.1f} -> est. test pairs "
          f"{vol['est_test_pairs_union_M']:.0f}M (fused top10: {vol['est_test_pairs_top10_M']:.0f}M)")
    print(f"  Runtime {summary['runtime_s']:.0f}s | peak process-tree memory {MEM.peak:,.0f} MB "
          f"(budget 5,500 MB: {'OK' if MEM.peak <= 5500 else 'OVER'})")
    gate = summary["recall_union"] >= 0.995
    print(f"  GATE G1 (union recall >= 0.995): {'PASS' if gate else 'FAIL'}")
    print("=" * 88)
    st = summary["stage1"]
    print("  GATE G1b - Stage-1 pruner (LightGBM on training-fold targets), same held-out links")
    print("=" * 88)
    print("  Recall@M      : " + " | ".join(f"@{m} {v:.4f}" for m, v in st["recall_at_m"].items()))
    print("  Kept/target   : " + " | ".join(f"@{m} {v:.1f}" for m, v in st["kept_per_target_at_m"].items()))
    print(f"  Top-{a.m_max} + prob floor {st['prob_floor']:.2e} (tuned on training fold): recall "
          f"{st['recall_top20_floor']:.4f}, kept/target {st['kept_per_target_top20_floor']:.1f}")
    print(f"  Est. test pairs: top-{a.m_max} {st['est_test_pairs_top20_M']:.0f}M | with floor "
          f"{st['est_test_pairs_top20_floor_M']:.0f}M (was {vol['est_test_pairs_union_M']:.0f}M)")
    print(f"  Oracle macro F0.5 ceiling after pruning: {st['oracle_macro_f05_top20']:.4f}")
    print(f"  Top features: {list(st['feature_gain'])[:8]}")
    print(f"  Runtime {summary['runtime_s']:.0f}s | peak process-tree memory {MEM.peak:,.0f} MB "
          f"(budget 5,500 MB: {'OK' if MEM.peak <= 5500 else 'OVER'})")
    g1b = st["recall_at_m"][a.m_max] >= 0.995
    print(f"  GATE G1b (recall@{a.m_max} >= 0.995): {'PASS' if g1b else 'FAIL'}")
    print("=" * 88)


def parse_args():
    p = argparse.ArgumentParser(description="Gate G1 recall diagnostic for V6 retrieval")
    p.add_argument("--data-dir", default="dataset/train")
    p.add_argument("--out-dir", default="output/diagnostics_v6")
    p.add_argument("--frac", type=float, default=1.0, help="fraction of S1 to use (debug)")
    p.add_argument("--holdout-frac", type=float, default=0.10)
    p.add_argument("--random-frac", type=float, default=0.02, help="share of all targets queried for volume")
    p.add_argument("--n-pairs", type=int, default=1_000_000, help="fit-fold pairs for learning")
    p.add_argument("--n-idf", type=int, default=600_000)
    p.add_argument("--max-df-frac", type=float, default=0.003)
    p.add_argument("--em-iters", type=int, default=6)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--k-deep", type=int, default=100, help="depth for address-less / unresolved-tie queries")
    p.add_argument("--tie-ratio", type=float, default=0.9, help="k-th score >= ratio * best -> unresolved")
    p.add_argument("--n-stage1-train", type=int, default=120_000, help="fit-fold targets for the stage-1 model")
    p.add_argument("--s1-block", type=int, default=40_000, help="queries per stage-1 feature block")
    p.add_argument("--m-max", type=int, default=20)
    p.add_argument("--floor-loss", type=float, default=0.0005, help="max recall loss allowed for the prob floor")
    p.add_argument("--procs", type=int, default=os.cpu_count() or 1)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
