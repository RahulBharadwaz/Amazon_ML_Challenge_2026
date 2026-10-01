#!/usr/bin/env python3
"""
ML Challenge 2026 - V6 candidate retrieval (Target -> Source 1).

Every Source 2 / Source 3 record queries an index of Source 1 records and keeps
its own top-K parents. There is no per-anchor candidate cap: an S1 record with
11 true matches receives all 11, because each target chooses independently.

Channels (union = candidate set):
  C1 name     char 3-gram / word / phonetic / initials TF-IDF, sparse top-k matmul
  C2 address  token + fuzzy-number (deletion neighbourhood) + number|street TF-IDF
  C3 encoder  hashed-feature siamese encoder (contrastive), FAISS HNSW over S1

Everything that is learned is learned from TRAINING data only:
  - Indic -> Latin transliteration: Indic scripts share Unicode's parallel block
    layout, so all are mapped onto one canonical block; akshara units are then
    aligned to the Latin Source 1 names of their training matches with EM.
  - Phonetic consonant key: vowel letters, confusable consonant classes and
    aspiration-h are read off the learned transliteration table.
  - Abbreviation lexicon (rd -> road, pvt -> private, ...), noise tokens
    (legal forms, honorifics), IDF weights and the encoder.
Test-set statistics are never used: features unseen in training get a neutral
(median) IDF.
"""

import array
import math
import multiprocessing as mp
import os
import pickle
import re
import time
import unicodedata
import zlib
from collections import Counter, defaultdict
from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp

FEAT_BITS = 21
FEAT_DIM = 1 << FEAT_BITS
FEAT_MASK = FEAT_DIM - 1

# ==============================================================================
# 1. SCRIPT FOLDING (Unicode structure only - no data-specific rules)
# ==============================================================================

_NON_DECOMPOSABLE = str.maketrans({
    "ß": "ss", "ẞ": "ss", "æ": "ae", "Æ": "ae", "œ": "oe", "Œ": "oe",
    "ø": "o", "Ø": "o", "ł": "l", "Ł": "l", "đ": "d", "Đ": "d",
    "ð": "d", "Ð": "d", "þ": "th", "Þ": "th", "ı": "i",
    "‘": "'", "’": "'", "ʼ": "'", "`": "'",
})
_LATIN_MARKS_RE = re.compile(r"[̀-ͯ]")
_TOKEN_RE = re.compile(r"[a-z0-9]+|[ऀ-ॿ]+")

INDIC_BASE = 0x0900


def _build_indic_table() -> Dict[int, Optional[str]]:
    """Map every Indic block (Devanagari..Malayalam share one layout) onto the Devanagari block."""
    tbl: Dict[int, Optional[str]] = {0x200C: None, 0x200D: None}
    for cp in range(0x0900, 0x0E00):
        off, block = (cp - INDIC_BASE) & 0x7F, (cp - INDIC_BASE) >> 7
        if 0x66 <= off <= 0x6F:
            tbl[cp] = chr(0x30 + off - 0x66)          # native digits -> ASCII digits
        elif off in (0x64, 0x65):
            tbl[cp] = " "                             # danda / double danda
        elif block == 2 and off == 0x70:
            tbl[cp] = "ं"                        # Gurmukhi tippi: a nasal, like anusvara
        elif block == 2 and off == 0x71:
            tbl[cp] = None                            # Gurmukhi addak: gemination mark
        elif block:
            tbl[cp] = chr(INDIC_BASE + off)
    return tbl


_INDIC_TABLE = _build_indic_table()

# Offsets inside the canonical block
_MODS = frozenset(range(0x01, 0x04))                                 # candrabindu, anusvara, visarga
_INDEP_V = frozenset(list(range(0x04, 0x15)) + [0x60, 0x61])
_CONS = frozenset(list(range(0x15, 0x3A)) + list(range(0x58, 0x60)))
_NUKTA, _VIRAMA = 0x3C, 0x4D
_VSIGN = frozenset(list(range(0x3E, 0x4D)) + [0x4E, 0x4F, 0x55, 0x56, 0x57, 0x62, 0x63])


def fold(text) -> str:
    """Lowercase; fold Latin diacritics; map all Indic scripts to the canonical block."""
    if not text or not isinstance(text, str):
        return ""
    t = text.translate(_NON_DECOMPOSABLE)
    if not t.isascii():
        t = unicodedata.normalize("NFKD", t)
        t = _LATIN_MARKS_RE.sub("", t).translate(_INDIC_TABLE)
    return t.lower().replace("'", "")


def is_indic(tok: str) -> bool:
    return tok[0] >= "ऀ"


def raw_tokens(text) -> List[str]:
    """Tokens of the folded text; runs of single letters are joined ('l l c' -> 'llc')."""
    out: List[str] = []
    run: List[str] = []
    for w in _TOKEN_RE.findall(fold(text)):
        if len(w) == 1 and "a" <= w <= "z":
            run.append(w)
            continue
        if run:
            out.append("".join(run)) if len(run) > 1 else out.extend(run)
            run = []
        out.append(w)
    if run:
        out.append("".join(run)) if len(run) > 1 else out.extend(run)
    return out


def aksharas(tok: str) -> List[str]:
    """Split a canonical Indic token into per-character transliteration units.

    A consonant (with its nukta) is tagged by context so the inherent vowel can be learned:
    'C+' when a vowel sign or virama follows (no inherent vowel), 'C$' when bare and
    word-final, 'C' when bare inside the word. Vowel signs, virama and modifiers are units.
    """
    units, i, n = [], 0, len(tok)
    while i < n:
        o = ord(tok[i]) - INDIC_BASE
        if o in _CONS:
            j = i + 1
            if j < n and ord(tok[j]) - INDIC_BASE == _NUKTA:
                j += 1
            nxt = ord(tok[j]) - INDIC_BASE if j < n else None
            tag = "+" if nxt is not None and (nxt == _VIRAMA or nxt in _VSIGN) else ("$" if nxt is None else "")
            units.append(tok[i:j] + tag)
            i = j
        else:
            units.append(tok[i])
            i += 1
    return units


def _unit_len_range(u: str, max_len: int) -> Tuple[int, int]:
    o = ord(u[0]) - INDIC_BASE
    if o in _CONS:
        return 1, max_len - 1 if u[-1] == "+" else max_len
    if o in _INDEP_V:
        return 1, 3
    if o == _VIRAMA or o == _NUKTA:
        return 0, 0
    return 0, 2      # vowel signs, anusvara / visarga, others


# ==============================================================================
# 2. LEARNING FROM TRAINING PAIRS
# ==============================================================================

def collect_translit_pairs(pairs: Iterable[Tuple[str, str]]) -> Counter:
    """(Latin S1 name, target name) training pairs -> Counter of aligned (indic word, latin word)."""
    out: Counter = Counter()
    for s_name, t_name in pairs:
        t_toks = raw_tokens(t_name)
        if not t_toks or not any(is_indic(w) for w in t_toks):
            continue
        s_toks = raw_tokens(s_name)
        if len(s_toks) != len(t_toks) or any(is_indic(w) for w in s_toks):
            continue
        for iw, lw in zip(t_toks, s_toks):
            if is_indic(iw) and lw.isalpha():
                out[(iw, lw)] += 1
    return out


def learn_transliteration(word_pairs: Counter, iters: int = 6, max_len: int = 4,
                          max_pairs: int = 120_000, log=print):
    """EM alignment of transliteration units to Latin substrings (monotone, unit-typed lengths).

    Returns (unit -> {latin substring: prob}, word dictionary indic word -> latin word).
    """
    # Word dictionary: dominant Latin rendering of frequent Indic words
    by_word: Dict[str, Counter] = defaultdict(Counter)
    for (iw, lw), c in word_pairs.items():
        by_word[iw][lw] += c
    word_dict = {}
    for iw, cnt in by_word.items():
        lw, c = cnt.most_common(1)[0]
        if c >= 2 and c >= 0.5 * sum(cnt.values()):
            word_dict[iw] = lw

    data = []
    for (iw, lw), c in word_pairs.most_common(max_pairs):
        units = aksharas(iw)
        lo = sum(_unit_len_range(u, max_len)[0] for u in units)
        hi = sum(_unit_len_range(u, max_len)[1] for u in units)
        if units and lo <= len(lw) <= hi:
            data.append((units, lw, 1.0 + math.log(c)))
    log(f"  [translit] {len(word_pairs):,} aligned word pairs, {len(data):,} used for EM, "
        f"{len(word_dict):,} dictionary words")

    # Initialization: each unit emits its proportional share of the Latin word. EM from a
    # flat start collapses into degenerate many-to-many alignments on a small vocabulary.
    init: Dict[str, Counter] = defaultdict(Counter)
    for units, lw, w in data:
        m = len(lw)
        wts = [sum(_unit_len_range(u, max_len)) / 2.0 for u in units]
        tot, acc, b = sum(wts) or 1.0, 0.0, [0]
        for x in wts:
            acc += x
            b.append(round(acc * m / tot))
        for i, u in enumerate(units):
            init[u][lw[b[i]:b[i + 1]]] += w
    theta: Optional[Dict[str, Dict[str, float]]] = {
        u: {s: c / sum(cu.values()) for s, c in cu.items()} for u, cu in init.items()}
    floor = 1e-4      # unseen segments stay reachable, so no pair becomes unalignable
    for it in range(iters):
        counts: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        used, ll = 0, 0.0
        for units, lw, w in data:
            n, m = len(units), len(lw)
            rng = [_unit_len_range(u, max_len) for u in units]
            probs = [theta.get(u, {}) for u in units]
            alpha = [[0.0] * (m + 1) for _ in range(n + 1)]
            alpha[0][0] = 1.0
            for i in range(n):
                lo, hi = rng[i]
                ai, an, pu = alpha[i], alpha[i + 1], probs[i]
                for j in range(m + 1):
                    a = ai[j]
                    if a == 0.0:
                        continue
                    for L in range(lo, min(hi, m - j) + 1):
                        p = pu.get(lw[j:j + L], floor)
                        if p:
                            an[j + L] += a * p
            Z = alpha[n][m]
            if Z <= 0.0:
                continue
            beta = [[0.0] * (m + 1) for _ in range(n + 1)]
            beta[n][m] = 1.0
            for i in range(n - 1, -1, -1):
                lo, hi = rng[i]
                bi, bn, pu = beta[i], beta[i + 1], probs[i]
                for j in range(m + 1):
                    s = 0.0
                    for L in range(lo, min(hi, m - j) + 1):
                        b = bn[j + L]
                        if b:
                            p = pu.get(lw[j:j + L], floor)
                            s += p * b
                    bi[j] = s
            used += 1
            ll += w * math.log(Z)
            for i in range(n):
                lo, hi = rng[i]
                ai, bn, pu, cu = alpha[i], beta[i + 1], probs[i], counts[units[i]]
                for j in range(m + 1):
                    a = ai[j]
                    if a == 0.0:
                        continue
                    for L in range(lo, min(hi, m - j) + 1):
                        b = bn[j + L]
                        if b:
                            seg = lw[j:j + L]
                            p = pu.get(seg, floor)
                            if p:
                                cu[seg] += w * a * p * b / Z
        theta, mass = {}, {}
        for u, cu in counts.items():
            tot = sum(cu.values())
            theta[u] = {s: c / tot for s, c in cu.items() if c / tot >= 1e-3}
            mass[u] = tot
        log(f"  [translit] EM iter {it + 1}/{iters}: {used:,} pairs aligned, "
            f"{len(theta):,} units, weighted log-lik {ll:,.0f}")
    return theta or {}, word_dict, (mass if theta else {})


def learn_phonetics(theta: Dict[str, Dict[str, float]], mass: Dict[str, float],
                    min_mass: float = 20.0, min_share: float = 0.3):
    """Vowel letters, consonant equivalence classes and aspiration-h, read off the transliteration table.

    Only units with at least `min_mass` expected alignment counts are trusted.
    A letter is a vowel when most of its emission mass comes from Indic vowel units
    rather than from the onset of consonant units.
    """
    trusted = {u: d for u, d in theta.items() if mass.get(u, 0.0) >= min_mass}
    v_mass, c_mass = Counter(), Counter()
    for u, dist in trusted.items():
        o, w = ord(u[0]) - INDIC_BASE, mass[u]
        for s, p in dist.items():
            if not s:
                continue
            if o in _INDEP_V or o in _VSIGN:
                for ch in s:
                    v_mass[ch] += w * p / len(s)
            elif o in _CONS:
                c_mass[s[0]] += w * p
    vowels = frozenset(ch for ch in v_mass if v_mass[ch] > c_mass[ch])

    first: Dict[str, Counter] = defaultdict(Counter)
    h_pos = Counter()
    for u, dist in trusted.items():
        # consonant classes only from 'C+' units: no inherent vowel, least orthographic noise
        if ord(u[0]) - INDIC_BASE not in _CONS or not u.endswith("+"):
            continue
        key = u[:2] if len(u) > 1 and ord(u[1]) - INDIC_BASE == _NUKTA else u[0]
        for s, p in dist.items():
            cons = "".join(ch for ch in s if ch not in vowels)
            if not cons:
                continue
            first[key][cons[0]] += mass[u] * p
            for k, ch in enumerate(cons):
                if ch == "h":
                    h_pos["initial" if k == 0 else "after"] += mass[u] * p

    parent = {ch: ch for ch in "abcdefghijklmnopqrstuvwxyz"}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for key, cnt in first.items():
        t = sum(cnt.values())
        if t < min_mass:
            continue
        major = [ch for ch, c in cnt.items() if c / t >= min_share and ch in parent]
        for a in major[1:]:
            ra, rb = find(a), find(major[0])
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)
    letter_class = {ch: find(ch) for ch in parent}
    drop_h = h_pos["after"] > h_pos["initial"]
    return vowels, letter_class, drop_h


def _is_subseq(short: str, long: str) -> bool:
    it = iter(long)
    return all(ch in it for ch in short)


def learn_lexicon(token_pairs: Iterable[Tuple[List[str], List[str]]],
                  min_count: int = 30, min_share: float = 0.5) -> Dict[str, str]:
    """Target-side variant -> Source 1 (reference) form, for abbreviations and shortenings.

    A pair (t, s) is counted when t is in the target but not its S1 match, s is in the
    S1 record but not the target, both share the first letter and one is a subsequence
    of the other ('rd'/'road', 'pvt'/'private', 'alaska'/'ak').
    """
    unmatched: Counter = Counter()
    cand: Counter = Counter()
    s1_freq: Counter = Counter()
    for s_toks, t_toks in token_pairs:
        sset, tset = set(s_toks), set(t_toks)
        s1_freq.update(sset)
        s_only = [s for s in sset - tset if s.isalpha()]
        for t in tset - sset:
            if not t.isalpha() or len(t) < 2:
                continue
            unmatched[t] += 1
            for s in s_only:
                if s[0] == t[0] and s != t and (
                        (len(t) < len(s) and _is_subseq(t, s)) or (len(s) < len(t) and _is_subseq(s, t))):
                    cand[(t, s)] += 1
    best: Dict[str, Tuple[str, int]] = {}
    for (t, s), c in cand.items():
        if c >= min_count and c >= min_share * unmatched[t] and (t not in best or c > best[t][1]):
            best[t] = (s, c)
    # Learned links can run both ways (ltd->limited, limited->ltd): collapse each connected
    # group onto one canonical form, the member most frequent in Source 1 (the reference).
    parent: Dict[str, str] = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for t, (s, _) in best.items():
        rt, rs = find(t), find(s)
        if rt != rs:
            parent[rt] = rs
    groups: Dict[str, List[str]] = defaultdict(list)
    for x in list(parent):
        groups[find(x)].append(x)
    lex = {}
    for members in groups.values():
        canon = max(members, key=lambda m: (s1_freq[m], -len(m), m))
        for m in members:
            if m != canon:
                lex[m] = canon
    return lex


def learn_noise_tokens(token_pairs: Iterable[Tuple[List[str], List[str]]],
                       min_count: int = 200, max_keep_rate: float = 0.5) -> frozenset:
    """Name tokens whose presence is not preserved across matching records (legal forms, honorifics)."""
    in_s, in_t, both = Counter(), Counter(), Counter()
    for s_toks, t_toks in token_pairs:
        sset, tset = set(s_toks), set(t_toks)
        in_s.update(sset)
        in_t.update(tset)
        both.update(sset & tset)
    noise = set()
    for tok in set(in_s) | set(in_t):
        union = in_s[tok] + in_t[tok] - both[tok]
        if union >= min_count and both[tok] / union < max_keep_rate:
            noise.add(tok)
    return frozenset(noise)


# ==============================================================================
# 3. CANONICALIZER + FEATURES
# ==============================================================================

class Canonicalizer:
    """All learned text-normalization artefacts, plus name/address feature extraction."""

    W_WORD, W_GRAM, W_PHON, W_INIT = 2.0, 1.0, 1.0, 1.0
    W_NUM, W_DEL, W_MIX, W_COMP, W_ADDR = 1.5, 0.5, 1.0, 1.0, 1.0

    def __init__(self, theta, word_dict, vowels, letter_class, drop_h,
                 lex_name=None, lex_addr=None, noise=frozenset()):
        self.best = {u: max(d.items(), key=lambda kv: kv[1])[0] for u, d in theta.items() if d}
        self.word_dict = word_dict
        self.vowels = vowels
        self.letter_class = letter_class
        self.drop_h = drop_h
        self.lex_name = lex_name or {}
        self.lex_addr = lex_addr or {}
        self.noise = noise
        self._cache: Dict[str, str] = {}

    # --- transliteration / phonetics -----------------------------------------
    def translit(self, tok: str) -> str:
        r = self._cache.get(tok)
        if r is not None:
            return r
        r = self.word_dict.get(tok)
        if r is None:
            r = self.translit_chars(tok)
        if len(self._cache) < 2_000_000:
            self._cache[tok] = r
        return r

    def translit_chars(self, tok: str) -> str:
        """Character-level (akshara) transliteration, used for words outside the learned dictionary."""
        parts = []
        for u in aksharas(tok):
            s = self.best.get(u)
            if s is None:
                base = u.rstrip("+$")
                for alt in (base, base + "+", base + "$"):
                    s = self.best.get(alt)
                    if s is not None:
                        break
            parts.append(s or "")
        return "".join(parts)

    def phon(self, word: str) -> str:
        out: List[str] = []
        prev_vowel = True
        for k, ch in enumerate(word):
            if not ("a" <= ch <= "z"):
                continue
            is_v = ch in self.vowels
            if is_v:
                if k == 0:
                    out.append("a")
                prev_vowel = True
                continue
            if ch == "h" and self.drop_h and not prev_vowel and out:
                continue
            c = self.letter_class.get(ch, ch)
            if not out or out[-1] != c:
                out.append(c)
            prev_vowel = False
        return "".join(out)

    # --- tokens --------------------------------------------------------------
    def tokens(self, text, lex: Dict[str, str]) -> List[str]:
        out = []
        for w in raw_tokens(text):
            if is_indic(w):
                w = self.translit(w)
                if not w:
                    continue
            out.append(lex.get(w, w))
        return out

    def name_core(self, text) -> List[str]:
        toks = self.tokens(text, self.lex_name)
        core = [t for t in toks if t not in self.noise]
        return core or toks

    def addr_tokens(self, text) -> List[str]:
        return self.tokens(text, self.lex_addr)

    # --- features ------------------------------------------------------------
    def name_features(self, text) -> Dict[str, float]:
        core = self.name_core(text)
        f: Dict[str, float] = defaultdict(float)
        if not core:
            return f
        for t in core:
            f["w" + t] += self.W_WORD
            if t.isalpha() and len(t) >= 2:
                p = self.phon(t)
                if len(p) >= 2:
                    f["p" + p] += self.W_PHON
        s = "^" + "".join(core) + "$"
        for i in range(len(s) - 2):
            f["g" + s[i:i + 3]] += self.W_GRAM
        if len(core) >= 2:
            f["i" + "".join(t[0] for t in core)] += self.W_INIT
        elif core[0].isalpha() and 2 <= len(core[0]) <= 4:
            f["i" + core[0]] += self.W_INIT       # an acronym name ('tt') meets initials
        return f

    def raw_features(self, text) -> Dict[str, float]:
        """Surface form of the name: transliterated but NOT canonicalized (legal forms, honorifics
        and abbreviations kept), so 'Gold It Pvt Ltd' and 'Gold It Limited' stay distinguishable."""
        s = "^" + " ".join(self.tokens(text, {})) + "$"
        f: Dict[str, float] = defaultdict(float)
        for i in range(len(s) - 2):
            f["r" + s[i:i + 3]] += 1.0
        return f

    def addr_features(self, text) -> Dict[str, float]:
        toks = self.addr_tokens(text)
        f: Dict[str, float] = defaultdict(float)
        n_tok = len(toks)
        for i, t in enumerate(toks):
            if t.isalpha():
                if len(t) >= 2:
                    f["a" + t] += self.W_ADDR
                continue
            if not t.isdigit():
                f["m" + t] += self.W_MIX
            for run in re.findall(r"\d+", t):
                n = run.lstrip("0") or "0"
                f["n" + n] += self.W_NUM
                if len(n) >= 2:
                    f["d" + n] += self.W_DEL
                if len(n) >= 3:
                    for j in range(len(n)):
                        f["d" + n[:j] + n[j + 1:]] += self.W_DEL
                for nb in (i - 1, i + 1):
                    if 0 <= nb < n_tok and toks[nb].isalpha() and len(toks[nb]) >= 3:
                        f["c" + n + "|" + toks[nb]] += self.W_COMP
        return f


# ==============================================================================
# 4. HASHED SPARSE MATRICES (multiprocess featurization)
# ==============================================================================

_CANON: Optional[Canonicalizer] = None


def _pack(dicts_iter) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    indptr = array.array("q", [0])
    idx = array.array("i")
    val = array.array("f")
    crc = zlib.crc32
    for f in dicts_iter:
        for k, v in f.items():
            idx.append(crc(k.encode("utf-8")) & FEAT_MASK)
            val.append(1.0 + math.log(v) if v > 1.0 else v)
        indptr.append(len(idx))
    return (np.frombuffer(indptr, dtype=np.int64).copy(),
            np.frombuffer(idx, dtype=np.int32).copy(),
            np.frombuffer(val, dtype=np.float32).copy())


def _featurize_chunk(args):
    names, addrs = args
    c = _CANON
    return (_pack(c.name_features(t) for t in names),
            _pack(c.addr_features(t) for t in addrs))


def _featurize_raw_chunk(names):
    return _pack(_CANON.raw_features(t) for t in names)


def _to_csr(parts, n_rows) -> sp.csr_matrix:
    indptrs, idxs, vals = [], [], []
    base = 0
    for k, (ip, ix, v) in enumerate(parts):
        indptrs.append(ip[1:] + base if k else ip + base)
        base += len(ix)
        idxs.append(ix)
        vals.append(v)
    m = sp.csr_matrix((np.concatenate(vals) if vals else np.zeros(0, np.float32),
                       np.concatenate(idxs) if idxs else np.zeros(0, np.int32),
                       np.concatenate(indptrs) if indptrs else np.zeros(1, np.int64)),
                      shape=(n_rows, FEAT_DIM))
    m.sum_duplicates()
    m.indptr = m.indptr.astype(np.int64)
    m.indices = m.indices.astype(np.int32)
    return m


def featurize(canon: Canonicalizer, names: Sequence[str], addrs: Sequence[str],
              n_procs: int = 1, chunk: int = 20_000) -> Tuple[sp.csr_matrix, sp.csr_matrix]:
    """Raw (tf-weighted, un-normalized) hashed name and address matrices."""
    global _CANON
    _CANON = canon
    jobs = [(names[i:i + chunk], addrs[i:i + chunk]) for i in range(0, len(names), chunk)]
    if n_procs > 1 and "fork" in mp.get_all_start_methods():
        with mp.get_context("fork").Pool(n_procs) as pool:
            res = pool.map(_featurize_chunk, jobs, chunksize=1)
    else:
        res = [_featurize_chunk(j) for j in jobs]
    n = len(names)
    return _to_csr([r[0] for r in res], n), _to_csr([r[1] for r in res], n)


def featurize_raw(canon: Canonicalizer, names: Sequence[str], n_procs: int = 1,
                  chunk: int = 20_000) -> sp.csr_matrix:
    """Raw hashed surface-form name matrix (see Canonicalizer.raw_features)."""
    global _CANON
    _CANON = canon
    jobs = [names[i:i + chunk] for i in range(0, len(names), chunk)]
    if n_procs > 1 and "fork" in mp.get_all_start_methods():
        with mp.get_context("fork").Pool(n_procs) as pool:
            res = pool.map(_featurize_raw_chunk, jobs, chunksize=1)
    else:
        res = [_featurize_raw_chunk(j) for j in jobs]
    return _to_csr(res, len(names))


class IdfModel:
    """IDF weights fitted on training Source 1 records; unseen features get the median IDF."""

    def __init__(self, raw: sp.csr_matrix, max_df_frac: float):
        n = raw.shape[0]
        df = np.bincount(raw.indices, minlength=FEAT_DIM).astype(np.float64)
        idf = np.log((n + 1.0) / (df + 1.0)) + 1.0
        seen = df > 0
        idf[~seen] = np.median(idf[seen]) if seen.any() else 1.0
        self.idf = idf.astype(np.float32)
        self.prune = df > max_df_frac * n
        self.n_fit = n

    def transform(self, raw: sp.csr_matrix, copy: bool = False) -> sp.csr_matrix:
        """IDF-weight and L2-normalize rows (in place unless copy=True)."""
        m = raw.copy() if copy else raw
        m.data *= self.idf[m.indices]
        rows = np.repeat(np.arange(m.shape[0]), np.diff(m.indptr))
        norms = np.sqrt(np.bincount(rows, weights=m.data.astype(np.float64) ** 2, minlength=m.shape[0]))
        norms[norms == 0] = 1.0
        m.data /= norms[rows].astype(np.float32)
        return m


# ==============================================================================
# 5. RETRIEVAL CHANNELS
# ==============================================================================

def _topn_to_dense(C: sp.csr_matrix, k: int) -> Tuple[np.ndarray, np.ndarray]:
    nq = C.shape[0]
    I = np.full((nq, k), -1, dtype=np.int32)
    D = np.zeros((nq, k), dtype=np.float32)
    counts = np.diff(C.indptr)
    rows = np.repeat(np.arange(nq), counts)
    rank = np.arange(len(C.indices)) - C.indptr[rows]
    I[rows, rank] = C.indices
    D[rows, rank] = C.data
    return I, D


class SparseIndex:
    """Cosine top-k over hashed TF-IDF rows of Source 1.

    Features too frequent in training (IdfModel.prune) are dropped from the index only;
    they carry little evidence and dominate the matmul cost. The input matrix is
    consumed (pruned in place) to avoid holding a second copy.
    """

    def __init__(self, index_mat: sp.csr_matrix, prune: np.ndarray, n_threads: int):
        # Transposed copy (feature x record); rows of pruned features are emptied.
        # The input matrix is left intact (stage 1 needs exact cosines).
        it = index_mat.T.tocsr()
        idx_t = np.int32 if it.nnz < 2**31 else np.int64
        it.indptr = it.indptr.astype(idx_t, copy=False)
        it.indices = it.indices.astype(idx_t, copy=False)
        it.data[np.repeat(prune[:it.shape[0]], np.diff(it.indptr))] = 0.0
        it.eliminate_zeros()
        self.it = it
        self.n_threads = n_threads

    def search(self, queries: sp.csr_matrix, k: int, chunk: int = 50_000) -> Tuple[np.ndarray, np.ndarray]:
        from sparse_dot_topn import sp_matmul_topn
        out_I = np.full((queries.shape[0], k), -1, dtype=np.int32)
        out_D = np.zeros((queries.shape[0], k), dtype=np.float32)
        for s in range(0, queries.shape[0], chunk):
            block = queries[s:s + chunk]
            block.indices = block.indices.astype(np.int32)
            block.indptr = block.indptr.astype(np.int32)
            C = sp_matmul_topn(block, self.it, top_n=k, sort=True, n_threads=self.n_threads)
            out_I[s:s + chunk], out_D[s:s + chunk] = _topn_to_dense(C.tocsr(), k)
        return out_I, out_D


class HnswIndex:
    """HNSW over int8 scalar-quantized embeddings (4x smaller than float32), added in chunks."""

    def __init__(self, vecs: np.ndarray, n_threads: int, M: int = 32, ef_construction: int = 64,
                 train_size: int = 200_000, chunk: int = 100_000, seed: int = 0):
        import faiss
        faiss.omp_set_num_threads(n_threads)
        d = vecs.shape[1]
        self.index = faiss.IndexHNSWSQ(d, faiss.ScalarQuantizer.QT_8bit, M, faiss.METRIC_INNER_PRODUCT)
        self.index.hnsw.efConstruction = ef_construction
        rng = np.random.default_rng(seed)
        sample = rng.choice(len(vecs), size=min(train_size, len(vecs)), replace=False)
        self.index.train(np.ascontiguousarray(vecs[np.sort(sample)], dtype=np.float32))
        for s in range(0, len(vecs), chunk):
            self.index.add(np.ascontiguousarray(vecs[s:s + chunk], dtype=np.float32))

    def search(self, queries: np.ndarray, k: int, ef_search: int = 96, chunk: int = 100_000):
        self.index.hnsw.efSearch = max(ef_search, k)
        out_I = np.empty((len(queries), k), dtype=np.int32)
        out_D = np.empty((len(queries), k), dtype=np.float32)
        for s in range(0, len(queries), chunk):
            D, I = self.index.search(np.ascontiguousarray(queries[s:s + chunk], dtype=np.float32), k)
            out_I[s:s + chunk], out_D[s:s + chunk] = I, D
        return out_I, out_D


def unresolved(D: np.ndarray, tie_ratio: float) -> np.ndarray:
    """Rows whose k-th score is within tie_ratio of the best: the ranking cannot separate the
    candidates, so the true parent may sit just beyond k."""
    top, last = D[:, 0], D[:, -1]
    return (top > 0) & (last > 0) & (last >= tie_ratio * top)


def trim_memory():
    """Return freed heap pages to the OS (glibc); no-op elsewhere."""
    import gc
    gc.collect()
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


# ==============================================================================
# 6. LEARNED ENCODER (hashed feature bag -> 128-d, contrastive on training pairs)
# ==============================================================================

ENC_BITS = 18


def encoder_inputs(name_mat: sp.csr_matrix, addr_mat: sp.csr_matrix):
    """Row-wise concatenation of normalized name and address features, hashed to ENC_BITS."""
    m = sp.hstack([name_mat, addr_mat], format="csr")
    return m.indptr.astype(np.int64), (m.indices & ((1 << ENC_BITS) - 1)).astype(np.int32), \
        m.data.astype(np.float32)


def _gather(indptr, idx, w, rows):
    lens = indptr[rows + 1] - indptr[rows]
    offsets = np.zeros(len(rows), dtype=np.int32)
    np.cumsum(lens[:-1], out=offsets[1:])
    pos = np.repeat(indptr[rows] - offsets, lens) + np.arange(lens.sum())
    return idx[pos], offsets, w[pos]


def build_encoder(dim: int = 128):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.EmbeddingBag(1 << ENC_BITS, dim, mode="sum", sparse=True)
            nn.init.normal_(self.emb.weight, 0.0, 0.05)
            self.mlp = nn.Sequential(nn.Linear(dim, 2 * dim), nn.ReLU(), nn.Linear(2 * dim, dim))

        def forward(self, idx, offsets, w):
            h = self.emb(idx, offsets, per_sample_weights=w)
            return F.normalize(h + self.mlp(h), dim=-1)

    return Encoder()


def train_encoder(s_in, t_in, groups: np.ndarray, epochs: int = 2, batch: int = 4096,
                  temperature: float = 0.05, n_threads: int = 4, seed: int = 0, log=print):
    """InfoNCE over (S1, target) training pairs; batches drawn within one country (harder negatives)."""
    import torch
    import torch.nn.functional as F
    torch.manual_seed(seed)
    torch.set_num_threads(n_threads)
    model = build_encoder()
    opt_sparse = torch.optim.SparseAdam(list(model.emb.parameters()), lr=3e-3)
    opt_dense = torch.optim.Adam(model.mlp.parameters(), lr=1e-3)
    rng = np.random.default_rng(seed)
    n = len(groups)
    for ep in range(epochs):
        batches = []
        for g in np.unique(groups):
            rows = rng.permutation(np.flatnonzero(groups == g))
            batches += [rows[i:i + batch] for i in range(0, len(rows) - batch // 4, batch)]
        rng.shuffle(batches)
        t0, tot, steps = time.time(), 0.0, 0
        for rows in batches:
            a = [torch.from_numpy(x) for x in _gather(*s_in, rows)]
            b = [torch.from_numpy(x) for x in _gather(*t_in, rows)]
            ea, eb = model(*a), model(*b)
            logits = ea @ eb.T / temperature
            labels = torch.arange(len(rows))
            loss = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))
            opt_sparse.zero_grad()
            opt_dense.zero_grad()
            loss.backward()
            opt_sparse.step()
            opt_dense.step()
            tot += loss.item()
            steps += 1
        log(f"  [encoder] epoch {ep + 1}/{epochs}: {steps} batches over {n:,} pairs, "
            f"mean loss {tot / max(1, steps):.4f}, {time.time() - t0:.0f}s")
    model.eval()
    return model


def encode(model, inputs, batch: int = 32_768) -> np.ndarray:
    """Embeddings as float16 (half the memory; HNSW quantizes to int8 anyway)."""
    import torch
    indptr, idx, w = inputs
    n = len(indptr) - 1
    out = np.zeros((n, model.emb.embedding_dim), dtype=np.float16)
    with torch.no_grad():
        for s in range(0, n, batch):
            rows = np.arange(s, min(n, s + batch))
            out[rows] = model(*[torch.from_numpy(x) for x in _gather(indptr, idx, w, rows)]).numpy()
    return out


# ==============================================================================
# 7. ARTEFACT PERSISTENCE
# ==============================================================================

def save_artifacts(path: str, canon: Canonicalizer, idf_name: IdfModel, idf_addr: IdfModel, model):
    os.makedirs(path, exist_ok=True)
    canon._cache = {}
    with open(os.path.join(path, "canon.pkl"), "wb") as f:
        pickle.dump(canon, f)
    with open(os.path.join(path, "idf.pkl"), "wb") as f:
        pickle.dump({"name": idf_name, "addr": idf_addr}, f)
    if model is not None:
        import torch
        torch.save(model.state_dict(), os.path.join(path, "encoder.pt"))
