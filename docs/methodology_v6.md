# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** Hackathon Tensors  
**Team Members:** Rahul M. & Team  
**Submission Date:** September 27, 2026  

---

## 1. Executive Summary

V6 is a three-stage learned entity-resolution system: **Target→Source 1 retrieval**, a **learned Stage-1 candidate pruner**, and a **Stage-2 LightGBM matcher with one-parent-per-target decisions**. Every learned component (Indic transliteration, phonetic classes, abbreviation lexicon, noise tokens, IDF weights, a contrastive hashed-n-gram encoder, and both LightGBM models) is learned from the training data only. There are no hand-written matching rules and no test-set adaptation. The whole pipeline runs in under 5 GB of RAM on 8 GB AWS Free Tier instances.

On 220,756 held-out training anchors (a 10% entity split, searched against the full 2.2M-record Source 1 index):
- Retrieval recall rose from **0.64 (V5) to 0.9957**.
- The Stage-1 pruner cut candidates from 78 to at most 20 per target (1.4 on average after its probability floor), keeping recall at 0.9918–0.9924.
- A perfect classifier on these candidates would score macro F0.5 = **0.998**.

## 2. Methodology

### 2.1 Problem Analysis
- **Structure (verified on train ground truth):** every Source 2/3 record belongs to **at most one** Source 1 entity (0 of 7.64M matched targets appear under two). 26% of targets match nothing, and 5.6% of Source 1 entities are singletons.
- **V5 failure analysis:** V5 blocking recall was 0.64. Its misses broke down as: 61% cut by the per-anchor top-30 cap, 35% lost to document-frequency pruning, and 4% with no shared key.
- **Hardest noise types:**
  - Indic-script names: Devanagari, Gujarati, Telugu, Kannada, Gurmukhi and others.
  - Names replaced entirely, where only the address matches.
  - Digit typos in house numbers (9320→320, 1618→168).
  - Neighbouring-locality substitutions.
  - Empty addresses (3%) combined with very common names.
- **Unseen country (France):** all features are country- and language-agnostic, and country labels are only used as an open-set partition key.

### 2.2 Solution Strategy
**Approach:** learned multi-channel retrieval in the target→Source 1 direction, followed by a learned pruner and a learned matcher.  
**Core idea:** each target searches for its single parent in a deduplicated Source 1 index. That removes the per-anchor candidate cap by construction and turns the one-parent structure into a decision constraint.

## 3. Candidate Generation (Blocking)

- **Canonicalization, all learned from training pairs:**
  - **Transliteration:** Indic scripts are folded onto one canonical block using Unicode's parallel layout. A per-character EM alignment (context-tagged consonants, vowel signs) and a word dictionary map them to Latin, both learned from (Latin S1 name, Indic target name) training pairs.
  - **Phonetic key:** vowels, consonant classes and aspiration-h are derived from the learned transliteration table.
  - **Abbreviation lexicon:** learned by subsequence alignment of S1 vs target tokens (rd→road, pvt→private, texas→tx).
  - **Noise tokens:** tokens that are not preserved across matching pairs (dba, fka, smt, llc…).
- **Channels, per country, K = 10 each:**
  1. **Name:** char-3-gram, word, phonetic and initials TF-IDF, via sparse top-k matmul.
  2. **Address:** tokens, numbers with single-digit-deletion neighbourhoods, and number|street compounds.
  3. **Encoder:** a hashed-feature siamese encoder (128-d, InfoNCE on 1M training pairs), searched with HNSW over int8-quantized vectors.
- **Adaptive depth (K = 100):** used when a target has no address or no name, or its top-10 scores are tied (10th score ≥ 0.9 × best).
- **Stage-1 pruner:** LightGBM on 27 cheap features. These are exact name, address, encoder and raw-surface-name cosines for every pair, channel ranks, and each target's gap to its best score and rank on every signal. Each target keeps its top ≤20 candidates above a probability floor tuned on training data.
- **Validation, 220k held-out anchors:**
  - Union recall 0.9957 and pruned recall 0.9924 at top 20.
  - About 0.13% of links are provably unrecoverable: address-less targets in exact-tie groups of more than 20 identical Source 1 names.
  - Peak memory 4.9 GB.

## 4. Matching Model

### 4.1 Features Used
- **Stage-1 evidence:** 27 features plus the Stage-1 probability.
- **Target-side competition:** rank among the target's candidates, and margin to the next candidate.
- **Source 1-side competition:** candidate count, number of targets whose top choice is this S1, sum and max of Stage-1 probability, this pair's gap to the S1's best, and the target's rank within the S1.
- **RapidFuzz scores:** ratio, token sort, token set and partial ratio, and Jaro-Winkler on raw folded names; ratio and token set on canonical core names; exact-equality flags.
- **Address:** token-set and partial scores (raw and canonical), shared house numbers, number conflicts, near-miss digits, and emptiness and length flags.

### 4.2 Model Type
LightGBM binary classifier, trained on the train fold of a sampled training corpus (Source 1 entity folds: 80% train, 10% tune, 10% holdout).

### 4.3 Threshold Selection Method
- **One parent per target:** each target is linked only to its highest-scoring Source 1 candidate.
- **Acceptance threshold τ:** grid-searched to maximize macro F0.5 on the tune fold, which is disjoint from training. Singletons are handled natively: an entity with no accepted target is predicted empty.

## 5. Results & Error Analysis

### 5.1 Validation & Submission Metrics
| Measure | Value |
|---|---|
| Retrieval union recall (220,756 held-out anchors, full 2.2M index) | 0.9957 |
| Stage-1 pruned recall (top-20 / with probability floor) | 0.9924 / 0.9918 |
| Oracle macro F0.5 after pruning (perfect decisions) | 0.9978 |
| Stage-2 held-out macro F0.5 (6% training sample, entity-level holdout fold)* | 0.9883 (P 0.993, R 0.976, singletons correct 99.2%) |
| Test: Source 1 entities with ≥1 match | 1,651,175 / 1,732,544 (95.3%) |
| Test: accepted links (India / US / France) | 2,831,318 / 2,386,079 / 831,769 |
| Test: candidate pairs after Stage 1 (India / US / France) | 7.81M / 4.88M / 7.67M |
| Peak memory (any stage, swap disabled) | 4.6 GB |

\* This estimate is optimistic: the sampled corpus has fewer same-name competitors than the full test corpus. The official validator passes, including the ID-existence check.

### 5.2 Error Analysis
- **Remaining retrieval losses** are dominated by targets with an empty address and a very common name. Where more than 20 Source 1 records share that exact name, no evidence can separate them, so these links are unrecoverable by design.
- **Precision risks** come from same-name entities in different cities. The Source 1-side competition features and the one-parent constraint reduce them.

## 6. Conclusion
Flipping the retrieval direction and learning every normalization from training data removed V5's recall ceiling (0.80 → 0.998 oracle). The Stage-1 pruner makes an expressive Stage-2 model affordable within 8 GB.

## Appendix

### A. Code Artefacts & Structure
- `src/v6_retrieval.py`: canonicalization learning, transliteration EM, lexicons, hashed TF-IDF, sparse and HNSW indices, encoder.
- `src/stage1_pruner.py`: Stage-1 pair features, training and pruning.
- `src/v6_pipeline.py`: `fit`, `candidates` (streaming, sliceable across workers), `stage2`, `predict` (writes both output files).
- `src/diagnose_v6_retrieval.py`: Gate G1/G1b recall diagnostics on held-out anchors.

### B. Computational & Resource Efficiency
- **Hardware:** AWS m7i-flex.large (2 vCPU, 8 GB, swap disabled). Peak memory ≤ 5 GB.
- **Test candidate generation:** streamed in 50k-target chunks against a resident per-country index, and parallelized across 4 instances by hashing target ids.
