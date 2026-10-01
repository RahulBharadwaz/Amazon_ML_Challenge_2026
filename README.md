# Business Entity Resolution at Scale — Amazon ML Challenge 2026

Links 1.73M reference business records (Source 1) to 9.97M noisy records from two other sources, across India, the US, and **France (unseen in training)**. The metric is macro F0.5, and everything runs within **8 GB of RAM**.

> **Final submission:** [`submission.zip`](submission.zip) (validated with the official checker, including ID-existence checks)
> **Predictions:** [`output/matching_results.tsv`](output/), [`output/candidate_pairs.tsv`](output/)
> **Methodology:** [`docs/methodology_v6.md`](docs/methodology_v6.md)

## Architecture (V6)

```
S2/S3 target ──► learned canonicalization ──► 3 retrieval channels ──► Stage-1 pruner ──► Stage-2 matcher ──► one parent per target
                 (Indic transliteration EM,    name TF-IDF            LightGBM,           LightGBM, fuzzy +    threshold tuned
                  phonetics, lexicon,          address + fuzzy #s     ≤20 cands/target    competition feats    on a tune fold
                  noise tokens — train only)   hashed encoder + HNSW
```

- **Target → Source 1 retrieval.** Every S2/S3 record searches for its single parent. This removes V5's per-anchor candidate cap and uses a verified property of the data: no target belongs to two Source 1 records.
- **Everything learned from training data only.** No hand-written matching rules and no test-set adaptation.
- **Memory-bounded.** Targets are streamed in 50k chunks against a per-country index, with int8 HNSW and on-disk pair features. Peak memory is ≤ 5 GB, measured with swap disabled.

## Results

| Stage | V5 | V6 |
|---|---|---|
| Retrieval recall (220k held-out anchors) | 0.64 | **0.9957** |
| Candidate recall after pruning (top 20) | – | 0.9924 |
| Oracle macro F0.5 on candidates | 0.80 | **0.998** |
| Stage-2 macro F0.5 (6% training sample, held-out fold)* | 0.751 | 0.988 |
| Leaderboard | 0.745 | *pending* |

\* Optimistic: the sampled corpus has fewer same-name competitors than the test set. Gate reports are in [`results/v6/`](results/v6/).

## Quick start

```bash
pip install -r requirements.txt   # torch: pip install torch --index-url https://download.pytorch.org/whl/cpu

# Reproduce the submission from the shipped models (no retraining)
mkdir -p work/v6 && cp -r models/v6_submission work/v6/artifacts
cp models/v6_submission/stage2_* work/v6/
python src/v6_pipeline.py candidates --prefix test --data-dir dataset/test --work-dir work/v6
python src/v6_pipeline.py predict   --data-dir dataset/test --work-dir work/v6 --output-dir output
python scripts/validate_submission.py --matching output/matching_results.tsv \
       --candidate output/candidate_pairs.tsv --test-dir dataset/test

# Retrain from scratch
python src/v6_pipeline.py fit --data-dir dataset/train --work-dir work/v6   # artefacts + stage-1 pruner
python src/v6_pipeline.py prep --frac 0.06 --work-dir work/v6s && cp -r work/v6/artifacts work/v6s/
python src/v6_pipeline.py candidates --prefix train --data-dir dataset/train --frac 0.06 --work-dir work/v6s
python src/v6_pipeline.py stage2 --frac 0.06 --work-dir work/v6s              # Gate G2 + stage-2 model

# Retrieval / pruning diagnostics (Gates G1, G1b) on 10% held-out anchors
python src/diagnose_v6_retrieval.py --out-dir results/v6/new_run
```

To parallelize candidate generation across machines, use `candidates --country <C> --slice k --n-slices n`. Then run `predict --country <C>` per country and combine the results with `scripts/merge_submission.py`. `infra/ec2_session.sh` launches, runs and terminates the AWS workers.

## Repository layout

| Path | Contents |
|---|---|
| `src/` | V6 pipeline: `v6_pipeline.py` (fit / candidates / stage2 / predict), `v6_retrieval.py`, `stage1_pruner.py`, `diagnose_v6_retrieval.py` (G1/G1b diagnostics and shared I/O helpers) |
| `scripts/` | `merge_submission.py`, `validate_submission.py` (official), `evaluate_f05.py`, `analyze_stage1_misses.py` |
| `models/v6_submission/` | Exact artefacts behind the submission (canonicalizer, IDF, encoder, stage-1 and stage-2 LightGBM, config) |
| `output/` | Final test predictions |
| `results/v6/` | Gate summaries, logs, missed-link samples, and validation metrics |
| `docs/` | V6 methodology, conceptual deep dive, `challenge/` (problem statement, rules, EDA) |
| `infra/` | EC2 session tooling, IAM CloudFormation template, launch logs |
| `dataset/`, `work/`, `.secrets/` | Gitignored: raw data, regenerable intermediates, SSH key |
