# Amazon ML Challenge 2026 — Entity Resolution Conceptual Deep Dive
### From our 0.575 rule engine to a competitive ML system

*Audience: every member of the team, from "I just joined" to "I wrote the index."*
*Scope: concepts, math, and architecture. No code. Every number below is either measured from our own files or explicitly marked as an estimate or hypothesis.*

---

## How to Read This Guide

| If you want to... | Read |
| :--- | :--- |
| Understand what we are actually being asked to do | Section 1 |
| Know what the data looks like and how it lies to us | Section 2 |
| Understand why inference runs on AWS, not a laptop | Section 3 |
| See how our current pipeline works end to end | Section 4 |
| Understand *why* we are stuck at 0.575 | Section 5 |
| Decide what to build next | Section 6 |

**Sources of truth used for this guide:** `context/challenge_constraints.txt`, `context/eda_summary.txt`, `context/aws_setup.txt`, `context/unstop_rules.txt`, and the current `cloud_er_pipeline.py` (V4 engine).

---

## 1. The Real-World Business Problem

### 1.1 What Entity Resolution is

**Entity Resolution (ER)** answers one question, billions of times: *"Do these two records describe the same real-world thing?"*

> **Analogy — the school reunion.** You have the official class photo with everyone's printed name (the clean list). Then you get three hundred RSVP cards: "Bob Smith", "Robert J. Smith", "Bobby S., now in Denver", "R. Smith (née Jones)". Your job is to staple every RSVP card to the right face in the photo — and to leave a face bare if that person simply never replied.

For Amazon, the "faces" are businesses and the "RSVP cards" are records from sellers, carriers, registries and partner feeds. When ER fails, one real business looks like several (split reviews, duplicate listings, bypassed fraud rules, conflicting delivery instructions) or several businesses look like one (the far more dangerous error: wrong payouts, wrong tax records, wrong addresses).

### 1.2 Anchor vs. Target: the shape of *our* problem

Our task is **asymmetric record linkage**, not general deduplication.

```
                    ┌──────────────────────────────┐
                    │  SOURCE 1  —  the ANCHORS    │
                    │  Clean, already deduplicated │
                    │  One row = one real business │
                    └──────────────┬───────────────┘
                                   │  "Which of you belong to me?"
               ┌───────────────────┴───────────────────┐
               ▼                                       ▼
   ┌───────────────────────┐               ┌───────────────────────┐
   │ SOURCE 2 — TARGETS    │               │ SOURCE 3 — TARGETS    │
   │ Noisy, may contain    │               │ Noisy, may contain    │
   │ several copies of the │               │ several copies of the │
   │ same business         │               │ same business         │
   └───────────────────────┘               └───────────────────────┘
```

Three rules fall out of this shape:

1. **Anchors never merge with each other.** Two Source 1 rows are, by definition, two different businesses.
2. **One anchor can own many targets.** In training, a non-singleton anchor owns between 1 and 11 target records (median 4, mean 3.67), split roughly evenly between Source 2 (48%) and Source 3 (52%).
3. **We index the targets and query with the anchors.** ~10 million targets sit in memory; each of ~1.7–2.2 million anchors asks "who is mine?"

> **Analogy — the lost-and-found desk.** Source 1 is the list of people who filed a claim. Sources 2 and 3 are the bins of found items. You don't compare items with each other; you take each claim and search the bins for everything that belongs to that person.

### 1.3 The metric: Macro F0.5, explained without fear

For **each** Source 1 anchor we compute a score between 0 and 1, then take the plain average over all anchors. That is what "macro" means: a tiny, obscure business counts exactly as much as a chain with eleven records.

For one anchor:

- **Precision (P)** = of the targets we predicted, what fraction are correct?
- **Recall (R)** = of the targets that truly belong to it, what fraction did we find?
- **F0.5** = `1.25 × P × R / (0.25 × P + R)`

The "0.5" tilts the score toward precision: per the challenge rules, **precision is weighted 2× relative to recall**. False matches hurt more than missed ones — *but not infinitely more*, which matters later.

### 1.4 The singleton trap (and its mirror image)

A **singleton** is an anchor with **zero** true matches in Sources 2 and 3. The rules score them as all-or-nothing:

| Anchor type | We predict | Score |
| :--- | :--- | :---: |
| Singleton | nothing | **1.0** |
| Singleton | anything at all | **0.0** |
| Has true matches | nothing | **0.0** |
| Has true matches | at least one correct, no wrong | **≥ 0.625** (see table below) |

> **Analogy — the smoke detector.** A singleton is a room with no fire. A detector that stays quiet earns full marks; one false alarm and it earns nothing. But a room that *is* on fire, where the detector also stays quiet, earns nothing too.

**What the data says about the trap:** in training, singletons are **5.58%** of anchors (123,247 of 2,206,821). So:

- Predicting "empty" for everyone would score only **≈ 0.056**. Silence is *not* a safe strategy here.
- The 94.4% of anchors that do have matches are where almost all of the score lives.

**How the score grows with recall** (assuming every prediction is correct, P = 1, for an anchor with 4 true targets):

| Correct targets found (of 4) | Recall | Entity F0.5 |
| :---: | :---: | :---: |
| 0 | 0.00 | **0.000** |
| 1 | 0.25 | **0.625** |
| 2 | 0.50 | **0.833** |
| 3 | 0.75 | **0.938** |
| 4 | 1.00 | **1.000** |

**The single most valuable event in this metric** is moving a non-singleton anchor from "nothing found" to "one confident correct match": **+0.625 for that anchor**.

**What one wrong match costs** (same anchor, 4 true targets):

| Prediction | P | R | F0.5 |
| :--- | :---: | :---: | :---: |
| 1 correct | 1.00 | 0.25 | 0.625 |
| 1 correct + 1 wrong | 0.50 | 0.25 | 0.417 |
| 2 correct | 1.00 | 0.50 | 0.833 |
| 4 correct + 1 wrong | 0.80 | 1.00 | 0.833 |

Read that table twice. On a non-singleton anchor, adding one wrong match costs about as much (−0.21) as adding one right match earns (+0.21). "Precision-weighted" does **not** mean "be timid everywhere." It means *be timid on anchors that might be singletons, and be thorough on anchors that clearly have a match.*

---

## 2. The Data Ecosystem

### 2.1 Files and schemas

All files are **tab-separated (`.tsv`)**, never comma-separated — a disqualification-level rule.

| File | Rows | Countries | Missing address |
| :--- | ---: | :--- | ---: |
| `train_source1.tsv` (anchors) | 2,206,821 | US 60%, India 40% | 0.00% |
| `train_source2.tsv` (targets) | 5,034,616 | US 60%, India 40% | 3.36% |
| `train_source3.tsv` (targets) | 5,285,603 | US 60%, India 40% | 3.33% |
| `train_ground_truth.tsv` (labels) | 2,206,821 | — | — |
| `test_source1.tsv` (anchors) | 1,732,544 | India 47%, US 38%, **France 15%** | 0.00% |
| `test_source2.tsv` (targets) | 4,887,273 | India 47%, US 38%, **France 14%** | 2.65% |
| `test_source3.tsv` (targets) | 5,082,316 | India 47%, US 38%, **France 14%** | 2.68% |

**Source files** have exactly four columns:

```
entity_id   |  business_name   |  business_address          |  country
S1-925783039|  Orelee's Barbershop | 1795 Westchester Drive, High Point, NC | US
```

There is **no** phone, postcode, city, or tax-ID column. City, state and postcode live *inside* the free-text address, in inconsistent order.

**Ground truth and our submission** share one shape — one row per Source 1 anchor:

```
source1_entity_id  |  matched_entity_ids
S1-965667          |  S2-681193310,S2-743505751,S3-775321672,...
S1-xxxxxxx         |  (empty = singleton)
```

In training that is **7,638,365** true anchor→target links in total.

**Required deliverables** (from the rules):

- `output/matching_results.tsv` — our final answer; drives the leaderboard.
- `output/candidate_pairs.tsv` — every target we *considered* per anchor. Every ID in the matching file **must** also appear here. Think of it as "show your working."

### 2.2 How the data lies to us: a noise taxonomy from real rows

Every example below comes from actual training clusters in our EDA report.

| Noise pattern | Real example (anchor → target) | Why it breaks naive matching |
| :--- | :--- | :--- |
| Case and spacing | `Quality Good, Inc` → `Quality Good,  Inc` | Double spaces, SHOUTING |
| Injected accents | `Upper Select Digital LLC` → `Upper Sélect Digital LLC`, `QUALITY GÓOD` | `é ≠ e` byte-for-byte |
| Suffix formatting | `LLC` → `L.L.C.` | Same legal form, different characters |
| Character typos | `Digital` → `DITIL`, `Good` → `Gsd`, `Marez` → `Mlaz` | Letters dropped, swapped, substituted |
| Truncation | `Davisson and Marez Audio Installation LLC` → `Davisson` | Most of the name is simply gone |
| Extra words | → `Center Davisson and Marez Audio LLC` | A prefix pushes names apart |
| Street-type abbreviations | `Trail` → `TRL`, `Road` → `RD`, `Avenue` → `AVE` | Different tokens, same meaning |
| Number formatting | `1860 Warner Road` → `01860 WARNER RD` | Leading zeros |
| Number words / bad ordinals | `12th Avenue` → `TWELFTH AVENUE`, `12nd Ave` | Digits vs words; broken suffixes |
| Component reordering | `1860 Warner Road, Tempe, AZ` → `AZ, TEMPE, 01860 WARNER ROAD` | Order-sensitive comparisons fail |
| Null placeholders | `235 TWELFTH AVENUE, <NULL>, ANCHORAGE` | Literal `<NULL>` text in the field |
| Missing address | ~3% of targets have **no address at all** | Only the name is left to judge by |

Two further noise sources matter for the leaderboard even though the EDA sample does not show them:

- **Indic scripts and transliteration.** In a 300,000-row training sample, ~19,000 names/addresses contain Devanagari. Latin transliterations also drift (`Shree` / `Sri` / `Shri`, `Ganesh` / `Ganesha`). We have **not yet measured** whether true pairs cross scripts (Devanagari on one side, Latin on the other). If they do, no character-level trick can match them — that is a job for transliteration or a multilingual model.
- **The unseen French data.** France is **~15% of test anchors and 0% of training.** French brings accents (`é`, `è`, `ç`), ligatures (`œ`), elisions (`l'École`), legal forms (`SARL`, `SAS`, `EURL`, `SCI`), street types (`Rue`, `Bd`, `Av.`), unit markers (`12bis`, `12ter`), and five-digit postcodes. Critically: **anything we *learn* from training data has never seen a French example.** The rules forbid dropping or hard-coding countries, so French handling must *generalize*, not be special-cased.

> **Analogy — the language exam you didn't study for.** Training is a practice paper in English and Hindi-flavoured English. The real exam adds a French section. You can't memorise French answers; you can only learn *skills* (comparing spellings, matching numbers) that transfer.

---

## 3. The Infrastructure: Why AWS EC2 and Not a Laptop

### 3.1 The combinatorial wall

Comparing every test anchor with every test target means **1,732,544 × 9,969,589 ≈ 17.3 trillion comparisons.** At an optimistic 100,000 fuzzy comparisons per second, that is roughly **5.5 years** of CPU time. Blocking (Section 4, Stage 2) is what makes the problem possible at all. The second question is where that blocking index lives.

### 3.2 What we measured on the laptop

We ran the V4 pipeline on the full training set locally (16.8 GB RAM, 16 logical cores) and stopped it at 27% because it locked the machine. Measured numbers:

| Measurement | Value |
| :--- | :--- |
| Target records indexed | 10,320,219 |
| Distinct exact names | 5,362,050 |
| Discriminative tokens indexed | 1,477,481 (2,895 generic tokens pruned) |
| Index build time | **619 s (~10 min)** |
| Resident memory after build | **~5.2 GB** |
| Matching throughput | **685 – 1,030 anchors/s** (single process) |
| Projected full training run | **~40 – 55 min**, machine unusable meanwhile |

**Honest conclusion:** today's rule engine *fits* on a laptop. The reason to move to AWS is not the current pipeline — it is **the pipeline we need to build next.**

### 3.3 Why the *next* pipeline cannot live on a laptop

> **Analogy — kitchen counter space.** RAM is counter space: everything you're actively cooking must be on it. Disk is the pantry: big, but every trip costs time. When the counter overflows, the cook spends all their time walking to the pantry ("swapping") and dinner stops.

Memory math for the ML options in Section 6 (estimates):

| Component | Arithmetic | Approximate RAM |
| :--- | :--- | ---: |
| Current V4 index | measured | ~5.2 GB |
| Training pairs for a classifier | 2.2 M anchors × 30 candidates × 25 features × 4 bytes | ~6.6 GB |
| Dense embeddings of all targets (384-dim, float32) | 10.3 M × 384 × 4 bytes | **~15.9 GB** |
| Same, compressed to float16 | half of the above | ~7.9 GB |

Any of the ML routes pushes past a 16 GB laptop, where the OS starts paging to disk and throughput collapses by orders of magnitude.

Latency hierarchy, to see why "just use disk" doesn't work:

```
   CPU cache ............. ~1 ns        ████ the chef's hands
   RAM ................... ~100 ns      ████████ the counter
   Local NVMe SSD ........ ~100 µs      (≈1,000× RAM)  the pantry
   Network block storage . ~1 ms        (≈10,000× RAM) the warehouse across town
```

An inverted index is millions of random lookups. It must live in RAM.

### 3.4 How AWS fits together for us

```
   ┌──────────────────────────────────────────────┐
   │  S3 bucket (ap-south-1, Mumbai)              │   the warehouse:
   │  raw TSVs · submissions · model artifacts    │   cheap, durable, shared
   └──────────────────────┬───────────────────────┘   across teammates
                          │ same-region transfer (no cross-region fees)
                          ▼
   ┌──────────────────────────────────────────────┐
   │  EC2 instance (rented by the hour)           │   the factory:
   │  • memory-optimised for the index + features │   start it, run the job,
   │  • GPU instance only if we compute embeddings│   STOP it
   └──────────────────────┬───────────────────────┘
                          ▼
            matching_results.tsv + candidate_pairs.tsv  →  back to S3
```

Operating rules from `context/aws_setup.txt` and our constraints:

- **Stay in `ap-south-1`.** Cross-region transfer eats credits.
- **Credits are finite** ($100 base, up to $200 total, plus $100 for top-500 teams at 48 h). An idle instance burns money; stop it when a job finishes.
- **Memory-optimised instance families** (e.g. AWS "r" family) give the most RAM per credit for index-heavy work; **GPU instances** are only worth it for the embedding option.
- **SageMaker free-tier hours** (m4/m5.xlarge training) are an option for training a classifier, but training jobs can't be shared across accounts — only artifacts via S3.

### 3.5 Parallelism: what vCPUs can and can't do for us

A vCPU is one hardware thread. Sixteen vCPUs *could* process anchors ~16× faster — but in Python, each worker process that needs the 5 GB index either rebuilds it or shares it through fork, and Python's reference counting slowly copies shared pages. Two practical routes:

1. **Shard across machines/processes by anchor range.** Each worker loads the index once and handles 1/N of Source 1. Note: `cloud_er_pipeline.py` accepts `--chunk` / `--total-chunks` flags but **does not use them yet** — it always processes every row.
2. **Move the heavy index into shared, non-Python memory** (arrays rather than dicts of lists) so workers read it without copying.

---

## 4. The 4-Stage Resolution Pipeline

Every serious ER system is a **funnel**: cheap steps discard the impossible, expensive steps judge the plausible.

```
   10.3 M targets        per anchor: ~10 M possibilities
         │
   ┌─────▼──────────────────┐
   │ 1  NORMALISE           │  make equal things look equal
   └─────┬──────────────────┘
   ┌─────▼──────────────────┐
   │ 2  BLOCK / RETRIEVE    │  10 M  →  ≤ 30 candidates      ← decides RECALL CEILING
   └─────┬──────────────────┘
   ┌─────▼──────────────────┐
   │ 3  COMPARE (features)  │  describe each pair with numbers
   └─────┬──────────────────┘
   ┌─────▼──────────────────┐
   │ 4  DECIDE (threshold)  │  keep / reject / "singleton"   ← decides PRECISION
   └─────┬──────────────────┘
         ▼
   matching_results.tsv
```

> **Analogy — hiring.** Stage 1 is formatting every CV the same way. Stage 2 is the recruiter's keyword filter: 10 million CVs to 30. Stage 3 is the interview scorecard. Stage 4 is the hire / no-hire decision. **A brilliant interviewer cannot hire someone the keyword filter threw away.**

### Stage 1 — Text Normalisation *(current state, after this week's upgrade)*

Goal: two strings that mean the same thing should become the *same* string.

| Step | Example |
| :--- | :--- |
| Null placeholders removed | `235 TWELFTH AVENUE, <NULL>, ANCHORAGE` → `<NULL>` dropped |
| Accents folded to ASCII (Unicode NFKD + a table for letters NFKD can't split) | `Café` → `cafe`, `Straße` → `strasse`, `Œuvre` → `oeuvre`, `Łódź` → `lodz` |
| Hindi script preserved intact | `आरती ट्रेडर्स` keeps its vowel signs (earlier versions silently deleted them) |
| French elision | `l'École` → `ecole`, `d'Art` → `art` |
| Dotted acronyms collapsed | `S.A.R.L.` → `sarl`, `L.L.C.` → `llc`, `J.P.` → `jp` |
| Legal forms stripped (US, India, France, Germany) | `llc inc corp ltd pvt ltd private limited sarl sas sasu eurl sci sa gmbh ag kg …` |
| Safety net | a name made only of legal words (`SA`, `Services Inc`) keeps its words instead of becoming empty |
| Address abbreviations expanded | `rd → road`, `ave/av → avenue`, `blvd/bd → boulevard`, `r → rue` … |
| Street numbers extracted | `01860` → `1860`; `12B` → {`12`, `12b`}; `B1102` → {`1102`, `b1102`}; ordinals like `12th` / `1er` are skipped |

**Known gaps:** number words (`TWELFTH` ≠ `12`), transliteration variants (`Shree`/`Sri`), and cross-script pairs are not handled.

### Stage 2 — Blocking / Candidate Generation *(current state)*

> **Analogy — a book index.** Instead of reading all 10 million pages, look up a word in the index at the back and read only the pages it lists.

What V4 does:

1. **Exact-name lookup** — key = (country, normalised name). A hit adds a huge score.
2. **Name-token inverted index** — key = (country, word), for words of ≥ 3 letters that aren't stop words. Each shared word adds 1 point.
3. **Rarity pruning** — any word that appears in more than **800** target names per country is deleted from the index (2,895 words were pruned in training). Common words like "traders" or "services" can't flood the candidate list.
4. **Top-30 cut** — candidates are ranked by points; only the top 30 go forward.

What V4 does **not** do: it never uses the **address** to find candidates, and it never looks at **pieces of words** (character fragments), so a typo in every rare word makes a record invisible.

### Stage 3 — Feature Comparison *(current state)*

For each of the ≤ 30 candidates, V4 computes a handful of similarity scores:

- **Name:** token-sort ratio (word order ignored), token-set ratio (subset-friendly), plain character ratio.
- **Address:** token-sort and token-set ratios.
- **Street numbers:** do both addresses contain numbers, and do they share at least one?

> **Analogy — a detective's checklist.** Same name? Same street? Same house number? Each answer is a clue; none is proof alone.

### Stage 4 — Decision Thresholding *(current state)*

V4 accepts a candidate if it passes **any one of four hand-written tiers**:

| Tier | Rule of thumb |
| :--- | :--- |
| 1 · Exact name | Identical normalised names. If both have addresses: no number conflict, and the addresses are loosely similar. |
| 2 · Strong fuzzy name | Name sort ≥ 70 **and** set ≥ 85. If an address is missing, demand ≥ 88 on character and sort ratios. |
| 3 · Moderate name + strong address | Catches typos like `DITIL` / `DIGITAL` at the same location. |
| 4 · Near-identical location | Same number, very similar long address, even when names differ. |

Any candidate that fails all four tiers is rejected. If every candidate is rejected — or none was retrieved — the anchor is declared a **singleton**.

> **Analogy — a bouncer with a laminated rulebook.** Fast and consistent, but every edge case needs a new laminated page, and the pages start to contradict each other.

---

## 5. The Score Gap: 0.575 vs. 0.99+

*(The 0.99+ figure is the top of the leaderboard as reported to the team; we don't know what those teams actually built. The techniques below are the standard, well-documented ways to get there.)*

### 5.1 Where can 0.425 points of score hide?

Macro F0.5 fails in exactly three ways. Every lost point sits in one of these buckets:

| Bucket | What happens | Entity score |
| :--- | :--- | :---: |
| **A · Retrieval misses** | A true target never reaches the top-30 list | Caps recall; 0 if *all* are missed |
| **B · Over-strict decisions** | A true target is retrieved but rejected by the rules | Lowers recall |
| **C · False accepts** | A wrong target is accepted — fatal on singletons | Lowers precision; singleton → 0 |

**What our partial training run already tells us** (first ~600,000 anchors, before we stopped it):

- V4 predicted "no match" for **~11.7%** of anchors.
- Only **5.58%** of anchors are true singletons.
- So **at least ~6% of anchors** that *do* have matches got nothing — each scoring 0 instead of ≥ 0.625.

That accounts for only about 0.06 of the 0.425 gap. **The rest must come from low recall on anchors where we found *something*, from false accepts, or both.** As a hypothesis: if precision were near-perfect, an average F0.5 around 0.59 on the remaining anchors would mean we typically find only **~1 of ~4** true targets.

**We don't need to guess.** `evaluate_f05.py` (written this week) reports macro precision, recall, singleton false-match counts and forced-empty counts. One extra measurement would complete the picture: **blocking recall** — the share of true links that appear in `candidate_pairs.tsv` at all. If that number is low, no amount of threshold tuning can help; the funnel is the problem.

### 5.2 Why hand-written rules hit a ceiling

**1. Rules draw boxes; reality draws curves.**

```
   address        TRUE MATCHES really live here          our rules accept
   similarity            ___________                    only this box
      1.0 │         .-'  ●  ●  ●  ●  '-.                 ┌──────────┐
          │      .-'  ●  ●  ●  ●  ●  ●  '-.              │  ACCEPT  │
      0.6 │    /  ●  ●  ●  ○  ●  ●  ●  ●  \        ──────┼──────────┤
          │   │  ○  ○  ●  ○  ○  ●  ○   ○   │             │  REJECT  │
      0.0 └─────────────────────────────────           ──┴──────────┴──
         0.0       name similarity       1.0          0.0     0.85  1.0
```

A truncated name (`Davisson`) with an identical address *is* a match; a perfect name (`Subway`) with a different house number is *not*. A box with straight edges can't express "low name score is fine **if** the address is identical **and** the house number is rare."

**2. Every clue is treated as equally strong.** Sharing the word `ganesh` in India is weak evidence; sharing `zymurgy` is overwhelming. V4 prunes the most common words but otherwise gives every shared word 1 point.

**3. Weights are guessed, not learned.** Each tier has several thresholds (70, 85, 88, 65, 40, 60 …). With twenty interacting knobs, hand-tuning finds a local optimum at best — and we have **7.6 million labelled links** that could tune them for us.

**4. Decisions are per pair, not per anchor.** The metric scores *anchors*. The best decision for candidate #7 depends on what else we accepted for the same anchor (if we already found 3 strong matches, a borderline 4th is less valuable).

### 5.3 The three ideas behind top scores, in plain English

#### Idea 1 — Pairwise classifiers: "let the labels set the thresholds"

> **Analogy — a wine taster in training.** Instead of a rulebook ("tannin > 7 means Bordeaux"), the taster tastes 10,000 labelled glasses and develops judgement that weighs many signals at once.

1. For each training anchor, retrieve candidates exactly as at test time.
2. Label each (anchor, candidate) pair **1** if the ground truth links them, else **0**. The rejected-but-similar candidates are the valuable **hard negatives**.
3. Describe each pair with ~20–40 numeric **features**: all our current similarity ratios, number agreement, word rarity of shared tokens, length differences, missing-address flags, and so on.
4. Train a **gradient-boosted tree model** (LightGBM or XGBoost — both permissively licensed and CPU-friendly). It learns thousands of small, curved decision rules automatically.
5. Output: a **probability** that the pair is a match — not a yes/no.
6. Tune the acceptance threshold **directly against Macro F0.5** on held-out anchors.

**Why this suits the France problem:** the model sees *similarity numbers*, not raw words. "Name similarity 0.93, same house number" means the same thing in Paris as in Pune. That is transferable skill, not memorised vocabulary.

#### Idea 2 — TF-IDF: "rare evidence counts more"

> **Analogy — fingerprints vs. hair colour.** Two suspects with brown hair tells you little; matching fingerprints tells you almost everything. TF-IDF automatically works out which features are "fingerprints."

- **TF (term frequency):** how prominent a piece of text is within one record.
- **IDF (inverse document frequency):** how *rare* it is across all 10 million records. Rare → high weight.
- Applied to **character fragments** (e.g. `quality` → `qua`, `ual`, `ali`, `lit`, `ity`), typos only damage a few fragments: `Gsd` vs `Good` or `DITIL` vs `DIGITAL` still share most of their pieces.
- Each record becomes a sparse list of weighted fragments; similarity is a fast **cosine** score (the angle between two such lists).

This fixes two V4 weaknesses at once: retrieval that survives typos, and word rarity that is continuous rather than an 800-count cliff.

#### Idea 3 — Embeddings: "meaning as coordinates"

> **Analogy — a map of meaning.** Every name is placed on a map so that similar-meaning names land near each other. "Hospital" and "medical centre" become neighbours even though they share no letters.

- A small pretrained language model (e.g. a MiniLM-class sentence encoder, Apache-2.0, ~20–120 M parameters — far inside our **≤ 8 B, MIT/Apache** limit) turns each name+address into a list of ~384 numbers.
- **Approximate-nearest-neighbour search** (FAISS-style) finds the closest points among 10 million in milliseconds.
- **Multilingual** variants place French, English and Hindi text in the same space, which helps with French data and cross-script pairs.
- **Cost:** ~16 GB of vectors for our target pool, and a GPU to encode 10 million strings in reasonable time.
- **Caveat for our data:** most of our noise is typos, truncation and formatting rather than synonyms. Embeddings are a *complement* to character-level evidence, not a replacement.

**Rules and constraints to respect for all three:** everything runs offline, trained only on the provided data; **no** external APIs, geocoders, lookups or internet data.

---

## 6. Actionable Strategy & Brainstorming Roadmap

### Step 0 — Measure before we build (half a day)

Every option below attacks a different bucket from §5.1. Three numbers, computed on the training set on AWS, tell us which bucket is biggest:

| Diagnostic | Answers | Tool |
| :--- | :--- | :--- |
| **Blocking recall** — % of true links present in `candidate_pairs.tsv` | Is the funnel losing matches? (Bucket A) | small extension to `evaluate_f05.py` |
| **Decision recall** — of retrieved true links, % accepted | Are the rules too strict? (Bucket B) | same |
| **Macro precision** and **singleton false-match rate** | Are we accepting junk? (Bucket C) | `evaluate_f05.py` today |

**Validation discipline for everything that follows:**

- Split training data **by anchor**, never by pair, so one business's records never leak across train and validation.
- **Simulate the unseen country:** train on US only, validate on India (and the reverse). The drop we see is our best estimate of how badly we'll do on France.

### The three architectural options

```
                         ┌────────────────────────┐
                         │  V4 today  ·  0.575    │
                         └───────────┬────────────┘
             ┌───────────────────────┼───────────────────────┐
             ▼                       ▼                       ▼
   ┌───────────────────┐   ┌───────────────────┐   ┌───────────────────┐
   │ A · FIX THE       │   │ B · LEARN THE     │   │ C · HYBRID        │
   │     FUNNEL        │   │     DECISION      │   │     SEMANTIC      │
   │ recall-first      │   │ GBDT reranker on  │   │ + embeddings, ANN │
   │ multi-key blocking│   │ A's candidates    │   │ + anchor-level    │
   │ + TF-IDF, rules   │   │                   │   │   reasoning       │
   └───────────────────┘   └───────────────────┘   └───────────────────┘
      CPU only · low risk     CPU only · medium       GPU · high effort
```

A → B → C is a ladder, not a menu: **B needs A's candidates, and C extends B.**

---

### Option A — Fix the Funnel (recall-first retrieval, rules kept)

> **Analogy:** before hiring better interviewers, stop the keyword filter from binning good CVs.

**What changes**

- **Multiple, independent blocking keys**, with candidates pooled across them:
  - exact normalised name (as today);
  - rare name tokens, **weighted by rarity** instead of a hard 800 cut-off;
  - **character-fragment TF-IDF** on names, so typos still retrieve;
  - **address keys**: street number + first street word (e.g. `1860|warner`), and postcode-like tokens pulled from the address text.
- A larger candidate list (e.g. top 50–100) — `candidate_pairs.tsv` has no size penalty, only the final matches are scored.
- The V4 tiers stay as the decision layer, perhaps loosened only where Step 0 shows they're too strict.

| | |
| :--- | :--- |
| **Fixes** | Bucket A (retrieval misses); the ~6% forced empties |
| **Pros** | CPU-only; small, interpretable change; every later option depends on it anyway |
| **Cons** | Still hand-tuned decisions — Buckets B and C are untouched; more candidates may slow the rule layer |
| **Effort** | Low–medium (1–2 days) |
| **Success signal** | Blocking recall rises toward ~99%; forced-empty rate falls toward the true 5.6% singleton rate |

---

### Option B — Learn the Decision (supervised pairwise GBDT) · **Recommended target**

> **Analogy:** replace the bouncer's laminated rulebook with an experienced door manager who has seen millions of guests.

**What changes**

1. Run Option A's retrieval over all training anchors (~2.2 M anchors × up to 30–100 candidates).
2. Build a feature vector for each pair: name ratios (sort / set / plain / fragment-cosine), address ratios, number agreement (shared / conflicting / missing), shared-token rarity, length ratios, missing-field flags, candidate rank, and **anchor-level context** (how many strong candidates this anchor has).
3. Train LightGBM on the pairs using the ground-truth labels.
4. Decide per anchor with **two learned thresholds**:
   - a **match threshold** — accept candidates above probability *t*;
   - a **singleton gate** — if even the best candidate is below *t₀*, predict empty.
   Both are tuned on validation anchors to maximise Macro F0.5 directly.
5. Only language-neutral features, so the model transfers to France.

| | |
| :--- | :--- |
| **Fixes** | Buckets B and C; replaces ~20 hand-set thresholds with learned, calibrated ones |
| **Pros** | CPU-only (a memory-optimised EC2 instance); fast to train; feature importances explain the model; directly optimises our metric |
| **Cons** | Needs a feature pipeline shared by training and inference; ~7 GB+ of pair features in RAM; risk of overfitting to US/India patterns (mitigated by the country-holdout check) |
| **Effort** | Medium (2–4 days on top of A) |
| **Success signal** | Clear validation gain over V4 on the same candidates, holding up under the US→India holdout |

---

### Option C — Hybrid Semantic (embeddings + GBDT + anchor-level reasoning)

> **Analogy:** the door manager now also carries a multilingual phrasebook and knows which guests arrived together.

**What changes**

- **Dense retrieval:** a multilingual sentence encoder embeds all names+addresses; ANN search adds candidates that neither spelling nor address keys found (synonyms, reordered or partially translated names, possibly cross-script pairs).
- **Richer features:** embedding cosine added to Option B's feature vector.
- **Anchor-level reasoning:** use the fact that the true targets of one anchor tend to resemble *each other*. If candidates X and Y are near-duplicates and X is a confident match, Y's evidence strengthens. This is a *cautious* form of graph clustering: anchors are never merged, and propagation needs strong edges, because one bad link spreads false matches and the metric punishes that hard.

| | |
| :--- | :--- |
| **Fixes** | Residual Bucket A (semantic and cross-script misses) and residual Bucket B |
| **Pros** | Highest ceiling; strongest story for French and Indic data |
| **Cons** | GPU instance and ~8–16 GB of vectors; many moving parts (encoder, ANN index, clustering); must verify every model licence is MIT/Apache and ≤ 8 B parameters |
| **Effort** | High (4–6 days on top of B) |
| **Success signal** | Gains concentrated in the India→US holdout and in anchors with no exact or fragment match |

---

### Side-by-side

| | V4 today | A · Funnel | B · GBDT | C · Hybrid |
| :--- | :---: | :---: | :---: | :---: |
| Main bucket attacked | — | A (retrieval) | B + C (decisions) | residual A + B |
| Decision logic | 4 hand tiers | 4 hand tiers | learned probabilities | learned + anchor context |
| Compute | CPU | CPU | CPU (high RAM) | GPU + CPU |
| RAM (estimate) | ~5 GB | ~8–12 GB | ~15–20 GB | ~25–35 GB |
| French/unseen robustness | rule-dependent | rule-dependent | good (neutral features) | best (multilingual) |
| Effort | done | 1–2 days | +2–4 days | +4–6 days |

*We deliberately don't print "expected leaderboard scores" per option. Any such number would be invented until Step 0 tells us which bucket holds the missing 0.425.*

### Recommended sequence

1. **Step 0 on AWS** — full training run, then `evaluate_f05.py` plus a blocking-recall measurement. *(Half a day.)*
2. **Option A** — widen and diversify blocking until blocking recall is near-complete. Re-measure.
3. **Option B** — train the LightGBM reranker with a singleton gate; tune both thresholds on held-out anchors; check the US↔India holdout.
4. **Option C only if** B's error analysis shows remaining misses are semantic or cross-script rather than typo/format.

### Brainstorming prompts for the next team session

- Which bucket (A, B or C) does the Step 0 report say is largest — and does that match our intuition?
- Should the singleton gate depend on anchor features (e.g. very common names, missing address)?
- Is there cheap signal we're ignoring — postcode-like numbers inside addresses, state/city tokens, target-to-target duplicates?
- How do we handle number words (`TWELFTH` → `12`) and Indic transliteration without hard-coding a country?
- What is our rollback plan if a new model scores worse on the leaderboard than on validation?

---

## Glossary

| Term | Plain meaning |
| :--- | :--- |
| **Anchor** | A Source 1 record; the "claim" we search on behalf of |
| **Target** | A Source 2 or 3 record; a "found item" that may belong to an anchor |
| **Singleton** | An anchor with no true targets; the correct answer is an empty list |
| **Blocking** | Cheaply shrinking 10 M possibilities to a short candidate list |
| **Inverted index** | Word → list of records containing it; the book index at the back |
| **Blocking recall** | Share of true links that survive blocking; the upper bound on final recall |
| **Hard negative** | A candidate that looks similar but is *not* a match; the most useful training example |
| **GBDT** | Gradient-boosted decision trees (LightGBM, XGBoost); many small trees voting |
| **TF-IDF** | Weighting that makes rare evidence count more than common evidence |
| **Embedding** | A list of numbers placing text on a "map of meaning" |
| **ANN** | Approximate nearest neighbour search; fast "who's closest on the map" |
| **Macro F0.5** | Per-anchor precision-leaning score, averaged equally over all anchors |

---

*Rewritten 2026-09-27 against the V4 pipeline, the EDA report and measured training-run statistics. The previous version is preserved as `Amazon_ER_Conceptual_Deep_Dive.prev.md`.*
