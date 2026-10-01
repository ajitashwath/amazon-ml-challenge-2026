# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** Masala_Dosa_Extra_Ghee
**Team Members:** Ajit Ashwath and Shalini Kumari
**Submission Date:** 2026-10-01

---

## 1. Executive Summary

The final pipeline has five stages:

1. normalization of names and addresses;
2. multi-key bucketing per country;
3. a small LightGBM ranker that keeps 10 candidates per Source 1 entity;
4. a LightGBM matcher that scores each pair;
5. a raw-text re-scorer for pairs the matcher is unsure about.

The re-scorer uses information that normalization throws away, above all the **legal form** of the business. The dataset's look-alike records differ from the real business in exactly one attribute, most often a replaced or added legal form. Genuine noisy copies drop or reorder the legal form but never swap it. Everything is CPU only, with a peak RAM of 4 to 5 GB.

This package is a complete, from-scratch retrain of the pipeline in this folder (prepare, ranker, matcher, re-scorer, predict), run end to end on the provided data. Both output files and every model in `model/` were produced by the code in `src/`.

Validation (20% of train Source 1 entities, held out by hash; test-like setting with 20% of the remaining Source 1 entities removed so that their records act as orphans):

| metric | India | US | overall |
| --- | --- | --- | --- |
| candidates per Source 1 entity | 9.95 | 9.96 | 9.96 |
| candidate recall (top 10) | 0.963 | 0.982 | |
| matcher alone, best threshold 0.65 | 0.949 | 0.975 | 0.964 |
| **macro F0.5 (final decision rule)** | **0.965** | **0.982** | **0.975** |
| pair precision | 0.989 | 0.995 | 0.993 |
| pair recall | 0.927 | 0.957 | 0.945 |
| singleton accuracy | 0.967 | 0.982 | 0.976 |

France appears only in test and has no validation split; it uses the same models as the other countries.

---

## 2. Methodology

### 2.1 Problem Analysis

**Scale**
- Train: 2.2M Source 1 entities and about 10.3M Source 2/3 records.
- Test: 1.73M Source 1 entities (US, India, France) and about 10M Source 2/3 records.
- Country never differs within a true pair, so all work runs per country.
- 5.6% of Source 1 entities are singletons; the mean is 3.46 matches and the maximum 11.
- Every Source 2/3 record belongs to at most one Source 1 entity.

**Noise found by inspecting clusters**
- Legal suffix changes.
- Reordered name tokens and typos.
- Injected accents and digit look-alikes (`5mart`).
- Glued names (`joyfulconsultancycom`) and domain-style names.
- DBA / fka aliases.
- Names written in Indic scripts (Devanagari, Bengali, Gujarati, Gurmukhi, Tamil, Telugu, Kannada, Malayalam).
- House-number noise: leading zeros, `18BIS`, appended unit numbers.
- Empty or `None` addresses.

**How look-alike records are built (found through error analysis)**
- Each look-alike record copies a real entity and changes one attribute: the legal form (replaced uniformly at random among the country's forms, or added when the original had none), the house number, or one name token.
- Within the uncertain score band, a conflicting legal form is almost never a true match (0.0% to 0.7% depending on the pattern).
- The same name and legal form with a different house number or street is still 94% to 98% true.

**Test vs train**
- Test has about 23% more Source 2/3 records per Source 1 entity.
- Its candidate-universe statistics for India and US match validation closely.
- France appears only in test.

### 2.2 Solution Strategy

**Approach type:** bucketing, then a candidate ranker, then a pairwise matcher, then an uncertain-band re-scorer.

**Core ideas**
1. Normalization that makes noisy and transliterated variants produce the same bucket keys.
2. Many cheap bucket keys, unioned, each with a block-size cap.
3. A learned ranker that cuts the union to 10 candidates per entity with almost no recall loss.
4. Competition and house-number features in the matcher.
5. A re-scorer that uses raw-text signals, in particular the legal form, only for pairs the matcher is unsure about.

---

## 3. Candidate Generation (Blocking)

### 3.1 Normalization (`textnorm.py`)

**Names**
- Indic scripts are mapped onto the Devanagari layout (the Brahmic blocks are parallel) and transliterated.
- Accents are stripped.
- For DBA and similar markers, the trade name after the marker is kept.
- Domain names are reduced to their stem.
- A token with one digit and at least three letters has its digit read as a letter.
- Glued suffixes (`...private`, `...com`) and prefixes (`sri...`) are removed from long tokens.
- Digit-only tokens and repeated tokens are dropped.
- Legal forms (US, India, France) and honorifics are removed.
- Outputs:
  - the *core* name;
  - the *compact* name;
  - a phonetic *skeleton*: voicing pairs merged, soft c/g, tion/ture rules, vowels dropped, trailing s/j stripped.

**Addresses**
- Non-Latin tokens and filler values are removed.
- State names become codes.
- Digits are split from letters, and leading zeros are stripped.
- Abbreviations are expanded.
- Outputs: numbers, significant words, and the street word.

### 3.2 Bucket keys (`blocking.py`)

There are 16 keys, all hashed to u64 and joined with polars hash joins per country. A bucket larger than its cap (30 to 60 records) is dropped.

| key | content |
| --- | --- |
| full, compact, skel | name identity: sorted core tokens, compact name, sorted skeleton |
| tok_num, skel_num | name token or skeleton token + each of the first two address numbers |
| skel_nums | full skeleton + each of the first six numbers |
| num_street, num_word | number + street word, or number + each address word |
| name_word, skel_word, compact_word, prefix_word | name (or skeleton, compact name, or first 8 letters of it) + each address word (first 6 and last 3 words) |
| tok_street, tok_pair | name token + street word, or pairs of name tokens |
| blank_tok, blank_skel | index only empty-address records by name token or skeleton; this catches name-only records of common names without giant buckets |

The union holds about 42 to 70 records per Source 1 entity. Hits are packed into one u64 per pair and merged in Source 1 slices, which keeps peak RAM at 4 to 5 GB.

### 3.3 Candidate ranker

The ranker works on cheap vectorized features (rapidfuzz `cpdist`):
- IDF-weighted name overlap;
- token_set ratios of name, skeleton and address;
- compact ratio and compact partial ratio;
- a flag for each bucket key.

It is a 200-tree LightGBM, trained only on non-validation entities. It keeps the **top 10** candidates per entity; these form `candidate_pairs.tsv`. Candidate recall on validation is 0.963 for India and 0.982 for US (the original token blocking had 0.66).

---

## 4. Matching Model

### 4.1 Matcher

A LightGBM model (600 rounds, 63 leaves) with 48 features:
- the ranker score and ranker features;
- name similarities (ratio, token sort, partial, Jaro-Winkler, shared and missing IDF mass);
- address similarity and number/street/word overlap;
- house-number features: edit distance of the first numbers, edit distance of the first two numbers joined, containment, log numeric difference;
- source flag;
- competition features over the country's candidate lists: rank and score gap of the pair within its Source 1 entity, and within the set of entities competing for the same record.

### 4.2 Raw-text re-scorer (`rawtext.py`)

Applied only to pairs with a matcher score in [0.05, 0.97). Its inputs are the matcher score plus seven categorical signals computed from the raw records:

| signal | categories |
| --- | --- |
| legal | same, overlap, conflict, both none, Source 1 only, candidate only (legal forms read from raw names, including Indic scripts after transliteration) |
| name_edit | same tokens, tokens added, removed, substituted |
| raw_marker | junk prefix, alias marker, domain, honorific, all caps, plain |
| addr_extra | PO box or unit present on one side only |
| num_pattern | first numbers equal, prefix, suffix, same length, different length, missing |
| num_keep | Source 1 house number kept, partially kept, one digit changed, replaced, missing |
| generic | generic business words (services, group, club, amicale, ...): same, overlap, conflict, one side, none |

The re-scorer is trained on 812k train entities that were used neither for the matcher nor for validation, so the matcher scores it sees are unbiased.

### 4.3 Decision rule

- Accept a pair if the matcher score is 0.97 or higher.
- In [0.05, 0.97), accept it if the re-scorer score is 0.70 or higher.
- Otherwise reject it.

A global threshold probe (0.85 instead of 0.70) lowered the public score, so tightening all uncertain pairs is not the fix. The re-scorer is selective instead.

---

## 5. Results & Error Analysis

- **Validation macro F0.5:** 0.975 (India 0.965, US 0.982). The public leaderboard score of this exact retrain was not measured before packaging; earlier versions of the same design scored 0.944 to 0.963 on the public leaderboard, below their validation scores, so a similar gap should be expected.
- **Remaining error budget on validation** (F0.5 gained by fixing each category perfectly):
  - blocking misses +0.010;
  - matcher false negatives +0.012;
  - false merges +0.008;
  - singleton errors +0.002.
- **Rejected true pairs** are dominated by records with the same name and a blank or noisy address. They are indistinguishable by content from look-alikes, and the second-hop comparison against the entity's other records was tested and adds no separation.
- **Ruled out:** ID or row-order leakage, train/test record overlap, rules in per-entity match counts, and French legal-form noise (French look-alikes use the same uniform legal replacement as the US).
- **Tested and rejected because they did not transfer to the leaderboard:**
  - agreement features (candidates agreeing with each other); they inflated uncertain predictions on test and dropped the LB to 0.926;
  - orphan-simulated retraining of the full model;
  - an expected-F0.5 decision rule.

---

## 6. Conclusion

The largest gains came from three things:
- candidate generation that makes noisy and transliterated variants collide;
- a matcher that is kept simple, so that it transfers to test;
- recovering information that normalization discards, above all the legal form, which exposes how look-alike records are generated.

Every design change was accepted only if it improved held-out validation and did not inflate test-time predictions on the uncertain band.

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/src/`:
`prepare.py` -> `train_ranker.py` -> `train_model.py` -> `train_rescorer.py` -> `predict.py`. Shared modules: `textnorm.py`, `blocking.py`, `features.py`, `candidates.py`, `rawtext.py`, `evaluate.py`.

`python prepare.py test` followed by `python predict.py` regenerates both output files with the shipped models. The README has details. No pretrained or external models are used; everything is trained on the provided train data only.

### B. Additional Results

Test output: 9.95 candidates per Source 1 entity and 3.25 (India), 3.34 (US), 3.43 (France) matches per Source 1 entity. Both files have one row for each of the 1,732,544 test Source 1 entities.
