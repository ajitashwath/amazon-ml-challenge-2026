# Business Entity Resolution

CPU only pipeline:

1. normalization of names and addresses (including Indic script transliteration and a phonetic skeleton);
2. multi-key bucketing (16 bucket keys, hash joins per country);
3. a small LightGBM ranker that keeps the top 10 candidates per Source 1 entity (this is `candidate_pairs.tsv`);
4. a LightGBM matcher that scores every candidate pair;
5. a raw-text re-scorer for the uncertain band, built on information that normalization discards (legal form, raw name markers, house-number change type, address extras, generic business words).

Final decision per pair: accept if matcher score >= 0.97; if the score is in [0.05, 0.97), accept if the re-scorer score >= 0.70; otherwise reject.

## Setup

Python 3.11, no GPU.

```bash
pip install -r requirements.txt
```

Data layout (override with environment variables if needed):

```
<root>/dataset/train/train_source{1,2,3}.tsv, train_ground_truth.tsv
<root>/dataset/test/test_source{1,2,3}.tsv
<root>/code/business_entity_resolution/src/...
```

`<root>` is three levels above `src/`. `BER_DATA` points at another dataset folder, `BER_CACHE` at the cache folder, `BER_OUT` at the output folder.

## Reproduce the submission (uses the shipped models)

From `code/business_entity_resolution/src/`:

```bash
python prepare.py test         # normalize the test sources into cache/
python predict.py              # candidates, matcher, re-scorer -> output/matching_results.tsv, output/candidate_pairs.tsv
```

This regenerates both output files from the shipped models.

## Retrain from scratch

```bash
python prepare.py train
python train_ranker.py         # bucketing on a train sample, candidate ranker -> model/ranker.txt
python train_model.py          # train candidates (20% of train Source 1 entities are held out of the universe so their records act as orphans, as on test), matcher -> model/matcher.txt
python train_rescorer.py       # re-scorer on train entities unseen by the matcher and outside validation -> model/rescorer.txt, model/config.json
```

Validation throughout is a fixed 20% of train Source 1 entities (hash split), never used for training any model.

Provenance of the shipped models: all four (`ranker.txt`, `matcher.txt`, `rescorer.txt`, `rescorer_corruption.txt`) and `config.json` were produced by the retrain commands above, in that order, in a single run. Validation (20% hold-out by hash) macro F0.5 is 0.9748 (India 0.9648, US 0.9815). The ranker trains on a sample of 80,000 Source 1 entities per country.

## Resources

- Peak RAM about 4 to 5 GB (bucketing of the largest country). Candidate generation runs one subprocess per country with `MIMALLOC_PURGE_DELAY=0`, and the bucket merge runs in Source 1 slices.
- Normalization uses a pool of 4 workers; everything else is a single process with polars joins.
- Run one script at a time.
- Runtime on a consumer CPU: normalization about 9 minutes per split, test candidates about 35 minutes, scoring a few minutes.

## Files

| file | role |
| --- | --- |
| `textnorm.py` | name and address normalization, Indic transliteration, phonetic skeleton, name un-gluing |
| `prepare.py` | reads the TSV files and writes normalized parquet caches |
| `blocking.py` | the 16 bucket keys and the hash join that produces candidate pairs |
| `features.py` | ranker features, pair features, house-number features, competition features |
| `candidates.py` | per country candidate generation: buckets, ranker, top 10, features |
| `rawtext.py` | legal forms from raw names, raw-text categories for the re-scorer |
| `train_ranker.py` | trains the candidate ranker |
| `train_model.py` | builds train candidates and trains the matcher |
| `train_rescorer.py` | trains the uncertain-band re-scorer and writes the decision config |
| `predict.py` | scores the test candidates and writes both output files |
| `evaluate.py` | ground truth loading and macro F0.5 |
