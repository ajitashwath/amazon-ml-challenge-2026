import json
import os
import sys

import lightgbm as lgb
import polars as pl

from candidates import build_all, countries, load_features, s1_ids
from config import CACHE_DIR, MODEL_DIR, OUT_DIR
from rawtext import band_features, corruption_features, encode, encode_corruption, legal_forms
from train_model import predict

KEY = ["s1_id", "cand_id"]


def write_lists(path, header, all_ids, pairs):
    grouped = pairs.group_by("s1_id").agg(pl.col("cand_id").unique().sort().str.join(",").alias("ids"))
    out = all_ids.join(grouped, on="s1_id", how="left").with_columns(pl.col("ids").fill_null(""))
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(f"source1_entity_id\t{header}\n")
        for sid, lst in out.iter_rows():
            fh.write(f"{sid}\t{lst}\n")


def load_models():
    with open(os.path.join(MODEL_DIR, "config.json")) as fh:
        config = json.load(fh)
    matcher = lgb.Booster(model_file=os.path.join(MODEL_DIR, "matcher.txt"))
    rescorer = {
        "base": lgb.Booster(model_file=os.path.join(MODEL_DIR, "rescorer.txt")),
        "corruption": lgb.Booster(model_file=os.path.join(MODEL_DIR, "rescorer_corruption.txt")),
    }
    return config, matcher, rescorer


def score(split, country, config, matcher, rescorer, legal, keep=None):
    df = load_features(split, country, keep)
    df = df.with_columns(pl.Series("p", predict(matcher, df)))
    band = df.filter(pl.col("p").is_between(config["band_low"], config["accept"], closed="left")).select(KEY + ["p"])
    band = band_features(band, split, legal)
    if country in config["corruption_countries"]:
        band = corruption_features(band)
        band = band.with_columns(pl.Series("r", rescorer["corruption"].predict(encode_corruption(band))))
    else:
        band = band.with_columns(pl.Series("r", rescorer["base"].predict(encode(band))))
    sure = df.filter(pl.col("p") >= config["accept"]).select(KEY)
    matches = pl.concat([sure, band.filter(pl.col("r") >= config["rescore"]).select(KEY)]).unique()
    return df.select(KEY), matches


def main():
    config, matcher, rescorer = load_models()
    if "--skip-build" not in sys.argv:
        build_all("test")
    legal = legal_forms("test")
    cands, matches = [], []
    for country in countries("test"):
        c, m = score("test", country, config, matcher, rescorer, legal)
        cands.append(c)
        matches.append(m)
        n = s1_ids("test", country).height
        print(f"{country}: s1={n} candidates/s1={c.height / n:.2f} matches/s1={m.height / n:.3f}", flush=True)
    all_ids = pl.read_parquet(os.path.join(CACHE_DIR, "test_s1.parquet"), columns=["entity_id"]).rename({"entity_id": "s1_id"})
    os.makedirs(OUT_DIR, exist_ok=True)
    write_lists(os.path.join(OUT_DIR, "candidate_pairs.tsv"), "candidate_entity_ids", all_ids, pl.concat(cands))
    write_lists(os.path.join(OUT_DIR, "matching_results.tsv"), "matched_entity_ids", all_ids, pl.concat(matches))
    print("wrote", OUT_DIR)


if __name__ == "__main__":
    main()
