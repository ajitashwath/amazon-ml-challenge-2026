import gc
import os
import shutil
import subprocess
import sys
import time

import lightgbm as lgb
import polars as pl

from blocking import add_parts, block
from config import CACHE_DIR, MODEL_DIR
from features import RANK_COLS, TEXT_COLS, context_features, join_text, pair_features, rank_features, token_idf

TOP_K = 10
CHUNK = 25_000
ORPHAN_SHARE = 5
RANKER_PATH = os.path.join(MODEL_DIR, "ranker.txt")


def load_side(split, sources, country, columns=None):
    frames = [pl.scan_parquet(os.path.join(CACHE_DIR, f"{split}_s{s}.parquet")).filter(pl.col("country") == country) for s in sources]
    if columns:
        frames = [f.select(["country"] + columns) for f in frames]
    return add_parts(pl.concat(frames).collect().with_row_index("i"))


def countries(split):
    return sorted(pl.scan_parquet(os.path.join(CACHE_DIR, f"{split}_s1.parquet")).select("country").unique().collect()["country"].to_list())


def feature_dir(split, country, tag=""):
    return os.path.join(CACHE_DIR, f"feat{tag}_{split}_{country}")


def build_country(split, country, top_k=TOP_K, sample=None, dump_blocked=False):
    out_dir = feature_dir(split, country, "_blocked" if dump_blocked else "")
    shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir)
    t0 = time.time()
    s1 = load_side(split, [1], country)
    if split == "train":
        s1 = s1.filter(pl.col("entity_id").hash(seed=2) % ORPHAN_SHARE != 0).drop("i").with_row_index("i")
    if sample:
        s1 = s1.sample(min(sample, s1.height), seed=7).drop("i").with_row_index("i")
    other = load_side(split, [2, 3], country, ["name", "compact", "skel", "nums", "words", "street"])
    idf, default_idf = token_idf(other)
    pairs = block(s1, other).sort("i")
    other = load_side(split, [2, 3], country)
    n_s1 = s1.height
    print(f"{split} {country}: s1={n_s1} other={other.height} blocked={pairs.height} ({pairs.height / max(n_s1, 1):.1f}/s1) {time.time() - t0:.0f}s", flush=True)
    s1 = s1.select(["i", pl.col("entity_id").alias("s1_id")] + TEXT_COLS)
    other = other.select(["i", "entity_id"] + TEXT_COLS)
    s1.select("i", "s1_id").write_parquet(os.path.join(out_dir, "s1.parquet"))
    ranker = None if dump_blocked else lgb.Booster(model_file=RANKER_PATH)
    kept = 0
    for part, start in enumerate(range(0, n_s1, CHUNK)):
        df = join_text(pairs.filter(pl.col("i").is_between(start, start + CHUNK - 1)), s1, other)
        rank = rank_features(df, idf, default_idf)
        if dump_blocked:
            pl.concat([df.select("i", "j", "s1_id", "cand_id"), rank], how="horizontal").write_parquet(os.path.join(out_dir, f"part{part}.parquet"))
            continue
        q = ranker.predict(rank.to_numpy()) if df.height else []
        df = pl.concat([df, rank, pl.DataFrame({"q": q}, schema={"q": pl.Float32})], how="horizontal")
        df = df.filter(pl.col("q").rank("ordinal", descending=True).over("i") <= top_k)
        feats = pl.concat([df.select(["i", "j", "s1_id", "cand_id", "q"] + RANK_COLS), pair_features(df, idf, default_idf)], how="horizontal")
        feats.write_parquet(os.path.join(out_dir, f"part{part}.parquet"))
        kept += feats.height
        del df, rank, feats
        gc.collect()
    del pairs, s1, other
    gc.collect()
    if not dump_blocked:
        slim = pl.scan_parquet(os.path.join(out_dir, "part*.parquet")).select("i", "j", "q").collect()
        context_features(slim).write_parquet(os.path.join(out_dir, "context.parquet"))
        print(f"{split} {country}: candidates={kept} ({kept / max(n_s1, 1):.2f}/s1) done in {time.time() - t0:.0f}s", flush=True)


def build_all(split):
    for country in countries(split):
        env = dict(os.environ, MIMALLOC_PURGE_DELAY="0")
        subprocess.run([sys.executable, os.path.abspath(__file__), split, country], check=True, env=env)


def load_features(split, country, keep=None):
    d = feature_dir(split, country)
    feats = pl.scan_parquet(os.path.join(d, "part*.parquet"))
    if keep is not None:
        feats = feats.join(keep.lazy().select("s1_id"), on="s1_id", how="semi")
    ctx = pl.scan_parquet(os.path.join(d, "context.parquet"))
    return feats.join(ctx, on=["i", "j"]).collect()


def s1_ids(split, country):
    return pl.read_parquet(os.path.join(feature_dir(split, country), "s1.parquet"))


if __name__ == "__main__":
    split = sys.argv[1]
    for c in sys.argv[2:] or countries(split):
        build_country(split, c)
