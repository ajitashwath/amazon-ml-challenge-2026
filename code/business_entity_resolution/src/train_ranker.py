import os

import lightgbm as lgb
import polars as pl

from candidates import RANKER_PATH, build_country, countries, feature_dir
from config import MODEL_DIR
from evaluate import add_labels, truth_pairs
from features import RANK_COLS

SAMPLE = 80_000


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    frames = []
    for country in countries("train"):
        build_country("train", country, sample=SAMPLE, dump_blocked=True)
        df = pl.read_parquet(os.path.join(feature_dir("train", country, "_blocked"), "part*.parquet"))
        truth = truth_pairs(pl.read_parquet(os.path.join(feature_dir("train", country, "_blocked"), "s1.parquet")))
        df = add_labels(df, truth).with_columns(pl.lit(truth.height).alias("n_truth"), pl.lit(country).alias("country"))
        frames.append(df)
    df = pl.concat(frames)
    is_val = (pl.col("s1_id").hash(seed=1) % 5) == 0
    tr, va = df.filter(~is_val), df.filter(is_val)
    params = dict(n_estimators=200, learning_rate=0.1, num_leaves=31, min_child_samples=50, subsample=0.5, subsample_freq=1, verbose=-1)
    model = lgb.LGBMClassifier(**params).fit(tr.select(RANK_COLS).to_numpy(), tr["y"].to_numpy())
    va = va.with_columns(pl.Series("q", model.predict_proba(va.select(RANK_COLS).to_numpy())[:, 1]))
    va = va.with_columns(pl.col("q").rank("ordinal", descending=True).over("s1_id").alias("r"))
    for country in va["country"].unique().sort():
        v = va.filter(pl.col("country") == country)
        n_val = v["s1_id"].n_unique()
        total = truth_pairs(v.select("s1_id").unique()).height
        print(f"{country}: blocked {v.height / n_val:.1f}/s1 recall {v['y'].sum() / total:.4f}")
        for k in (5, 8, 10, 12, 15, 20):
            s = v.filter(pl.col("r") <= k)
            print(f"  top{k:<3d} {s.height / n_val:5.2f}/s1 recall {s['y'].sum() / total:.4f}")
    model = lgb.LGBMClassifier(**params).fit(df.select(RANK_COLS).to_numpy(), df["y"].to_numpy())
    model.booster_.save_model(RANKER_PATH)
    print("saved", RANKER_PATH)


if __name__ == "__main__":
    main()
