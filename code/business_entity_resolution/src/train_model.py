import os
import sys

import lightgbm as lgb
import numpy as np
import polars as pl

from candidates import build_all, countries, load_features, s1_ids
from config import MODEL_DIR
from evaluate import add_labels, decide, macro_f05, truth_pairs
from features import FEATURE_COLS

TRAIN_S1 = 600_000
THRESHOLDS = [round(0.30 + 0.05 * k, 2) for k in range(13)]


def predict(model, df):
    out = []
    for start in range(0, df.height, 1_000_000):
        out.append(model.predict(df.slice(start, 1_000_000).select(FEATURE_COLS).to_numpy()))
    return np.concatenate(out) if out else np.array([])


def split_ids():
    ids = pl.concat([s1_ids("train", c).with_columns(pl.lit(c).alias("country")) for c in countries("train")])
    is_val = (pl.col("s1_id").hash(seed=1) % 5) == 0
    val_ids = ids.filter(is_val)
    train_ids = ids.filter(~is_val)
    return ids, val_ids, train_ids.sample(min(TRAIN_S1, train_ids.height), seed=3)


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    if "--skip-build" not in sys.argv:
        build_all("train")
    ids, val_ids, train_ids = split_ids()
    truth = truth_pairs(ids)
    val_truth = truth.join(val_ids, on="s1_id", how="semi")
    tr = add_labels(pl.concat([load_features("train", c, train_ids) for c in countries("train")]), truth)
    va = add_labels(pl.concat([load_features("train", c, val_ids).with_columns(pl.lit(c).alias("country")) for c in countries("train")]), truth)
    for country in val_ids["country"].unique().sort():
        c_ids = val_ids.filter(pl.col("country") == country)
        c = va.filter(pl.col("country") == country)
        c_truth = val_truth.join(c_ids, on="s1_id", how="semi")
        sizes = c.group_by("s1_id").len()["len"]
        print(f"{country} val: s1={c_ids.height} cand/s1={c.height / c_ids.height:.2f} "
              f"p50={sizes.median()} p90={sizes.quantile(0.9)} max={sizes.max()} candidate recall={c['y'].sum() / c_truth.height:.4f}")
    print(f"train pairs={tr.height} pos={tr['y'].sum()} val pairs={va.height}", flush=True)
    dtrain = lgb.Dataset(tr.select(FEATURE_COLS).to_numpy(), tr["y"].to_numpy(), feature_name=FEATURE_COLS, free_raw_data=True)
    del tr
    params = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=100, feature_fraction=0.8,
                  bagging_fraction=0.7, bagging_freq=1, lambda_l2=1.0, num_threads=8, seed=7, verbose=-1)
    model = lgb.train(params, dtrain, num_boost_round=600)
    model.save_model(os.path.join(MODEL_DIR, "matcher.txt"))

    va = va.with_columns(pl.Series("p", predict(model, va)))
    best = None
    for t in THRESHOLDS:
        m = macro_f05(decide(va, t), val_truth, val_ids)
        print(f"t={t:.2f} F0.5={m['f05']:.4f} P={m['pair_precision']:.4f} R={m['pair_recall']:.4f} "
              f"single={m['singleton_acc']:.3f} pred/s1={m['pred_per_s1']:.2f} empty={m['empty_pred_rate']:.3f}")
        if best is None or m["f05"] > best[0] + 1e-4:
            best = (m["f05"], t)
    f, t = best
    for country in val_ids["country"].unique().sort():
        c_ids = val_ids.filter(pl.col("country") == country)
        m = macro_f05(decide(va.join(c_ids, on="s1_id", how="semi"), t), val_truth.join(c_ids, on="s1_id", how="semi"), c_ids)
        print(country, {k: round(v, 4) for k, v in m.items() if v is not None})
    imp = sorted(zip(FEATURE_COLS, model.feature_importance("gain")), key=lambda x: -x[1])
    print("top features:", [(n, int(g)) for n, g in imp[:15]])
    print("matcher alone: best threshold", t, "val F0.5", round(f, 4))


if __name__ == "__main__":
    main()
