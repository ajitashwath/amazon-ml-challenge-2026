import os

import polars as pl

from config import DATA_DIR


def truth_pairs(s1_ids=None):
    gt = pl.read_csv(os.path.join(DATA_DIR, "train", "train_ground_truth.tsv"), separator="\t", infer_schema_length=0, quote_char=None)
    gt = gt.rename({"source1_entity_id": "s1_id"})
    if s1_ids is not None:
        gt = gt.join(s1_ids.select("s1_id"), on="s1_id", how="semi")
    gt = gt.with_columns(pl.col("matched_entity_ids").str.split(",").alias("cand_id")).explode("cand_id")
    return gt.filter(pl.col("cand_id").is_not_null() & (pl.col("cand_id") != "")).select("s1_id", "cand_id")


def add_labels(pairs, truth):
    return pairs.join(truth.with_columns(pl.lit(1, dtype=pl.Int8).alias("y")), on=["s1_id", "cand_id"], how="left").with_columns(pl.col("y").fill_null(0))


def macro_f05(pred, truth, s1_ids):
    tp = pred.join(truth, on=["s1_id", "cand_id"], how="semi").group_by("s1_id").len().rename({"len": "tp"})
    n_pred = pred.group_by("s1_id").len().rename({"len": "n_pred"})
    n_true = truth.group_by("s1_id").len().rename({"len": "n_true"})
    df = s1_ids.select("s1_id").join(tp, on="s1_id", how="left").join(n_pred, on="s1_id", how="left").join(n_true, on="s1_id", how="left").fill_null(0)
    p = pl.col("tp") / pl.col("n_pred")
    r = pl.col("tp") / pl.col("n_true")
    f = (
        pl.when((pl.col("n_pred") == 0) & (pl.col("n_true") == 0)).then(1.0)
        .when(pl.col("tp") == 0).then(0.0)
        .otherwise(1.25 * p * r / (0.25 * p + r))
    )
    df = df.with_columns(f.alias("f"))
    single = df.filter(pl.col("n_true") == 0)
    return {
        "f05": df["f"].mean(),
        "pair_precision": df["tp"].sum() / max(df["n_pred"].sum(), 1),
        "pair_recall": df["tp"].sum() / max(df["n_true"].sum(), 1),
        "singleton_acc": (single["n_pred"] == 0).mean() if single.height else None,
        "pred_per_s1": df["n_pred"].mean(),
        "empty_pred_rate": (df["n_pred"] == 0).mean(),
    }


def decide(scored, threshold, exclusive=True):
    picked = scored.filter(pl.col("p") >= threshold)
    if exclusive:
        picked = picked.filter(pl.col("p") == pl.col("p").max().over("cand_id"))
    return picked.select("s1_id", "cand_id")
