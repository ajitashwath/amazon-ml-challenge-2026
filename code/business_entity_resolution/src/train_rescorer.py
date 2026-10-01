import json
import os

import lightgbm as lgb
import polars as pl

from config import MODEL_DIR
from evaluate import macro_f05, truth_pairs
from predict import KEY, score
from rawtext import band_features, corruption_features, encode, encode_corruption, legal_forms
from candidates import countries, load_features
from train_model import predict, split_ids

CONFIG = {"accept": 0.97, "band_low": 0.05, "rescore": 0.70, "corruption_countries": ["India", "US"]}
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=300, feature_fraction=0.9, bagging_fraction=0.8,
              bagging_freq=1, num_threads=8, seed=7, verbose=-1)


def main():
    matcher = lgb.Booster(model_file=os.path.join(MODEL_DIR, "matcher.txt"))
    ids, val_ids, matcher_ids = split_ids()
    free = ids.join(val_ids, on="s1_id", how="anti").join(matcher_ids, on="s1_id", how="anti")
    print("rescorer training entities (unseen by the matcher, outside validation):", free.height, flush=True)
    legal = legal_forms("train")
    truth = truth_pairs(free).with_columns(pl.lit(1, dtype=pl.Int8).alias("y"))
    parts = []
    for country in countries("train"):
        df = load_features("train", country, free.filter(pl.col("country") == country))
        df = df.with_columns(pl.Series("p", predict(matcher, df)))
        band = df.filter(pl.col("p").is_between(CONFIG["band_low"], CONFIG["accept"], closed="left")).select(KEY + ["p"])
        band = corruption_features(band_features(band, "train", legal)).join(truth, on=KEY, how="left").with_columns(pl.col("y").fill_null(0))
        parts.append(band)
        print(country, "band pairs", band.height, flush=True)
    band = pl.concat(parts, how="diagonal")
    rescorer = {
        "base": lgb.train(PARAMS, lgb.Dataset(encode(band), band["y"].to_numpy()), num_boost_round=400),
        "corruption": lgb.train(PARAMS, lgb.Dataset(encode_corruption(band), band["y"].to_numpy()), num_boost_round=400),
    }
    rescorer["base"].save_model(os.path.join(MODEL_DIR, "rescorer.txt"))
    rescorer["corruption"].save_model(os.path.join(MODEL_DIR, "rescorer_corruption.txt"))
    with open(os.path.join(MODEL_DIR, "config.json"), "w") as fh:
        json.dump(CONFIG, fh, indent=2)

    val_truth = truth_pairs(val_ids)
    preds = []
    for country in countries("train"):
        _, m = score("train", country, CONFIG, matcher, rescorer, legal, val_ids.filter(pl.col("country") == country))
        preds.append(m)
        c_ids = val_ids.filter(pl.col("country") == country)
        r = macro_f05(m, val_truth.join(c_ids, on="s1_id", how="semi"), c_ids)
        print(country, {k: round(v, 4) for k, v in r.items() if v is not None}, flush=True)
    r = macro_f05(pl.concat(preds), val_truth, val_ids)
    print("validation", {k: round(v, 4) for k, v in r.items() if v is not None})


if __name__ == "__main__":
    main()
