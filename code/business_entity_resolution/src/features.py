import math

import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist
from rapidfuzz.distance import JaroWinkler, Levenshtein

from blocking import KEY_CAPS

TEXT_COLS = ["name", "compact", "skel", "addr", "nums", "words", "num", "street"]
RANK_COLS = ["idf_jacc", "name_set", "skel_set", "addr_set", "compact_ratio", "compact_partial", "n_bits"] + [f"k_{k}" for k in KEY_CAPS]
PAIR_COLS = [
    "name_ratio", "name_sort", "name_partial", "name_jw", "idf_shared", "idf_missing1", "idf_missing2", "tok_diff",
    "name_len1", "name_len2", "addr_ratio", "num_eq", "num_jacc", "street_eq", "words_jacc", "addr1_empty", "addr2_empty", "is_s2",
    "hn_edit", "hn_join_edit", "hn_contains", "hn_logdiff",
]
CONTEXT_COLS = ["n_cand", "rank_i", "gap_i", "second_i", "n_s1", "rank_j", "gap_j", "second_j"]
MATCHER_KEYS = ["full", "compact", "skel", "tok_num", "skel_num", "num_street", "num_word", "name_word", "tok_street", "tok_pair"]
FEATURE_COLS = RANK_COLS[:7] + [f"k_{k}" for k in MATCHER_KEYS] + PAIR_COLS + CONTEXT_COLS
FEATURE_COLS = ["q"] + FEATURE_COLS


def token_idf(other):
    df = other.select(pl.col("name").str.split(" ").list.unique().alias("t")).explode("t").group_by("t").len()
    n = other.height + 1
    idf = {t: math.log(n / c) for t, c in df.iter_rows()}
    return idf, math.log(n)


def jacc(a, b):
    return len(a & b) / len(a | b) if a or b else 0.0


def join_text(pairs, s1, other):
    return pairs.join(s1.select(["i", "s1_id"] + TEXT_COLS), on="i").join(
        other.select(["i", "entity_id"] + TEXT_COLS).rename({"i": "j", "entity_id": "cand_id"}), on="j", suffix="_2"
    )


def idf_overlap(df, idf, default_idf):
    out = np.empty((df.height, 4), dtype=np.float32)
    for r, (n1, n2) in enumerate(df.select("name", "name_2").iter_rows()):
        t1, t2 = set(n1.split()), set(n2.split())
        w_shared = sum(idf.get(t, default_idf) for t in t1 & t2)
        w_1 = sum(idf.get(t, default_idf) for t in t1)
        w_2 = sum(idf.get(t, default_idf) for t in t2)
        out[r] = (w_shared, w_1, w_2, abs(len(t1) - len(t2)))
    return out


def pairwise(df, scorer, a, b, guard=False):
    x, y = df[a].to_list(), df[b].to_list()
    score = cpdist(x, y, scorer=scorer, workers=4, dtype=np.float32)
    if guard:
        empty = (df[a].str.len_chars() == 0) | (df[b].str.len_chars() == 0)
        score = np.where(empty.to_numpy(), np.float32(-1.0), score)
    return score


def rank_features(df, idf, default_idf):
    w = idf_overlap(df, idf, default_idf)
    union = w[:, 1] + w[:, 2] - w[:, 0]
    cols = [
        pl.Series("idf_jacc", np.divide(w[:, 0], union, out=np.zeros_like(union), where=union > 0)),
        pl.Series("name_set", pairwise(df, fuzz.token_set_ratio, "name", "name_2")),
        pl.Series("skel_set", pairwise(df, fuzz.token_set_ratio, "skel", "skel_2")),
        pl.Series("addr_set", pairwise(df, fuzz.token_set_ratio, "addr", "addr_2", guard=True)),
        pl.Series("compact_ratio", pairwise(df, fuzz.ratio, "compact", "compact_2")),
        pl.Series("compact_partial", pairwise(df, fuzz.partial_ratio, "compact", "compact_2")),
    ]
    bits = df["bits"].to_numpy().astype(np.int32)
    flags = [((bits >> b) & 1).astype(np.int8) for b in range(len(KEY_CAPS))]
    cols += [pl.Series("n_bits", sum(flags).astype(np.int8))] + [pl.Series(f"k_{k}", f) for k, f in zip(KEY_CAPS, flags)]
    return pl.DataFrame(cols)


def pair_features(df, idf, default_idf):
    w = idf_overlap(df, idf, default_idf)
    jac = np.empty((df.height, 2), dtype=np.float32)
    for r, (u1, u2, w1, w2) in enumerate(df.select("nums", "nums_2", "words", "words_2").iter_rows()):
        jac[r] = (
            jacc(set(u1.split()), set(u2.split())) if u1 and u2 else -1.0,
            jacc(set(w1.split()), set(w2.split())) if w1 and w2 else -1.0,
        )
    ntok1 = df["name"].str.split(" ").list.unique().list.len().to_numpy().astype(np.float32)
    ntok2 = df["name_2"].str.split(" ").list.unique().list.len().to_numpy().astype(np.float32)
    ntok1[(df["name"] == "").to_numpy()] = 0
    ntok2[(df["name_2"] == "").to_numpy()] = 0

    def missing(own):
        return np.divide(own - w[:, 0], own, out=np.zeros_like(own), where=own > 0)

    def same(a, b):
        both = (df[a] != "") & (df[b] != "")
        return np.where(both.to_numpy(), (df[a] == df[b]).to_numpy().astype(np.float32), np.float32(-1.0))

    cols = {
        "name_ratio": pairwise(df, fuzz.ratio, "name", "name_2"),
        "name_sort": pairwise(df, fuzz.token_sort_ratio, "name", "name_2"),
        "name_partial": pairwise(df, fuzz.partial_ratio, "name", "name_2"),
        "name_jw": pairwise(df, JaroWinkler.normalized_similarity, "name", "name_2"),
        "idf_shared": w[:, 0], "idf_missing1": missing(w[:, 1]), "idf_missing2": missing(w[:, 2]), "tok_diff": w[:, 3],
        "name_len1": ntok1, "name_len2": ntok2,
        "addr_ratio": pairwise(df, fuzz.ratio, "addr", "addr_2", guard=True),
        "num_eq": same("num", "num_2"), "num_jacc": jac[:, 0], "street_eq": same("street", "street_2"), "words_jacc": jac[:, 1],
        "addr1_empty": (df["addr"] == "").to_numpy().astype(np.float32), "addr2_empty": (df["addr_2"] == "").to_numpy().astype(np.float32),
        "is_s2": df["cand_id"].str.starts_with("S2-").to_numpy().astype(np.float32),
    }
    both = (pl.col("num") != "") & (pl.col("num_2") != "")
    joined = [pl.col(c).str.split(" ").list.head(2).list.join("").alias(c) for c in ("nums", "nums_2")]
    as_int = [pl.col(c).str.slice(0, 9).cast(pl.Int64, strict=False) for c in ("num", "num_2")]
    hn = df.select(
        pl.when(both & (pl.col("num").str.contains(pl.col("num_2"), literal=True) | pl.col("num_2").str.contains(pl.col("num"), literal=True)))
        .then(1.0).when(both).then(0.0).otherwise(-1.0).alias("contains"),
        pl.when(both).then((as_int[0] - as_int[1]).abs().log1p()).otherwise(-1.0).alias("logdiff"),
    )
    cols["hn_edit"] = pairwise(df, Levenshtein.distance, "num", "num_2", guard=True)
    cols["hn_join_edit"] = pairwise(df.select(joined), Levenshtein.distance, "nums", "nums_2", guard=True)
    cols["hn_contains"] = hn["contains"].to_numpy()
    cols["hn_logdiff"] = hn["logdiff"].to_numpy()
    return pl.DataFrame({c: np.asarray(cols[c], dtype=np.float32) for c in PAIR_COLS})


def context_features(slim):
    second = pl.col("q").sort(descending=True).slice(1, 1).first()
    return slim.with_columns(
        pl.len().over("i").alias("n_cand"),
        pl.col("q").rank("ordinal", descending=True).over("i").alias("rank_i"),
        (pl.col("q").max().over("i") - pl.col("q")).alias("gap_i"),
        second.over("i").fill_null(0).alias("second_i"),
        pl.len().over("j").alias("n_s1"),
        pl.col("q").rank("ordinal", descending=True).over("j").alias("rank_j"),
        (pl.col("q").max().over("j") - pl.col("q")).alias("gap_j"),
        second.over("j").fill_null(0).alias("second_j"),
    ).select(["i", "j"] + [pl.col(c).cast(pl.Float32) for c in CONTEXT_COLS])
