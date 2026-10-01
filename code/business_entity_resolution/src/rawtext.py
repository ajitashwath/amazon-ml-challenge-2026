import os
import re

import polars as pl
from rapidfuzz.distance import Levenshtein

from config import CACHE_DIR, DATA_DIR
from textnorm import deva_to_latin, skeleton

FORMS = {
    "inc": "inc", "incorporated": "inc", "llc": "llc", "llp": "llp", "lp": "lp", "ltd": "ltd", "limited": "ltd", "pvt": "pvt",
    "private": "pvt", "corp": "corp", "corporation": "corp", "co": "co", "company": "co", "plc": "plc", "pc": "pc", "pllc": "pllc",
    "pa": "pa", "sarl": "sarl", "sas": "sas", "sasu": "sasu", "sa": "sa", "eurl": "eurl", "sci": "sci", "snc": "snc",
}
GENERIC = {
    "services", "service", "trading", "traders", "enterprises", "enterprise", "industries", "solutions", "center", "centre",
    "partners", "group", "international", "holdings", "associates", "consultants", "consultancy", "ventures", "global",
    "technologies", "systems", "products", "exports", "infra", "projects", "developers", "builders", "brothers", "sons",
    "plus", "overseas", "india", "agency", "store", "stores", "mart", "hub", "works", "network", "management", "resources",
    "club", "ecole", "amicale", "association", "societe", "comite", "fils", "freres",
}
VOCAB = {
    "legal": ["both none", "s1 only", "cand only", "same", "overlap", "conflict"],
    "name_edit": ["same", "added", "removed", "substituted"],
    "raw_marker": ["plain", "caps", "junk", "alias", "domain", "honorific"],
    "addr_extra": ["same", "pobox", "unit"],
    "num_pattern": ["missing", "equal", "prefix", "suffix", "samelen", "difflen"],
    "generic": ["none", "one side", "same", "overlap", "conflict"],
    "num_keep": ["no s1 number", "missing", "kept", "partial", "one digit", "replaced"],
}
CATS = list(VOCAB)


def raw_path(split, source):
    return os.path.join(DATA_DIR, split, f"{split}_source{source}.tsv")


def legal_forms(split):
    path = os.path.join(CACHE_DIR, f"legal_{split}.parquet")
    if os.path.exists(path):
        return pl.read_parquet(path)
    out = []
    for source in (1, 2, 3):
        df = pl.scan_csv(raw_path(split, source), separator="\t", infer_schema_length=0, quote_char=None).select("entity_id", "business_name").collect()
        names = []
        for n in df["business_name"].fill_null("").to_list():
            if any("ऀ" <= ch <= "ൿ" for ch in n):
                n = " ".join("private" if skeleton(w) in ("prpt", "prp") else "limited" if skeleton(w) in ("lmtt", "lmt") else w for w in deva_to_latin(n).split())
            names.append(n)
        t = pl.col("business_name").str.normalize("NFKD").str.replace_all(r"\p{M}", "").str.to_lowercase().str.replace_all(r"\.", "")
        df = df.with_columns(pl.Series("business_name", names)).select("entity_id", t.str.replace_all(r"[^a-z0-9]+", " ").str.strip_chars().str.split(" ").alias("t")).explode("t")
        df = df.with_columns(pl.col("t").replace_strict(FORMS, default=None).alias("f")).drop_nulls("f")
        out.append(df.group_by("entity_id").agg(pl.col("f").unique().sort().str.join(",").alias("legal")))
    legal = pl.concat(out)
    legal.write_parquet(path)
    return legal


def keep_kind(n1, l2):
    if not n1:
        return "no s1 number"
    if not l2:
        return "missing"
    if n1 in l2:
        return "kept"
    if any(len(x) >= 2 and (x in n1 or n1 in x) for x in l2):
        return "partial"
    if any(len(x) == len(n1) and sum(a != b for a, b in zip(x, n1)) == 1 for x in l2):
        return "one digit"
    return "replaced"


def band_features(pairs, split, legal):
    cols = ["entity_id", "name", "nums"]
    s1 = pl.read_parquet(os.path.join(CACHE_DIR, f"{split}_s1.parquet"), columns=cols).rename({"entity_id": "s1_id"}).join(pairs.select("s1_id").unique(), on="s1_id")
    need = pairs.select(pl.col("cand_id").alias("entity_id")).unique()
    other = pl.concat([pl.read_parquet(os.path.join(CACHE_DIR, f"{split}_s{s}.parquet"), columns=cols) for s in (2, 3)]).join(need, on="entity_id").rename({"entity_id": "cand_id"})
    raw = pl.concat([pl.scan_csv(raw_path(split, s), separator="\t", infer_schema_length=0, quote_char=None).join(need.lazy(), on="entity_id", how="semi").collect() for s in (2, 3)])
    raw = raw.select(pl.col("entity_id").alias("cand_id"), pl.col("business_name").fill_null("").alias("rn"), pl.col("business_address").fill_null("").alias("ra"))
    raw1 = pl.scan_csv(raw_path(split, 1), separator="\t", infer_schema_length=0, quote_char=None).join(pairs.select(pl.col("s1_id").alias("entity_id")).unique().lazy(), on="entity_id", how="semi").collect()
    raw1 = raw1.select(pl.col("entity_id").alias("s1_id"), pl.col("business_address").fill_null("").alias("ra1"))
    x = pairs.join(s1, on="s1_id").join(other, on="cand_id", suffix="_2").join(raw, on="cand_id").join(raw1, on="s1_id")
    x = x.join(legal.rename({"entity_id": "s1_id", "legal": "l1"}), on="s1_id", how="left").join(legal.rename({"entity_id": "cand_id", "legal": "l2"}), on="cand_id", how="left")
    x = x.with_columns(pl.col("l1").fill_null(""), pl.col("l2").fill_null(""))
    toks = lambda c: pl.col(c).str.split(" ")
    generic = pl.lit(sorted(GENERIC))
    x = x.with_columns(
        toks("name").list.set_difference(toks("name_2")).list.len().alias("miss"),
        toks("name_2").list.set_difference(toks("name")).list.len().alias("extra"),
        toks("nums").list.first().fill_null("").alias("n1"),
        toks("nums_2").list.first().fill_null("").alias("n2"),
        toks("name").list.set_intersection(generic).alias("g1"),
        toks("name_2").list.set_intersection(generic).alias("g2"),
    )
    overlap = pl.col("l1").str.split(",").list.set_intersection(pl.col("l2").str.split(",")).list.len() > 0
    pobox = r"(?i)\b(po box|pmb|p\.o\.)"
    unit = r"(?i)\b(unit|suite|ste|apt|#)"
    x = x.with_columns(
        pl.when((pl.col("l1") == "") & (pl.col("l2") == "")).then(pl.lit("both none")).when(pl.col("l2") == "").then(pl.lit("s1 only"))
        .when(pl.col("l1") == "").then(pl.lit("cand only")).when(pl.col("l1") == pl.col("l2")).then(pl.lit("same"))
        .when(overlap).then(pl.lit("overlap")).otherwise(pl.lit("conflict")).alias("legal"),
        pl.when((pl.col("miss") == 0) & (pl.col("extra") == 0)).then(pl.lit("same")).when(pl.col("miss") == 0).then(pl.lit("added"))
        .when(pl.col("extra") == 0).then(pl.lit("removed")).otherwise(pl.lit("substituted")).alias("name_edit"),
        pl.when(pl.col("rn").str.contains(r"^\s*(--|<<|\[|\(|-)")).then(pl.lit("junk"))
        .when(pl.col("rn").str.contains(r"(?i)\b(dba|d/b/a|fka|aka|trading as|doing business as)\b")).then(pl.lit("alias"))
        .when(pl.col("rn").str.contains(r"(?i)\.(com|in|fr|net|org)\b")).then(pl.lit("domain"))
        .when(pl.col("rn").str.contains(r"(?i)^(dr|sri|shri|smt|m/s|the|mr|ms)\b")).then(pl.lit("honorific"))
        .when(pl.col("rn") == pl.col("rn").str.to_uppercase()).then(pl.lit("caps")).otherwise(pl.lit("plain")).alias("raw_marker"),
        pl.when(pl.col("ra").str.contains(pobox) != pl.col("ra1").str.contains(pobox)).then(pl.lit("pobox"))
        .when(pl.col("ra").str.contains(unit) != pl.col("ra1").str.contains(unit)).then(pl.lit("unit")).otherwise(pl.lit("same")).alias("addr_extra"),
        pl.when((pl.col("n1") == "") | (pl.col("n2") == "")).then(pl.lit("missing")).when(pl.col("n1") == pl.col("n2")).then(pl.lit("equal"))
        .when(pl.col("n1").str.starts_with(pl.col("n2")) | pl.col("n2").str.starts_with(pl.col("n1"))).then(pl.lit("prefix"))
        .when(pl.col("n1").str.ends_with(pl.col("n2")) | pl.col("n2").str.ends_with(pl.col("n1"))).then(pl.lit("suffix"))
        .when(pl.col("n1").str.len_chars() == pl.col("n2").str.len_chars()).then(pl.lit("samelen")).otherwise(pl.lit("difflen")).alias("num_pattern"),
        pl.when((pl.col("g1").list.len() == 0) & (pl.col("g2").list.len() == 0)).then(pl.lit("none"))
        .when((pl.col("g1").list.len() == 0) | (pl.col("g2").list.len() == 0)).then(pl.lit("one side"))
        .when(pl.col("g1").list.sort() == pl.col("g2").list.sort()).then(pl.lit("same"))
        .when(pl.col("g1").list.set_intersection(pl.col("g2")).list.len() > 0).then(pl.lit("overlap")).otherwise(pl.lit("conflict")).alias("generic"),
    )
    kinds = [keep_kind(a, [t for t in b.split() if t]) for a, b in zip(x["n1"].to_list(), x["nums_2"].to_list())]
    return x.with_columns(pl.Series("num_keep", kinds))


def encode(x):
    return x.select(["p"] + [pl.col(c).replace_strict({v: i for i, v in enumerate(VOCAB[c])}, return_dtype=pl.Int8).alias(c) for c in CATS]).to_numpy()


CORRUPTION = ["ed_ins", "ed_del", "ed_rep", "reordered", "concat", "rep_token", "addr_survival", "cand_addr_toks", "ind_name", "ind_addr", "is_s2", "name_len_diff"]
INDIC = re.compile("[ऀ-ൿ]")
ADDR_TOK = re.compile(r"[a-z0-9]{3,}")


def corruption_features(x):
    rows = []
    for n1, n2, rn, ra, ra1, cid in x.select("name", "name_2", "rn", "ra", "ra1", "cand_id").iter_rows():
        ops = Levenshtein.editops(n1, n2)
        ins = sum(1 for o in ops if o.tag == "insert")
        dele = sum(1 for o in ops if o.tag == "delete")
        t1, t2 = n1.split(), n2.split()
        raw_toks = rn.lower().split()
        a1 = set(ADDR_TOK.findall(ra1.lower()))
        a2 = ADDR_TOK.findall(ra.lower())
        rows.append((
            ins, dele, len(ops) - ins - dele, 1.0 if t1 != t2 and sorted(t1) == sorted(t2) else 0.0, 1.0 if (" " in n1) != (" " in n2) else 0.0,
            1.0 if len(raw_toks) != len(set(raw_toks)) else 0.0, len(a1 & set(a2)) / len(a1) if a1 else -1.0, len(a2),
            1.0 if INDIC.search(rn) else 0.0, 1.0 if INDIC.search(ra) else 0.0, 1.0 if cid.startswith("S2") else 0.0, abs(len(n1) - len(n2)),
        ))
    return x.with_columns([pl.Series(c, [r[k] for r in rows], dtype=pl.Float32) for k, c in enumerate(CORRUPTION)])


def encode_corruption(x):
    return x.select(["p"] + [pl.col(c).replace_strict({v: i for i, v in enumerate(VOCAB[c])}, return_dtype=pl.Int8).alias(c) for c in CATS] + CORRUPTION).to_numpy()
