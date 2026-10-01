import polars as pl

WORDS = pl.concat_list(pl.col("words").str.split(" ").list.head(6), pl.col("words").str.split(" ").list.tail(3)).list.unique(maintain_order=True)

SLICE = 1_000_000
MERGE_PARTS = 8
KEY_CAPS = {
    "full": 60,
    "compact": 60,
    "skel": 60,
    "tok_num": 30,
    "skel_num": 30,
    "num_street": 40,
    "num_word": 30,
    "name_word": 30,
    "tok_street": 30,
    "tok_pair": 30,
    "skel_nums": 30,
    "skel_word": 30,
    "compact_word": 30,
    "prefix_word": 30,
    "blank_tok": 30,
    "blank_skel": 30,
}


def add_parts(df):
    nums = pl.col("nums").str.split(" ")
    return df.drop("country").with_columns(
        nums.list.first().fill_null("").alias("num"),
        nums.list.head(2).list.join(" ").fill_null("").alias("num2"),
    )


def keys_for(df, kind, side="s1"):
    toks = pl.col("name").str.split(" ")
    if kind.startswith("blank") and side == "other":
        df = df.filter((pl.col("words") == "") & (pl.col("nums") == ""))
    if kind == "full":
        out = df.select("i", toks.list.sort().list.join(" ").alias("k"))
    elif kind == "compact":
        out = df.select("i", pl.col("compact").alias("k"))
    elif kind == "skel":
        out = df.select("i", pl.col("skel").str.split(" ").list.sort().list.join(" ").alias("k"))
    elif kind in ("tok_num", "skel_num"):
        src = toks if kind == "tok_num" else pl.col("skel").str.split(" ")
        out = df.filter(pl.col("num") != "").select("i", pl.col("num2").str.split(" ").alias("num"), src.alias("t"))
        out = out.explode("num").explode("t").filter(pl.col("t").str.len_chars() > 1)
        out = out.select("i", (pl.col("t") + "|" + pl.col("num")).alias("k"))
    elif kind == "num_street":
        out = df.filter((pl.col("num") != "") & (pl.col("street") != ""))
        out = out.select("i", (pl.col("num") + "|" + pl.col("street")).alias("k"))
    elif kind == "num_word":
        out = df.filter(pl.col("num") != "").select("i", pl.col("num2").str.split(" ").alias("num"), WORDS.alias("w"))
        out = out.explode("num").explode("w")
        out = out.filter(pl.col("w").str.len_chars() > 1).select("i", (pl.col("num") + "|" + pl.col("w")).alias("k"))
    elif kind == "name_word":
        out = df.select("i", toks.list.sort().list.join(" ").alias("n"), WORDS.alias("w")).explode("w")
        out = out.filter(pl.col("w").str.len_chars() > 1).select("i", (pl.col("n") + "|" + pl.col("w")).alias("k"))
    elif kind == "tok_street":
        out = df.filter(pl.col("street") != "").select("i", "street", toks.alias("t")).explode("t")
        out = out.filter(pl.col("t").str.len_chars() > 1).select("i", (pl.col("t") + "|" + pl.col("street")).alias("k"))
    elif kind in ("skel_word", "compact_word", "prefix_word"):
        n = {"skel_word": pl.col("skel").str.split(" ").list.sort().list.join(" "), "compact_word": pl.col("compact"),
             "prefix_word": pl.when(pl.col("compact").str.len_chars() >= 6).then(pl.col("compact").str.slice(0, 8)).otherwise(pl.lit(""))}[kind]
        out = df.select("i", n.alias("n"), WORDS.alias("w")).filter(pl.col("n") != "").explode("w")
        out = out.filter(pl.col("w").str.len_chars() > 1).select("i", (pl.col("n") + "|" + pl.col("w")).alias("k"))
    elif kind == "skel_nums":
        out = df.filter(pl.col("nums") != "").select(
            "i", pl.col("skel").str.split(" ").list.sort().list.join(" ").alias("n"), pl.col("nums").str.split(" ").list.head(6).alias("m")
        ).explode("m").filter(pl.col("m").str.len_chars() > 0).select("i", (pl.col("n") + "|" + pl.col("m")).alias("k"))
    elif kind == "blank_tok":
        out = df.select("i", toks.alias("t")).explode("t").filter(pl.col("t").str.len_chars() > 2).select("i", pl.col("t").alias("k"))
    elif kind == "blank_skel":
        out = df.select("i", pl.col("skel").str.split(" ").list.sort().list.join(" ").alias("k"))
    elif kind == "tok_pair":
        out = df.select("i", toks.list.head(4).alias("a")).filter(pl.col("a").list.len() >= 3)
        out = out.with_columns(pl.col("a").alias("b")).explode("a").explode("b").filter(pl.col("a") < pl.col("b"))
        out = out.select("i", (pl.col("a") + " " + pl.col("b")).alias("k"))
    out = out.filter(pl.col("k").str.len_chars() > 0)
    return out.select("i", pl.col("k").hash().alias("k")).unique()


def block(s1, other, kinds=None):
    hits = []
    for kind in kinds or KEY_CAPS:
        cap = KEY_CAPS[kind]
        bit = list(KEY_CAPS).index(kind)
        ko = pl.concat([keys_for(other.slice(start, SLICE), kind, "other") for start in range(0, other.height, SLICE)])
        small = ko.group_by("k").len().filter(pl.col("len") <= cap).select("k")
        ks = pl.concat([keys_for(s1.slice(start, SLICE), kind) for start in range(0, s1.height, SLICE)])
        ks = ks.join(small, on="k", how="semi")
        del small
        ko = ko.join(ks.select("k").unique(), on="k", how="semi")
        hit = ks.join(ko.rename({"i": "j"}), on="k")
        hit = hit.select((pl.col("i").cast(pl.UInt64) * 2**32 + pl.col("j").cast(pl.UInt64)).alias("p")).unique()
        hit = hit.with_columns(pl.lit(1 << bit, dtype=pl.UInt32).alias("bits"))
        del ko, ks
        hits.append(hit)
        del hit
    step = (s1.height // MERGE_PARTS + 1) * 2**32
    parts = []
    for lo in range(0, s1.height * 2**32 + 1, step):
        part = pl.concat([h.filter(pl.col("p").is_between(lo, lo + step - 1)) for h in hits])
        parts.append(part.group_by("p").agg(pl.col("bits").sum().cast(pl.UInt32)))
    del hits
    pairs = pl.concat(parts)
    return pairs.select((pl.col("p") // 2**32).cast(pl.UInt32).alias("i"), (pl.col("p") % 2**32).cast(pl.UInt32).alias("j"), "bits")
