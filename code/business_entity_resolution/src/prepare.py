import os
import sys
from multiprocessing import Pool

import polars as pl

from config import DATA_DIR, CACHE_DIR
from textnorm import normalize_name, normalize_address

CHUNK = 200_000


def normalize_chunk(rows):
    out = []
    for eid, name, addr, country in rows:
        n = normalize_name(name)
        a = normalize_address(addr)
        out.append((eid, country or "", n[0], n[1], n[2], a[0], a[1], a[2], a[3]))
    return out


def prepare(split, source, pool):
    path = os.path.join(DATA_DIR, split, f"{split}_source{source}.tsv")
    out_path = os.path.join(CACHE_DIR, f"{split}_s{source}.parquet")
    if os.path.exists(out_path):
        return
    df = pl.read_csv(path, separator="\t", infer_schema_length=0, quote_char=None)
    df = df.select("entity_id", "business_name", "business_address", "country")
    cols = ["entity_id", "country", "name", "compact", "skel", "addr", "nums", "words", "street"]
    parts = []
    step = CHUNK * 4
    for start in range(0, df.height, step):
        chunks = [df.slice(i, CHUNK).rows() for i in range(start, min(start + step, df.height), CHUNK)]
        for res in pool.map(normalize_chunk, chunks):
            parts.append(pl.DataFrame(res, schema=cols, orient="row"))
    del df
    pl.concat(parts).write_parquet(out_path)
    print(split, source, "done", flush=True)


if __name__ == "__main__":
    os.makedirs(CACHE_DIR, exist_ok=True)
    splits = sys.argv[1:] or ["train", "test"]
    with Pool(4) as pool:
        for split in splits:
            for source in (1, 2, 3):
                prepare(split, source, pool)
