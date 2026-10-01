import os

os.environ.setdefault("POLARS_MAX_THREADS", "6")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DATA_DIR = os.environ.get("BER_DATA", os.path.join(ROOT, "dataset"))
CACHE_DIR = os.environ.get("BER_CACHE", os.path.join(ROOT, "cache"))
OUT_DIR = os.environ.get("BER_OUT", os.path.join(ROOT, "output"))
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "model")
