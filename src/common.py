"""Paths and small shared helpers."""
import os, re
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA, RES, FIG = (os.path.join(ROOT, d) for d in ("data", "results", "figures"))
for d in (DATA, RES, FIG):
    os.makedirs(d, exist_ok=True)

MIN = 60                    # seconds per minute
DAY = 24 * 60 * MIN

_URL = re.compile(r"https?://\S+")
_WS = re.compile(r"\s+")
_NONW = re.compile(r"[^a-z0-9# ]+")


def norm_text(t):
    """Normalisation used for exact-duplicate matching (as in CooRTweet-style tools)."""
    t = _URL.sub(" ", str(t).lower())
    return _WS.sub(" ", _NONW.sub(" ", t)).strip()


def read_lines(path):
    with open(path, encoding="utf8") as f:
        return [l.rstrip("\n").strip('"') for l in f if l.strip()]
