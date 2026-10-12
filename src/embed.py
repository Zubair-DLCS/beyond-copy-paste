"""
Sentence embeddings without a GPU or model hub: SIF-weighted GloVe-Twitter vectors
(Arora et al., 2017) with the first principal component removed. Weaker than the
transformer embeddings used by coorsim; this is stated as a limitation.
"""
import gzip, os, re, subprocess
from collections import Counter
import numpy as np
from common import DATA

GLOVE_URL = ("https://github.com/RaRe-Technologies/gensim-data/releases/download/"
             "glove-twitter-50/glove-twitter-50.gz")
GLOVE = os.path.join(DATA, "glove-twitter-50.gz")
TOK = re.compile(r"<\w+>|#?\w+(?:'\w+)?")


def tokens(t):
    t = re.sub(r"https?://\S+", " <url> ", str(t).lower())
    t = re.sub(r"#(\w+)", r" <hashtag> \1 ", t)
    return TOK.findall(t)


class SIF:
    def __init__(self, texts, a=1e-3):
        toks = [tokens(t) for t in texts]
        counts = Counter(w for ts in toks for w in ts)
        total = sum(counts.values())
        if not os.path.exists(GLOVE):
            subprocess.run(["curl", "-sL", "-o", GLOVE, GLOVE_URL], check=True)
        self.vec = {}
        with gzip.open(GLOVE, "rt", encoding="utf8", errors="ignore") as f:
            next(f)
            for line in f:
                w, rest = line.split(" ", 1)
                if w in counts:
                    self.vec[w] = np.array(rest.split(), dtype=np.float32)
        self.w = {w: a / (a + c / total) for w, c in counts.items() if w in self.vec}
        E = self._raw(toks)
        self.pc = np.linalg.svd(E[np.abs(E).sum(1) > 0][:20000], full_matrices=False)[2][0]

    def _raw(self, toks):
        E = np.zeros((len(toks), 50), np.float32)
        for i, ts in enumerate(toks):
            ws = [(self.w[t], self.vec[t]) for t in ts if t in self.w]
            if ws:
                wt = np.array([w for w, _ in ws])[:, None]
                E[i] = (wt * np.stack([v for _, v in ws])).sum(0) / wt.sum()
        return E

    def transform(self, texts):
        E = self._raw([tokens(t) for t in texts])
        E = E - np.outer(E @ self.pc, self.pc)                  # remove common component
        n = np.linalg.norm(E, axis=1, keepdims=True)
        return E / np.where(n == 0, 1, n)
