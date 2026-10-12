"""
Coordination detectors. Each turns posts into account-level suspicion scores.

Content co-posting (pairs of accounts posting "the same thing" within WINDOW):
  exact     identical normalised text            (CooRTweet-style co-tweet / co-retweet)
  url       identical link                       (CooRnet-style coordinated link sharing)
  tfidf     character 3-5-gram cosine >= TAU_TFIDF   (near-duplicate text)
  semantic  sentence-embedding cosine >= calibrated threshold   (coorsim-style)
Behaviour:
  timing    similarity of accounts' activity time series (no content at all)
Fusion:
  graph     Louvain communities on the union of all layers (label-free)
  fused     gradient boosting on all signals, trained on DIFFERENT campaigns

Thresholds are fixed in advance or calibrated on organic-only null pairs, never on
the test set. An account-pair edge needs at least MIN_REPEAT co-posts, as in the
coordination literature (one shared viral headline is not coordination).
"""
from collections import defaultdict
import numpy as np
import pandas as pd
import networkx as nx
from scipy.ndimage import gaussian_filter1d
from sklearn.feature_extraction.text import TfidfVectorizer
from common import norm_text, MIN

WINDOW = 15 * MIN
MIN_REPEAT = 2
TAU_TFIDF = 0.6
MIN_CHARS = 25        # very short generic replies ("thank you!") are not evidence
LAYERS = ["exact", "url", "tfidf", "semantic", "timing"]


# ---------------------------------------------------------------- pair machinery

def _pairs_by_key(df, key):
    """Account pairs that posted the same key within WINDOW; one count per key and pair."""
    w = defaultdict(int)
    d = df[df[key] != ""][["account", "ts", key]].sort_values([key, "ts"])
    for _, g in d.groupby(key, sort=False):
        if g.account.nunique() < 2:
            continue
        a, t = g.account.to_numpy(), g.ts.to_numpy()
        seen, j0 = set(), 0
        for i in range(len(t)):
            while t[i] - t[j0] > WINDOW:
                j0 += 1
            for j in range(j0, i):
                if a[i] != a[j]:
                    p = (a[i], a[j]) if a[i] < a[j] else (a[j], a[i])
                    if p not in seen:
                        seen.add(p); w[p] += 1
    return w


def _pairs_by_similarity(df, X, tau, dense=False):
    """Account pairs whose posts within WINDOW have similarity >= tau. X: one row per post."""
    order = np.argsort(df.ts.to_numpy())
    t, a = df.ts.to_numpy()[order], df.account.to_numpy()[order]
    X = X[order]
    b = (t // WINDOW).astype(int)
    starts = np.searchsorted(b, np.arange(b.max() + 2))
    w = defaultdict(int)
    for k in range(b.max() + 1):
        i0, i1, i2 = starts[k], starts[k + 1], starts[min(k + 2, len(starts) - 1)]
        if i1 == i0:
            continue
        S = X[i0:i1] @ X[i0:i2].T
        S = S if dense else S.toarray()
        dt = np.abs(t[i0:i1, None] - t[None, i0:i2])
        ii, jj = np.nonzero((S >= tau) & (dt <= WINDOW))
        ii, jj = ii + i0, jj + i0
        m = (jj > ii) & (a[ii] != a[jj])
        seen = set()
        for x, y in zip(a[ii[m]], a[jj[m]]):
            p = (x, y) if x < y else (y, x)
            if p not in seen:
                seen.add(p); w[p] += 1
    return w


def _scores(pairs, accts):
    s = pd.Series(0.0, index=accts)
    for (x, y), c in pairs.items():
        if c >= MIN_REPEAT:
            s[x] += c; s[y] += c
    return s


# ---------------------------------------------------------------- layers

def layer_exact(df, accts, **_):
    p = _pairs_by_key(df.assign(k=df.text.map(norm_text)), "k")
    return _scores(p, accts), p


def layer_url(df, accts, **_):
    p = _pairs_by_key(df, "url")
    return _scores(p, accts), p


def layer_tfidf(df, accts, **_):
    X = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True,
                        preprocessor=norm_text).fit_transform(df.text)
    p = _pairs_by_similarity(df, X.tocsr(), TAU_TFIDF)
    return _scores(p, accts), p


def calibrate_semantic(texts, sif, rng, q=0.999, n=200_000):
    """Threshold = q-quantile of cosine between random pairs of posts with DIFFERENT text
    (organic-only data), i.e. how similar unrelated messages get by chance."""
    u = pd.Series(texts).map(norm_text).drop_duplicates()
    u = u[u.str.len() >= MIN_CHARS]
    E = sif.transform(u.tolist())
    i, j = rng.integers(0, len(E), n), rng.integers(0, len(E), n)
    keep = i != j
    return float(np.quantile((E[i[keep]] * E[j[keep]]).sum(1), q))


def layer_semantic(df, accts, sif=None, tau=None, **_):
    E = sif.transform(df.text.tolist())
    p = _pairs_by_similarity(df, E, tau, dense=True)
    return _scores(p, accts), p


def activity_matrix(df, accts, bin_min=10, smooth=2.0):
    days = df.ts.max() / 86400
    nb = int(np.ceil(days * 1440 / bin_min)) + 1
    idx = pd.Series(np.arange(len(accts)), index=accts)
    M = np.zeros((len(accts), nb), np.float32)
    np.add.at(M, (idx[df.account].to_numpy(), (df.ts // (bin_min * MIN)).astype(int).to_numpy()), 1)
    return M, smooth


def timing_from_matrix(M, smooth, accts, k=10, min_posts=5):
    X = gaussian_filter1d(M, smooth, axis=1) if smooth else M.copy()
    active = M.sum(1) >= min_posts
    X[~active] = 0
    n = np.linalg.norm(X, axis=1, keepdims=True)
    X = X / np.where(n == 0, 1, n)
    S = X @ X.T
    np.fill_diagonal(S, 0)
    top = -np.sort(-S, axis=1)[:, :k]
    score = pd.Series(top.mean(1), index=accts)
    tau = np.quantile(S[np.triu_indices_from(S, 1)][::7], 0.999)   # edges: rare-similarity pairs
    pairs = {}
    rows, cols = np.nonzero(np.triu(S >= tau, 1))
    for r, c in zip(rows, cols):
        pairs[(accts[r], accts[c])] = 2                              # counts as a repeated tie
    return score, pairs


def layer_timing(df, accts, **_):
    M, sm = activity_matrix(df, accts)
    return timing_from_matrix(M, sm, accts)


LAYER_FN = dict(exact=layer_exact, url=layer_url, tfidf=layer_tfidf, semantic=layer_semantic, timing=layer_timing)


# ---------------------------------------------------------------- fusion

def graph_scores(pair_sets, accts, min_comm=5):
    G = nx.Graph()
    for name, pairs in pair_sets.items():
        if not pairs:
            continue
        mx = max(pairs.values())
        for (x, y), c in pairs.items():
            if c >= MIN_REPEAT:
                w = c / mx
                G.add_edge(x, y, weight=G[x][y]["weight"] + w if G.has_edge(x, y) else w)
    s = pd.Series(0.0, index=accts)
    comm_size = pd.Series(0, index=accts)
    if G.number_of_edges():
        for comm in nx.community.louvain_communities(G, weight="weight", seed=0):
            for n in comm:
                comm_size[n] = len(comm)
                if len(comm) >= min_comm:
                    s[n] = G.degree(n, weight="weight")
    return s, comm_size


def features(df, sif, tau_sem, layers=LAYERS):
    accts = np.array(sorted(df.account.unique()))
    content = df[df.text.map(norm_text).str.len() >= MIN_CHARS]
    out, pair_sets = {}, {}
    for name in layers:
        s, p = LAYER_FN[name](df if name == "timing" else content, accts, sif=sif, tau=tau_sem)
        out[name], pair_sets[name] = s, p
    out["graph"], csize = graph_scores(pair_sets, accts)
    F = pd.DataFrame(out)
    F["log_comm_size"] = np.log1p(csize)
    F["n_layers"] = (F[layers] > 0).sum(1)
    F["n_posts"] = np.log1p(df.account.value_counts().reindex(accts).to_numpy())
    return F
