"""
Part B: how much user data does coordination detection need?

Threat model
  Asset      the data a platform or researcher stores to run the detector.
  Adversary  anyone who later obtains that store: a breach, an insider, a subpoena,
             or reuse for unrelated profiling.
  Goal       minimise what such an adversary learns about ORDINARY users while keeping
             detection of coordinated accounts.
  Not covered: hiding who is coordinated (that is the purpose of detection), or
             attacks on the live platform itself.

Data tiers (each detector is retrained on its tier's data only)
  full        raw text, links, timestamps
  embeddings  no raw text: sentence vectors + hashed links + timestamps
              (data minimisation only; embeddings can leak content through inversion)
  hashed      salted hashes of text and links + timestamps: supports exact matching only
  timing      activity time series only (no content)
  timing-10min / -1h / -6h   activity time series at decreasing resolution
  timing-DP(eps)  event-level differential privacy: each account's hourly activity
             counts are released through the Laplace mechanism. One post changes one
             cell by 1 (L1 sensitivity 1), so Laplace noise with scale 1/eps gives
             eps-DP for every individual posting event; everything after the release
             (clipping, smoothing, similarity, classification) is post-processing.
             Cells below 1/eps are then zeroed (post-processing). The guarantee protects
             WHETHER/WHEN a single post happened, not an account's overall pattern,
             which is exactly what the detector needs.
Content tiers are evaluated with the fused detector retrained on the tier's signals;
timing tiers with the raw timing score, so they differ only in the data they see.

Only the timing channel gets a formal guarantee. The content tiers are data
minimisation, and the README says so.
"""
import json, os, pickle, time
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score
from benchmark import Adversary, simulate, accounts
from detectors import activity_matrix, timing_from_matrix, LAYERS
from experiments import LADDER, SEEDS, TRAIN_CAMPS, TEST_CAMPS, recall_at
from common import RES

CONTENT_TIERS = {"full": LAYERS, "embeddings": ["semantic", "url", "timing"],
                 "hashed": ["exact", "url", "timing"]}
EPSILONS = [0.5, 1.0, 2.0, 5.0, 10.0]
DP_BIN_MIN = 60             # coarse bins: fewer cells to noise (DP histograms of sparse data need this)
TIMING_TIERS = {"timing-10min": dict(), "timing-1h": dict(bin_min=60, smooth=1.0),
                "timing-6h": dict(bin_min=360, smooth=0.5),
                **{f"timing-DP(eps={e:g})": dict(bin_min=DP_BIN_MIN, smooth=1.0, eps=e) for e in EPSILONS}}
EVAL_LEVELS = [0, 2, 3, 5]          # indices into LADDER


def laplace_release(M, eps, rng):
    """Laplace mechanism on a count matrix with L1 sensitivity 1 per event."""
    return M + rng.laplace(0.0, 1.0 / eps, size=M.shape).astype(np.float32)


def timing_variant(adv, seed, camps, bin_min=10, smooth=2.0, eps=None):
    df = simulate(adv, campaigns=camps, seed=seed)
    accts = np.array(sorted(df.account.unique()))
    M, _ = activity_matrix(df, accts, bin_min=bin_min)
    if eps is not None:
        M = laplace_release(M, eps, np.random.default_rng(10_000 + seed))
        M = np.where(M >= 1.0 / eps, M, 0.0).astype(np.float32)   # post-processing: drop noise-level cells
    # min_posts uses the (noisy) released counts only: no raw data touches the detector
    s, _ = timing_from_matrix(M, smooth, accts, min_posts=5)
    A = accounts(df).set_index("account").loc[accts]
    return pd.DataFrame({"timing": s.to_numpy(), "n_posts": np.log1p(M.sum(1).clip(0)),
                         "bot": A.bot.to_numpy()}, index=accts)


def fit_eval(train, tests, feats):
    m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, random_state=0).fit(train[feats], train.bot)
    aps = [average_precision_score(F.bot, m.predict_proba(F[feats])[:, 1]) for F in tests]
    r1s = [recall_at(F.bot, m.predict_proba(F[feats])[:, 1]) for F in tests]
    return float(np.mean(aps)), float(np.mean(r1s))


def main():
    t0 = time.time()
    store = pickle.load(open(os.path.join(RES, "feature_cache.pkl"), "rb"))
    cache, Tr = store["cache"], store["train"]
    rows = []
    # content tiers: reuse cached layer scores, retrain on the tier's columns only
    for tier, layers in CONTENT_TIERS.items():
        tr = Tr.assign(n_layers=(Tr[layers] > 0).sum(axis=1))
        for li in EVAL_LEVELS:
            name = LADDER[li][0]
            tests = [cache[("ladder", name, s)].assign(n_layers=lambda F: (F[layers] > 0).sum(axis=1)) for s in SEEDS]
            ap, r1 = fit_eval(tr, tests, layers + ["n_layers", "n_posts"])
            rows.append(dict(tier=tier, level=name, ap=ap, r1=r1))
    # timing tiers: recompute the timing channel from coarsened or DP-released data
    for tier, kw in TIMING_TIERS.items():
        # unsupervised: the timing score itself, so tiers differ only in the data they see
        for li in EVAL_LEVELS:
            name, adv = LADDER[li]
            tests = Parallel(n_jobs=2)(delayed(timing_variant)(adv, s, TEST_CAMPS, **kw) for s in SEEDS)
            ap = float(np.mean([average_precision_score(F.bot, F.timing) for F in tests]))
            r1 = float(np.mean([recall_at(F.bot, F.timing) for F in tests]))
            rows.append(dict(tier=tier, level=name, ap=ap, r1=r1))
        print(f"  {tier} done ({time.time()-t0:.0f}s)")
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(RES, "privacy_results.csv"), index=False)
    print(R.pivot(index="tier", columns="level", values="ap").round(2).to_string())


if __name__ == "__main__":
    main()
