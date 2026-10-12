"""
All detection experiments (part A). Writes results/*.json and features caches.

E0  pair level: can each similarity measure recognise an AI paraphrase as the same message?
E1  one adversary control at a time (paraphrase, timing jitter, URL obfuscation, cover traffic)
E2  escalation ladder: controls switched on one after another
E3  signal ablation of the fused detector under the strongest attack
Fused detector: gradient boosting trained on campaigns 0-3 under randomly drawn
adversary settings, tested on campaigns 4-7 (different content, different accounts).

    python src/experiments.py      (~30-40 min on 2 cores)
"""
import json, os, pickle, time
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from benchmark import Adversary, NAIVE, ADAPTIVE, simulate, accounts, _pools
from detectors import features, calibrate_semantic, LAYERS, TAU_TFIDF
from embed import SIF
from common import RES, norm_text

SEEDS = (1, 2, 3)
TRAIN_CAMPS, TEST_CAMPS = range(0, 4), range(4, 8)
DETECTORS = LAYERS + ["graph", "fused"]
FEATS = LAYERS + ["graph", "log_comm_size", "n_layers", "n_posts"]

LADDER = [("naive copy-paste", NAIVE),
          ("+ AI paraphrase & edits", Adversary(p_para=1, p_edit=1)),
          ("+ unique short links", Adversary(p_para=1, p_edit=1, p_url=1)),
          ("+ 60-min timing jitter", Adversary(p_para=1, p_edit=1, p_url=1, jitter=60)),
          ("+ cover traffic (2x)", Adversary(p_para=1, p_edit=1, p_url=1, jitter=60, cover=2)),
          ("+ 120-min jitter (full)", ADAPTIVE)]
SWEEPS = {"paraphrase": [Adversary(p_para=p, p_edit=p) for p in (0, .25, .5, .75, 1)],
          "jitter": [Adversary(jitter=j) for j in (3, 15, 30, 60, 120, 240)],
          "url": [Adversary(p_url=u) for u in (0, .5, 1)],
          "cover": [Adversary(cover=c) for c in (0, .5, 1, 2, 4)]}


def setup():
    chat, heads, camps = _pools()
    msgs = [m["seed"] for c in camps for m in c["messages"]] + [p for c in camps for m in c["messages"] for p in m["para"]]
    sif = SIF(chat + heads + msgs)
    org = simulate(NAIVE, campaigns=[], seed=99)
    tau = calibrate_semantic(org.text.tolist(), sif, np.random.default_rng(0))
    return sif, tau


def run_one(adv, seed, camps, sif, tau):
    df = simulate(adv, campaigns=camps, seed=seed)
    F = features(df, sif, tau)
    A = accounts(df).set_index("account").loc[F.index]
    F["bot"], F["campaign"] = A.bot.to_numpy(), A.campaign.to_numpy()
    return F


def recall_at(y, s, fpr=0.01):
    f, t, _ = roc_curve(y, s)
    return float(t[f <= fpr].max())


def evaluate(F, model, feats=FEATS):
    y = F.bot.to_numpy()
    out = {}
    for d in DETECTORS:
        s = model.predict_proba(F[feats])[:, 1] if d == "fused" else F[d].to_numpy()
        out[d] = dict(ap=float(average_precision_score(y, s)), r1=recall_at(y, s))
    return out


def e0_pairs(sif, tau):
    camps = _pools()[2]
    heads = _pools()[1]
    groups = [[m["seed"]] + m["para"] for c in camps for m in c["messages"]]
    texts = [t for g in groups for t in g]
    gid = np.repeat(np.arange(len(groups)), 5)
    tf = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1, sublinear_tf=True,
                         preprocessor=norm_text).fit(texts + heads[:5000])
    T = tf.transform(texts); E = sif.transform(texts)
    i, j = np.triu_indices(len(texts), 1)
    same = gid[i] == gid[j]
    S_tf = np.asarray((T @ T.T).todense())[i, j]
    S_se = (E @ E.T)[i, j]
    exact = np.array([norm_text(texts[a]) == norm_text(texts[b]) for a, b in zip(i, j)])
    res = {}
    for name, s, thr in (("exact", exact.astype(float), 0.5), ("tfidf", S_tf, TAU_TFIDF), ("semantic", S_se, tau)):
        res[name] = dict(auc=float(roc_auc_score(same, s)), recognised_at_threshold=float((s[same] >= thr).mean()),
                         false_match_rate=float((s[~same] >= thr).mean()), threshold=float(thr))
    res["n_positive_pairs"], res["n_negative_pairs"] = int(same.sum()), int((~same).sum())
    return res


def main():
    t0 = time.time()
    sif, tau = setup()
    print(f"semantic threshold (99.9% of unrelated pairs): {tau:.3f}")
    res = {"semantic_tau": tau, "e0_pairs": e0_pairs(sif, tau)}
    print("E0", json.dumps(res["e0_pairs"]))

    # ---- training data for the fused detector: campaigns 0-3, random adversaries
    rng = np.random.default_rng(42)
    train_advs = [Adversary(p_para=float(rng.choice([0, .5, 1])), p_edit=float(rng.choice([0, 1])),
                            jitter=float(rng.choice([3, 30, 60, 120])), p_url=float(rng.choice([0, 1])),
                            cover=float(rng.choice([0, 1, 2]))) for _ in range(12)]
    train = Parallel(n_jobs=2)(delayed(run_one)(a, 100 + k, TRAIN_CAMPS, sif, tau) for k, a in enumerate(train_advs))
    Tr = pd.concat(train)
    model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0).fit(Tr[FEATS], Tr.bot)
    print(f"trained fused detector on {len(Tr):,} accounts ({time.time()-t0:.0f}s)")

    # ---- test runs: campaigns 4-7
    jobs = [("ladder", name, adv) for name, adv in LADDER] + \
           [("sweep:" + k, adv.label(), adv) for k, advs in SWEEPS.items() for adv in advs]
    tasks = [(kind, name, adv, s) for kind, name, adv in jobs for s in SEEDS]
    outs = Parallel(n_jobs=2)(delayed(run_one)(adv, s, TEST_CAMPS, sif, tau) for _, _, adv, s in tasks)
    cache = {}
    rows = []
    for (kind, name, adv, s), F in zip(tasks, outs):
        cache[(kind, name, s)] = F
        for d, m in evaluate(F, model).items():
            rows.append(dict(experiment=kind, setting=name, seed=s, detector=d, **m, **{f"adv_{k}": v for k, v in adv.__dict__.items()}))
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(RES, "detection_results.csv"), index=False)

    # ---- E3 ablation of the fused detector under the full attack (retrain without each signal)
    abl = {}
    full = [cache[("ladder", LADDER[-1][0], s)] for s in SEEDS]
    mid = [cache[("ladder", LADDER[3][0], s)] for s in SEEDS]
    for drop in [None] + LAYERS:
        feats = [f for f in FEATS if f != drop]
        m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0).fit(Tr[feats], Tr.bot)
        sc = lambda Fs: float(np.mean([average_precision_score(F.bot, m.predict_proba(F[feats])[:, 1]) for F in Fs]))
        abl["all signals" if drop is None else f"without {drop}"] = dict(full_attack_ap=sc(full), jitter60_ap=sc(mid))
    res["e3_ablation"] = abl
    res["fused_feature_names"] = FEATS
    with open(os.path.join(RES, "feature_cache.pkl"), "wb") as f:
        pickle.dump(dict(cache=cache, train=Tr), f)
    pickle.dump(model, open(os.path.join(RES, "fused_model.pkl"), "wb"))
    json.dump(res, open(os.path.join(RES, "detection_summary.json"), "w"), indent=2)
    print(R[R.experiment == "ladder"].groupby(["setting", "detector"]).ap.mean().unstack().round(2).to_string())
    print("ablation", json.dumps(abl, indent=1))
    print(f"done in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
