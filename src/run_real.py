"""
Run the detectors on a real dataset in the common schema.

Example for Cresci-2017 (request it from the MIB team or the Bot Repository; it must
not be redistributed). Merge the genuine accounts' tweets with one social-spambot
group, add a `campaign` column (-1 for genuine, 0 for the spambot group), then:

    python src/run_real.py data/cresci_merged.csv \
        --account user_id --ts timestamp --text text --url expanded_url --campaign campaign

Outputs per-detector average precision and recall at 1% FPR, using the fused model
trained on the benchmark (no retraining on the real data).
"""
import argparse, json, os, pickle
import numpy as np
from sklearn.metrics import average_precision_score
from benchmark import load_real, accounts, _pools
from detectors import features, calibrate_semantic
from embed import SIF
from experiments import DETECTORS, FEATS, recall_at
from common import RES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    for k in ("account", "ts", "text", "url", "campaign"):
        ap.add_argument(f"--{k}", required=k in ("account", "ts", "text"))
    a = ap.parse_args()
    colmap = {k: getattr(a, k) for k in ("account", "ts", "text", "url") if getattr(a, k)}
    df = load_real(a.csv, colmap, campaign_col=a.campaign)
    df["ts"] = df.ts - df.ts.min()
    df["account"] = df.account.astype(str)
    chat, heads, _ = _pools()
    sif = SIF(df.text.tolist() + chat[:20000])
    organic = df[df.campaign < 0]
    tau = calibrate_semantic(organic.text.tolist(), sif, np.random.default_rng(0))
    F = features(df, sif, tau)
    y = accounts(df).set_index("account").loc[F.index].bot.to_numpy()
    model = pickle.load(open(os.path.join(RES, "fused_model.pkl"), "rb"))
    out = {"semantic_tau": tau, "accounts": int(len(y)), "coordinated": int(y.sum())}
    for d in DETECTORS:
        s = model.predict_proba(F[FEATS])[:, 1] if d == "fused" else F[d].to_numpy()
        out[d] = dict(ap=float(average_precision_score(y, s)), recall_at_1pct_fpr=recall_at(y, s))
    json.dump(out, open(os.path.join(RES, "real_data_results.json"), "w"), indent=2)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
