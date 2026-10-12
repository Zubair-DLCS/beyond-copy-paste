"""Draws every figure from saved results."""
import json, os, pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import RES, FIG

S1, S2, S3, S4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
SURF, INK, INK2, GRID, NEUTRAL = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#b9b8b2"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.titlecolor": INK, "axes.titleweight": "bold", "axes.titlesize": 11,
    "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "lines.linewidth": 2, "font.size": 9.5, "legend.frameon": False, "figure.dpi": 150,
})
NAMES = {"exact": "exact text", "url": "same link", "tfidf": "TF-IDF", "semantic": "semantic",
         "timing": "timing", "graph": "graph (label-free)", "fused": "fused (trained)"}


def title(fig, t, sub=None):
    h = fig.get_size_inches()[1]
    fig.get_layout_engine().set(rect=(0, 0, 1, 1 - (0.62 if sub else 0.4) / h))
    fig.text(0.008, 1 - 0.08 / h, t, ha="left", va="top", fontweight="bold", fontsize=13)
    if sub:
        fig.text(0.008, 1 - 0.36 / h, sub, ha="left", va="top", color=INK2, fontsize=9.5)


def save(fig, name):
    fig.savefig(os.path.join(FIG, name), bbox_inches="tight"); plt.close(fig)


def heat(ax, M, rows, cols, fmt="{:.2f}"):
    im = ax.imshow(M, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.grid(False); [s.set_visible(False) for s in ax.spines.values()]
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols); ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows)
    ax.tick_params(length=0)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, fmt.format(M[i, j]), ha="center", va="center", fontsize=9, color=SURF if M[i, j] > 0.55 else INK)
    return im


def fig1_pairs():
    e = json.load(open(os.path.join(RES, "detection_summary.json")))["e0_pairs"]
    names = ["exact", "tfidf", "semantic"]
    rec = [100 * e[n]["recognised_at_threshold"] for n in names]
    fal = [100 * e[n]["false_match_rate"] for n in names]
    fig, ax = plt.subplots(figsize=(8, 3.3), constrained_layout=True)
    y = np.arange(3)[::-1]
    ax.barh(y + 0.18, rec, height=0.34, color=S1, label="AI paraphrases recognised as the same message")
    ax.barh(y - 0.18, fal, height=0.34, color=S2, label="different messages wrongly matched")
    for yi, r, f in zip(y, rec, fal):
        ax.text(r + 1, yi + 0.18, f"{r:.0f}%", va="center", fontsize=9)
        ax.text(f + 1, yi - 0.18, f"{f:.1f}%", va="center", fontsize=9)
    ax.set_yticks(y); ax.set_yticklabels([NAMES[n] for n in names]); ax.grid(axis="y", visible=False)
    ax.set_xlim(0, 108); ax.set_xlabel("% of pairs, at each method's standard threshold")
    ax.legend(loc="upper right", fontsize=8.5, bbox_to_anchor=(1.0, 1.0))
    title(fig, "Can a similarity measure tell that two different wordings are the same message?",
          f"{e['n_positive_pairs']} pairs of AI paraphrases of the same seed vs {e['n_negative_pairs']:,} pairs of different seeds")
    save(fig, "fig1_paraphrase_recognition.png")


def fig2_ladder():
    R = pd.read_csv(os.path.join(RES, "detection_results.csv"))
    from experiments import LADDER
    steps = [n for n, _ in LADDER]
    dets = ["exact", "url", "tfidf", "timing", "semantic", "graph", "fused"]
    M = R[R.experiment == "ladder"].groupby(["detector", "setting"]).ap.mean().unstack().loc[dets, steps].to_numpy()
    fig, ax = plt.subplots(figsize=(11, 4.0), constrained_layout=True)
    im = heat(ax, M, [NAMES[d] for d in dets], [s.replace(" & ", " &\n").replace("+ ", "+ ").replace(" (", "\n(") for s in steps])
    ax.axhline(4.5, color=SURF, lw=3)
    cb = fig.colorbar(im, ax=ax, pad=0.01, shrink=0.9); cb.set_label("average precision"); cb.outline.set_visible(False)
    title(fig, "Escalating the attack: each tactic switches off a signal; trained fusion holds",
          "Average precision on held-out campaigns (4 campaigns, 120 coordinated among ~3,100 accounts; chance ≈ 0.04); 3 runs each")
    save(fig, "fig2_escalation.png")


def fig3_sweeps():
    R = pd.read_csv(os.path.join(RES, "detection_results.csv"))
    spec = [("sweep:paraphrase", "adv_p_para", "share of posts paraphrased by AI", False),
            ("sweep:jitter", "adv_jitter", "timing jitter (minutes)", True),
            ("sweep:url", "adv_p_url", "share of links replaced by unique short URLs", False),
            ("sweep:cover", "adv_cover", "cover posts per coordinated post", False)]
    dets = [("exact", S1), ("semantic", S2), ("timing", S3), ("fused", S4)]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.4), constrained_layout=True, sharey=True)
    for ax, (exp, col, xl, logx) in zip(axes, spec):
        d = R[R.experiment == exp].groupby([col, "detector"]).ap.mean().unstack()
        for k, c in dets + ([("url", NEUTRAL)] if exp == "sweep:url" else []):
            ax.plot(d.index, d[k], "-o", color=c, ms=5, label=NAMES[k])
        if logx:
            ax.set_xscale("log"); ax.axvspan(130, 260, color=GRID, alpha=0.6, lw=0)
            ax.text(180, 0.12, "beyond\ntraining\nrange", ha="center", fontsize=7.5, color=INK2)
        ax.set_xlabel(xl, fontsize=8.5); ax.set_ylim(0, 1.05)
    axes[0].set_ylabel("average precision")
    axes[0].legend(loc="lower left", fontsize=7.5)
    axes[2].legend(loc="lower left", fontsize=7.5)
    title(fig, "One evasion tactic at a time", "Every other control left at its naive setting; mean of 3 runs on held-out campaigns")
    save(fig, "fig3_single_tactics.png")


def fig4_fingerprint():
    from experiments import LADDER, SEEDS
    st = pickle.load(open(os.path.join(RES, "feature_cache.pkl"), "rb"))["cache"]
    F = pd.concat([st[("ladder", LADDER[-1][0], s)] for s in SEEDS])
    S = F[F.semantic > 0]
    share = S.groupby("bot").exact.apply(lambda x: 100 * (x > 0).mean())
    from sklearn.metrics import average_precision_score as AP
    R = pd.read_csv(os.path.join(RES, "detection_results.csv"))
    fused = R[(R.experiment == "ladder") & (R.setting == LADDER[-1][0]) & (R.detector == "fused")].ap.mean()
    vals = [("semantic score", AP(F.bot, F.semantic)),
            ("semantic, no exact copies", AP(F.bot, np.where(F.exact == 0, F.semantic, 0))),
            ("fused (trained)", fused)]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 3.3), constrained_layout=True, gridspec_kw=dict(width_ratios=[1, 1.2]))
    a1.barh([1, 0], [share[0], share[1]], color=[S1, S2], height=0.55)
    for y, v in zip([1, 0], [share[0], share[1]]):
        a1.text(v + 1.5, y, f"{v:.0f}%", va="center")
    a1.set_yticks([1, 0]); a1.set_yticklabels(["ordinary accounts", "coordinated accounts"]); a1.set_xlim(0, 80)
    a1.grid(axis="y", visible=False); a1.set_xlabel("% that ALSO post exact copies")
    a1.set_title("Among accounts with semantic matches")
    a2.barh(range(3)[::-1], [v for _, v in vals], color=S1, height=0.55)
    for y, (_, v) in zip(range(3)[::-1], vals):
        a2.text(v + 0.01, y, f"{v:.2f}", va="center")
    a2.set_yticks(range(3)[::-1]); a2.set_yticklabels([n for n, _ in vals]); a2.set_xlim(0, 1.1)
    a2.grid(axis="y", visible=False); a2.set_xlabel("average precision, full attack")
    a2.set_title("Using the fingerprint")
    title(fig, "The paraphrase fingerprint: people copy, paraphrasing bots don't",
          "Full attack (AI paraphrase, unique links, 120-min jitter, 2x cover traffic), held-out campaigns, 3 runs pooled")
    save(fig, "fig4_paraphrase_fingerprint.png")


def fig5_privacy():
    P = pd.read_csv(os.path.join(RES, "privacy_results.csv"))
    from experiments import LADDER
    levels = [LADDER[i][0] for i in (0, 2, 3, 5)]
    tiers = ["full", "embeddings", "hashed", "timing-10min", "timing-1h", "timing-6h"]
    labels = {"full": "raw text + links + times", "embeddings": "sentence embeddings, no raw text",
              "hashed": "hashed text & links + times", "timing-10min": "timing only (10-min)",
              "timing-1h": "timing only (1-hour)", "timing-6h": "timing only (6-hour)"}
    M = P.pivot(index="tier", columns="level", values="ap").loc[tiers, levels].to_numpy()
    eps = sorted([t for t in P.tier.unique() if "DP" in t], key=lambda t: float(t.split("=")[1][:-1]))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 3.9), constrained_layout=True, gridspec_kw=dict(width_ratios=[1.5, 1]))
    heat(a1, M, [labels[t] for t in tiers], [l.replace(" (", "\n(").replace("+ ", "+ ").replace("-min timing jitter", "-min\njitter").replace("unique short links", "unique\nlinks") for l in levels])
    a1.axhline(1.5, color=SURF, lw=3)
    a1.set_title("Detection (average precision) by what the detector may store")
    ev = [float(t.split("=")[1][:-1]) for t in eps]
    for lvl, c in ((levels[0], S1), (levels[2], S2)):
        v = [P[(P.tier == t) & (P.level == lvl)].ap.iloc[0] for t in eps]
        a2.plot(ev, v, "-o", color=c, ms=6, label=f"{lvl}")
        ref = P[(P.tier == "timing-1h") & (P.level == lvl)].ap.iloc[0]
        a2.axhline(ref, color=c, lw=1, ls="--")
    a2.text(ev[0], 1.0, "dashed: same detector without DP", fontsize=7.5, color=INK2, va="bottom")
    a2.set_xscale("log"); a2.set_xticks(ev); a2.set_xticklabels([f"{e:g}" for e in ev]); a2.set_ylim(0, 1.08)
    a2.set_xlabel("privacy budget ε (event-level; smaller = stronger privacy)"); a2.set_ylabel("average precision")
    a2.legend(loc="upper left", fontsize=8, bbox_to_anchor=(0.0, 0.93)); a2.set_title("Differentially private timing")
    title(fig, "Privacy vs detection: the signal that survives AI paraphrase is the one privacy removes",
          "Each tier retrained/re-run on its own data; held-out campaigns, 3 runs. Formal guarantee only for the DP timing channel")
    save(fig, "fig5_privacy_tradeoff.png")


def fig6_ablation():
    a = json.load(open(os.path.join(RES, "detection_summary.json")))["e3_ablation"]
    names = list(a.keys())
    v = [a[n]["full_attack_ap"] for n in names]
    fig, ax = plt.subplots(figsize=(7.5, 3.2), constrained_layout=True)
    y = np.arange(len(names))[::-1]
    ax.barh(y, v, color=[S2 if n == "all signals" else S1 for n in names], height=0.6)
    for yi, x in zip(y, v):
        ax.text(x + 0.005, yi, f"{x:.2f}", va="center", fontsize=9)
    ax.set_yticks(y); ax.set_yticklabels(names); ax.grid(axis="y", visible=False); ax.set_xlim(0.7, 1.02)
    ax.set_xlabel("average precision under the full attack")
    title(fig, "Which signals does the fused detector need?", "Retrained without one signal at a time")
    save(fig, "fig6_signal_ablation.png")


if __name__ == "__main__":
    for f in (fig1_pairs, fig2_ladder, fig3_sweeps, fig4_fingerprint, fig5_privacy, fig6_ablation):
        f(); print("drew", f.__name__)
