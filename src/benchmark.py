"""
Semi-synthetic coordination benchmark with adversary controls.

Every post is a row of a common schema, so a real dataset can replace the
simulation without touching the detectors:

    account (str) | ts (float, seconds) | text (str) | url (str or "") | campaign (int, -1 = organic)

Organic accounts (real text, simulated behaviour)
  - chat posts: human-written Reddit comments (GoEmotions)
  - news shares: headline + article URL from a 12k-headline pool with Zipf popularity;
    each article has a publication time and its shares decay after it. Popular stories
    are shared by many unrelated people within minutes, the classic false positive
    for copy-paste and URL detectors.
  - reposts of viral chat items (exact duplicates, also bursty)
  Human daily rhythm and session-clumped activity.

Campaigns (real seed text, simulated coordination)
  - each campaign pushes its 12 seed messages, one per "wave", posted by many accounts
  - adversary controls:
      p_para    share of posts that use an AI paraphrase instead of the seed
      p_edit    share of posts with light surface edits (hashtag, prefix, punctuation)
      jitter    spread of posting times inside a wave (minutes)
      p_url     share of posts whose link is replaced by a unique shortened URL
      cover     cover traffic: ordinary-looking posts per coordinated post
"""
import json, os, string
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
from common import DATA, DAY, MIN

PREFIXES = ["", "", "Wow: ", "Breaking: ", "Read this: ", "Must read - ", "Thoughts? ", "This! "]
TAGS = ["", "", "", " #news", " #breaking", " #mustread", " #share", " #truth", " #today"]


@dataclass
class Adversary:
    p_para: float = 0.0
    p_edit: float = 0.0
    jitter: float = 3.0          # minutes
    p_url: float = 0.0
    cover: float = 0.0           # cover posts per coordinated post

    def label(self):
        return f"para={self.p_para:g},edit={self.p_edit:g},jit={self.jitter:g},url={self.p_url:g},cover={self.cover:g}"


NAIVE = Adversary()
ADAPTIVE = Adversary(p_para=1.0, p_edit=1.0, jitter=120, p_url=1.0, cover=2.0)


def _pools():
    chat = pd.read_csv(os.path.join(DATA, "organic_chat.txt"), header=None, keep_default_na=False)[0].tolist()
    heads = pd.read_csv(os.path.join(DATA, "headlines.txt"), header=None, keep_default_na=False)[0].tolist()
    camps = json.load(open(os.path.join(DATA, "campaign_messages.json"), encoding="utf8"))["campaigns"]
    return chat, heads, camps


def _diurnal(rng, n, days):
    h = np.arange(24)
    pdf = 0.15 + np.exp(-0.5 * ((h - 11) / 2.5) ** 2) + 1.2 * np.exp(-0.5 * ((h - 20) / 2.0) ** 2)
    hour = rng.choice(24, n, p=pdf / pdf.sum())
    return rng.integers(0, days, n) * DAY + hour * 3600 + rng.uniform(0, 3600, n)


def _short(rng):
    return "https://bit.ly/" + "".join(rng.choice(list(string.ascii_letters + string.digits), 7))


def _edit(rng, text):
    t = rng.choice(PREFIXES) + text.rstrip(" .") + rng.choice(TAGS)
    if rng.random() < 0.3:
        t = t + rng.choice(["!", "!!", " ...", " 👀", " 🔥"])
    return t


def simulate(adv=NAIVE, campaigns=range(8), seed=0, n_org=3000, bots_per_campaign=30, days=14):
    rng = np.random.default_rng(seed)
    chat, heads, camps = _pools()
    rows = []

    # ---- shared organic objects: news articles and viral chat items
    n_art = len(heads)
    pop = 1 / np.arange(1, n_art + 1) ** 0.8; pop /= pop.sum()
    art_order = rng.permutation(n_art)                 # random which headline is popular
    art_time = rng.uniform(0, days * DAY, n_art)
    N_VIRAL = 2000
    viral = rng.choice(len(chat), N_VIRAL, replace=False)
    vpop = 1 / np.arange(1, N_VIRAL + 1) ** 0.9; vpop /= vpop.sum()
    viral_time = rng.uniform(0, days * DAY, N_VIRAL)

    def organic_posts(acct, n, t_window=None):
        kinds = rng.choice(3, n, p=[0.60, 0.35, 0.05])
        sess = _diurnal(rng, max(1, n // 3), days)
        for k in kinds:
            if k == 0:
                rows.append((acct, sess[rng.integers(len(sess))] + rng.exponential(20 * MIN),
                             chat[rng.integers(len(chat))], "", -1))
            elif k == 1:
                a = art_order[rng.choice(n_art, p=pop)]
                ts = art_time[a] + rng.exponential(4 * 3600)       # shares decay after publication
                rows.append((acct, ts, heads[a], f"https://news.example/{a}", -1))
            else:
                v = rng.choice(N_VIRAL, p=vpop)
                rows.append((acct, viral_time[v] + rng.exponential(2 * 3600), chat[viral[v]], "", -1))

    for i in range(n_org):
        organic_posts(f"u{i:05d}", 1 + rng.negative_binomial(1.3, 0.08))

    # ---- campaigns
    for c in campaigns:
        msgs = camps[c]["messages"]
        waves = rng.uniform(0.5 * DAY, (days - 0.5) * DAY, len(msgs))
        arts = [f"https://site{c}.example/story{j}" for j in range(len(msgs))]
        for b in range(bots_per_campaign):
            acct = f"c{c}_b{b:02d}"
            n_coord = 0
            for j, (m, t0) in enumerate(zip(msgs, waves)):
                if rng.random() > 0.8:
                    continue
                text = m["para"][rng.integers(4)] if rng.random() < adv.p_para else m["seed"]
                if rng.random() < adv.p_edit:
                    text = _edit(rng, text)
                url = _short(rng) if rng.random() < adv.p_url else arts[j]
                ts = t0 + abs(rng.normal(0, adv.jitter * MIN)) + rng.uniform(0, 30)
                rows.append((acct, ts, text, url, c)); n_coord += 1
            if adv.cover > 0:                                   # look like an ordinary user
                organic_posts(acct, rng.poisson(adv.cover * n_coord))

    df = pd.DataFrame(rows, columns=["account", "ts", "text", "url", "campaign"])
    df = df[(df.ts >= 0) & (df.ts < days * DAY)].sort_values("ts").reset_index(drop=True)
    # campaign label is an account property
    acct_c = {a: int(a.split("_")[0][1:]) if a.startswith("c") else -1 for a in df.account.unique()}
    df["campaign"] = df.account.map(acct_c)
    return df


def accounts(df):
    """Account table with ground truth (1 = coordinated)."""
    a = df.groupby("account").agg(n=("ts", "size"), campaign=("campaign", "first")).reset_index()
    a["bot"] = (a.campaign >= 0).astype(int)
    return a


def load_real(path, colmap, campaign_col=None):
    """Load a real dataset into the common schema (e.g. Cresci-2017 after download).
    colmap maps schema names to the file's column names, e.g.
    {"account": "user_id", "ts": "timestamp", "text": "text", "url": "expanded_url"}."""
    d = pd.read_csv(path, low_memory=False)
    out = pd.DataFrame({k: d[v] for k, v in colmap.items()})
    if not pd.api.types.is_numeric_dtype(out.ts):
        out["ts"] = pd.to_datetime(out.ts, utc=True).astype("int64") / 1e9
    out["url"] = out["url"].fillna("").astype(str) if "url" in out else ""
    out["text"] = out["text"].fillna("").astype(str)
    out["campaign"] = d[campaign_col] if campaign_col else -1
    return out.sort_values("ts").reset_index(drop=True)


if __name__ == "__main__":
    for adv in (NAIVE, ADAPTIVE):
        df = simulate(adv)
        a = accounts(df)
        print(adv.label(), f"posts={len(df):,} accounts={len(a):,} bots={a.bot.sum()} "
              f"bot posts={int((df.campaign >= 0).sum()):,} with url={(df.url != '').mean():.2f}")
    print(df.sample(5, random_state=1)[["account", "text", "url"]].to_string())
