"""
Build the two text pools used by ordinary (organic) accounts.

  data/organic_chat.txt   everyday comments: GoEmotions, human-written Reddit comments
                          (Demszky et al., 2020; downloaded from GitHub)
  data/headlines.txt      news headlines that ordinary users share. Drawn from the
                          IRA archive's news-feed posts but DISJOINT from every campaign
                          seed, so headline style cannot identify a campaign.

The headline pool ships with the repository; rebuilding it needs the IRA archive
(`--rebuild-headlines PATH_TO_tweets.parquet`).
"""
import argparse, json, os, re, subprocess
import pandas as pd
from common import DATA

GOEMO = "https://raw.githubusercontent.com/google-research/google-research/master/goemotions/data/train.tsv"
BAD = re.compile(r"\b(kill|dead|death|shot|shoot|rape|sex|nigg|fuck|shit|bitch|whore|gay|muslim|islam|black|"
                 r"white|terror|murder|gun|drug|police|cop|trump|hillary|obama|clinton)\w*", re.I)


def build_chat():
    path = os.path.join(DATA, "organic_chat.txt")
    if os.path.exists(path):
        return path
    raw = os.path.join(DATA, "goemotions_train.tsv")
    if not os.path.exists(raw):
        subprocess.run(["curl", "-sL", "-o", raw, GOEMO], check=True)
    d = pd.read_csv(raw, sep="\t", header=None, names=["text", "labels", "id"], quoting=3)
    t = d.text.str.replace("[NAME]", "Sam", regex=False).str.strip()
    t = t[t.str.len().between(15, 220)].drop_duplicates()
    t.to_csv(path, index=False, header=False)
    return path


def build_headlines(parquet, n=12000, seed=0):
    seeds = {m["seed"] for c in json.load(open(os.path.join(DATA, "campaign_messages.json")))["campaigns"]
             for m in c["messages"]}
    t = pd.read_parquet(parquet, columns=["content", "language", "account_category", "retweet"])
    t = t[(t.language == "English") & (t.retweet == 0) & (t.account_category == "NewsFeed")]
    c = (t.content.str.replace(r"https?://\S+", "", regex=True).str.replace(r"@\w+", "", regex=True)
         .str.replace(r"\s+", " ", regex=True).str.strip())
    c = c[c.str.len().between(50, 180) & ~c.str.contains(BAD) & ~c.isin(seeds)].drop_duplicates()
    c.sample(n, random_state=seed).to_csv(os.path.join(DATA, "headlines.txt"), index=False, header=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild-headlines", metavar="TWEETS_PARQUET")
    a = ap.parse_args()
    if a.rebuild_headlines:
        build_headlines(a.rebuild_headlines)
    print(build_chat())
