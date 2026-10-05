import json, re
from pathlib import Path
import yfinance as yf
import anthropic
import random
import requests
from collections import defaultdict

from moves import watchlist
from news import get_news_for_ticker
from explain import explain_move

client = anthropic.Anthropic()
JUDGE = "claude-sonnet-5-5"
CASES = Path("eval_cases.json")

#baseline from volatility window of 'lookback', 'days' recent days to test 
def past_flagged(tickers, days=15, lookback=30, z_thr=1.6):
    rows = []
    for t in tickers:
        h = yf.Ticker(t).history(period = "90d")
        h["ret"] = h["Close"].pct_change()
        #downloads 60 trading days and computes daily returns
        for i in range(-days, 0):
            trailing = h["ret"].iloc[i - lookback : i]
            z = (h["ret"].iloc[i] - trailing.mean()) / trailing.std()
            if abs(z) > z_thr:
                rows.append({
                    "ticker": t,
                    "date": h.index[i].strftime("%Y-%m-%d"),
                    "return_pct": round(h["ret"].iloc[i] * 100, 2),
                    "z_score": round(z, 2),
                })
    return rows

def freeze_cases(n_max=30, per_ticker=4, seed=42):
    moves = past_flagged(watchlist)
    #keeps the eval set diverse and random for reproducibility
    random.Random(seed).shuffle(moves)
    counts = defaultdict(int)
    cases = []

    for r in moves:
        if len(cases) >= n_max:
            break
        if counts[r["ticker"]] >= per_ticker:
            continue
        try:
            articles = get_news_for_ticker(r["ticker"], r["date"])
        except Exception as e:
            print(f"Skipping {r['ticker']} {r['date']}: HTTP {e.response.status_code}")
            continue
        if not articles:
            print(f"Skipping {r['ticker']} {r['date']}: no articles")
            continue

        r["direction"] = "UP" if r["return_pct"] > 0 else "DOWN"
        r["articles"] = articles
        #following will be filled by hand
        r["gold_has_driver"] = None
        r["gold_driver"] = ""
        r["article_relevance"] = [None] * len(r["articles"])
        cases.append(r)
        counts[r["ticker"]] += 1

    CASES.write_text(json.dumps(cases, indent=2))
    print(f"Froze {len(cases)} cases across {len(counts)} tickers")

def is_abstain(text):
    return text.strip().lstrip("*#> ").upper().startswith("NO CLEAR DRIVER")

def citations_valid(text, n):
    if is_abstain(text):
        return True
    #breaks the output into pieces checker can inspect (each sentence has at least 1 citation)
    text = re.sub(r"([.!?])\s*((?:\[\d+\]\s*)+)", lambda m: " " + m.group(2).strip() + m.group(1) + " ", text)
    cited = [int(x) for x in re.findall(r"\[(\d+)\]", text)]
    sents = [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]
    return bool(cited) and all(1 <= c <= n for c in cited) \
        and all(re.search(r"\[\d+\]", s) for s in sents)

#not quoting a number from article
def no_number_leak(text, return_pct):
    found = [float(x) for x in re.findall(r"(\d+(?:\.\d+)?)%", text)]
    return all(abs(f - abs(return_pct)) < 0.1 for f in found)

#LLM Judge
JUDGE_PROMPT = """You are grading a stock-move explanation.
Move (given to the model as fact, treat as true): {move}
Excerpts: {excerpts}
Gold driver (human-written): {gold}
Model output: {output}
Return ONLY a JSON object with these keys, no code fences, no comments:
faithful (true/false): every claim about WHY the stock moved is supported by excerpt
matches_gold (true/false): the output identifies the gold driver
unsupported (list of strings): claims not supported by the cited excerpt"""

def parse_json(text):
    start, end = text.find("{"),text.rfind("}")
    return json.loads(text[start:end + 1])

def get_text(response):
    return "".join(b.text for b in response.content if b.type == "text")
def judge(case, output):
    ex = "\n".join(
        f"[{i}] {a['title']}\n{(a['description'] or '')[:300]}"
        for i, a in enumerate(case["articles"], 1)
    )
    move = f"{case['ticker']} moved {case['direction']} {case['return_pct']}% on {case['date']}."
    r = client.messages.create(
        model= JUDGE, max_tokens= 4000,
        messages=[{"role": "user", "content": JUDGE_PROMPT.format(
            move=move, excerpts=ex, gold=case["gold_driver"], output=output) }],
    )
    try:
        return parse_json(get_text(r))
    except Exception as e:
        print(f"Judge parse failed for {case['ticker']} {case['date']}: {e}")
        return None

def run():
    cases = json.loads(CASES.read_text())
    #converts eval_cases.json into python list of dicts
    labeled = [c for c in cases
               if c["gold_has_driver"] is not None
               and all(x is not None for x in c["article_relevance"])]
                #only keeping the fully labeled cases
    print(f"{len(labeled)}/{len(cases)} cases labeled")
    if not labeled:
        return
    #true positive, false negative, true negative, false positive
    tp = fn = tn = fp = 0
    #precision@5, at least 1 article is relevant, citations_valid?, no_number_leak?..
    p5, hit5, cite, leak, faith, match = [], [], [], [], [], []
    failures = []

    for c in labeled:
        tag = f"{c['ticker']} {c['date']}"
        rel = c["article_relevance"]
        p5.append(sum(rel) / len(rel)) #fraction of articles that were relevant
        hit5.append(int(any(rel)))
    #the case dict c holds articles, date, move size, direction
        out = explain_move(c["ticker"], c)
        abstained = is_abstain(out)

        if c["gold_has_driver"]:
            tp += not abstained
            fn += abstained
            if abstained:
                failures.append((tag, "missed driver (FN)", out))
        else:
            tn += abstained
            fp += not abstained
            if not abstained:
                failures.append((tag, "hallucinated driver (FP)", out))

        ok = citations_valid(out, len(c["articles"]))
        cite.append(ok)
        if not ok:
            failures.append((tag, "bad citations", out))

        ok = no_number_leak(out, c["return_pct"])
        leak.append(ok)
        if not ok:
            failures.append((tag, "number leak", out))

        if not abstained:
            j = judge(c, out)
            if j:
                faith.append(j["faithful"])
                if c["gold_has_driver"]:
                    match.append(j["matches_gold"])
                if not j["faithful"]:
                    failures.append((tag, f"unfaithful: {j['unsupported']}", out))
                if c["gold_has_driver"] and not j["matches_gold"]:
                    failures.append((tag, f"gold mismatch (gold: {c['gold_driver']})", out))

    avg = lambda x: f"{sum(x)} / {len(x)}" if x else "n/a"
    #like 18/20
    print("\n=== RESULTS ===")
    print(f"retrieval precision@5 = {round(sum(p5)/len(p5), 2)} hit@5 = {avg(hit5)}")
    print(f"abstention  TP={tp} FN={fn} | TN={tn} FP={fp}")
    print(f"rules       citations={avg(cite)}  no_number_leak={avg(leak)}")
    print(f"quality     faithful={avg(faith)}  matches_gold={avg(match)}")
    print("\n=== FAILURES ===")
    for tag, reason, out in failures:
        print(f"\n[{tag}] {reason}\n{out}")

def dry_run():
    moves = past_flagged(watchlist, z_thr=0)
    moves.sort(key=lambda m: abs(m["z_score"]), reverse=True)
    print(len(moves), "days scored")
    for m in moves[:15]:
        print(m)

if __name__ == "__main__":
    if not CASES.exists():
        freeze_cases()
        print("Cases frozen. Label gold_* fields in eval_cases.json, then rerun.")
    else:
        run()

    



