import os
import requests
from moves import get_flagged_moves, watchlist
from datetime import datetime, timedelta

NEWSAPI_KEY = os.environ.get("NEWSAPI_KEY")

COMPANY_NAMES = {
    "NVDA": "Nvidia", "AAPL": "Apple", "MSFT": "Microsoft",
    "GOOGL": "Google", "META": "Meta", "AMZN": "Amazon",
    "TSLA": "Tesla", "AMD": "AMD", "PLTR": "Palantir",
    "COIN": "Coinbase", "MSTR": "MicroStrategy", "JPM": "JPMorgan",
    "XOM": "Exxon", "KO": "Coca-Cola", "JNJ": "Johnson & Johnson",
    "CRWV": "CoreWeave", "CRCL": "Circle",
}

def get_news_for_ticker(ticker, move_date, lookback_days_news = 3, max_articles=5):
    name = COMPANY_NAMES.get(ticker, ticker)
    move_dt = datetime.strptime(move_date, "%Y-%m-%d")
    from_date = (move_dt - timedelta(days=lookback_days_news)).strftime("%Y-%m-%d")
    to_date = move_dt.strftime("%Y-%m-%dT23:59:59")
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": f'"{name}" AND (stock OR shares OR analyst OR earnings OR selloff OR plunge OR tumbles OR downgrade OR sinks OR drops)',
        "to": to_date,
        "from": from_date,
        "sortBy": "relevancy",
        "language": "en",
        "pageSize": max_articles,
        "apiKey": NEWSAPI_KEY
        
    }
    response = requests.get(url, params=params)
    #converts that dictionary into the ?key=value&key=value format
    response.raise_for_status()
    data = response.json()
    #converts the raw json to python dictionary

    articles = []
    for a in data.get("articles", []):
        articles.append({
            "title": a["title"],
            "source": a["source"]["name"],
            "published_at": a["publishedAt"],
            "url": a["url"],
            "description": a.get("description", "")
        })
    return articles

def retreive_for_flagged(df):
    flagged_df = df[df["flagged"] == True]
    #compares the true/false column in the table generated in phase 1
    news_by_ticker = {}

    #returns data and index from each row
    for _,row in flagged_df.iterrows():
        ticker = row["ticker"]
        direction = "UP" if row["return_pct"] > 0 else "DOWN"
        print(f"Fetching news for {ticker} ({direction} {row['return_pct']}%, z={row['z_score']})...")
        news_by_ticker[ticker] = {
            "date": row["date"],
            "return_pct": row["return_pct"],
            "z_score": row["z_score"],
            "direction": direction,
            "articles": get_news_for_ticker(ticker, row["date"]),
        }
    return news_by_ticker
if __name__ == "__main__":
    df = get_flagged_moves(watchlist)
    news = retreive_for_flagged(df)

    for ticker, info in news.items():
        print(f"\n=== {ticker}: {info['direction']} {info['return_pct']}% (z={info['z_score']}) ===")
        for a in info["articles"]:
            print(f"- {a['title']} ({a['source']}, {a['published_at'][:10]})")