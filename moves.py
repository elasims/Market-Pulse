import yfinance as yf
import pandas as pd

watchlist = [
    "NVDA", "AAPL", "MSFT", "GOOGL", "META", "AMZN", "TSLA",
    "AMD", "PLTR", "COIN", "MSTR",
    "JPM", "XOM", "KO", "JNJ",
    "CRWV", "CRCL",
]

#yf.ticker creates a ticker object for t
#hist is a data frame, each row is one trading day

def get_flagged_moves (tickers, lookback_days=30, z_treshold = 2):
    flagged = [] #empty list to collect flagged tickers
    for t in tickers:
        try:
            hist = yf.Ticker(t).history(period = f"{lookback_days + 10}d")
            if len(hist) < lookback_days:
                print(f"Skipping {t} not enough history({len(hist)} days)")
                continue
            hist["return"] = hist["Close"].pct_change()
            trailing = hist["return"].iloc[-(lookback_days - 1) : -1] #excludes today
            mean_ret = trailing.mean()
            std_ret = trailing.std()

            today_return = hist["return"].iloc[-1]
            z_score = (today_return - mean_ret) / std_ret

            result = {
                "ticker": t,
                "date": hist.index[-1].strftime("%Y-%m-%d"),
                "return_pct": round(today_return * 100, 2),
                "z_score": round(z_score, 2),
                "flagged": abs(z_score) > z_treshold
            }
            flagged.append(result)
        except Exception as e:
            print(f"Error with {t}: {e}")
    return pd.DataFrame(flagged)
        
if __name__ == "__main__":
    df = get_flagged_moves(watchlist)
    print(df.sort_values("z_score", key=abs, ascending=False))

