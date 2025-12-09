import os
import pandas as pd
from datetime import datetime
from utils.data_loader import fetch_ticker
from models.train_model import train, feature_engineer

MODEL_PATH = "models/price_model.joblib"
OUTPUT_DIR = "output"

def compute_financial_indicators(df):
    df = df.copy().sort_values("Date").reset_index(drop=True)
    df['Returns'] = df['Adj Close'].pct_change().fillna(0)
    mean_return = df['Returns'].mean()
    volatility = df['Returns'].std()
    sharpe = (mean_return / volatility) * (252**0.5) if volatility != 0 else 0
    df['MA5'] = df['Adj Close'].rolling(5).mean().fillna(method='bfill')
    df['MA10'] = df['Adj Close'].rolling(10).mean().fillna(method='bfill')
    df['Volatility'] = df['Returns'].rolling(10).std().fillna(0)
    indicators = {
        'mean_return': float(mean_return),
        'volatility': float(volatility),
        'sharpe_ratio': float(sharpe),
        'last_price': float(df['Adj Close'].iloc[-1]) if not df.empty else None,
        'data_points': int(len(df))
    }
    return indicators, df

def ensure_dirs():
    os.makedirs('models', exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

def save_outputs(indicators, df, ticker):
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    df_path = os.path.join(OUTPUT_DIR, f"{ticker}_data_{ts}.csv")
    sum_path = os.path.join(OUTPUT_DIR, f"{ticker}_summary_{ts}.csv")
    df.to_csv(df_path, index=False)
    pd.DataFrame([indicators]).to_csv(sum_path, index=False)
    return df_path, sum_path

def run_pipeline(ticker='AAPL', period='1y', interval='1d'):
    ensure_dirs()
    print(f"[INFO] Fetching data for {ticker}...")
    df = fetch_ticker(ticker=ticker, period=period, interval=interval)
    if df is None or df.empty:
        print("[ERROR] No data fetched.")
        return
    print("[INFO] Computing indicators...")
    indicators, df_ind = compute_financial_indicators(df)
    print("[INFO] Training model...")
    score = train(df)
    print(f"[INFO] Model Score: {score}")
    df_path, sum_path = save_outputs(indicators, df_ind, ticker)
    print(f"[INFO] Saved processed data: {df_path}")
    print(f"[INFO] Saved summary: {sum_path}")

if __name__ == "__main__":
    run_pipeline('AAPL')
