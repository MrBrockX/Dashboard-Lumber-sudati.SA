#!/usr/bin/env python3
"""
scripts/update_data.py
Baixa dados de LBR=F via yfinance e previsões/histórico Open-Meteo.
Gera CSVs em Dados/.
"""
import os
from datetime import datetime, timedelta
import requests
import pandas as pd
import yfinance as yf

DATA_DIR = os.getenv("DATA_DIR", "Dados")
os.makedirs(DATA_DIR, exist_ok=True)

def fetch_lumber(ticker="LBR=F", period="365d"):
    try:
        df = yf.download(tickers=ticker, period=period, interval="1d", progress=False)
        if df is None or df.empty:
            print("yfinance: sem dados retornados")
            return
        df.reset_index(inplace=True)
        df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d %H:%M:%S')
        out = os.path.join(DATA_DIR, f"lumber_futures_{ticker}_latest.csv")
        df.to_csv(out, index=False)
        print("Saved", out)
    except Exception as e:
        print("yfinance error:", e)

def fetch_open_meteo_forecast(lat=-23.0, lon=-51.0, days=14):
    try:
        base = "https://api.open-meteo.com/v1/forecast"
        params = {"latitude": lat, "longitude": lon, "daily": "precipitation_sum", "timezone": "UTC", "forecast_days": days}
        r = requests.get(base, params=params, timeout=30)
        r.raise_for_status()
        j = r.json()
        times = j.get("daily", {}).get("time", [])
        prec = j.get("daily", {}).get("precipitation_sum", [None]*len(times))
        df = pd.DataFrame({"Date": pd.to_datetime(times).strftime('%Y-%m-%d'), "precip_mm": prec})
        fname = os.path.join(DATA_DIR, f"open_meteo_forecast_{lat}_{lon}.csv")
        df.to_csv(fname, index=False)
        print("Saved", fname)
    except Exception as e:
        print("open-meteo forecast error:", e)

def fetch_open_meteo_history(lat=-23.0, lon=-51.0, days=365):
    try:
        end = datetime.utcnow().date()
        start = end - timedelta(days=days)
        base = "https://archive-api.open-meteo.com/v1/archive"
        params = {"latitude": lat, "longitude": lon, "start_date": start.strftime('%Y-%m-%d'),
                  "end_date": end.strftime('%Y-%m-%d'), "timezone": "UTC", "daily": "precipitation_sum"}
        r = requests.get(base, params=params, timeout=60)
        r.raise_for_status()
        j = r.json()
        times = j.get("daily", {}).get("time", [])
        prec = j.get("daily", {}).get("precipitation_sum", [None]*len(times))
        df = pd.DataFrame({"Date": pd.to_datetime(times).strftime('%Y-%m-%d'), "precip_mm": prec})
        fname = os.path.join(DATA_DIR, f"open_meteo_history_{lat}_{lon}.csv")
        df.to_csv(fname, index=False)
        print("Saved", fname)
    except Exception as e:
        print("open-meteo history error:", e)

if __name__ == "__main__":
    LAT = float(os.getenv("LAT", -23.0))
    LON = float(os.getenv("LON", -51.0))
    fetch_lumber()
    fetch_open_meteo_forecast(lat=LAT, lon=LON)
    fetch_open_meteo_history(lat=LAT, lon=LON)
