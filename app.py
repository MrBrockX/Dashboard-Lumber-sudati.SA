# app.py
# Sudati — Lumber Futures & Weather Hedge Toolkit (arquivo unificado)
import os
import io
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pandas.api import types as ptypes
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error

# Prophet optional
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except Exception:
    PROPHET_AVAILABLE = False

# ---------- CONFIG ----------
st.set_page_config(page_title="Sudati — Lumber Hedge Toolkit", layout="wide", initial_sidebar_state="expanded")
DATA_DIR = "Dados"
OUTPUT_DIR = "lumber_analysis"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

LOCAL_PRICE_CSV = os.path.join(DATA_DIR, "lumber_futures_LBR=F_latest.csv")
LOCAL_FORECAST_PATTERN = "open_meteo_forecast_{lat}_{lon}.csv"
YF_TICKER = "LBR=F"

# ---------- UTIL (FRED + file helpers + meteo) ----------
@st.cache_data(ttl=3600)
def get_fred_data():
    """Get FRED lumber index (WPU081) as DataFrame"""
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=WPU081"
    try:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        df = pd.read_csv(io.BytesIO(r.content))
        if df.shape[1] >= 2:
            df.columns = ['Date', 'Price_Index']
        else:
            df.columns = ['Date', 'Price_Index']
        df['Date'] = pd.to_datetime(df['Date'], errors="coerce")
        return df
    except Exception:
        return None

@st.cache_data(ttl=300)
def fetch_yf(ticker: str, period: str = "365d", interval: str = "1d"):
    try:
        df = yf.download(tickers=ticker, period=period, interval=interval, progress=False)
        if df is None or df.empty:
            return None
        df = df.reset_index()
        if "Adj Close" in df.columns and "Close" not in df.columns:
            df = df.rename(columns={"Adj Close": "Close"})
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce", utc=True).dt.tz_convert(None)
        return df
    except Exception:
        return None

@st.cache_data(ttl=600)
def fetch_open_meteo_history(lat, lon, start_date, end_date):
    base = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": float(lat), "longitude": float(lon),
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "timezone": "UTC", "daily": "precipitation_sum"
    }
    try:
        resp = requests.get(base, params=params, timeout=30)
        resp.raise_for_status()
        j = resp.json()
        if "daily" not in j or "time" not in j["daily"]:
            return None
        df = pd.DataFrame({"Date": pd.to_datetime(j["daily"]["time"], utc=True).tz_convert(None),
                           "precip_mm": j["daily"].get("precipitation_sum", [0]*len(j["daily"]["time"]))})
        return df
    except Exception:
        return None

@st.cache_data(ttl=600)
def fetch_open_meteo_forecast(lat, lon, days=14):
    base = "https://api.open-meteo.com/v1/forecast"
    params = {"latitude": float(lat), "longitude": float(lon),
              "daily": "precipitation_sum", "timezone": "UTC", "forecast_days": days}
    try:
        resp = requests.get(base, params=params, timeout=20)
        resp.raise_for_status()
        j = resp.json()
        if "daily" not in j or "time" not in j["daily"]:
            return None
        df = pd.DataFrame({"Date": pd.to_datetime(j["daily"]["time"], utc=True).tz_convert(None),
                           "precip_mm": j["daily"].get("precipitation_sum", [0]*len(j["daily"]["time"]))})
        return df
    except Exception:
        return None

def read_csv_safe(path_or_buffer):
    try:
        if isinstance(path_or_buffer, (str, os.PathLike)):
            if not os.path.exists(path_or_buffer):
                return None
            df = pd.read_csv(path_or_buffer)
        else:
            path_or_buffer.seek(0)
            df = pd.read_csv(path_or_buffer)
        return df
    except Exception:
        return None

def drop_header_row_with_ticker(df: pd.DataFrame):
    if df is None or df.empty:
        return df
    df = df.copy()
    try:
        first = df.iloc[0].astype(str).apply(lambda x: x.strip())
    except Exception:
        first = df.iloc[0].astype(str)
    col_names = [str(c).strip() for c in list(df.columns)]
    eq_colname = np.array([first.iloc[i] == col_names[i] if i < len(first) else False for i in range(len(col_names))])
    eq_ticker = np.array([first.iloc[i] == str(YF_TICKER) if i < len(first) else False for i in range(len(col_names))])
    if int(np.sum(eq_colname) + np.sum(eq_ticker)) >= max(1, len(col_names) // 2):
        try:
            df = df.drop(df.index[0]).reset_index(drop=True)
        except Exception:
            pass
    return df

def detect_and_make_date(df: pd.DataFrame):
    if df is None:
        return None
    if not isinstance(df, pd.DataFrame):
        try:
            df = pd.DataFrame(df)
        except Exception:
            return None
    df = df.copy()
    df = drop_header_row_with_ticker(df)
    candidates = ["Date", "date", "DATE", "datetime", "timestamp", "time", "DateTime"]
    for c in candidates:
        if c in df.columns:
            try:
                parsed = pd.to_datetime(df[c], errors="coerce", utc=True)
                df[c] = parsed
                df = df.dropna(subset=[c]).copy()
                df["Date"] = df[c].dt.tz_convert(None)
                if c != "Date":
                    try:
                        df = df.drop(columns=[c])
                    except Exception:
                        pass
                return df
            except Exception:
                continue
    for c in df.columns:
        try:
            if ptypes.is_numeric_dtype(df[c]):
                continue
            parsed = pd.to_datetime(df[c], errors="coerce", utc=True)
            non_null = parsed.notna().sum()
            if non_null >= max(5, len(df) // 10):
                df["Date"] = parsed.dt.tz_convert(None)
                df = df.dropna(subset=["Date"]).copy()
                return df
        except Exception:
            continue
    if isinstance(df.index, pd.DatetimeIndex):
        df2 = df.reset_index()
        idx_name = df2.columns[0]
        df2 = df2.rename(columns={idx_name: "Date"})
        try:
            df2["Date"] = pd.to_datetime(df2["Date"], errors="coerce", utc=True).dt.tz_convert(None)
            df2 = df2.dropna(subset=["Date"]).copy()
            return df2
        except Exception:
            return df
    return df

def standardize_columns(df: pd.DataFrame):
    if df is None:
        return None
    df = df.copy()
    df = drop_header_row_with_ticker(df)
    if "Price" in df.columns and "Close" not in df.columns:
        df = df.rename(columns={"Price": "Close"})
    if "Price_Index" in df.columns and "Index" not in df.columns:
        df = df.rename(columns={"Price_Index": "Index"})
    for c in ["Close", "Open", "High", "Low", "Volume", "Index"]:
        if c in df.columns:
            try:
                col = df[c]
                if isinstance(col, pd.DataFrame):
                    col = col.squeeze()
                nonnull = col.dropna()
                if not nonnull.empty:
                    first_val = str(nonnull.iloc[0]).strip()
                    if first_val == str(YF_TICKER) or first_val == str(c):
                        mask = df[c].astype(str).str.strip() == first_val
                        if mask.any():
                            df = df.loc[~mask].reset_index(drop=True)
                            col = df[c]
                df[c] = pd.to_numeric(col, errors="coerce")
            except Exception:
                pass
    return df

# ---------- DATA LOAD ----------
def load_data(prefer_local=True, uploaded_recent=None, period="365d", interval="1d"):
    recent = None
    if prefer_local and os.path.exists(LOCAL_PRICE_CSV):
        recent = read_csv_safe(LOCAL_PRICE_CSV)
        if recent is not None:
            st.info(f"Usando CSV local gerado: {LOCAL_PRICE_CSV}")
    if recent is None and uploaded_recent is not None:
        recent = read_csv_safe(uploaded_recent)
        if recent is not None:
            st.info("Usando CSV enviado para dados recentes.")
    if recent is None:
        recent = fetch_yf(YF_TICKER, period=period, interval=interval)
        if recent is not None:
            st.info("Usando dados do yfinance (fallback).")
        else:
            st.warning("Não foi possível obter dados do yfinance e não há CSV local/upload.")
    if recent is not None:
        recent = detect_and_make_date(recent)
        recent = standardize_columns(recent)
    return recent

# ---------- FEATURES & MODEL PREP ----------
def prepare_for_model(price_df, weather_hist_df, weather_forecast_df=None):
    p = detect_and_make_date(price_df).sort_values("Date").reset_index(drop=True)
    w = detect_and_make_date(weather_hist_df).sort_values("Date").reset_index(drop=True)
    merged = pd.merge_asof(p[["Date","Close"]], w[["Date","precip_mm"]], on="Date", direction="backward", tolerance=pd.Timedelta("2D"))
    merged = merged.dropna(subset=["Close"]).reset_index(drop=True)
    merged["precip_mm"] = merged["precip_mm"].fillna(0.0)
    merged["month"] = merged["Date"].dt.month
    month_dummies = pd.get_dummies(merged["month"], prefix="m", drop_first=True)
    for lag in [7,14,30]:
        merged[f"precip_lag_{lag}"] = merged["precip_mm"].rolling(lag, min_periods=1).sum().shift(1).fillna(0.0)
    exog = pd.concat([merged[["precip_mm"]] , month_dummies, merged[[f"precip_lag_{lag}" for lag in [7,14,30]]]], axis=1).fillna(0.0)
    future_exog = None
    if weather_forecast_df is not None:
        wf = detect_and_make_date(weather_forecast_df).sort_values("Date").reset_index(drop=True)
        wf["precip_mm"] = wf["precip_mm"].fillna(0.0)
        future_rows = []
        for d in wf["Date"]:
            month = d.month
            row = {"precip_mm": float(wf.loc[wf["Date"] == d, "precip_mm"].iloc[0])}
            for c in exog.columns:
                if c.startswith("m_"):
                    row[c] = 1 if int(c.split("_")[1]) == month else 0
            for lag in [7,14,30]:
                row[f"precip_lag_{lag}"] = 0.0
            future_rows.append(row)
        if future_rows:
            future_exog = pd.DataFrame(future_rows, columns=exog.columns).fillna(0.0)
    return merged, exog, future_exog

# ---------- MODELING ----------
def sarimax_forecast(y, exog, future_exog=None, order=(1,1,1), seasonal_order=(1,1,1,12)):
    try:
        model = SARIMAX(y, exog=exog, order=order, seasonal_order=seasonal_order,
                        enforce_stationarity=False, enforce_invertibility=False)
        res = model.fit(disp=False, maxiter=200)
        if future_exog is None:
            future_exog = np.repeat(exog.iloc[-1:].values, 30, axis=0)
        f = res.get_forecast(steps=len(future_exog), exog=future_exog)
        mean = f.predicted_mean.reset_index(drop=True)
        conf = f.conf_int().reset_index(drop=True)
        return mean, conf, res
    except Exception:
        return None, None, None

def prophet_forecast(merged_df, future_weather_df=None, steps=30):
    try:
        dfp = merged_df[["Date","Close"]].rename(columns={"Date":"ds","Close":"y"})
        m = Prophet(daily_seasonality=False, yearly_seasonality=True, weekly_seasonality=False)
        if "precip_mm" in merged_df.columns:
            m.add_regressor("precip_mm")
            dfp["precip_mm"] = merged_df["precip_mm"].values
        m.fit(dfp)
        future = m.make_future_dataframe(periods=steps)
        if future_weather_df is not None:
            fw = future_weather_df.rename(columns={"Date":"ds"})
            future = future.merge(fw, how="left", on="ds")
            future["precip_mm"] = future["precip_mm"].fillna(method="ffill").fillna(0.0)
        else:
            future["precip_mm"] = 0.0
        fc = m.predict(future)
        mean = fc["yhat"].iloc[-steps:].reset_index(drop=True)
        conf = pd.DataFrame({"lower": fc["yhat_lower"].iloc[-steps:].values, "upper": fc["yhat_upper"].iloc[-steps:].values})
        return mean, conf, m
    except Exception:
        return None, None, None

def walk_forward_backtest(y, exog, train_window=180, forecast_horizon=14, max_rounds=50):
    results = []
    y = pd.Series(y).reset_index(drop=True)
    exog = exog.reset_index(drop=True)
    n = len(y)
    rounds = 0
    for start in range(train_window, n - forecast_horizon):
        if rounds >= max_rounds:
            break
        train_y = y[:start]
        train_exog = exog.iloc[:start]
        try:
            model = SARIMAX(train_y, exog=train_exog, order=(1,1,1), seasonal_order=(1,1,1,12),
                            enforce_stationarity=False, enforce_invertibility=False)
            res = model.fit(disp=False, maxiter=200)
            future_exog = exog.iloc[start:start+forecast_horizon]
            f = res.get_forecast(steps=forecast_horizon, exog=future_exog)
            pred = f.predicted_mean
            for i in range(forecast_horizon):
                idx = start + i
                if idx >= len(y): break
                results.append({"idx": idx, "pred": float(pred.iloc[i]), "obs": float(y.iloc[idx])})
        except Exception:
            pass
        rounds += 1
    if not results:
        return None
    return pd.DataFrame(results)

def generate_simple_hedge_signal(forecast_series, recent_price, threshold_pct=0.05):
    if forecast_series is None or recent_price is None:
        return "no_signal"
    mean_forecast = float(np.nanmean(forecast_series))
    change = (mean_forecast - recent_price) / recent_price
    if change >= threshold_pct:
        return "buy_inventory"
    elif change <= -threshold_pct:
        return "sell_inventory"
    else:
        return "hold"

# ---------- UI ----------
st.title("Sudati — Lumber Forecast & Hedge Toolkit")
st.markdown("Previsão de preços de madeira com exógenas meteorológicas (Open-Meteo).")

with st.sidebar:
    st.header("Config / Dados")
    prefer_local = st.checkbox("Preferir CSVs locais gerados (Dados/)", value=True)
    uploaded_recent = st.file_uploader("Upload CSV recent (opcional)", type=["csv"])
    period = st.selectbox("Periodo yfinance (fallback)", ["90d","180d","365d"], index=1)
    interval = st.selectbox("Intervalo yfinance", ["1d","1h"], index=0)
    st.markdown("---")
    st.header("Localização / Meteo")
    lat = st.number_input("Latitude", value=-23.0, format="%.4f")
    lon = st.number_input("Longitude", value=-51.0, format="%.4f")
    hist_start = st.date_input("Hist. meteologia início", value=(datetime.utcnow() - timedelta(days=365)).date())
    hist_end = st.date_input("Hist. meteologia fim", value=datetime.utcnow().date())
    st.markdown("---")
    st.header("Model / Hedge")
    steps = st.selectbox("Horizonte forecast (dias)", [7,14,30,60,90], index=1)
    use_prophet = st.checkbox("Usar Prophet (se instalado)", value=False)
    hedge_ratio = st.number_input("Hedge ratio (contratos por unidade)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    simulate = st.button("Rodar / Atualizar")
    st.markdown("---")
    st.header("Debug / Export")
    show_debug = st.checkbox("Mostrar debug", value=False)
    if st.button("Limpar cache"):
        try:
            st.cache_data.clear()
            st.experimental_rerun()
        except Exception:
            pass

# Load price data (prefer local)
recent = load_data(prefer_local=prefer_local, uploaded_recent=uploaded_recent, period=period, interval=interval)
if recent is None:
    st.error("Sem série de preço disponível. Forneça CSV ou permita yfinance.")
    st.stop()

# FRED data (optional long-term index)
fred_df = get_fred_data()
if fred_df is not None:
    st.sidebar.success("FRED index carregado (WPU081)")

# Load weather: prefer saved forecast file in Dados/
weather_hist = None
weather_forecast = None
local_forecast_name = LOCAL_FORECAST_PATTERN.format(lat=str(lat), lon=str(lon))
local_forecast_path = os.path.join(DATA_DIR, local_forecast_name)
if prefer_local and os.path.exists(local_forecast_path):
    weather_forecast = read_csv_safe(local_forecast_path)
    if weather_forecast is not None:
        st.info(f"Usando forecast Open-Meteo local: {local_forecast_path}")
if weather_forecast is None:
    try:
        weather_forecast = fetch_open_meteo_forecast(lat, lon, days=steps)
    except Exception:
        weather_forecast = None
try:
    weather_hist = fetch_open_meteo_history(lat, lon, pd.to_datetime(hist_start), pd.to_datetime(hist_end))
except Exception:
    weather_hist = None
if weather_hist is None:
    weather_hist = pd.DataFrame({"Date": recent["Date"].unique(), "precip_mm": 0.0})

merged, exog, future_exog = prepare_for_model(recent, weather_hist, weather_forecast)

# KPIs
def compute_kpis_from_series(series):
    if series is None or series.empty:
        return {"last": np.nan, "ma30": np.nan, "ma90": np.nan, "vol30_annual": np.nan}
    s = pd.Series(series).dropna().reset_index(drop=True)
    last = float(s.iloc[-1])
    ma30 = float(s.tail(30).mean()) if len(s) >= 1 else np.nan
    ma90 = float(s.tail(90).mean()) if len(s) >= 1 else np.nan
    vol30 = float(s.pct_change().tail(30).std() * np.sqrt(252)) if len(s) >= 2 else np.nan
    return {"last": last, "ma30": ma30, "ma90": ma90, "vol30_annual": vol30}

kpis = compute_kpis_from_series(merged["Close"])

forecast_mean = None
forecast_conf = None
model_obj = None
wf_backtest = None

if simulate:
    with st.spinner("Executando modelos (pode demorar)..."):
        if use_prophet and PROPHET_AVAILABLE:
            fm, fc, mobj = prophet_forecast(merged, future_weather_df=weather_forecast, steps=steps)
            forecast_mean, forecast_conf, model_obj = fm, fc, mobj
        else:
            fm, fc, mobj = sarimax_forecast(merged["Close"], exog, future_exog=future_exog)
            forecast_mean, forecast_conf, model_obj = fm, fc, mobj
        try:
            wf_backtest = walk_forward_backtest(merged["Close"], exog, train_window=min(180, max(90, len(merged)-30)), forecast_horizon=14)
        except Exception:
            wf_backtest = None

# Plot builder
def build_figure(merged, forecast_mean=None, forecast_conf=None):
    template = "plotly_dark"
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.12,0.62,0.26], vertical_spacing=0.06)
    last_val = kpis["last"] if not np.isnan(kpis["last"]) else 0
    fig.add_trace(go.Indicator(mode="number", value=float(last_val), title={"text":"Último (Close)"}, number={"valueformat":".2f"}), row=1, col=1)
    fig.add_trace(go.Indicator(mode="number", value=float(kpis["ma30"] if not np.isnan(kpis["ma30"]) else 0), title={"text":"MA30"}, number={"valueformat":".2f"}), row=1, col=1)
    dfr = merged.sort_values("Date").reset_index(drop=True)
    dfr["Close"] = pd.to_numeric(dfr["Close"], errors="coerce")
    fig.add_trace(go.Scatter(x=dfr["Date"], y=dfr["Close"], name="Close", mode="lines", line=dict(color="#2ecc71")), row=2, col=1)
    if len(dfr) >= 30:
        dfr["MA30"] = dfr["Close"].rolling(30, min_periods=1).mean()
        fig.add_trace(go.Scatter(x=dfr["Date"], y=dfr["MA30"], name="MA30", line=dict(color="#27ae60", width=2)), row=2, col=1)
    if forecast_mean is not None:
        last_date = merged["Date"].max()
        fut_dates = [last_date + timedelta(days=i+1) for i in range(len(forecast_mean))]
        fig.add_trace(go.Scatter(x=fut_dates, y=forecast_mean, name="Forecast mean", line=dict(color="orange", dash="dash")), row=2, col=1)
        if forecast_conf is not None:
            lower = forecast_conf.iloc[:,0] if hasattr(forecast_conf, "iloc") else forecast_conf["lower"]
            upper = forecast_conf.iloc[:,1] if hasattr(forecast_conf, "iloc") else forecast_conf["upper"]
            fig.add_trace(go.Scatter(x=fut_dates, y=lower, name="Lower CI", line=dict(color="rgba(255,165,0,0.2)"), showlegend=False), row=2, col=1)
            fig.add_trace(go.Scatter(x=fut_dates, y=upper, name="Upper CI", line=dict(color="rgba(255,165,0,0.2)"), fill='tonexty', fillcolor="rgba(255,165,0,0.1)", showlegend=False), row=2, col=1)
    if "precip_mm" in merged.columns:
        fig.add_trace(go.Bar(x=merged["Date"], y=merged["precip_mm"], name="Precip (mm)", marker_color="royalblue", opacity=0.4), row=3, col=1)
    fig.update_layout(height=900, template=template, showlegend=True, paper_bgcolor='black', plot_bgcolor='black',
                      title_text=f"Madeira — Dashboard (atualizado {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')})")
    fig.update_xaxes(title_text="Data", row=2, col=1)
    fig.update_yaxes(title_text="Preço (USD)", row=2, col=1)
    return fig

fig = build_figure(merged, forecast_mean=forecast_mean, forecast_conf=forecast_conf)
st.plotly_chart(fig, use_container_width=True)

# KPIs + signals
col1, col2, col3, col4 = st.columns(4)
col1.metric("Último preço", f"{kpis['last']:.2f}" if not np.isnan(kpis["last"]) else "N/A")
col2.metric("MA30", f"{kpis['ma30']:.2f}" if not np.isnan(kpis["ma30"]) else "N/A")
col3.metric("MA90", f"{kpis['ma90']:.2f}" if not np.isnan(kpis["ma90"]) else "N/A")
col4.metric("Vol anual (proxy)", f"{kpis['vol30_annual']:.4f}" if not np.isnan(kpis["vol30_annual"]) else "N/A")

st.markdown("### Sinais & Recomendação de Hedge")
if forecast_mean is None:
    st.info("Sem forecast (clique em Rodar / Atualizar).")
else:
    mean_val = float(np.nanmean(forecast_mean))
    signal = generate_simple_hedge_signal(forecast_mean, kpis["last"], threshold_pct=0.05)
    st.write(f"Forecast média ({len(forecast_mean)} dias): {mean_val:.2f}")
    st.write(f"Sinal: {signal}")
    if signal == "buy_inventory":
        st.success(f"Comprar estoque agora (previsão de alta). Hedge ratio sugerido: {hedge_ratio:.2f}")
    elif signal == "sell_inventory":
        st.warning("Reduzir estoque (previsão de queda).")
    else:
        st.info("Manter posição.")

st.markdown("### Walk-forward backtest")
if wf_backtest is None:
    st.info("Sem backtest disponível.")
else:
    try:
        wf_mse = mean_squared_error(wf_backtest["obs"].dropna(), wf_backtest["pred"].loc[wf_backtest["obs"].notna()]) if not wf_backtest["obs"].isna().all() else None
        st.write(f"Walk-forward MSE (proxy): {wf_mse:.6f}" if wf_mse is not None else "MSE: N/A")
        st.line_chart(wf_backtest.set_index("idx")[["obs","pred"]].dropna().rename(columns={"obs":"Observed","pred":"Predicted"}))
    except Exception:
        st.write("Erro ao exibir backtest.")

if show_debug:
    st.markdown("### Debug")
    st.write("Merged head:")
    st.dataframe(merged.head(8))
    st.write("Exog cols:")
    st.write(list(exog.columns))
    if model_obj is not None:
        try:
            st.text(str(model_obj.summary()))
        except Exception:
            st.write("Resumo indisponível.")
    if fred_df is not None:
        st.write("FRED head:")
        st.dataframe(fred_df.tail(5))

st.markdown("---")
st.header("Export")
# Export forecast CSV / Excel
if forecast_mean is not None:
    csv_buf = io.StringIO()
    last_date = merged["Date"].max()
    fut_dates = [last_date + timedelta(days=i+1) for i in range(len(forecast_mean))]
    out = pd.DataFrame({"Date": fut_dates, "forecast": list(map(float, forecast_mean))})
    out.to_csv(csv_buf, index=False)
    st.download_button("Baixar forecast CSV", csv_buf.getvalue(), file_name="forecast_lumber.csv", mime="text/csv")

# Consolidated Excel: recent prices + forecast + fred (if available)
def create_consolidated_excel(prices_df, forecast_df=None, fred_df=None):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine='openpyxl') as writer:
        prices_df_copy = prices_df.copy()
        if "Date" in prices_df_copy.columns:
            prices_df_copy["Date"] = pd.to_datetime(prices_df_copy["Date"]).dt.strftime("%Y-%m-%d %H:%M:%S")
        prices_df_copy.to_excel(writer, sheet_name="Recent_Prices", index=False)
        if forecast_df is not None:
            fdf = forecast_df.copy()
            fdf["Date"] = pd.to_datetime(fdf["Date"]).dt.strftime("%Y-%m-%d %H:%M:%S")
            fdf.to_excel(writer, sheet_name="Forecast", index=False)
        if fred_df is not None:
            ff = fred_df.copy()
            ff["Date"] = pd.to_datetime(ff["Date"]).dt.strftime("%Y-%m-%d")
            ff.to_excel(writer, sheet_name="FRED_Index", index=False)
    buf.seek(0)
    return buf

if st.button("Gerar Excel consolidado"):
    forecast_df = None
    if forecast_mean is not None:
        last_date = merged["Date"].max()
        fut_dates = [last_date + timedelta(days=i+1) for i in range(len(forecast_mean))]
        forecast_df = pd.DataFrame({"Date": fut_dates, "forecast": list(map(float, forecast_mean))})
    excel_buf = create_consolidated_excel(recent, forecast_df=forecast_df, fred_df=fred_df)
    st.download_button("Baixar Excel consolidado", data=excel_buf, file_name=f"lumber_consolidated_{datetime.utcnow().strftime('%Y%m%d_%H%M')}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

st.write("Última atualização (UTC):", datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"))
