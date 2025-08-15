# app.py
# Sudati — Dashboard Estável com FRED e Open-Meteo
import os
import io
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- REMOVIDO: Importações de yfinance, statsmodels, etc. ---

# ---------- CONFIG ----------
st.set_page_config(page_title="Sudati — Dashboard Estável", layout="wide", initial_sidebar_state="expanded")
DATA_DIR = "Dados"
os.makedirs(DATA_DIR, exist_ok=True)

# Nomes de arquivos locais
FRED_DATA_FILE = os.path.join(DATA_DIR, "lumber_fred_historical_latest.csv")
FORECAST_DATA_FILE = os.path.join(DATA_DIR, "open_meteo_forecast_-23.0_-51.0.csv")
HISTORY_DATA_FILE = os.path.join(DATA_DIR, "open_meteo_history_-23.0_-51.0.csv")

# ---------- UTIL (leitura de arquivos) ----------
# ADICIONADO: @st.cache_data para otimizar o carregamento de dados
@st.cache_data(ttl=3600)
def read_csv_safe(file_path):
    try:
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            # Conversão de tipo para garantir que a data esteja correta
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            elif 'observation_date' in df.columns:
                df = df.rename(columns={'observation_date': 'Date'})
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            return df.dropna(subset=['Date'])
        return None
    except Exception as e:
        st.error(f"Erro ao carregar o arquivo {file_path}: {e}")
        return None

# ---------- UI & EXECUÇÃO ----------
st.title("Sudati — Dashboard de Preços de Madeira (FRED)")
st.markdown("Análise de longo prazo baseada no índice de preços do FRED e dados meteorológicos.")

with st.sidebar:
    st.header("Configurações")
    lat = st.number_input("Latitude", value=-23.0, format="%.4f")
    lon = st.number_input("Longitude", value=-51.0, format="%.4f")
    st.markdown("---")
    st.info("Os dados são atualizados automaticamente via GitHub Actions. \n\nPara atualizar manualmente, rode `python scripts/update_data.py` no seu terminal.")

# Carregar dados usando a função com caching
fred_df = read_csv_safe(FRED_DATA_FILE)
weather_forecast_df = read_csv_safe(FORECAST_DATA_FILE)
weather_hist_df = read_csv_safe(HISTORY_DATA_FILE)

# Verificação e Exibição de Gráficos
if fred_df is None or fred_df.empty:
    st.warning("Sem dados de preços do FRED disponíveis. Rode o script de atualização para obter.")
else:
    st.success("Dados do FRED carregados com sucesso.")

    # KPI - Cálculo simples
    last_price_index = fred_df['Price_Index'].iloc[-1] if 'Price_Index' in fred_df.columns else "N/A"
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Último Índice FRED", f"{last_price_index:.2f}" if last_price_index != "N/A" else "N/A")
    
    st.markdown("### Tendência Histórica de Preços (FRED)")
    
    fig = go.Figure()
    # CORREÇÃO AQUI: 'WPU081' mudado para 'Price_Index'
    fig.add_trace(go.Scatter(x=fred_df['Date'], y=fred_df['Price_Index'], mode='lines', name='Índice de Preços (WPU081)'))
    fig.update_layout(title='Índice de Preços de Madeira (FRED)', yaxis_title='Índice (1982=100)', template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("### Dados Meteorológicos (Precipitação)")
    
    if weather_hist_df is not None and not weather_hist_df.empty:
        fig_weather = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1)
        
        # Histórico de precipitação
        fig_weather.add_trace(go.Bar(x=weather_hist_df['Date'], y=weather_hist_df['precip_mm'], name='Precipitação Histórica (mm)'), row=1, col=1)
        fig_weather.update_yaxes(title_text="Precipitação (mm)", row=1, col=1)
        
        # Previsão de precipitação
        if weather_forecast_df is not None and not weather_forecast_df.empty:
            fig_weather.add_trace(go.Bar(x=weather_forecast_df['Date'], y=weather_forecast_df['precip_mm'], name='Previsão de Precipitação (mm)', marker_color='orange'), row=2, col=1)
            fig_weather.update_yaxes(title_text="Previsão (mm)", row=2, col=1)
        
        fig_weather.update_layout(title_text="Precipitação Histórica e Previsão", template="plotly_dark")
        st.plotly_chart(fig_weather, use_container_width=True)

    else:
        st.info("Sem dados meteorológicos disponíveis. Rode o script de atualização para obter.")
