# app.py
# Sudati — Dashboard Estável com FRED, Open-Meteo e Mix de Modelos
import os
import io
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- CONFIGURAÇÕES ---
st.set_page_config(page_title="Sudati — Dashboard Completo", layout="wide", initial_sidebar_state="expanded")
DATA_DIR = "Dados"
os.makedirs(DATA_DIR, exist_ok=True)

# Nomes de arquivos locais
FRED_DATA_FILE = os.path.join(DATA_DIR, "lumber_fred_historical_latest.csv")
FORECAST_DATA_FILE = os.path.join(DATA_DIR, "open_meteo_forecast_-23.0_-51.0.csv")
HISTORY_DATA_FILE = os.path.join(DATA_DIR, "open_meteo_history_-23.0_-51.0.csv")

# --- UTILITÁRIOS ---
@st.cache_data(ttl=3600)
def read_csv_safe(file_path):
    """
    Função segura para ler arquivos CSV com caching.
    """
    try:
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
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

# --- MODELOS E FUNÇÕES DE ANÁLISE ---

def calculate_vaR(data, confidence_level=0.95):
    """
    Calcula o Valor em Risco (VaR) de um DataFrame.
    """
    if data.empty:
        return None
    returns = data.pct_change().dropna()
    return_std = returns.std()
    return_mean = returns.mean()
    var = - (return_mean + 1.65 * return_std) # 1.65 for 95% confidence
    return var

# --- DASHBOARD UI ---

st.title("Sudati — Dashboard de Análise Estratégica")
st.markdown("Plataforma unificada de análise de cenários, sazonalidade e monitoramento meteorológico.")

# Carregar dados
fred_df = read_csv_safe(FRED_DATA_FILE)
weather_forecast_df = read_csv_safe(FORECAST_DATA_FILE)
weather_hist_df = read_csv_safe(HISTORY_DATA_FILE)

# --- MODELO 1: PAINEL DE ANÁLISE DE CENÁRIOS DE HEDGE ---
st.header("1. Análise de Cenários de Hedge")
st.markdown("Simule o impacto de diferentes estratégias de proteção de preços.")

if fred_df is not None and not fred_df.empty:
    current_price_index = fred_df['Price_Index'].iloc[-1]
    hedge_target = st.number_input("Definir Preço-Alvo de Hedge", value=current_price_index, format="%.2f")
    hedge_percentage = st.slider("Percentual de Hedge", 0, 100, 50)
    
    # Simulação simplificada de cenários
    # Cenário otimista (preço sobe 10%)
    optimistic_price = current_price_index * 1.10
    optimistic_profit = (optimistic_price - hedge_target) * (1 - hedge_percentage/100)
    
    # Cenário pessimista (preço cai 10%)
    pessimistic_price = current_price_index * 0.90
    pessimistic_profit = (pessimistic_price - hedge_target) * (1 - hedge_percentage/100)
    
    # Cenário neutro (preço se mantém)
    neutral_profit = (current_price_index - hedge_target) * (1 - hedge_percentage/100)
    
    col_hedge_1, col_hedge_2, col_hedge_3 = st.columns(3)
    with col_hedge_1:
        st.metric("Lucro Cenário Otimista", f"${optimistic_profit:.2f}")
    with col_hedge_2:
        st.metric("Lucro Cenário Neutro", f"${neutral_profit:.2f}")
    with col_hedge_3:
        st.metric("Lucro Cenário Pessimista", f"${pessimistic_profit:.2f}")

    # VaR (Valor em Risco)
    vaR = calculate_vaR(fred_df['Price_Index'])
    if vaR is not None:
        st.info(f"O Valor em Risco (VaR) para o índice de preços é de aproximadamente {vaR:.2f}%")
else:
    st.warning("Dados do FRED necessários para a Análise de Cenários de Hedge.")

# --- MODELO 2: PAINEL DE SAZONALIDADE E ANÁLISE DE LONGO PRAZO ---
st.header("2. Análise de Sazonalidade e Tendência")
st.markdown("Identifique padrões de longo prazo e ciclos de mercado.")

if fred_df is not None and not fred_df.empty:
    fred_df['Year'] = fred_df['Date'].dt.year
    fred_df['Month'] = fred_df['Date'].dt.month

    # Cálculo das médias móveis
    fred_df['MA_12'] = fred_df['Price_Index'].rolling(window=12).mean()
    fred_df['MA_60'] = fred_df['Price_Index'].rolling(window=60).mean()

    # Gráfico de Tendência (com médias móveis)
    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(x=fred_df['Date'], y=fred_df['Price_Index'], mode='lines', name='Índice FRED'))
    fig_trend.add_trace(go.Scatter(x=fred_df['Date'], y=fred_df['MA_12'], mode='lines', name='Média Móvel 12 meses', line=dict(dash='dot')))
    fig_trend.add_trace(go.Scatter(x=fred_df['Date'], y=fred_df['MA_60'], mode='lines', name='Média Móvel 60 meses', line=dict(dash='dash')))
    fig_trend.update_layout(title='Tendência de Preços com Médias Móveis', yaxis_title='Índice (1982=100)', template="plotly_dark")
    st.plotly_chart(fig_trend, use_container_width=True)

    # Gráfico de Sazonalidade (cálculo de médias mensais)
    monthly_avg = fred_df.groupby(fred_df['Date'].dt.month_name())['Price_Index'].mean().reindex(
        ['January', 'February', 'March', 'April', 'May', 'June',
         'July', 'August', 'September', 'October', 'November', 'December']
    )
    
    fig_seasonal = go.Figure()
    fig_seasonal.add_trace(go.Bar(x=monthly_avg.index, y=monthly_avg.values, name='Preço Médio Mensal'))
    fig_seasonal.update_layout(title='Sazonalidade Média de Preços', yaxis_title='Índice Médio (1982=100)', template="plotly_dark")
    st.plotly_chart(fig_seasonal, use_container_width=True)
else:
    st.warning("Dados do FRED necessários para a Análise de Sazonalidade e Tendência.")

# --- MODELO 3: PAINEL DE MONITORAMENTO METEOROLÓGICO ---
st.header("3. Monitoramento Meteorológico")
st.markdown("Monitore as condições de clima nas áreas de fornecimento.")

if weather_hist_df is not None and not weather_hist_df.empty and weather_forecast_df is not None and not weather_forecast_df.empty:
    fig_weather = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1)
    
    # Histórico de precipitação
    fig_weather.add_trace(go.Bar(x=weather_hist_df['Date'], y=weather_hist_df['precip_mm'], name='Precipitação Histórica (mm)'), row=1, col=1)
    fig_weather.update_yaxes(title_text="Precipitação (mm)", row=1, col=1)
    
    # Previsão de precipitação
    fig_weather.add_trace(go.Bar(x=weather_forecast_df['Date'], y=weather_forecast_df['precip_mm'], name='Previsão de Precipitação (mm)', marker_color='orange'), row=2, col=1)
    fig_weather.update_yaxes(title_text="Previsão (mm)", row=2, col=1)
    
    fig_weather.update_layout(title_text="Precipitação Histórica e Previsão", template="plotly_dark")
    st.plotly_chart(fig_weather, use_container_width=True)
else:
    st.info("Dados meteorológicos necessários para o Monitoramento de Clima.")

# --- EXPORTAÇÃO DE DADOS ---
st.markdown("---")
st.header("Exportação")
if fred_df is not None and not fred_df.empty:
    excel_buf = io.BytesIO()
    with pd.ExcelWriter(excel_buf, engine='openpyxl') as writer:
        fred_df.to_excel(writer, sheet_name='FRED Data', index=False)
    excel_buf.seek(0)
    st.download_button("Baixar Dados do FRED (Excel)", data=excel_buf, file_name="fred_lumber_data.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
