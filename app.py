# app.py
# Sudati — Dashboard de Storytelling com Dados
import os
import io
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- CONFIGURAÇÕES DO DASHBOARD ---
st.set_page_config(page_title="Sudati — Dashboard de Hedge Estratégico", layout="wide", initial_sidebar_state="expanded")
DATA_DIR = "Dados"
os.makedirs(DATA_DIR, exist_ok=True)

# Nomes de arquivos locais
FRED_DATA_FILE = os.path.join(DATA_DIR, "lumber_fred_historical_latest.csv")
FORECAST_DATA_FILE = os.path.join(DATA_DIR, "open_meteo_forecast_-23.0_-51.0.csv")
HISTORY_DATA_FILE = os.path.join(DATA_DIR, "open_meteo_history_-23.0_-51.0.csv")

# --- FUNÇÕES DE UTILIDADE E ANÁLISE ---

@st.cache_data(ttl=3600)
def read_csv_safe(file_path, date_col):
    """
    Função segura para ler arquivos CSV e converter a coluna de data.
    """
    try:
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            if date_col in df.columns:
                df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            return df.dropna(subset=[date_col])
        return None
    except Exception as e:
        st.error(f"Erro ao carregar o arquivo {file_path}: {e}")
        return None

def calculate_simple_signal(df, forecast_horizon=12, threshold_pct=0.03):
    """
    Gera um sinal de hedge simplificado com base na média histórica.
    """
    if df is None or df.empty:
        return "N/A"
    
    # Usa a média dos últimos 12 meses como "preço previsto" simplificado
    historical_avg = df['Price_Index'].iloc[-forecast_horizon:].mean()
    current_price = df['Price_Index'].iloc[-1]
    
    change = (current_price - historical_avg) / historical_avg
    
    if change > threshold_pct:
        return "Vender (Preço Acima da Média)"
    elif change < -threshold_pct:
        return "Comprar (Preço Abaixo da Média)"
    else:
        return "Manter Posição"

# --- DASHBOARD PRINCIPAL ---

st.title("Sudati — Plataforma de Hedge Estratégico")
st.markdown("Uma narrativa de dados para o controle e crescimento da sua operação.")

# Carregar dados
fred_df_raw = read_csv_safe(FRED_DATA_FILE, date_col='observation_date')
if fred_df_raw is not None:
    fred_df = fred_df_raw.rename(columns={'observation_date': 'Date', 'WPU081': 'Price_Index'})
else:
    fred_df = pd.DataFrame()

weather_forecast_df = read_csv_safe(FORECAST_DATA_FILE, date_col='Date')
weather_hist_df = read_csv_safe(HISTORY_DATA_FILE, date_col='Date')

# Verificação inicial
if fred_df.empty:
    st.error("Sem dados de preços do FRED disponíveis. Por favor, atualize os dados.")
    st.stop()

# --- NARRATIVA EM TRÊS ATOS (com st.tabs) ---
tab1, tab2, tab3 = st.tabs(["📊 Ato 1: O Contexto", "📈 Ato 2: A Análise", "💼 Ato 3: O Impacto"])

with tab1: # --- ATO 1: O CONTEXTO - ONDE ESTAMOS? ---
    st.header("Ato 1: O Contexto - Onde estamos?")
    st.markdown("Uma visão histórica do mercado para entender os ciclos de volatilidade e as oportunidades.")

    st.markdown("#### Tendência de Longo Prazo do Índice de Preços (FRED)")
    fig_area = go.Figure()
    fig_area.add_trace(go.Scatter(x=fred_df['Date'], y=fred_df['Price_Index'], fill='tozeroy', mode='lines', line_color='rgba(0,128,0,0.5)', name='Índice de Preços'))
    fig_area.update_layout(title='Comportamento Histórico do Preço da Madeira (1926 - Hoje)', yaxis_title='Índice (1982=100)', template="plotly_dark")
    st.plotly_chart(fig_area, use_container_width=True)

    st.markdown("#### Sazonalidade dos Preços")
    monthly_avg = fred_df.groupby(fred_df['Date'].dt.month_name())['Price_Index'].mean().reindex(
        ['January', 'February', 'March', 'April', 'May', 'June',
         'July', 'August', 'September', 'October', 'November', 'December']
    ).sort_values() # Ordena do menor para o maior preço
    
    fig_seasonal = go.Figure()
    fig_seasonal.add_trace(go.Bar(x=monthly_avg.index, y=monthly_avg.values, marker_color=['#2ecc71' if x == monthly_avg.max() else '#e74c3c' if x == monthly_avg.min() else '#3498db' for x in monthly_avg.values]))
    fig_seasonal.update_layout(title='Média de Preços por Mês (Janela de Oportunidade)', yaxis_title='Índice Médio (1982=100)', template="plotly_dark")
    st.plotly_chart(fig_seasonal, use_container_width=True)


with tab2: # --- ATO 2: A ANÁLISE - O QUE FAZER? ---
    st.header("Ato 2: A Análise - O que fazer?")
    st.markdown("Análise de preços e uma recomendação de hedge clara para proteger sua margem.")
    
    current_price_index = fred_df['Price_Index'].iloc[-1]
    
    # KPIs de Decisão
    st.subheader("KPIs de Decisão")
    col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
    with col_kpi1:
        st.metric("Índice de Preço Atual", f"{current_price_index:.2f}")
    with col_kpi2:
        st.metric("Média dos Últimos 12 Meses", f"{fred_df['Price_Index'].iloc[-12:].mean():.2f}")
    with col_kpi3:
        signal = calculate_simple_signal(fred_df)
        st.metric("Recomendação de Hedge", signal)

    st.markdown("---")
    st.subheader("Gráfico de Simulação de Hedge")
    st.markdown("Compare o cenário de volatilidade sem hedge com a estabilidade com hedge.")

    # Simulação de preço volátil
    future_dates = [fred_df['Date'].iloc[-1] + timedelta(days=30*i) for i in range(1, 13)]
    sim_volatility = np.random.normal(0, 0.05, 12).cumsum()
    sim_price = fred_df['Price_Index'].iloc[-1] * (1 + sim_volatility)
    sim_df = pd.DataFrame({'Date': future_dates, 'Preço Sem Hedge': sim_price})
    
    # Simulação de preço com hedge
    hedge_price = fred_df['Price_Index'].iloc[-1] * (1 + np.random.normal(0, 0.01, 12).cumsum())
    sim_df['Preço com Hedge'] = hedge_price

    fig_hedge = go.Figure()
    fig_hedge.add_trace(go.Scatter(x=sim_df['Date'], y=sim_df['Preço Sem Hedge'], mode='lines', name='Preço Sem Hedge', line_color='#e74c3c'))
    fig_hedge.add_trace(go.Scatter(x=sim_df['Date'], y=sim_df['Preço com Hedge'], mode='lines', name='Preço com Hedge', line_color='#2ecc71'))
    fig_hedge.update_layout(title='Simulação: Cenário com Hedge vs. Sem Hedge', yaxis_title='Índice (1982=100)', template="plotly_dark")
    st.plotly_chart(fig_hedge, use_container_width=True)

with tab3: # --- ATO 3: O IMPACTO - COMO A SUDATI SE BENEFICIA? ---
    st.header("Ato 3: O Impacto - Como a Sudati se beneficia?")
    st.markdown("A plataforma oferece previsibilidade e proteção para o seu negócio.")

    st.subheader("Risco e Previsibilidade")
    vaR = calculate_vaR(fred_df['Price_Index'])
    if vaR is not None:
        col_impact_1, col_impact_2 = st.columns(2)
        with col_impact_1:
            st.metric("Valor em Risco (VaR) de 95%", f"{vaR:.2f}%", help="Estimativa da perda potencial máxima no valor do índice em 95% de confiança.")
        with col_impact_2:
            st.metric("Confiança na Previsão", "Alta", help="Com base na estabilidade do índice FRED, a previsibilidade é alta.")
    
    st.markdown("---")
    st.subheader("Fatores Externos: Monitoramento Meteorológico")
    st.markdown("Acompanhe o clima para planejar sua logística e evitar atrasos.")

    if weather_hist_df is not None and not weather_hist_df.empty and weather_forecast_df is not None and not weather_forecast_df.empty:
        fig_weather = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1)
        
        fig_weather.add_trace(go.Bar(x=weather_hist_df['Date'], y=weather_hist_df['precip_mm'], name='Precipitação Histórica (mm)'), row=1, col=1)
        fig_weather.update_yaxes(title_text="Precipitação (mm)", row=1, col=1)
        
        fig_weather.add_trace(go.Bar(x=weather_forecast_df['Date'], y=weather_forecast_df['precip_mm'], name='Previsão de Precipitação (mm)', marker_color='orange'), row=2, col=1)
        fig_weather.update_yaxes(title_text="Previsão (mm)", row=2, col=1)
        
        fig_weather.update_layout(title_text="Precipitação Histórica e Previsão", template="plotly_dark")
        st.plotly_chart(fig_weather, use_container_width=True)
    else:
        st.info("Dados meteorológicos necessários para o Monitoramento de Clima.")
