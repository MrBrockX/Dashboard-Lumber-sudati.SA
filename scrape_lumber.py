import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import os

def scrape_lumber_historical_data():
    """
    Realiza o web scraping de dados históricos de futuros de madeira do Investing.com
    e retorna um DataFrame do pandas.
    """
    url = "https://www.investing.com/commodities/lumber-historical-data"

    # Cabeçalho para simular um navegador e evitar ser bloqueado
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    try:
        print("Baixando dados de futuros de madeira...")
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Lança um erro para códigos de status HTTP ruins

        # Parseia o conteúdo HTML da página
        soup = BeautifulSoup(response.content, 'html.parser')

        # Encontra a tabela de dados históricos. O ID da tabela pode mudar.
        table = soup.find('table', {'id': 'curr_table'})
        
        if not table:
            print("Erro: Tabela de dados históricos não encontrada. O seletor do site pode ter mudado.")
            return None

        # Extrai os cabeçalhos da tabela
        headers = []
        for th in table.find_all('th'):
            headers.append(th.text.strip())

        # Extrai os dados das linhas da tabela
        data = []
        for row in table.find('tbody').find_all('tr'):
            cols = row.find_all('td')
            # Garante que a linha contém dados e não apenas HTML vazio
            if cols:
                row_data = [col.text.strip() for col in cols]
                data.append(row_data)

        # Cria um DataFrame com os dados
        df = pd.DataFrame(data, columns=headers)

        # --- Limpeza e processamento de dados ---
        
        print("Dados baixados com sucesso. Processando...")

        # Converte a coluna 'Date' para o formato de data
        df['Date'] = pd.to_datetime(df['Date'], format='%b %d, %Y')
        
        # Converte colunas numéricas (Price, Open, etc.) para float
        for col in ['Price', 'Open', 'High', 'Low']:
            df[col] = df[col].str.replace(',', '', regex=True).astype(float)
        
        # Limpa e converte a coluna 'Vol.' (Volume)
        def clean_volume(volume):
            if volume == '-':
                return 0.0
            if 'M' in volume:
                return float(volume.replace('M', '')) * 1_000_000
            if 'K' in volume:
                return float(volume.replace('K', '')) * 1_000
            return float(volume)

        df['Volume'] = df['Vol.'].apply(clean_volume)
        
        # Limpa e converte a coluna 'Change %'
        df['Change %'] = df['Change %'].str.replace('%', '', regex=True).astype(float)
        
        # Renomeia as colunas para um formato mais fácil de usar
        df = df.rename(columns={'Vol.': 'Volume (raw)', 'Change %': 'Change (%)'})
        
        # Reorganiza as colunas
        df = df[['Date', 'Price', 'Open', 'High', 'Low', 'Volume', 'Change (%)']]
        
        print("Dados processados e prontos!")
        return df

    except requests.exceptions.RequestException as e:
        print(f"Erro de requisição HTTP: {e}")
    except Exception as e:
        print(f"Ocorreu um erro: {e}")
    
    return None

if __name__ == "__main__":
    # Roda a função de scraping
    lumber_df = scrape_lumber_historical_data()
    
    if lumber_df is not None:
        # Cria a pasta 'Dados' se ela não existir
        output_dir = "Dados"
        os.makedirs(output_dir, exist_ok=True)
        
        # Salva o DataFrame em um arquivo CSV na pasta 'Dados'
        file_path = os.path.join(output_dir, f"lumber_investing_{datetime.now().strftime('%Y-%m-%d')}.csv")
        lumber_df.to_csv(file_path, index=False)
        
        print(f"\nDataFrame salvo em: {file_path}")
        print("\nPrimeiras 5 linhas do DataFrame:")
        print(lumber_df.head())
        print("\nÚltimas 5 linhas do DataFrame:")
        print(lumber_df.tail())
