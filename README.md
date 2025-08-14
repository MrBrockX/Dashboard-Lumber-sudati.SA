# Painel-Lumber-sudati.SA

Aplicativo Streamlit unificado para previsão de preços de madeira (LBR=F) com exógenas meteorológicas (Open-Meteo), SARIMAX/Prophet, visualização Plotly e export CSV/Excel.

Estrutura esperada
- app.py                         # app Streamlit principal
- requirements.txt               # dependências
- Dados/                         # CSVs gerados/armazenados (workflow grava aqui)
- .github/workflows/update_data.yml

Como usar (local)
1. Criar venv:
   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate

2. Instalar dependências:
   pip install -r requirements.txt

3. Executar:
   streamlit run app.py

Deploy no Streamlit Cloud
- Conectar repositório ao Streamlit Cloud.
- Criar nova app apontando para `app.py` no branch `principal` (ou seu branch).
- Se usar Prophet, prefira testá-lo localmente; depende de `cmdstanpy` e builds pesados.

Automação (GitHub Actions)
- Há um workflow em `.github/workflows/update_data.yml` que roda periodicamente e gera CSVs em `Dados/`.
- Caso queira alterar frequência, edite o workflow.

Observações
- Mantenha `requirements.txt` sem pacotes pesados se pretende deploy rápido no Streamlit Cloud.
- Se quiser, posso abrir um PR com esses arquivos já adicionados.
