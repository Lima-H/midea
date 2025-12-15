#!/bin/zsh

# Script opcional para preparar ambiente e rodar a aplicação Streamlit
# Uso: chmod +x run_streamlit.sh && ./run_streamlit.sh

python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
