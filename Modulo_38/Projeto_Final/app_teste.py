import streamlit as st
import pandas as pd
from pycaret.classification import *
import os

if os.path.exists('dataset.csv'):
    df = pd.read_csv('dataset.csv', index_col=None)

with st.sidebar:
    st.title('Teste Pycaret')
    choice = st.radio('Selecione uma das opções:', ['Arquivo', 'Previsão'])

if choice == 'Arquivo':
    st.title('Carregue seu dataset')
    file = st.file_uploader('Carregue seu dataset')
    if file:
        df = pd.read_csv(file, index_col=None)
        df.to_csv('dataset.csv', index=None)
        st.dataframe(df)

if choice == 'Previsão':
    with open('lightgbm_model_final.pkl', 'rb') as f:
        st.download_button('Download do modelo treinado', f, file_name='lightgbm_model_final.pkl')
