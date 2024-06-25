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
        # Converta a coluna 'data_ref' para datetime
        df['data_ref'] = pd.to_datetime(df['data_ref'], errors='coerce')
        df.drop(['data_ref','index', 'Unnamed: 0'], axis=1, inplace=True)
        def valores_missing(df):
            for coluna in df.columns:
                if df[coluna].isnull().sum() > 0:
                    if df[coluna].dtype in [np.float64, np.int64]:
                        df[coluna] = df[coluna].fillna(df[coluna].mean())
                    else:
                        df[coluna] = df[coluna].fillna(df[coluna].mode()[0])
            return df
        df_sem_missing = valores_missing(df)

if choice == 'Previsão':
    with open('lightgbm_model_final.pkl', 'rb') as f:
        model = load_model(f)
        predictions = predict_model(model, data=df_sem_missing)
        st.download_button("Baixar Previsões", predictions.to_csv(index=False), file_name="predict_credit_scorring.csv")
