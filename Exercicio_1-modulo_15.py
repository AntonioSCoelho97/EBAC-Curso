import streamlit as st
import numpy as np
import pandas as pd

st.set_page_config(page_title = 'Exercício Streamlit',
                    page_icon='https://i.ytimg.com/vi/4ys7otXf_PA/maxresdefault.jpg',
                    layout='wide')

st.title('Exercício_1 - Módulo_15')

@st.cache_data
def carregar_df():
    df = pd.DataFrame({
        'Primeira coluna': [1, 2, 3, 4],
        'Segunda coluna': [10, 20, 30, 40]
        })
    return df
df = carregar_df()

if st.checkbox('Mostrar base de dados'):
    st.write("Aqui está nossa primeira tentativa de usar dados para criar uma tabela:")
    st.dataframe(df.style.highlight_max(axis=0))


st.write("Utilizando o selectbox para escolha do número na variável 'Primeira coluna'")
option = st.selectbox(
    'Qual número você mais gosta?',
     df['Primeira coluna'])

'Você selecionou: ', option

st.write("Utilizando um botão para escolha do número na variável 'Primeira coluna'")
numero =  df['Primeira coluna'].to_list()
numero = list(map(str,numero))
num_pedido = st.text_input('Digite o número desejado')

if st.button('Verificar'):
    consulta = num_pedido in numero
    if consulta:
        f'Você selecionou: {num_pedido}'
    else:
        'Este número não está na tabela. Deseja Incluir?'

st.write("Utilizando o 'data editor' para inclusão ou exclusão de dados do dataframe")
edited_df = st.data_editor(df, num_rows="dynamic")

st.write("Visualizando os dados atualizados")
st.dataframe(edited_df.style.highlight_max(axis=0))

st.write("Fazendo download do dataframe")
if st.button('Download'):
    edited_df.to_csv('./download.csv',index=False, sep=';')

