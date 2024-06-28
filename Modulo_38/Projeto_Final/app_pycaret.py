import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pycaret.classification import *
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, precision_score, f1_score, cohen_kappa_score, matthews_corrcoef
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from io import StringIO, BytesIO
import os
import pickle




# Função principal da aplicação
#@st.cache
def main():
    # Configuração inicial da página da aplicação
    st.set_page_config(
        page_title="EBAC - Exercício 02 do Módulo 38",
        page_icon=':bar_chart:', 
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Configuração do sidebar com markdown e HTML
    st.sidebar.markdown('''
                        <!-- Div centralizada para alinhar o conteúdo -->
                        <div style="text-align:center; white-space: nowrap;">
                            <!-- Imagem com largura ajustada para 50% -->
                            <img src="https://raw.githubusercontent.com/AntonioSCoelho97/EBAC-Curso/main/Modulo_38/Projeto_Final/newebac_logo_black_half.png"  width=50%>
                        </div>

                        # **Profissão: Cientista de Dados**
                        ### **Streamlit VI e Pycaret**

                        <!-- Informações do autor e data -->
                        **Por:** [Antônio Coelho](https://www.linkedin.com/in/antonio-coelho-datascience/)<br>
                        **Data:** 22 de Junho de 2024<br>

                        <!-- Linha horizontal para separação visual -->
                        ---
                        ''', unsafe_allow_html=True)

    # Visualização inicial
    st.markdown('''
                <div style="text-align:center">
                    <img src="https://raw.githubusercontent.com/AntonioSCoelho97/EBAC-Curso/main/Modulo_31/Exercicio_02/cabecalho_notebook.png" alt="cabecalho notebook" width="100%">
                </div>

                ---

                ### **Módulo 38** | Streamlit VI e Pycaret
                ####  Projeto Final: Construção de um Modelo de Credit Scoring para Cartão de Crédito
                

                **Aluno:** [Antônio Coelho](https://www.linkedin.com/in/antonio-coelho-datascience/)<br>
                **Data:** 22 de Junho de 2024.

                ---
                ''', unsafe_allow_html=True)

    st.markdown('''
                <a name="intro"></a> 

                Neste projeto, estamos construindo um credit scoring para cartão de crédito, em um desenho amostral com 15 safras, 
                e utilizando 12 meses de performance.

                ''', unsafe_allow_html=True)   
     
    # Carregar o arquivo CSV
    uploaded_file = st.sidebar.file_uploader("Carregar arquivo CSV", type=["csv", "txt"])

    # Apresentando os dados
    if uploaded_file is not None:
        st.header('Base de Dados')  # Este cabeçalho será exibido somente após o carregamento do arquivo
        df = pd.read_csv(uploaded_file)
        df['mau'] = df['mau'].astype(bool)
        st.write(df.head())

        st.header('Estrutura da Base de Dados')
        # Criando um DataFrame com as informações relevantes
        info_df = pd.DataFrame({
            'Column': df.columns,
            'Non-Null Count': df.notnull().sum(),
            'Dtype': df.dtypes
        }).reset_index(drop=True)
        st.write(info_df)

        # Filtrar as colunas com valores nulos
        colunas_com_missing = info_df[info_df['Non-Null Count'] < len(df)]

        # Criar um novo DataFrame apenas com as colunas que têm dados ausentes
        colunas_missing_df = colunas_com_missing[['Column', 'Non-Null Count']]

        # Escrever as colunas com dados ausentes
        st.write("Colunas com dados ausentes:")
        st.write(colunas_missing_df)

        # Converta a coluna 'data_ref' para datetime
        df['data_ref'] = pd.to_datetime(df['data_ref'], errors='coerce')

        # Ordenando a base de dados
        df = df.sort_values(by='data_ref')

        # Visualização gráfica número de linhas para cada mês
        fig, ax = plt.subplots(figsize=(16,8))
        mes = df['data_ref'].dt.strftime('%Y-%m')
        sns.countplot(data=df, x=mes,hue=mes, palette='Set1', ax=ax, legend=False)
        ax.set_xlabel('Mês')
        ax.set_ylabel('Número de linhas')
        ax.set_title('Número de linhas para cada mês da base de dados')
        st.pyplot(fig)

        st.header('Analisando a Variável Resposta')
        resposta = df['mau'].value_counts()
        st.write(resposta)
        if resposta[1] != resposta[0]:
            st.write(f'Os dados estão desbalanceados! Há {resposta[1]} clientes classificados como "mau" e {resposta[0]} clientes classificados como "bom"')

        st.header('Análise Descritiva Univariada dos Dados')
        # Visualização Gráfica das Variáveis Qualitativas e/ou Quantitativas Discretas
        st.write('Visualização gráfica das variáveis qualitativas e/ou quantitativas discretas')
        plt.rc('figure', figsize=(16, 12))
        fig, axes = plt.subplots(3, 3)
        sns.countplot(ax=axes[0, 0], x='sexo', data=df, hue='sexo', palette='Set1', legend=False)
        sns.countplot(ax=axes[0, 1], x='posse_de_veiculo', data=df, hue='posse_de_veiculo', palette='Set1', legend=False)
        sns.countplot(ax=axes[0, 2], x='posse_de_imovel', data=df, hue='posse_de_imovel', palette='Set1', legend=False)
        sns.countplot(ax=axes[1, 0], x='qtd_filhos', data=df, hue='qtd_filhos', palette='Set1', legend=False)
        sns.countplot(ax=axes[1, 1], x='tipo_renda', data=df, hue='tipo_renda', palette='Set1', legend=False)
        axes[1, 1].set_xticks(range(len(df['tipo_renda'].unique())))
        axes[1, 1].set_xticklabels(df['tipo_renda'].unique(), rotation=45, ha='right')
        sns.countplot(ax=axes[1, 2], x='educacao', data=df, hue='educacao', palette='Set1', legend=False)
        axes[1, 2].set_xticks(range(len(df['educacao'].unique())))
        axes[1, 2].set_xticklabels(df['educacao'].unique(), rotation=45, ha='right')
        sns.countplot(ax=axes[2, 0], x='estado_civil', data=df, hue='estado_civil', palette='Set1', legend=False)
        axes[2, 0].set_xticks(range(len(df['estado_civil'].unique())))
        axes[2, 0].set_xticklabels(df['estado_civil'].unique(), rotation=45, ha='right')
        sns.countplot(ax=axes[2, 1], x='tipo_residencia', data=df, hue='tipo_residencia', palette='Set1', legend=False)
        axes[2, 1].set_xticks(range(len(df['tipo_residencia'].unique())))
        axes[2, 1].set_xticklabels(df['tipo_residencia'].unique(), rotation=45, ha='right')
        sns.countplot(ax=axes[2, 2], x='qt_pessoas_residencia', data=df, hue='qt_pessoas_residencia', palette='Set1', legend=False)
        # Ajustar o espaçamento
        plt.subplots_adjust(wspace=0.7, hspace=0.7)
        st.pyplot(fig)

        # Visualização Gráfica das Variáveis Quantitativas Contínuas via histograma
        st.write('Visualização gráfica das variáveis quantitativas contínuas via histograma')
        plt.rc('figure', figsize=(16, 6))
        fig, axes = plt.subplots(1, 3, figsize=(16, 6))
        sns.histplot(df['idade'], bins=10, kde = True, ax = axes[0])
        sns.histplot(df['tempo_emprego'], bins=10, kde = True, ax = axes[1])
        # Apresentando o gráfico de renda com limitação de escala em função da distorção do eixo y em relação ao eixo x
        sns.histplot(df['renda'], bins=10, kde=True, ax=axes[2])
        axes[2].set_ylim(0, 6000)  # Defina o limite superior do eixo y
        #ajustar o espaçamento
        plt.subplots_adjust(wspace=0.3)
        st.pyplot(fig)

        st.header('Análise Descritiva Bivariada dos Dados')
        st.write('Visualização Gráfica das Variáveis Qualitativas e/ou Quantitativas Discretas')
        plt.rc('figure', figsize=(16, 12))
        fig, axes = plt.subplots(3, 3)
        sns.countplot(ax = axes[0, 0], x='sexo', hue='mau', data=df)
        sns.countplot(ax = axes[0, 1], x='posse_de_veiculo', hue='mau', data=df)
        sns.countplot(ax = axes[0, 2], x='posse_de_imovel', hue='mau', data=df)
        sns.countplot(ax = axes[1, 0], x='qtd_filhos', hue='mau', data=df) 
        sns.countplot(ax = axes[1, 1], x='tipo_renda', hue='mau', data=df)
        axes[1, 1].set_xticks(range(len(df['tipo_renda'].unique())))
        axes[1, 1].set_xticklabels(axes[1, 1].get_xticklabels(), rotation=45, ha='right')
        sns.countplot(ax = axes[1, 2], x='educacao', hue='mau', data=df)
        axes[1, 2].set_xticks(range(len(df['educacao'].unique())))
        axes[1, 2].set_xticklabels(axes[1, 2].get_xticklabels(), rotation=45, ha='right')
        sns.countplot(ax = axes[2, 0], x='estado_civil', hue='mau', data=df)
        axes[2, 0].set_xticks(range(len(df['estado_civil'].unique())))
        axes[2, 0].set_xticklabels(axes[2, 0].get_xticklabels(), rotation=45, ha='right')
        sns.countplot(ax = axes[2, 1], x='tipo_residencia', hue='mau', data=df)
        axes[2, 1].set_xticks(range(len(df['tipo_residencia'].unique())))
        axes[2, 1].set_xticklabels(axes[2, 1].get_xticklabels(), rotation=45, ha='right')
        sns.countplot(ax = axes[2, 2], x='qt_pessoas_residencia', hue='mau', data=df)
        #ajustar o espaçamento
        plt.subplots_adjust(wspace=0.7, hspace=0.7)
        st.pyplot(fig)

        st.write('Visualização Gráfica das Variáveis Quantitativas Contínuas via histograma')
        plt.rc('figure', figsize=(16, 6))
        fig, axes = plt.subplots(1, 3, figsize=(16, 6))
        sns.histplot(data=df, x='idade', hue='mau', multiple='stack', kde = True, ax = axes[0])
        sns.histplot(data=df, x='tempo_emprego', hue='mau', multiple='stack', kde = True, ax = axes[1])
        # Apresentando o gráfico de renda com limitação de escala em função da distorção do eixo y em relação ao eixo x
        sns.histplot(data=df, x='renda', hue='mau', multiple='stack', kde = True, ax = axes[2])
        axes[2].set_xlim(0, 2000000)  # Defina o limite superior do eixo x
        axes[2].set_ylim(0, 200)  # Defina o limite superior do eixo y
        #ajustar o espaçamento
        plt.subplots_adjust(wspace=0.3)
        st.pyplot(fig)

        st.header('Iniciando o tratamento dos dados para a aplicação do modelo')
        st.write('Removendo as variáveis desnecessárias: data_ref, index e Unnamed: 0')
        df.drop(['data_ref','index', 'Unnamed: 0'], axis=1, inplace=True)
        # Criando um DataFrame com as informações relevantes
        info_df = pd.DataFrame({
            'Column': df.columns,
            'Non-Null Count': df.notnull().sum(),
            'Dtype': df.dtypes
        }).reset_index(drop=True)
        st.dataframe(info_df)

        st.write(' Tratando valores missing: média para variável numérica e moda para variável categórica')
        def valores_missing(df):
            for coluna in df.columns:
                if df[coluna].isnull().sum() > 0:
                    if df[coluna].dtype in [np.float64, np.int64]:
                        df[coluna] = df[coluna].fillna(df[coluna].mean())
                    else:
                        df[coluna] = df[coluna].fillna(df[coluna].mode()[0])
            return df
        df_sem_missing = valores_missing(df)
        # Criando um DataFrame com as informações relevantes
        info_df = pd.DataFrame({
            'Column': df_sem_missing.columns,
            'Non-Null Count': df_sem_missing.notnull().sum(),
            'Dtype': df_sem_missing.dtypes
        }).reset_index(drop=True)
        st.dataframe(info_df)

        st.header('Carregando o modelo')
        link_pkl = 'https://www.dropbox.com/scl/fi/s8xyl0s4zaf7zfeetyqku/lightgbm_model_final.pkl?rlkey=5zwlx24fhbrsrdp64s0y3o4mp&dl=0'
        model = load_model(link_pkl, 'rb')
        # model = load_model('./lightgbm_model_final.pkl', 'rb')
        st.write(model)

        st.header('Fazendo as predições')
        predictions = predict_model(model, data=df_sem_missing)
        st.write(predictions.head())

        st.header('Visualizando os resultados obtidos')
        y_true = predictions.mau
        y_pred = predictions.prediction_label
        st.write(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
        st.write(f"AUC: {roc_auc_score(y_true, y_pred):.4f}")
        st.write(f"Recall: {recall_score(y_true, y_pred):.4f}")
        st.write(f"Precisão: {precision_score(y_true, y_pred):.4f}")
        st.write(f"F1-score: {f1_score(y_true, y_pred):.4f}")
        st.write(f"Kappa: {cohen_kappa_score(y_true, y_pred):.4f}")
        st.write(f"MCC: {matthews_corrcoef(y_true, y_pred):.4f}")
        
        st.header('Visualizando a matriz de classificação')
        cm = confusion_matrix(y_true, y_pred)
        labels = ['Bom', 'Mau']
        plt.figure(figsize=(5, 3))
        sns.heatmap(cm, annot=True, fmt='d', cmap='coolwarm', cbar=False, xticklabels=labels, yticklabels=labels)
        plt.xlabel('Previsão')
        plt.ylabel('Real')
        plt.title('Matriz de Confusão')
        st.pyplot()

        # Adiciona um botão de download para as previsões com o nome do arquivo
        #if st.download_button("Baixar Previsões", predictions.to_csv('predict_credit_scorring.csv'), key="download_button"):
         #   st.success("Arquivo predict_credit_scorring.csv baixado com sucesso!")

        # Crie um botão para salvar as previsões em um arquivo CSV
        if st.download_button("Baixar Previsões", predictions.to_csv(index=False), file_name="predict_credit_scorring.csv"):
            st.success("Arquivo baixado com sucesso!")

# Desativar o aviso sobre o uso global do Pyplot
st.set_option('deprecation.showPyplotGlobalUse', False)
        

if __name__ == '__main__':
    main()
