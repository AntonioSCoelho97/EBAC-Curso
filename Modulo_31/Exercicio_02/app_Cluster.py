# Importando as bibliotecas
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import streamlit as st

from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from gower import gower_matrix
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler

# Função principal da aplicação
def main():
    # Configuração inicial da página da aplicação
    st.set_page_config(page_title = 'Tarefa - Agrupamento hierárquico', \
        layout="wide",
        initial_sidebar_state='expanded'
    )

    st.markdown('''
                <div style="text-align:center">
                    <img src="https://raw.githubusercontent.com/AntonioSCoelho97/EBAC-Curso/main/Modulo_31/Exercicio_02/cabecalho_notebook.png"  width=100%>
                </div>

                # **Curso: Ciência de Dados**
                ### **Projeto de Agrupamento Hierárquico**

                **Por:** [Antônio Coelho](https://www.linkedin.com/in/antonio-coelho-datascience/)<br>
                **Data:** 16 de abril de 2024.<br>

                ---
                ''', unsafe_allow_html=True)

    # Título principal da aplicação
    st.write("""
Neste exercício vamos usar a base [online shoppers purchase intention](https://archive.ics.uci.edu/ml/datasets/Online+Shoppers+Purchasing+Intention+Dataset) de Sakar, C.O., Polat, S.O., Katircioglu, M. et al. Neural Comput & Applic (2018). [Web Link](https://doi.org/10.1007/s00521-018-3523-0).
A base trata de registros de 12.330 sessões de acesso a páginas, cada sessão sendo de um único usuário em um período de 12 meses, para posteriormente estudarmos a relação entre o design da página e o perfil do cliente - "Será que clientes com comportamento de navegação diferentes possuem propensão a compra diferente?" 
Nosso objetivo agora é agrupar as sessões de acesso ao portal considerando o comportamento de acesso e informações da data, como a proximidade a uma data especial, fim de semana e o mês.

|Variavel                |Descrição          | 
|------------------------|:-------------------| 
|Administrative          | Quantidade de acessos em páginas administrativas| 
|Administrative_Duration | Tempo de acesso em páginas administrativas | 
|Informational           | Quantidade de acessos em páginas informativas  | 
|Informational_Duration  | Tempo de acesso em páginas informativas  | 
|ProductRelated          | Quantidade de acessos em páginas de produtos | 
|ProductRelated_Duration | Tempo de acesso em páginas de produtos | 
|BounceRates             | *Percentual de visitantes que entram no site e saem sem acionar outros *requests* durante a sessão  | 
|ExitRates               | * Soma de vezes que a página é visualizada por último em uma sessão dividido pelo total de visualizações | 
|PageValues              | * Representa o valor médio de uma página da Web que um usuário visitou antes de concluir uma transação de comércio eletrônico | 
|SpecialDay              | Indica a proximidade a uma data festiva (dia das mães etc) | 
|Month                   | Mês  | 
|OperatingSystems        | Sistema operacional do visitante | 
|Browser                 | Browser do visitante | 
|Region                  | Região | 
|TrafficType             | Tipo de tráfego                  | 
|VisitorType             | Tipo de visitante: novo ou recorrente | 
|Weekend                 | Indica final de semana | 
|Revenue                 | Indica se houve compra ou não |
 
\* variávels calculadas pelo google analytics""")
    st.markdown("---")

    # Gerando DataFrame
    df = pd.read_csv('https://raw.githubusercontent.com/AntonioSCoelho97/EBAC-Curso/main/Modulo_31/Exercicio_02/online_shoppers_intention.csv')

    st.write('#### Visualizando o DataFrame')
    st.dataframe(df.head())

    st.write('#### Extratificando os acessos (houve compra e não houve compra)')
    st.write(df.Revenue.value_counts(dropna=False))


    st.write('### Análise descritiva')
    st.write('#### Faça uma análise descritiva das variáveis do escopo.')
    st.write('#### - Verifique a distribuição dessas variáveis')
    st.write('#### - Veja se há valores *missing* e caso haja, decida o que fazer')
    st.write('#### - Faça mais algum tratamento nas variáveis caso ache pertinente')

    st.write('#### Separando as variáveis para análise')
    df_var_1 = df.iloc[:,:6]
    df_var_2 = df[['SpecialDay','Weekend','Month']]
    df_1 = pd.concat([df_var_1,df_var_2], axis=1)
    st.dataframe(df_1.head())

    # Criando uma cópia do DataFrame para visualização gráfica
    df_2 = df_1.copy()

    # Convertendo as variáveis 'Month' e 'Weekend' em numéricas para visualização gráfica
    dict_mes = {'Feb':2, 'Mar': 3, 'May': 5, 'June': 6, 'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10,
       'Nov': 11, 'Dec': 12}
    df_2['Month'] = df_2['Month'].map(dict_mes)
    df_2['Weekend'] = df_2['Weekend'].astype(int)

    st.write('#### Verificando a distribuição dessas variáveis')
    sns.pairplot(df_2.iloc[:,6:], hue = 'SpecialDay')
    st.pyplot(plt) 
    # Pela limitação do web service será apresentada a figura 

    # st.write('#### Verificando a distribuição dessas variáveis')
    # st.image("https://raw.githubusercontent.com/AntonioSCoelho97/EBAC-Curso/main/Modulo_31/Exercicio_02/pairplot.png")

    st.write('#### Procurando valores do tipo "missing"')
    st.write(df_1.isna().sum())


    st.write('### **Obs:**')
    st.write('#### - As seções de acesso apresentam pouco proximidade a uma data festiva, ocorrendo uma pequena parcela nos meses de fevereiro e maio;')
    st.write('#### - A variável informacional é a que apresenta a menor quantidade de acesso e a ProductRelated a que apresenta a maior; e')
    st.write('#### - Há necessidade de padronização das variáveis em função da variação das escalas.')

    st.write('### Variáveis de agrupamento')
    st.write('#### Liste as variáveis que você vai querer utilizar. Essa é uma atividade importante do projeto, e tipicamente não a recebemos pronta. Não há resposta pronta ou correta, mas apenas critérios e a sua decisão. Os critérios são os seguintes:')
    st.write('#### - Selecione para o agrupamento variáveis que descrevam o padrão de navegação na sessão.')
    st.write('#### - Selecione variáveis que indiquem a característica da data.')
    st.write('#### - Não se esqueça de que você vai precisar realizar um tratamento especial para variáveis qualitativas.')
    st.write('#### - Trate adequadamente valores faltantes.')

    # Selecionando para o agrupamento variáveis que descrevam o padrão de navegação na sessão.
    variaveis = df_1.columns.values[:6]

    # Selecionando variáveis que indiquem a característica da data.
    variaveis_cat = df_1.columns.values[6:]

    st.write('#### Gerando DataFrame com as variáveis "dummies"')
    df_3 = pd.get_dummies(df_1.dropna(), columns = variaveis_cat)
    st.dataframe(df_3.head())

    st.write('#### Padronizando as variáveis')
    padronizador = StandardScaler()
    df_padronizado = padronizador.fit_transform(df_3.iloc[:,:6])
    df_pd = pd.DataFrame(df_padronizado, columns = df_3.iloc[:,:6].columns)

    st.write('#### Ajustando o novo DataFrame')
    variaveis_cat_dummies = df_3.columns.values[6:]
    df_var_pd = pd.concat([df_pd,df_3[variaveis_cat_dummies]], axis=1)
    df_var_pd[variaveis_cat_dummies] = df_var_pd[variaveis_cat_dummies].astype(int)
    st.dataframe(df_var_pd.head())

    st.write('### Número de grupos') 
    st.write('#### Nesta atividade vamos adotar uma abordagem bem pragmática e avaliar agrupamentos hierárquicos com 3 e 4 grupos, por estarem bem alinhados com uma expectativa e estratégia do diretor da empresa.') 
    st.write('#### *Atenção*: Cuidado se quiser fazer o dendrograma, pois com muitas observações ele pode ser mais complicado de fazer, e dependendo de como for o comando, ele pode travar o *kernell* do seu python.')
    st.write(' ')
    st.write('### Avaliação dos grupos') 
    st.write('#### Construa os agrupamentos com a técnica adequada que vimos em aula. Não se esqueça de tratar variáveis qualitativas, padronizar escalas das quantitativas, tratar valores faltantes e utilizar a distância correta.')
    st.write('#### Faça uma análise descritiva para pelo menos duas soluções de agrupamentos (duas quantidades diferentes de grupos) sugeridas no item anterior, utilizando as variáveis que estão no escopo do agrupamento.')
    st.write('#### - Com base nesta análise e nas análises anteriores, decida pelo agrupamento final.')
    st.write('#### - Se puder, sugira nomes para os grupos.')

    # Criando lista com a identificação das variáveis categóricas
    vars_cat = [True if x in {'SpecialDay_0.0', 'SpecialDay_0.2', 'SpecialDay_0.4',
       'SpecialDay_0.6', 'SpecialDay_0.8', 'SpecialDay_1.0',
       'Weekend_False', 'Weekend_True', 'Month_Aug', 'Month_Dec',
       'Month_Feb', 'Month_Jul', 'Month_June', 'Month_Mar', 'Month_May',
       'Month_Nov', 'Month_Oct', 'Month_Sep'} else False for x in df_var_pd.columns]

    # Calculando a matriz de distâncias utilizando a distância de Gower
    distancia_gower = gower_matrix(df_var_pd, cat_features=vars_cat)

    # ajustando o formato da matriz de distâncias para alimentar o algoritmo
    gdv = squareform(distancia_gower,force='tovector')

    st.write('#### Treinando o agrupamento')
    Z = linkage(gdv, method='ward')
    Z_df = pd.DataFrame(Z,columns=['id1','id2','dist','n'])
    st.dataframe(Z_df.head())

    # st.write('#### Visualizando o Dendrograma')
    # fig,axs = plt.subplots(1,1,figsize=(12,12))
    # dn = dendrogram(Z, truncate_mode='level',p=30,show_leaf_counts=True,ax=axs, color_threshold=.24)
    # st.write(f"Leaves = {len(dn['leaves'])}")
    # st.pyplot(plt)
    # por limitação do web service não será apresentada a figura

    st.write('#### avaliando agrupamentos hierárquicos com 3 grupos')
    df_grupo_3 = pd.concat([df[variaveis], df[variaveis_cat], df[['BounceRates','Revenue']]], axis=1)
    df_grupo_3['grupo_3'] = fcluster(Z, 3, criterion='maxclust')
    st.write(df_grupo_3.grupo_3.value_counts())

    # Convertendo a varíavel 'grupo' em 'str' para melhor visualização
    df_grupo_3['grupo_3'] = df_grupo_3['grupo_3'].astype(str)

    st.write('#### Verificando a distribuição dessas variáveis')
    sns.pairplot(df_grupo_3.iloc[:,6:], hue = 'grupo_3')
    st.pyplot(plt)

    st.write('#### Verificando a composição do agrupamento com a decisão de compra considerando a proximidade de uma data especial')
    df_cross_3_1 = pd.crosstab(index=[df_grupo_3['Revenue'], df_grupo_3['SpecialDay']], 
                       columns=df_grupo_3['grupo_3'], values=df_grupo_3['SpecialDay'], 
                       aggfunc='count').fillna(0).astype(int)
    st.dataframe(df_cross_3_1)

    st.write('#### Verificando a composição do agrupamento com a decisão de compra considerando se é fim de semana ou não')
    df_cross_3_2 = pd.crosstab(index=[df_grupo_3['Revenue'], df_grupo_3['Weekend']], 
                       columns=df_grupo_3['grupo_3'], values=df_grupo_3['Weekend'], 
                       aggfunc='count').fillna(0).astype(int)
    st.dataframe(df_cross_3_2)

    st.write('#### avaliando agrupamentos hierárquicos com 4 grupos')
    df_grupo_4 = pd.concat([df[variaveis], df[variaveis_cat], df[['BounceRates','Revenue']]], axis=1)
    df_grupo_4['grupo_4'] = fcluster(Z, 4, criterion='maxclust')
    st.write(df_grupo_4.grupo_4.value_counts())

    # Convertendo a varíavel 'grupo' em 'str' para melhor visualização
    df_grupo_4['grupo_4'] = df_grupo_4['grupo_4'].astype(str)

    st.write('#### Verificando a distribuição dessas variáveis')
    sns.pairplot(df_grupo_4.iloc[:,6:], hue = 'grupo_4')
    st.pyplot(plt)

    st.write('#### Verificando a composição do agrupamento com a decisão de compra considerando a proximidade de uma data especial')
    df_cross_4_1 = pd.crosstab(index=[df_grupo_4['Revenue'], df_grupo_4['SpecialDay']], 
                       columns=df_grupo_4['grupo_4'], values=df_grupo_4['SpecialDay'], 
                       aggfunc='count').fillna(0).astype(int)
    st.dataframe(df_cross_4_1)

    st.write('#### Verificando a composição do agrupamento com a decisão de compra considerando se é fim de semana ou não')
    df_cross_2 = pd.crosstab(index=[df_grupo_4['Revenue'], df_grupo_4['Weekend']], 
                       columns=df_grupo_4['grupo_4'], values=df_grupo_4['Weekend'], 
                       aggfunc='count').fillna(0).astype(int)
    st.dataframe(df_cross_2)

    st.write('#### Visualizando o quantitativo de clientes com compras em cada grupo para o agrupamento com 4 grupos')
    crosstab_g4 = pd.crosstab(df_grupo_4['Revenue'], df_grupo_4['grupo_4'])
    st.dataframe(crosstab_g4)

    st.write('#### Verificando o percentual de compra em cada grupo')
    crosstab_g4_norm = 100*crosstab_g4.div(crosstab_g4.sum(axis=0), axis=1)
    st.dataframe(crosstab_g4_norm)


    st.write('### Análise:')
    st.write('#### - Foram testados os quatro tipos de ligações para o algoritmo e o que apresentou melhor resultado para o modelo em questão foi o tipo "Ward";')
    st.write('#### - Os demais tipos apresentaram grupos pequenos nas divisões de 03 e/ou 04 grupos;')
    st.write('#### - Os dois agrupamentos (com 3 e 4 grupos) diferem pelo desmembramento de um dos grupos; e')
    st.write('#### - Considerando que o desmembramento do grupo de número "3", como o novo agrupamento: 4 grupos, acrescentou um grupo com clientes mais propensos à compra, representando cerca de 17,90% do total de acessos; adotaremos o agrupamento com 4 grupos.')
    st.write(' ')
    st.write('### Avaliação de resultados') 
    st.write('#### Avalie os grupos obtidos com relação às variáveis fora do escopo da análise (minimamente *bounce rate* e *revenue*).')
    st.write('#### - Qual grupo possui clientes mais propensos à compra?')

    sns.pairplot(df_grupo_4.iloc[:,8:], hue = 'grupo_4')
    st.pyplot(plt)

    st.write('#### Visualizando o quantitativo de clientes com compras em cada grupo para o agrupamento com 3 grupos')
    st.dataframe(crosstab_g4_norm)


    st.write('### **Conclusão:**')
    st.write('#### - O **grupo 3**, **clientes que acessam 100% no meio da semana e 100% distantes de datas especiais**, apresentam o maior propensão à compras, num percentual de **24,92%**).')

if __name__ == '__main__':
	main()



