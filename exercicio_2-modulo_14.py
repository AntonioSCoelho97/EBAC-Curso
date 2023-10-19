import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

import sys

print ('Quantidade de argumentos:', len(sys.argv), 'argumentos.')
print ('Lista de argumentos:', sys.argv)

# Lista dos meses solicitados
lista_meses = sys.argv[1:]

def plota_pivot_table(df, value, index, func, ylabel, xlabel, opcao='nada'):
    if opcao == 'nada':
        pd.pivot_table(df, values=value, index=index,
                       aggfunc=func).plot(figsize=[15, 5])
    elif opcao == 'sort':
        pd.pivot_table(df, values=value, index=index,
                       aggfunc=func).sort_values(value).plot(figsize=[15, 5])
    elif opcao == 'unstack':
        pd.pivot_table(df, values=value, index=index,
                       aggfunc=func).unstack().plot(figsize=[15, 5])
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    return None

def gerar_graficos(mes):
    endereco = './input/SINASC_RO_2019_'+mes+'.csv'
    sinasc = pd.read_csv(endereco) # Conectando a base de dados
    sinasc.head()
    max_data = sinasc.DTNASC.max()[:7] # Identificando ano e mês
    os.makedirs('./output/figs/'+max_data, exist_ok=True) # Criando o diretório de destino
    # Gerando o gráfico 'media idade mae por sexo'
    plota_pivot_table(sinasc, 'IDADEMAE', ['DTNASC', 'SEXO'], 'mean', 'media idade mae','data de nascimento','unstack')
    plt.savefig('./output/figs/'+max_data+'/media idade mae por sexo.png')
    # Gerando o gráfico 'media peso bebe por sexo'
    plota_pivot_table(sinasc, 'PESO', ['DTNASC', 'SEXO'], 'mean', 'media peso bebe','data de nascimento','unstack')
    plt.savefig('./output/figs/'+max_data+'/media peso bebe por sexo.png')
    # Gerando o gráfico 'media apgar1 por escolaridade mae'
    plota_pivot_table(sinasc, 'PESO', 'ESCMAE', 'median', 'apgar1 medio','gestacao','sort')
    plt.savefig('./output/figs/'+max_data+'/media apgar1 por escolaridade mae.png')
    # Gerando o gráfico 'media apgar1 por gestacao'
    plota_pivot_table(sinasc, 'APGAR1', 'GESTACAO', 'mean', 'apgar1 medio','gestacao','sort')
    plt.savefig('./output/figs/'+max_data+'/media apgar1 por gestacao.png')
    # Gerando o gráfico 'media apgar5 por gestacao'
    plota_pivot_table(sinasc, 'APGAR5', 'GESTACAO', 'mean', 'apgar5 medio','gestacao','sort')
    plt.savefig('./output/figs/'+max_data+'/media apgar5 por gestacao.png')
    return None

# Gerando os gráficos e salvando nos respectivos diretórios
for mes in lista_meses:
    gerar_graficos(mes)