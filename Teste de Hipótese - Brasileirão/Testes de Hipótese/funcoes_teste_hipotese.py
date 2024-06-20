import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scipy.stats as stats
import sys


def pontuacao_porjogo_treinador(arquivo):
    # Ler o arquivo CSV e definir a primeira coluna como índice
    df = pd.read_csv(arquivo, index_col=0)
    
    # Plotar o gráfico de barras e a linha sobreposta
    plt.bar(df.index, df['Número de jogos'], color=sns.color_palette('deep'), label='Número de jogos')
    plt.plot(df.index, df['Pontuação média'], color='red', marker='o', label='Média de pontos por partida')

    # Adicionar valores da média de pontuação nas barras
    for i, valor in enumerate(df['Pontuação média']):
        plt.text(i, valor, str(round(valor, 2)), ha='center', va='bottom')

    # Adicionar rótulos e título ao gráfico
    plt.xlabel('Técnico')
    plt.ylabel('Pontos')
    plt.title('Pontuação média e número de jogos')
    plt.legend()
    
    # Exibir o gráfico
    plt.tight_layout()
    plt.show()


def barras_xG_treinador(arquivo):
    # Ler o arquivo CSV
    df = pd.read_csv(arquivo, index_col=0)
    
    # Determinar a largura das barras e os índices dos treinadores
    largura_barra = 0.35
    tecnico_indices = range(len(df.index))

    # Plotar as barras de xG a Favor médio
    plt.bar(tecnico_indices, df['xG a Favor médio'], width=largura_barra, color='blue', label='xG a Favor médio')

    # Plotar as barras de xG Contra médio com um pequeno deslocamento para a direita
    plt.bar([i + largura_barra for i in tecnico_indices], df['xG Contra médio'], width=largura_barra, color='red', label='xG Contra médio')

    # Adicionar rótulos e título ao gráfico
    plt.xlabel('Treinador')
    plt.ylabel('xG')
    plt.title('xG a Favor e xG Contra Médios por Treinador')
    plt.xticks([i + largura_barra / 2 for i in tecnico_indices], df.index)  # Colocar os rótulos no meio das barras
    plt.legend()
    
    # Exibir o gráfico
    plt.tight_layout()
    plt.show()


# Agora para as análises do teste de hipótese:
# Passo 1: Definindo as hipóteses.

# Por padrão (Média de pontos, xG a Favor Médio e xG Contra Médio):
# Ho: μB - μA = 0, ou seja, o desempenho do técnico seguinte foi igual ao técnico anterior
# H1: μB - μA != 0, ou seja, o desempenho do técnico seguinte foi diferente que o do técnico anterior

# Teste bicaudal, em todas as situações.

# Passo 2: Definir significância

# Por padrão (Média de pontos, xG a Favor Médio e xG Contra Médio):
# α = 0.05
# Ou seja, estamos dispostos a aceitar uma probabilidade de no máximo 5% de cometer um erro do tipo 1, que é rejeitar erroneamente H0, e afirmar que há evidências suficientes para dizer que o técnico seguinte foi mais eficaz que o anterior

# Passo 3: Definir o tipo de distribuição que será utilizada
# Como não temos os desvios padrão de todos os jogos desse treinador por essa agremiação e pela quantidade de dados ser pequena, iremos utilizar t-student.

# Passo 4: Calcular o p-valor

# Vamos fazer esse cálculo usando a biblioteca scipy.stats
# Graus de  liberdade é o menor valor de n1 -1 e n2 - 1

# Passo 5: Tomada de decisão

# Utilizaremos um função para o seguinte cálculo:
# Se o p-valor < 0.05 = Rejeitar H0. Há evidências que o técnico seguinte foi mais eficiente que o anterior nesse quesito.
# Se o p-valor >= 0.05 = Rejeitar H1. Há evidências que o técnico seguinte foi mais eficiente que o anterior nesse quesito.


# Método FCCD

def t_obs_media_pontos_fccd(arquivo, index1, index2):
    df = pd.read_csv(arquivo, index_col=0)
    a = 0.05
    
    # Dados treinador 1:
    nome_treinador1 = df.index[index1]
    media_treinador1 = df.iloc[index1]['Pontuação média']
    n1 = df.iloc[index1]['Número de jogos']
    desvio_padrao_media_treinador1 = df.iloc[index1]['Desvio padrão da média de pontos']
    variancia_treinador1 = desvio_padrao_media_treinador1 ** 2
    
    # Dados treinador 2:
    nome_treinador2 = df.index[index2]
    media_treinador2 = df.iloc[index2]['Pontuação média']
    n2 = df.iloc[index2]['Número de jogos']
    desvio_padrao_media_treinador2 = df.iloc[index2]['Desvio padrão da média de pontos']
    variancia_treinador2 = desvio_padrao_media_treinador2 ** 2

    estatistica_t = (media_treinador1 - media_treinador2) / math.sqrt((variancia_treinador1 / n1) + (variancia_treinador2 / n2))

    amostras = [n1, n2]

    # Calculando os graus de liberdade
    graus_liberdade = min(amostras) - 1
    print(f'Os graus de liberdade são: {graus_liberdade}')
    
    print(f'A estatística t é igual a {estatistica_t:.4f}')
    print()
    p_valor = 2 * (1 - stats.t.cdf(abs(estatistica_t), df=graus_liberdade)) # Teste bicaudal
    print(f'O p-valor nesse caso é de {p_valor:.8f}')
    print()
    if p_valor < a:
        print(f'Com base em uma análise com nível de significância de 5%, rejeitamos a hipótese nula, ou seja,')
        print(f'há evidências de que há uma diferença significativa na média de pontos entre os treinadores {nome_treinador1} e {nome_treinador2}.')
    else:
        print(f'Com base em uma análise com nível de significância de 5%, não rejeitamos a hipótese nula, ou seja,')
        print(f'não há evidências suficientes para concluir que há uma diferença significativa na\nmédia de pontos entre os treinadores {nome_treinador1} e {nome_treinador2}.')


def t_obs_media_pontos_fccd_com_tabela(arquivo, index1=0, index2=0):
    df = pd.read_csv(arquivo, index_col=0)
    a = 0.05
    
    # Função para calcular estatística t e p-valor
    def calcular_estatisticas(index1, index2):
        # Dados treinador 1:
        nome_treinador1 = df.index[index1]
        media_treinador1 = df.iloc[index1]['Pontuação média']
        n1 = df.iloc[index1]['Número de jogos']
        desvio_padrao_media_treinador1 = df.iloc[index1]['Desvio padrão da média de pontos']
        variancia_treinador1 = desvio_padrao_media_treinador1 ** 2

        # Dados treinador 2:
        nome_treinador2 = df.index[index2]
        media_treinador2 = df.iloc[index2]['Pontuação média']
        n2 = df.iloc[index2]['Número de jogos']
        desvio_padrao_media_treinador2 = df.iloc[index2]['Desvio padrão da média de pontos']
        variancia_treinador2 = desvio_padrao_media_treinador2 ** 2

        estatistica_t = (media_treinador1 - media_treinador2) / math.sqrt((variancia_treinador1 / n1) + (variancia_treinador2 / n2))

        amostras = [n1, n2]

        # Calculando os graus de liberdade
        graus_liberdade = min(amostras) - 1
        
        p_valor = 2 * (1 - stats.t.cdf(abs(estatistica_t), df=graus_liberdade)) # Teste bicaudal
        
        return p_valor < a
    
    # Criando a tabela comparativa
    tabela_comparativa = pd.DataFrame(index=df.index, columns=df.index)
    
    for i in range(len(df)):
        for j in range(len(df)):
            if i != j:
                if calcular_estatisticas(i, j):
                    tabela_comparativa.iloc[i, j] = 1
                else:
                    tabela_comparativa.iloc[i, j] = 0
            else:
                tabela_comparativa.iloc[i, j] = '-'
    
    return tabela_comparativa



def t_obs_xG_aFavor_fccd(arquivo, index1, index2):
    df = pd.read_csv(arquivo, index_col=0)
    a = 0.05
    
    # Dados treinador 1:
    nome_treinador1 = df.index[index1]
    xGaFavor_medio_treinador1 = df.iloc[index1]['xG a Favor médio']
    n1 = df.iloc[index1]['Número de jogos']
    desvio_padrao_xGaFavor_treinador1 = df.iloc[index1]['Desvio padrão do xG a Favor']
    variancia_treinador1 = desvio_padrao_xGaFavor_treinador1 ** 2
    
    # Dados treinador 2:
    nome_treinador2 = df.index[index2]
    xGaFavor_medio_treinador2 = df.iloc[index2]['xG a Favor médio']
    n2 = df.iloc[index2]['Número de jogos']
    desvio_padrao_xGaFavor_treinador2 = df.iloc[index2]['Desvio padrão do xG a Favor']
    variancia_treinador2 = desvio_padrao_xGaFavor_treinador2 ** 2

    estatistica_t = (xGaFavor_medio_treinador1 - xGaFavor_medio_treinador2) / math.sqrt((variancia_treinador1 / n1) + (variancia_treinador2 / n2))

    amostras = [n1, n2]

    # Calculando os graus de liberdade
    graus_liberdade = min(amostras) - 1
    print(f'Os graus de liberdade são: {graus_liberdade}')
    
    print(f'A estatística t é igual a {estatistica_t:.4f}')
    print()
    p_valor = 2 * (1 - stats.t.cdf(abs(estatistica_t), df=graus_liberdade)) # Teste bicaudal
    print(f'O p-valor nesse caso é de {p_valor:.8f}')
    print()
    if p_valor < a:
        print(f'Com base em uma análise com nível de significância de 5%, rejeitamos a hipótese nula, ou seja,')
        print(f'há evidências de que há uma diferença significativa na média de expectativa de gols a favor entre os treinadores {nome_treinador1} e {nome_treinador2}.')
    else:
        print(f'Com base em uma análise com nível de significância de 5%, não rejeitamos a hipótese nula, ou seja,')
        print(f'não há evidências suficientes para concluir que há uma diferença significativa na\nmédia de expectativa de gols a favor entre os treinadores {nome_treinador1} e {nome_treinador2}.')



def t_obs_xGAFavor_fccd_com_tabela(arquivo, index1=0, index2=0):
    df = pd.read_csv(arquivo, index_col=0)
    a = 0.05
    
    # Função para calcular estatística t e p-valor
    def calcular_estatisticas(index1, index2):
        # Dados treinador 1:
        nome_treinador1 = df.index[index1]
        xGaFavor_medio_treinador1 = df.iloc[index1]['xG a Favor médio']
        n1 = df.iloc[index1]['Número de jogos']
        desvio_padrao_xGaFavor_treinador1 = df.iloc[index1]['Desvio padrão do xG a Favor']
        variancia_treinador1 = desvio_padrao_xGaFavor_treinador1 ** 2

        # Dados treinador 2:
        nome_treinador2 = df.index[index2]
        xGaFavor_medio_treinador2 = df.iloc[index2]['xG a Favor médio']
        n2 = df.iloc[index2]['Número de jogos']
        desvio_padrao_xGaFavor_treinador2 = df.iloc[index2]['Desvio padrão do xG a Favor']
        variancia_treinador2 = desvio_padrao_xGaFavor_treinador2 ** 2

        estatistica_t = (xGaFavor_medio_treinador1 - xGaFavor_medio_treinador2) / math.sqrt((variancia_treinador1 / n1) + (variancia_treinador2 / n2))

        amostras = [n1, n2]

        # Calculando os graus de liberdade
        graus_liberdade = min(amostras) - 1
        
        p_valor = 2 * (1 - stats.t.cdf(abs(estatistica_t), df=graus_liberdade)) # Teste bicaudal
        
        return p_valor < a
    
    # Criando a tabela comparativa
    tabela_comparativa = pd.DataFrame(index=df.index, columns=df.index)
    
    for i in range(len(df)):
        for j in range(len(df)):
            if i != j:
                if calcular_estatisticas(i, j):
                    tabela_comparativa.iloc[i, j] = 1
                else:
                    tabela_comparativa.iloc[i, j] = 0
            else:
                tabela_comparativa.iloc[i, j] = '-'
    
    return tabela_comparativa



def t_obs_xG_Contra_fccd(arquivo, index1, index2):
    df = pd.read_csv(arquivo, index_col=0)
    a = 0.05
    
    # Dados treinador 1:
    nome_treinador1 = df.index[index1]
    xGcontra_medio_treinador1 = df.iloc[index1]['xG Contra médio']
    n1 = df.iloc[index1]['Número de jogos']
    desvio_padrao_xGcontra_treinador1 = df.iloc[index1]['Desvio padrão do xG Contra']
    variancia_treinador1 = desvio_padrao_xGcontra_treinador1 ** 2
    
    # Dados treinador 2:
    nome_treinador2 = df.index[index2]
    xGcontra_medio_treinador2 = df.iloc[index2]['xG Contra médio']
    n2 = df.iloc[index2]['Número de jogos']
    desvio_padrao_xGcontra_treinador2 = df.iloc[index2]['Desvio padrão do xG Contra']
    variancia_treinador2 = desvio_padrao_xGcontra_treinador2 ** 2

    estatistica_t = (xGcontra_medio_treinador1 - xGcontra_medio_treinador2) / math.sqrt((variancia_treinador1 / n1) + (variancia_treinador2 / n2))

    amostras = [n1, n2]

    # Calculando os graus de liberdade
    graus_liberdade = min(amostras) - 1
    print(f'Os graus de liberdade são: {graus_liberdade}')
    
    print(f'A estatística t é igual a {estatistica_t:.4f}')
    print()
    p_valor = 2 * (1 - stats.t.cdf(abs(estatistica_t), df=graus_liberdade)) # Teste bicaudal
    print(f'O p-valor nesse caso é de {p_valor:.8f}')
    print()
    if p_valor < a:
        print(f'Com base em uma análise com nível de significância de 5%, rejeitamos a hipótese nula, ou seja,')
        print(f'há evidências de que há uma diferença significativa na média de expectativa de gols contra entre os treinadores {nome_treinador1} e {nome_treinador2}.')
    else:
        print(f'Com base em uma análise com nível de significância de 5%, não rejeitamos a hipótese nula, ou seja,')
        print(f'não há evidências suficientes para concluir que há uma diferença significativa na\nmédia de expectativa de gols contra entre os treinadores {nome_treinador1} e {nome_treinador2}.')



def t_obs_xGContra_fccd_com_tabela(arquivo, index1=0, index2=0):
    df = pd.read_csv(arquivo, index_col=0)
    a = 0.05
    
    # Função para calcular estatística t e p-valor
    def calcular_estatisticas(index1, index2):
        # Dados treinador 1:
        nome_treinador1 = df.index[index1]
        xGaFavor_medio_treinador1 = df.iloc[index1]['xG a Favor médio']
        n1 = df.iloc[index1]['Número de jogos']
        desvio_padrao_xGaFavor_treinador1 = df.iloc[index1]['Desvio padrão do xG a Favor']
        variancia_treinador1 = desvio_padrao_xGaFavor_treinador1 ** 2

        # Dados treinador 2:
        nome_treinador2 = df.index[index2]
        xGaFavor_medio_treinador2 = df.iloc[index2]['xG a Favor médio']
        n2 = df.iloc[index2]['Número de jogos']
        desvio_padrao_xGaFavor_treinador2 = df.iloc[index2]['Desvio padrão do xG a Favor']
        variancia_treinador2 = desvio_padrao_xGaFavor_treinador2 ** 2

        estatistica_t = (xGaFavor_medio_treinador1 - xGaFavor_medio_treinador2) / math.sqrt((variancia_treinador1 / n1) + (variancia_treinador2 / n2))

        amostras = [n1, n2]

        # Calculando os graus de liberdade
        graus_liberdade = min(amostras) - 1
        
        p_valor = 2 * (1 - stats.t.cdf(abs(estatistica_t), df=graus_liberdade)) # Teste bicaudal
        
        return p_valor < a
    
    # Criando a tabela comparativa
    tabela_comparativa = pd.DataFrame(index=df.index, columns=df.index)
    
    for i in range(len(df)):
        for j in range(len(df)):
            if i != j:
                if calcular_estatisticas(i, j):
                    tabela_comparativa.iloc[i, j] = 1
                else:
                    tabela_comparativa.iloc[i, j] = 0
            else:
                tabela_comparativa.iloc[i, j] = '-'
    
    return tabela_comparativa