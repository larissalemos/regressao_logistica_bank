import pandas as pd # importando a biblioteca de manipulação de dados
import numpy as np # biblioteca para calculo
import scipy.stats as stats # biblioteca para modelagem
import statsmodels.api as sm # biblioteca para a regressão logística
import statistics
import seaborn as sns
import math
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, RegressorMixin
from scipy.stats import chi2_contingency
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def selecionar_pvalor_forward(var_dependente, var_independente, base, signif):
    """   
    Esta função realiza uma seleção forward stepwise com base no p-valor das variáveis independentes.
    A cada passo, adiciona a variável independente com o menor p-valor ao modelo, desde que o p-valor 
    seja menor que o nível de significância especificado.
    
    Parâmetros:
    var_dependente (str): Nome da variável dependente.
    var_independente (list): Lista de variáveis independentes a serem avaliadas.
    base (pd.DataFrame): Conjunto de dados contendo as variáveis dependentes e independentes.
    signif (float): Nível de significância para a inclusão das variáveis (por exemplo, 0.05).
    
    Retorna: 
    pd.DataFrame: DataFrame contendo as variáveis selecionadas e seus respectivos p-valores.
    
    Exemplo de uso:
        >>> import pandas as pd
        >>> df = pd.read_csv('https://raw.githubusercontent.com/Zack1803/Body-Fat-Prediction-Dataset/refs/heads/main/bodyfat.csv')
        >>> colunas_pvalor = selecionar_pvalor_forward(var_dependente='BodyFat', var_independente=df.drop('BodyFat', axis = 1).columns.to_list(), base=df, signif=0.05)
        >>> colunas_pvalor
    
    criada por Mateus Rocha - time ASN.Rocks
    """
    

    preditoras = []
    pvalor_preditoras = []
    Y = base[var_dependente]
    while True and var_independente != []:
        lista_pvalor = []
        lista_variavel = []
        for var in var_independente:
            X = sm.add_constant(base[ [var] +  preditoras ])
            
            modelo = sm.GLM(Y,X,family=sm.families.Binomial()).fit()
            
            if( preditoras == []):    
                
                pvalor = modelo.pvalues[1]
                variavel = modelo.pvalues.index[1]
            
            else:
                
                pvalor = modelo.pvalues.drop(preditoras)[1]
                variavel = modelo.pvalues.drop(preditoras).index[1]
                
            lista_pvalor.append(pvalor)
            lista_variavel.append(variavel)          
        
        if( lista_pvalor[ np.argmin(lista_pvalor) ] < signif ):
            preditoras.append( lista_variavel[np.argmin(lista_pvalor)] )
            pvalor_preditoras.append(lista_pvalor[ np.argmin(lista_pvalor) ])
            var_independente.remove( lista_variavel[ np.argmin(lista_pvalor)] )
        else:
            break
    info_final = pd.DataFrame({ 'var': preditoras, 'pvalor': pvalor_preditoras})
    return info_final


def selecionar_aic_forward(var_dependente, var_independente, base):
    """   
    Esta função realiza uma seleção forward stepwise com base no critério de informação de Akaike (AIC).
    A cada passo, adiciona a variável independente que minimiza o AIC ao modelo.
    
    Parâmetros:
      var_dependente (str): Nome da variável dependente.
      var_independente (list): Lista de variáveis independentes a serem avaliadas.
      base (pd.DataFrame): Conjunto de dados contendo as variáveis dependentes e independentes.
    
    Retorna: 
        pd.DataFrame: DataFrame contendo as combinações de variáveis selecionadas e seus respectivos AICs, 
        ordenados do menor para o maior AIC.
        
    Exemplo de uso:
            >>> import pandas as pd
            >>> df = pd.read_csv('https://raw.githubusercontent.com/Zack1803/Body-Fat-Prediction-Dataset/refs/heads/main/bodyfat.csv')
            >>> colunas_aicforward = selecionar_aic_forward(var_dependente='BodyFat', var_independente=df.drop('BodyFat', axis = 1).columns.to_list(), base=df)
            >>> colunas_aicforward
    
    criada por Mateus Rocha - time ASN.Rocks
    """    

    preditoras = []
    aic_preditoras = []
    Y = base[var_dependente]
    lista_final = []
    aic_melhor = float('inf')
    
    while True and var_independente != []:
        lista_aic = []
        lista_variavel = []
        lista_modelos =[]
        if(var_independente == []):
            break
        for var in var_independente:
            X = sm.add_constant(base[ [var] +  preditoras ])
            aic = sm.GLM(Y,X,family=sm.families.Binomial()).fit().aic
            variavel = var
                
            lista_aic.append(aic)
            
            lista_variavel.append(var)
            
            lista_modelos.append( [var] +  preditoras )
            
        if( lista_aic[ np.argmin(lista_aic) ] < aic_melhor ):
            
            lista_final.append(lista_modelos[ np.argmin(lista_aic)]  )
            
            preditoras.append( lista_variavel[np.argmin(lista_aic)] )
            
            aic_preditoras.append(lista_aic[ np.argmin(lista_aic) ])
            
            var_independente.remove( lista_variavel[ np.argmin(lista_aic)] )
            
            aic_melhor = lista_aic[ np.argmin(lista_aic) ] 
            
        else:
            break
        
    info_final = pd.DataFrame({ 'var': lista_final, 'aic': aic_preditoras}).sort_values(by = 'aic')
    return info_final


def selecionar_bic_forward(var_dependente, var_independente, base):
    
    """   
    Esta função realiza uma seleção forward stepwise com base no critério de informação bayesiano (BIC).
    A cada passo, adiciona a variável independente que minimiza o BIC ao modelo.
    
    Parâmetros:
      var_dependente (str): Nome da variável dependente.
      var_independente (list): Lista de variáveis independentes a serem avaliadas.
      base (pd.DataFrame): Conjunto de dados contendo as variáveis dependentes e independentes.
    
    Retorna: 
        pd.DataFrame: DataFrame contendo as combinações de variáveis selecionadas e seus respectivos BICs, 
        ordenados do menor para o maior BIC.
        
    Exemplo de uso:
            >>> import pandas as pd
            >>> df = pd.read_csv('https://raw.githubusercontent.com/Zack1803/Body-Fat-Prediction-Dataset/refs/heads/main/bodyfat.csv')
            >>> colunas_bicforward = selecionar_bic_forward(var_dependente='BodyFat', var_independente=df.drop('BodyFat', axis = 1).columns.to_list(), base=df)
            >>> colunas_bicforward
    
    criada por Mateus Rocha - time ASN.Rocks
    """
    preditoras = []
    bic_preditoras = []
    Y = base[var_dependente]
    lista_final = []
    bic_melhor = float('inf')
    
    while True and var_independente != []:
        lista_bic = []
        lista_variavel = []
        lista_modelos =[]
        if(var_independente == []):
            break
        for var in var_independente:
            X = sm.add_constant(base[ [var] +  preditoras ])
            bic = sm.GLM(Y,X,family=sm.families.Binomial()).fit().bic
            variavel = var
                
            lista_bic.append(bic)
            
            lista_variavel.append(var)
            
            lista_modelos.append( [var] +  preditoras )
            
        if( lista_bic[ np.argmin(lista_bic) ] < bic_melhor ):
            
            lista_final.append(lista_modelos[ np.argmin(lista_bic)]  )
            
            preditoras.append( lista_variavel[np.argmin(lista_bic)] )
            
            bic_preditoras.append(lista_bic[ np.argmin(lista_bic) ])
            
            var_independente.remove( lista_variavel[ np.argmin(lista_bic)] )
            
            aic_melhor = lista_bic[ np.argmin(lista_bic) ] 
            
        else:
            break
        
    info_final = pd.DataFrame({ 'var': lista_final, 'bic': bic_preditoras}).sort_values(by = 'bic')
    return info_final


def selecionar_pvalor_backward(var_dependente, var_independente, base, signif):
    """   
    Esta função realiza uma seleção backward stepwise com base no p-valor das variáveis independentes.
    A cada passo, remove a variável independente com o maior p-valor do modelo, 
    desde que seja maior que o nível de significância especificado.
    
    Parâmetros:
      var_dependente (str): Nome da variável dependente.
      var_independente (list): Lista de variáveis independentes a serem avaliadas.
      base (pd.DataFrame): Conjunto de dados contendo as variáveis dependentes e independentes.
      signif (float): Nível de significância para a inclusão das variáveis (por exemplo, 0.05).
      
    Retorna: 
        pd.DataFrame: DataFrame contendo as variáveis restantes após a seleção backward.
        
    Exemplo de uso:
            >>> import pandas as pd
            >>> df = pd.read_csv('https://raw.githubusercontent.com/Zack1803/Body-Fat-Prediction-Dataset/refs/heads/main/bodyfat.csv')
            >>> colunas_pvalorbackward = selecionar_pvalor_backward(var_dependente='BodyFat', var_independente=df.drop('BodyFat', axis = 1).columns.to_list(), signif = 0.05 ,base=df)
            >>> colunas_pvalorbackward
    
    criada por Mateus Rocha - time ASN.Rocks
    """

    Y = base[var_dependente]
    
    while True and var_independente != []:
        
        X_geral = sm.add_constant(base[var_independente])
        
        modelo = sm.GLM(Y,X_geral,family=sm.families.Binomial()).fit()
        
        pvalor_geral = modelo.pvalues
        
        variavel_geral = modelo.pvalues.index
        
        if(pvalor_geral[ np.argmax(pvalor_geral) ] > signif ):
            var_independente.remove( variavel_geral[ np.argmax(pvalor_geral) ] )
        else:
            break
    
    
    
    info_final = pd.DataFrame({ 'var': var_independente})
    return info_final


def selecionar_aic_backward(var_dependente, var_independente, base):
    """   
    Esta função realiza uma seleção backward stepwise com base no critério de informação de Akaike (AIC).
    A cada passo, adiciona a variável independente que minimiza o AIC ao modelo.
    
    Parâmetros:
      var_dependente (str): Nome da variável dependente.
      var_independente (list): Lista de variáveis independentes a serem avaliadas.
      base (pd.DataFrame): Conjunto de dados contendo as variáveis dependentes e independentes.
    
    Retorna: 
        pd.DataFrame: DataFrame contendo as combinações de variáveis selecionadas e seus respectivos AICs, 
        ordenados do menor para o maior AIC.
        
    Exemplo de uso:
            >>> import pandas as pd
            >>> df = pd.read_csv('https://raw.githubusercontent.com/Zack1803/Body-Fat-Prediction-Dataset/refs/heads/main/bodyfat.csv')
            >>> colunas_aicbackward = selecionar_aic_backward(var_dependente='BodyFat', var_independente=df.drop('BodyFat', axis = 1).columns.to_list(), base=df)
            >>> colunas_aicbackward
    
    criada por Mateus Rocha - time ASN.Rocks
    """
    Y = base[var_dependente]
    
    preditoras_finais = []
    
    aic_final = []
    
    while True and var_independente != []:
        
        lista_aic = []
        lista_preditoras = []

        X_geral = sm.add_constant(base[var_independente])
        
        aic_geral = sm.GLM(Y,X_geral,family=sm.families.Binomial()).fit().aic
    
        aic_final.append(aic_geral)
        
        preditoras_finais.append(base[var_independente].columns.to_list())
        
        for var in var_independente:
            
            lista_variaveis = var_independente.copy()
            lista_variaveis.remove(var)
            
            X = sm.add_constant(base[ lista_variaveis ])
            aic = sm.GLM(Y,X,family=sm.families.Binomial()).fit().aic    
            
            lista_aic.append(aic)
            
            lista_preditoras.append(var)
            
        if(lista_aic[ np.argmin(lista_aic) ] < aic_geral ):
            var_independente.remove( lista_preditoras[ np.argmin(lista_aic) ] )
            
        else:
            break
    
    
    info_final = pd.DataFrame({ 'var': preditoras_finais, 'aic':aic_final }).sort_values(by = 'aic')
    return info_final

def selecionar_bic_backward(var_dependente, var_independente, base):
    """   
    Esta função realiza uma seleção backward stepwise com base no critério de informação bayesiano (BIC).
    A cada passo, adiciona a variável independente que minimiza o BIC ao modelo.
    
    Parâmetros:
      var_dependente (str): Nome da variável dependente.
      var_independente (list): Lista de variáveis independentes a serem avaliadas.
      base (pd.DataFrame): Conjunto de dados contendo as variáveis dependentes e independentes.
    
    Retorna: 
        pd.DataFrame: DataFrame contendo as combinações de variáveis selecionadas e seus respectivos BICs, 
        ordenados do menor para o maior BIC.
        
    Exemplo de uso:
            >>> import pandas as pd
            >>> df = pd.read_csv('https://raw.githubusercontent.com/Zack1803/Body-Fat-Prediction-Dataset/refs/heads/main/bodyfat.csv')
            >>> colunas_bicbackward = selecionar_bic_backward(var_dependente='BodyFat', var_independente=df.drop('BodyFat', axis = 1).columns.to_list(), base=df)
            >>> colunas_bicbackward
    
    criada por Mateus Rocha - time ASN.Rocks
    """
    Y = base[var_dependente]
    
    preditoras_finais = []
    
    bic_final = []
    
    while True and var_independente != []:
        
        lista_bic = []
        lista_preditoras = []

        X_geral = sm.add_constant(base[var_independente])
        
        bic_geral = sm.GLM(Y,X_geral,family=sm.families.Binomial()).fit().bic
    
        bic_final.append(bic_geral)
        
        preditoras_finais.append(base[var_independente].columns.to_list())
        
        for var in var_independente:
            
            lista_variaveis = var_independente.copy()
            lista_variaveis.remove(var)
            
            X = sm.add_constant(base[ lista_variaveis ])
            bic = sm.GLM(Y,X,family=sm.families.Binomial()).fit().bic    
            
            lista_bic.append(bic)
            
            lista_preditoras.append(var)
            
        if(lista_bic[ np.argmin(lista_bic) ] < bic_geral ):
            var_independente.remove( lista_preditoras[ np.argmin(lista_bic) ] )
            
        else:
            break
    
    
    info_final = pd.DataFrame({ 'var': preditoras_finais, 'bic':bic_final }).sort_values(by = 'bic')
    return info_final

def stepwise( var_dependente , var_independente , base, metrica, signif = 0.05, epsilon = 0.0001):
      
    """   
    Esta função realiza a seleção stepwise de variáveis, usando os métodos forward e backward 
    com base em uma métrica específica (AIC, BIC ou p-valor).
    O processo consiste em primeiro aplicar a seleção forward com a métrica escolhida e, 
    em seguida, a backward, ajustando o modelo até que a diferença entre as métricas seja menor 
    que um valor de tolerância (epsilon).
    
    Parâmetros:
      var_dependente (str): Nome da variável dependente.
      var_independente (list): Lista de variáveis independentes a serem avaliadas.
      base (pd.DataFrame): Conjunto de dados contendo as variáveis dependentes e independentes.
      metrica (str): A métrica a ser usada no processo de seleção (pode ser 'aic', 'bic', ou 'pvalor').
      signif (float): Nível de significância usado para a seleção por p-valor (padrão 0.05).
      epsilon (float): Diferença mínima aceitável entre as métricas forward e backward para parar o processo (padrão 0.0001).
    Retorna: 
         Resultado da seleção de variáveis com base no método e métrica escolhidos.
        
    Exemplo de uso:
            >>> import pandas as pd
            >>> df = pd.read_csv('https://raw.githubusercontent.com/Zack1803/Body-Fat-Prediction-Dataset/refs/heads/main/bodyfat.csv')
            >>> colunas_stepwise = stepwise(var_dependente='BodyFat', var_independente=df.drop('BodyFat', axis = 1).columns.to_list(), base = df ,metrica='aic', signif=0.05)
            >>> colunas_stepwise
    
    criada por Mateus Rocha - time ASN.Rocks
    """

    lista_var = var_independente
    
    metrica_forward = 0
    
    metrica_backward = 0
    
    while True:
    
        if(metrica == 'aic'):
            resultado = selecionar_aic_forward(var_dependente = var_dependente, var_independente = var_independente, base = base)

            if (len(resultado) == 1):
                return resultado
            
            resultado_final = selecionar_aic_backward(var_dependente = var_dependente, var_independente = resultado['var'].to_list()[0], base = base)

            if(len(resultado_final) == 1):
                return resultado_final

            metrica_forward = resultado['aic'].to_list()[0]

            metrica_backward = resultado_final['aic'].to_list()[0]


        elif(metrica == 'bic'):
            resultado = selecionar_bic_forward(var_dependente = var_dependente, var_independente = var_independente, base = base)

            if (len(resultado) == 1):
                return resultado

            resultado_final = selecionar_bic_backward(var_dependente = var_dependente, var_independente = resultado['var'].to_list()[0], base = base)

            if(len(resultado_final) == 1):
                return resultado_final

            metrica_forward = resultado['bic'].to_list()[0]

            metrica_backward = resultado_final['bic'].to_list()[0]

        elif(metrica == 'pvalor'):
            resultado = selecionar_pvalor_forward(var_dependente = var_dependente, var_independente = var_independente, base = base, signif = signif)

            if (len(resultado) == 1):
                return resultado

            resultado_final = selecionar_pvalor_backward(var_dependente = var_dependente, var_independente = resultado['var'].to_list(), base = base, signif = signif)

            if(len(resultado_final) == 1):
                return resultado_final

            return resultado_final

        if( abs(metrica_forward - metrica_backward) < epsilon ):
            break
        else:
            var_independente = set(resultado_final['var'].to_list() + lista_var)

    
def step( var_dependente , var_independente , base, metodo, metrica, signif = 0.05):
    """   
    Esta função realiza a seleção de variáveis usando os métodos forward, backward ou stepwise, 
    com base em uma métrica escolhida (AIC, BIC ou p-valor).O usuário pode escolher o método de 
    seleção (forward, backward ou both) e a métrica desejada para o critério de inclusão ou exclusão de variáveis.
    
    Parâmetros:
      var_dependente (str): Nome da variável dependente.
      var_independente (list): Lista de variáveis independentes a serem avaliadas.
      base (pd.DataFrame): Conjunto de dados contendo as variáveis dependentes e independentes.
      metrica (str): A métrica a ser usada no processo de seleção (pode ser 'aic', 'bic', ou 'pvalor').
      metodo (str): Método de seleção ('forward', 'backward' ou 'both').
      signif (float): Nível de significância usado para a seleção por p-valor (padrão 0.05).
    Retorna: 
        Resultado da seleção de variáveis com base no método e métrica escolhidos.
        
    Exemplo de uso:
            >>> import pandas as pd
            >>> df = pd.read_csv('https://raw.githubusercontent.com/Zack1803/Body-Fat-Prediction-Dataset/refs/heads/main/bodyfat.csv')
            >>> colunas_step = step(var_dependente='BodyFat', var_independente=df.drop('BodyFat', axis = 1).columns.to_list(), base = df, metodo = 'forward' ,metrica='aic', signif=0.05)
            >>> colunas_step
    
    criada por Mateus Rocha - time ASN.Rocks
    """

    if( metodo == 'forward' and metrica == 'aic' ):
        resultado = selecionar_aic_forward(var_dependente = var_dependente, var_independente = var_independente, base = base)
    elif(metodo == 'forward' and metrica == 'bic' ):
        resultado = selecionar_bic_forward(var_dependente = var_dependente, var_independente = var_independente, base = base)
    elif(metodo == 'forward' and metrica == 'pvalor' ):
        resultado = selecionar_pvalor_forward(var_dependente = var_dependente, var_independente = var_independente, base = base, signif = signif)
    elif( metodo == 'backward' and metrica == 'aic' ):
        resultado = selecionar_aic_backward(var_dependente = var_dependente, var_independente = var_independente, base = base)
    elif(metodo == 'backward'and metrica == 'bic' ):
        resultado = selecionar_bic_backward(var_dependente = var_dependente, var_independente = var_independente, base = base)
    elif(metodo == 'backward' and metrica == 'pvalor' ):
        resultado = selecionar_pvalor_backward(var_dependente = var_dependente, var_independente = var_independente, base = base, signif = signif)
    elif(metodo == 'both'):
        resultado = stepwise( var_dependente = var_dependente , var_independente = var_independente , base = base, metrica = metrica, signif = signif)
        
    return resultado

def univariada_variavel_numerica(dado, variavel):
    """
    Gera uma matriz de gráficos (2x2) para uma variável contínua.

    [1,1] Histograma
    [1,2] Gráfico de violino
    [2,1] Box plot
    [2,2] Box plot com pontos sobrepostos

    Acima dos gráficos, exibe uma tabela com as estatísticas descritivas da variável,
    incluindo skewness, kurtosis e coeficiente de variação.

    Parâmetros:
        dado (pd.DataFrame): Base de dados contendo a variável
        variavel (str): Nome da variável a ser analisada

    Retorna:
        None

    Exemplo de uso:
        >> dado = pd.DataFrame({"variavel_exemplo": np.random.normal(loc=50, scale=10, size=100)})
        >> univariada_variavel_numerica(dado, "variavel_exemplo")
    """
    
    # Calcular as estatísticas descritivas básicas
    desc_stats = dado[variavel].describe().to_frame().T
    
    # Calcular e adicionar as métricas adicionais
    desc_stats['skewness'] = dado[variavel].skew()
    desc_stats['kurtosis'] = dado[variavel].kurtosis()
    desc_stats['coef_var'] = (dado[variavel].std() / dado[variavel].mean()) * 100  # em porcentagem
    
    # Renomear colunas para melhor legibilidade
    desc_stats = desc_stats.rename(columns={
        'count': 'n',
        'mean': 'média',
        'std': 'desvio_padrão',
        'min': 'mínimo',
        '25%': 'Q1',
        '50%': 'mediana',
        '75%': 'Q3',
        'max': 'máximo',
        'skewness': 'assimetria',
        'kurtosis': 'curtose',
        'coef_var': 'coef_var (%)'
    })
    
    # Ordenar as colunas de forma lógica
    desc_stats = desc_stats[['n', 'média', 'desvio_padrão', 'coef_var (%)', 
                            'mínimo', 'Q1', 'mediana', 'Q3', 'máximo',
                            'assimetria', 'curtose']]
    
    # Arredondar os valores
    desc_stats = desc_stats.round(4)

    # Configuração dos subplots
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(f"Análise da variável: {variavel}", fontsize=16, y=0.98)

    # Adicionar a tabela no topo (agora maior para acomodar mais colunas)
    ax_table = plt.subplot2grid((3, 2), (0, 0), colspan=2)
    ax_table.axis("off")
    table = ax_table.table(cellText=desc_stats.values,
                         colLabels=desc_stats.columns,
                         rowLabels=desc_stats.index,
                         cellLoc="center",
                         loc="center",
                         colWidths=[0.1]*len(desc_stats.columns))  # Ajustar larguras das colunas
    
    # Formatação da tabela
    table.auto_set_font_size(False)
    table.set_fontsize(10)  # Reduzir um pouco o tamanho da fonte para caber mais colunas
    table.scale(1, 1.5)  # Ajustar escala da tabela

    # [1,1] Histograma
    ax1 = plt.subplot2grid((3, 2), (1, 0))
    sns.histplot(dado[variavel], kde=True, ax=ax1, color="skyblue")

    # Adiciona linha vertical para a média
    media = dado[variavel].mean()
    ax1.axvline(media, color='red', linestyle='--', linewidth=1.5, label=f'Média: {media:.2f}')

    ax1.set_title("Histograma", fontsize=12)
    ax1.set_xlabel(variavel)
    ax1.legend()  # Adiciona a legenda para mostrar o valor da média

    # [1,2] Gráfico de violino
    ax2 = plt.subplot2grid((3, 2), (1, 1), sharex=ax1)
    sns.violinplot(x=dado[variavel], ax=ax2, color="lightgreen")
    ax2.set_title("Gráfico de violino", fontsize=12)
    ax2.set_xlabel(variavel)

    # [2,1] Box plot
    ax3 = plt.subplot2grid((3, 2), (2, 0), sharex=ax1)
    sns.boxplot(x=dado[variavel], ax=ax3, color="orange")
    ax3.set_title("Box plot", fontsize=12)
    ax3.set_xlabel(variavel)

    # [2,2] Box plot com pontos sobrepostos
    ax4 = plt.subplot2grid((3, 2), (2, 1), sharex=ax1)
    sns.boxplot(x=dado[variavel], ax=ax4, color="lightcoral")
    sns.stripplot(x=dado[variavel], ax=ax4, color="black", alpha=0.5, jitter=True)
    ax4.set_title("Box plot com pontos", fontsize=12)
    ax4.set_xlabel(variavel)

    # Ajustes finais
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def univariada_variavel_categoria(dado, variavel):
    """
    Análises para variáveis categóricas.

    1. Retorna o describe transposto e formatado em uma tabela.
    2. Retorna uma tabela com a frequência de cada nível (incluindo percentuais e total).
    3. Plota um gráfico de barras com a frequência e exibe os valores no topo.

    Parâmetros:
        dado (pd.DataFrame): O dataframe contendo os dados.
        variavel (str): O nome da variável categórica para análise.

    Retorna:
        None

    Exemplo de uso:
        >>> import pandas as pd
        >>> df = pd.DataFrame({'Categoria': ['A', 'B', 'A', 'C', 'B', 'A', 'C', 'C', 'A', 'B']})
        >>> univariada_variavel_categoria(df, 'Categoria')
    """
    # Verificar se a variável está no DataFrame
    if variavel not in dado.columns:
        raise ValueError(f"A variável '{variavel}' não está no DataFrame.")

    # 1. Describe transposto e formatado
    describe_table = dado[variavel].describe().to_frame()
    describe_table = describe_table.T
    describe_table.index = [variavel]

    # Exibir a tabela formatada
    print("Describe da variável categórica:")
    display(describe_table)

    # 2. Frequência de cada nível (com percentuais e total)
    frequency_table = dado[variavel].value_counts().reset_index()
    frequency_table.columns = [variavel, 'Frequência']
    frequency_table['Percentual (%)'] = (frequency_table['Frequência'] / len(dado) * 100).round(2)

    # Adicionar uma linha para o total
    total_row = pd.DataFrame({
        variavel: ['Total'],
        'Frequência': [frequency_table['Frequência'].sum()],
        'Percentual (%)': [100.0]
    })
    frequency_table = pd.concat([frequency_table, total_row], ignore_index=True)

    # Exibir a tabela formatada
    print("Tabela de frequência da variável categórica (com percentuais e total):")
    display(frequency_table)

    # 3. Gráfico de barras com frequência
    plt.figure(figsize=(8, 5))
    ax = sns.barplot(x=variavel, y='Frequência', data=frequency_table[:-1], errorbar=None)

    # Adicionar os valores no topo das barras
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}',
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='baseline', fontsize=10, color='black',
                    xytext=(0, 5), textcoords='offset points')

    # Configurar o gráfico
    plt.title(f'Gráfico de Frequência: {variavel}')
    plt.xlabel(variavel)
    plt.ylabel('Frequência')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def analise_var_numerica_por_percentil(data, x, y, q=10, grafico='none'):
    """
    Ordena a variável x, divide em percentis e sumariza estatísticas.

    Parâmetros:
        data (pd.DataFrame): O banco de dados contendo as variáveis.
        x (str): O nome da variável independente (explanatória).
        y (str): O nome da variável dependente (resposta).
        q (int): O número de percentis (default: 10).
        grafico (str): Opção de gráfico: 'p', 'logito', 'ambos', 'none' (default: 'none').

    Retorno:
        pd.DataFrame: DataFrame com as estatísticas por percentil, incluindo:
                      - Percentil
                      - n (número de linhas)
                      - Min de x
                      - Max de x
                      - p (média de y)
                      - logito de p

    Exemplo de uso
        >> data = pd.DataFrame({'x': np.random.uniform(0, 100, 1000), 
        'y': np.random.randint(0, 2, 1000)})
        >> resultado = analise_var_numerica_por_percentil(data, 'x', 'y', q=10, grafico='ambos')
        >> print(resultado)
    """
    # Certificar-se de que a variável y está no formato numérico
    data[y] = pd.to_numeric(data[y], errors='coerce')

    # Ordenar os dados pela variável x
    data = data.sort_values(by=x).reset_index(drop=True)

    # Criar os percentis
    data['percentil'] = pd.qcut(data[x], q=q, labels=[str(i) for i in range(1, q + 1)])

    # Sumarizar as estatísticas por percentil
    summary = data.groupby('percentil').agg(
        n=(x, 'count'),
        min_x=(x, 'min'),
        max_x=(x, 'max'),
        p=(y, 'mean')
    ).reset_index()

    # Calcular o logito de p
    summary['logito_p'] = np.log(summary['p'] / (1 - summary['p']))

    # Ajuste para lidar com casos onde p é 0 ou 1
    epsilon = 1e-10  # Pequeno valor para ajustar 0 e 1
    summary['logito_p'] = np.log(np.clip(summary['p'], epsilon, 1 - epsilon) / 
                                 (1 - np.clip(summary['p'], epsilon, 1 - epsilon)))

    # Calcular correlação de Pearson entre ponto médio e logito
    pearson_r = summary[['percentil', 'logito_p']].corr().iloc[0, 1]
    summary['pearson_r'] = pearson_r  # Adiciona a todos as linhas para referência

    # Opções de gráfico
    if grafico in ['p', 'logito', 'ambos']:
        plt.figure(figsize=(12, 6))

        if grafico == 'p':
            plt.scatter(summary['percentil'], summary['p'], color='blue')
            plt.title(f'Gráfico de Percentil de {x} x p')
            plt.xlabel(f'Percentil de {x}')
            plt.ylabel('p (média de y)')
            plt.grid(True)
            plt.show()

        elif grafico == 'logito':
            plt.scatter(summary['percentil'], summary['logito_p'], color='red')
            plt.title(f'Gráfico de Percentil de {x} x Logito de p')
            plt.xlabel(f'Percentil de {x}')
            plt.ylabel('Logito de p')
            plt.grid(True)
            plt.show()

        elif grafico == 'ambos':
            # Gráficos lado a lado
            fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharex=True)

            # Gráfico Percentil x p
            axes[0].scatter(summary['percentil'], summary['p'], color='blue')
            axes[0].set_title(f'Percentil de {x} x p')
            axes[0].set_xlabel(f'Percentil de {x}')
            axes[0].set_ylabel('p (média de y)')
            axes[0].grid(True)

            # Gráfico Percentil x Logito de p
            axes[1].scatter(summary['percentil'], summary['logito_p'], color='red')
            axes[1].set_title(f'Percentil de {x} x Logito de p')
            axes[1].set_xlabel(f'Percentil de {x}')
            axes[1].set_ylabel('Logito de p')
            axes[1].grid(True)

            plt.tight_layout()
            plt.show()

    return summary

def calcular_metricas(y_true, y_pred):
    """
    Calcula métricas de avaliação do modelo a partir dos rótulos reais e previstos.
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return accuracy, precision, recall, f1

def calcular_lift_flexivel(df, y_true_col, list_y_pred_col, list_thresholds=None, list_percentis=None):
    """
    Calcula o Lift para múltiplos modelos usando tanto thresholds fixos quanto percentis da população.
    
    Args:
        df (DataFrame): DataFrame contendo as colunas de verdadeiros e previstos.
        y_true_col (str): Nome da coluna com os valores reais.
        list_y_pred_col (list): Lista de nomes de colunas com as previsões dos diferentes modelos.
        list_thresholds (list, optional): Lista de thresholds para corte. Default=None.
        list_percentis (list, optional): Lista de percentis para corte. Default=None.
        
    Returns:
        DataFrame: Resultados consolidados com Lift para cada combinação.
    """
    resultados = []
    
    for y_pred_col in list_y_pred_col:
        # Verificar se o modelo existe no DataFrame
        if y_pred_col not in df.columns:
            print(f"Aviso: Coluna {y_pred_col} não encontrada no DataFrame. Pulando...")
            continue
            
        # Cálculo por thresholds fixos
        if list_thresholds is not None:
            for threshold in list_thresholds:
                # Cálculo tradicional por threshold
                df_above_threshold = df[df[y_pred_col] >= threshold]
                if len(df_above_threshold) == 0:
                    lift = 0
                    taxa_grupo = 0
                else:
                    taxa_grupo = df_above_threshold[y_true_col].mean()
                    taxa_global = df[y_true_col].mean()
                    lift = taxa_grupo / taxa_global if taxa_global != 0 else float('inf')
                
                resultados.append({
                    'Modelo': y_pred_col,
                    'Tipo_Corte': 'Threshold',
                    'Valor_Corte': threshold,
                    'Tamanho_Grupo': len(df_above_threshold),
                    'Taxa_Resposta_Grupo': taxa_grupo,
                    'Lift': lift
                })
        
        # Cálculo por percentis
        if list_percentis is not None:
            df_ordenado = df.sort_values(by=y_pred_col, ascending=False)
            for percentil in list_percentis:
                tamanho_grupo = int(len(df) * percentil / 100)
                top_observacoes = df_ordenado.head(tamanho_grupo)
                
                taxa_grupo = top_observacoes[y_true_col].mean()
                taxa_global = df[y_true_col].mean()
                lift = taxa_grupo / taxa_global if taxa_global != 0 else float('inf')
                
                resultados.append({
                    'Modelo': y_pred_col,
                    'Tipo_Corte': 'Percentil',
                    'Valor_Corte': percentil,
                    'Tamanho_Grupo': tamanho_grupo,
                    'Taxa_Resposta_Grupo': taxa_grupo,
                    'Lift': lift
                })
    
    return pd.DataFrame(resultados)

# DATA DESCRIPTION ---------------------------------------------------------------------------------------------

def data_description(data):
    """
    Função para exibir um resumo do dataset, incluindo:
    - Dimensões do dataset
    - Tipos de dados
    - Quantidade de valores ausentes
    - Quantidade de registros duplicados
    """
    # 1. Data Dimensions
    print('*' * 20 + ' DATA DIMENSIONS ' + '*' * 20)
    print('Quantidade de linhas: {:,}'.format(data.shape[0]))
    print('Quantidade de colunas: {:,}'.format(data.shape[1]))
    print()

    # 2. Data Types
    print('*' * 20 + ' DATA TYPES ' + '*' * 25)
    print(data.dtypes)
    print()

    # 3. Check NA's
    print('*' * 20 + ' CHECK NAs ' + '*' * 26)
    print(data.isna().sum())
    print()

    # 4. Check duplicated
    print('*' * 20 + ' CHECK DUPLICATED ' + '*' * 19)
    print('Quantidade de registros duplicados: {:,}'.format(data.duplicated().sum()))
    
def calcular_estatisticas_descritivas(dados: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula estatísticas descritivas básicas para um DataFrame, incluindo:
    - Média, mediana, desvio padrão, mínimo, máximo, percentis (via `.describe()`)
    - Amplitude (range)
    - Assimetria (skewness)
    - Curtose
    - Coeficiente de variação (CV)
    
    Parâmetros:
    -----------
    dados : pd.DataFrame
        DataFrame contendo as variáveis numéricas para análise.

    Retorno:
    --------
    pd.DataFrame
        DataFrame transposto com as estatísticas calculadas para cada variável.
    """

    # Obtém estatísticas descritivas básicas
    desc_stats = dados.describe().T  

    # Adiciona métricas estatísticas adicionais
    desc_stats['range'] = dados.max() - dados.min()  # Amplitude total da variável
    desc_stats['skew'] = dados.skew()                # Grau de assimetria da distribuição
    desc_stats['kurtosis'] = dados.kurtosis()        # Grau de achatamento da distribuição (curtose)
    desc_stats['cv'] = (dados.std() / dados.mean())  # Coeficiente de variação (medida de dispersão relativa)

    return desc_stats.round(2)  # Retorna o DataFrame com as estatísticas

# DATA WRANGLING -----------------------------------------------------------------------------------------------

def rename_columns(cols):
    """
    Padroniza os nomes das colunas:
    - Transforma para "Title Case"
    - Remove espaços
    - Remove acentos
    - Converte para snake_case
    """
    cols = list(map(lambda x: inflection.titleize(x), cols))
    cols = list(map(lambda x: x.replace(' ', ''), cols))
    cols = list(map(lambda x: unidecode(x), cols))
    cols = list(map(lambda x: inflection.underscore(x), cols))
    return cols
    
# EDA ----------------------------------------------------------------------------------------------------------

def plotar_graficos(coluna, dados):
    """
    Função para plotar três gráficos (histograma com KDE, boxplot e gráfico de violino) 
    para uma coluna específica de um conjunto de dados.

    Parâmetros:
    -----------
    coluna : str
        Nome da coluna do conjunto de dados que será analisada.
    dados : pandas.DataFrame
        DataFrame contendo os dados a serem visualizados.

    Retorna:
    --------
    None
        A função exibe os gráficos diretamente usando `plt.show()`.
    """
    
    # Cria uma figura com três subplots (1 linha, 3 colunas)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 4))

    # Histograma com KDE
    sns.histplot(dados[coluna], kde=True, bins=30, color="#39568CFF", ax=ax1)
    ax1.set_title(f'Histograma de {coluna}')
    ax1.set_xlabel(coluna)
    ax1.set_ylabel('Frequência')
    ax1.axvline(dados[coluna].mean(), color='red', linestyle='dashed', linewidth=1, label='Mean') # Linha da média
    ax1.legend()

    # Boxplot
    sns.boxplot(x=dados[coluna], color="#F8766D", ax=ax2)
    ax2.set_title(f'Boxplot de {coluna}')
    ax2.set_xlabel(coluna)

    # Gráfico de dispersão
    sns.violinplot(x=dados[coluna], color="#1F968BFF", ax=ax3)
    ax3.set_title(f'Gráfico violino de {coluna}')
    ax3.set_xlabel(coluna)

    # Ajusta o layout para evitar sobreposição de elementos
    plt.tight_layout()

    # Exibe a figura com os três gráficos
    plt.show()


def analise_categorias_vs_target(df, target='y', graficos=True, ordenar_por_media=True):
    """
    Analisa a relação entre variáveis categóricas e uma variável target binária.
    
    Parâmetros:
        df (pd.DataFrame): DataFrame contendo os dados
        target (str): Nome da variável target binária
        graficos (bool): Se True, gera gráficos para cada variável
        ordenar_por_media (bool): Se True, ordena as categorias pela média de y
    
    Retorna:
        tuple: (DataFrame com resultados estatísticos, 
                DataFrame com contagem e média por categoria)
    """
    # Identificar variáveis categóricas (excluindo a target)
    cat_vars = df.select_dtypes(include=['object', 'category']).columns.tolist()
    cat_vars = [var for var in cat_vars if var != target]
    
    # Listas para armazenar resultados
    resultados_temp = []
    estatisticas_temp = []
    
    for var in cat_vars:
        # Calcular estatísticas descritivas primeiro
        stats_df = df.groupby(var)[target].agg(['count', 'mean']).reset_index()
        stats_df.columns = [var, 'Contagem', f'Média_{target}']
        stats_df['Variável'] = var
        estatisticas_temp.append(stats_df)
        
        # Ordenar por média se solicitado
        if ordenar_por_media:
            stats_df = stats_df.sort_values(f'Média_{target}', ascending=False)
            order = stats_df[var].values
        else:
            order = None
        
        # Criar tabela de contingência
        tabela = pd.crosstab(df[target], df[var])
        
        # Reordenar a tabela se necessário
        if ordenar_por_media:
            tabela = tabela[order]
        
        # Calcular teste qui-quadrado
        chi2, p_valor, _, _ = chi2_contingency(tabela)
        
        # Calcular Cramér's V
        n = df.shape[0]
        min_dim = min(tabela.shape) - 1
        cramers_v = np.sqrt(chi2 / (n * min_dim))
        
        # Armazenar resultados estatísticos
        resultados_temp.append({
            'Variável': var,
            'p-valor': p_valor,
            'Cramér_V': cramers_v
        })
        
        # Gerar gráfico se solicitado
        if graficos:
            plt.figure(figsize=(14, 7))
            
            # Gráfico de contagem por categoria com ordenação
            ax = sns.countplot(data=df, x=var, hue=target, order=order)
            
            # Adicionar valores nas barras (contagem)
            for p in ax.patches:
                ax.annotate(f'{int(p.get_height())}', 
                           (p.get_x() + p.get_width() / 2., p.get_height()),
                           ha='center', va='bottom', 
                           xytext=(0, 5), textcoords='offset points',
                           fontsize=9)
            
            # Adicionar as médias acima de cada grupo
            for i, categoria in enumerate(order if ordenar_por_media else stats_df[var].values):
                mean_val = stats_df.loc[stats_df[var] == categoria, f'Média_{target}'].values[0]
                ax.text(i, ax.get_ylim()[1]*0.95, 
                       f'Média: {mean_val:.3f}',
                       ha='center', va='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                       fontsize=10)
            
            plt.title(f'Distribuição de {target} por {var}\n'
                     f'p-valor: {p_valor:.4f} - Cramér\'s V: {cramers_v:.4f}')
            plt.ylabel('Contagem de Observações')
            plt.xlabel(var)
            plt.legend(title=target, bbox_to_anchor=(1.05, 1))
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
    
    # Criar DataFrames de resultados
    resultados = pd.DataFrame(resultados_temp).sort_values('p-valor')
    estatisticas = pd.concat(estatisticas_temp, ignore_index=True)
    
    return resultados

# ML MODELING --------------------------------------------------------------------------------------------------
    
# Função das métricas de performance dos modelos de machine learning

def ml_error(model_name, y, yhat):
    mae = mean_absolute_error(y, yhat)
    mape = mean_absolute_percentage_error(y, yhat)
    mse = mean_squared_error(y, yhat)
    rmse = np.sqrt(mean_squared_error(y, yhat))
    return {
        'model_name': model_name,
        'RMSE': float(rmse),
        'MAPE': mape,
        'MSE' : mse,
        'MAE' : mae
    }

# Plots

def plot_countplots(df, columns, fig_width=16, subplot_width=5):
    """
    Plota countplots horizontais para múltiplas colunas em subplots lado a lado.
    
    Parâmetros:
    -----------
    df : DataFrame
        DataFrame contendo os dados
    columns : list
        Lista de nomes de colunas para plotar
    fig_width : int (opcional)
        Largura total da figura (default=16)
    subplot_width : int (opcional)
        Altura de cada subplot (default=5)
    """
    n_cols = len(columns)
    if n_cols == 0:
        return
    
    # Cria a figura com subplots
    fig, axes = plt.subplots(1, n_cols, figsize=(fig_width, subplot_width))
    
    # Se for apenas uma coluna, axes não é array - transformamos em lista para uniformizar
    if n_cols == 1:
        axes = [axes]
    
    # Itera sobre cada coluna e eixo correspondente
    for col, ax in zip(columns, axes):
        # Cria o countplot
        sns.countplot(data=df, y=col, order=df[col].value_counts().index, ax=ax)
        
        # Adiciona os valores nas barras
        for container in ax.containers:
            ax.bar_label(container, fmt='%d', label_type='edge', padding=5)
        
        # Remove bordas desnecessárias
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        
        # Configura labels e título
        ax.set_ylabel('Categoria')
        ax.set_xlabel('Quantidade')
        ax.set_title(f'Quantidade por categoria em {col.replace("_", " ").title()}')
    
    # Ajusta o layout para evitar sobreposição
    plt.tight_layout()
    plt.show()

def plot_boxplots(df, cat_columns, target_col, fig_width=16, subplot_height=5, rotation=30):
    """
    Plota boxplots comparando uma variável target com múltiplas colunas categóricas.
    
    Parâmetros:
    -----------
    df : DataFrame
        DataFrame contendo os dados
    cat_columns : list
        Lista de colunas categóricas para plotar no eixo X
    target_col : str
        Coluna numérica para plotar no eixo Y
    fig_width : int (opcional)
        Largura total da figura (default=16)
    subplot_height : int (opcional)
        Altura de cada subplot (default=5)
    rotation : int (opcional)
        Rotação dos labels do eixo X (default=30)
    """
    n_cols = len(cat_columns)
    if n_cols == 0:
        return
    
    # Cria a figura com subplots
    fig, axes = plt.subplots(1, n_cols, figsize=(fig_width, subplot_height))
    
    # Se for apenas uma coluna, axes não é array - transformamos em lista para uniformizar
    if n_cols == 1:
        axes = [axes]
    
    # Formatação do target para exibição
    target_title = target_col.replace("_", " ").title()
    
    # Itera sobre cada coluna categórica e eixo correspondente
    for col, ax in zip(cat_columns, axes):
        # Cria o boxplot
        sns.boxplot(data=df, x=col, y=target_col, ax=ax)
        
        # Remove bordas desnecessárias
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        
        # Configura labels e título
        ax.set_xlabel('Categoria')
        ax.set_ylabel(target_title)
        ax.set_title(f'Distribuição de {target_title}\npor {col.replace("_", " ").title()}')
        
        # Rotaciona labels e adiciona grid
        ax.tick_params(axis='x', rotation=rotation)
        ax.grid(axis='y')
    
    # Ajusta o layout para evitar sobreposição
    plt.tight_layout()
    plt.show()

def plot_line_interactions(df, x_columns, hue_columns, target_col, fig_width=20, subplot_height=5, rotation=30):
    """
    Plota lineplots mostrando a interação entre variáveis categóricas e uma variável target.
    
    Parâmetros:
    -----------
    df : DataFrame
        DataFrame contendo os dados
    x_columns : list
        Lista de colunas para o eixo X (pode ter repetições)
    hue_columns : list
        Lista de colunas para o hue (deve ter mesmo tamanho de x_columns)
    target_col : str
        Coluna numérica para o eixo Y
    fig_width : int (opcional)
        Largura total da figura (ajustável conforme número de gráficos)
    subplot_height : int (opcional)
        Altura de cada subplot
    rotation : int (opcional)
        Rotação dos labels do eixo X
    """
    n_plots = len(x_columns)
    if n_plots == 0 or len(hue_columns) != n_plots:
        print("Erro: x_columns e hue_columns devem ter o mesmo tamanho")
        return
    
    # Formatação do target para exibição
    target_label = target_col.replace('_', ' ').title()
    
    # Cria a figura com subplots
    fig, axes = plt.subplots(1, n_plots, figsize=(fig_width, subplot_height))
    
    # Se for apenas um gráfico, axes não é array - transformamos em lista
    if n_plots == 1:
        axes = [axes]
    
    # Prepara os dados agrupados
    grouped_dfs = []
    for x_col, hue_col in zip(x_columns, hue_columns):
        grouped = df.groupby([x_col, hue_col])[target_col].mean().reset_index()
        grouped_dfs.append(grouped)
    
    # Itera sobre cada combinação de colunas
    for idx, (x_col, hue_col, grouped_df, ax) in enumerate(zip(x_columns, hue_columns, grouped_dfs, axes)):
        # Cria o lineplot
        sns.lineplot(data=grouped_df, x=x_col, y=target_col, hue=hue_col, marker='o', ax=ax)
        
        # Remove bordas desnecessárias
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        
        # Configura labels e título
        ax.set_xlabel('Categoria')
        ax.set_ylabel(f'Média de {target_label}')
        ax.set_title(f'Interação {x_col.replace("_", " ").title()}\ne {hue_col.replace("_", " ").title()}')
        
        # Rotaciona labels e configura legenda
        ax.tick_params(axis='x', rotation=rotation)
        ax.legend(title=hue_col.replace("_", " ").title())
    
    # Ajusta o layout
    plt.tight_layout()
    plt.show()


def plotar_boxplots(colunas, dados, colunas_por_linha=5):
    """
    Plota múltiplos boxplots em uma grade de N linhas e 5 colunas.

    Parâmetros:
    -----------
    colunas : list
        Lista das colunas a serem plotadas.
    dados : pandas.DataFrame
        DataFrame com os dados.
    colunas_por_linha : int, opcional (padrão=5)
        Número de boxplots por linha.
    """
    num_colunas = len(colunas)
    num_linhas = (num_colunas + colunas_por_linha - 1) // colunas_por_linha
    
    # Ajusta o tamanho da figura (largura, altura)
    fig, axs = plt.subplots(
        num_linhas, 
        colunas_por_linha, 
        figsize=(20, 3 * num_linhas)  # 20 de largura, 3*N de altura
    )
    
    # Caso especial: apenas 1 linha (axs vira um array 1D)
    if num_linhas == 1:
        axs = axs.reshape(1, -1)
    
    # Plota cada boxplot
    for i, coluna in enumerate(colunas):
        linha = i // colunas_por_linha
        col = i % colunas_por_linha
        
        sns.boxplot(x=dados[coluna], color="#1F968BFF", ax=axs[linha, col])
        axs[linha, col].set_title(f'Boxplot de {coluna.replace("_", " ").title()}')
        axs[linha, col].set_xlabel(f'{coluna.replace("_", " ").title()}')
    
    # Desativa eixos vazios (se houver)
    for i in range(num_colunas, num_linhas * colunas_por_linha):
        linha = i // colunas_por_linha
        col = i % colunas_por_linha
        axs[linha, col].axis('off')
    
    plt.tight_layout()
    plt.show()

# Seleção de variáveis

class FeatureSelector(BaseEstimator, RegressorMixin):
    """
    Classe para seleção de variáveis usando métodos forward, backward ou stepwise (both),
    com critérios de seleção baseados em AIC, BIC ou p-valor.

    Parâmetros:
    -----------
    X : pd.DataFrame
        DataFrame contendo as variáveis independentes (features).
    y : pd.Series
        Série contendo a variável dependente (target).
    metodo : str, opcional (padrão='forward')
        Método de seleção: 'forward', 'backward' ou 'both'.
    metrica : str, opcional (padrão='aic')
        Critério de seleção: 'aic', 'bic' ou 'pvalor'.
    signif : float, opcional (padrão=0.05)
        Nível de significância para o critério de p-valor.
    epsilon : float, opcional (padrão=0.01)
        Tolerância para parar o processo no método stepwise.
    """

    def __init__(self, metodo='forward', metrica='aic', signif=0.05, epsilon=0.01):
        self.metodo = metodo
        self.metrica = metrica
        self.signif = signif
        self.epsilon = epsilon
        self.selected_features_ = []
        self.steps_ = []

    def fit(self, X, y):
        """
        Ajusta o modelo de seleção de variáveis com base no método e critério escolhidos.
        """
        self.X = X.copy()
        self.y = y.copy()
        var_independente = self.X.columns.tolist()

        # Remove 'const' da lista de variáveis independentes, se existir
        if 'const' in var_independente:
            var_independente.remove('const')

        if self.metodo == 'forward':
            self._forward_selection(var_independente)
        elif self.metodo == 'backward':
            self._backward_selection(var_independente)
        elif self.metodo == 'both':
            self._stepwise_selection(var_independente)
        else:
            raise ValueError("Método inválido. Escolha entre 'forward', 'backward' ou 'both'.")

        return self

    def _forward_selection(self, var_independente):
        """
        Seleção forward com base no critério escolhido.
        """
        aic_melhor = float('inf')
        bic_melhor = float('inf')
        pval_melhor = float('inf')

        while var_independente:
            lista_metricas = []
            lista_variaveis = []

            for var in var_independente:
                features = self.selected_features_ + [var]
                X_temp = self.X[features].copy()
                X_temp = sm.add_constant(X_temp)
                model = sm.OLS(self.y, X_temp).fit()

                if self.metrica == 'aic':
                    metrica = model.aic
                elif self.metrica == 'bic':
                    metrica = model.bic
                elif self.metrica == 'pvalor':
                    metrica = model.pvalues[var]
                else:
                    raise ValueError("Métrica inválida. Escolha entre 'aic', 'bic' ou 'pvalor'.")

                lista_metricas.append(metrica)
                lista_variaveis.append(var)

            melhor_metrica = min(lista_metricas)
            melhor_var = lista_variaveis[np.argmin(lista_metricas)]

            if (self.metrica in ['aic', 'bic'] and melhor_metrica < (aic_melhor if self.metrica == 'aic' else bic_melhor)) or \
               (self.metrica == 'pvalor' and melhor_metrica < self.signif):
                self.selected_features_.append(melhor_var)
                var_independente.remove(melhor_var)
                if self.metrica == 'aic':
                    aic_melhor = melhor_metrica
                elif self.metrica == 'bic':
                    bic_melhor = melhor_metrica
                elif self.metrica == 'pvalor':
                    pval_melhor = melhor_metrica

                self.steps_.append({
                    'tipo': 'forward',
                    'var': self.selected_features_.copy(),
                    self.metrica: melhor_metrica
                })
            else:
                break

    def _backward_selection(self, var_independente):
        """
        Seleção backward com base no critério escolhido.
        """
        self.selected_features_ = var_independente.copy()
        aic_melhor = float('inf')
        bic_melhor = float('inf')
        pval_melhor = float('inf')

        while len(self.selected_features_) > 1:
            X_temp = self.X[self.selected_features_].copy()
            X_temp = sm.add_constant(X_temp)
            model = sm.OLS(self.y, X_temp).fit()

            if self.metrica == 'aic':
                metrica = model.aic
                pvals = model.pvalues[1:]  # Ignora o intercepto
                pior_var = pvals.idxmax()
            elif self.metrica == 'bic':
                metrica = model.bic
                pvals = model.pvalues[1:]
                pior_var = pvals.idxmax()
            elif self.metrica == 'pvalor':
                pvals = model.pvalues[1:]
                pior_var = pvals.idxmax()
                metrica = pvals[pior_var]
            else:
                raise ValueError("Métrica inválida. Escolha entre 'aic', 'bic' ou 'pvalor'.")

            if (self.metrica in ['aic', 'bic'] and metrica < (aic_melhor if self.metrica == 'aic' else bic_melhor)) or \
               (self.metrica == 'pvalor' and metrica > self.signif):
                self.selected_features_.remove(pior_var)
                if self.metrica == 'aic':
                    aic_melhor = metrica
                elif self.metrica == 'bic':
                    bic_melhor = metrica
                elif self.metrica == 'pvalor':
                    pval_melhor = metrica

                self.steps_.append({
                    'tipo': 'backward',
                    'var': self.selected_features_.copy(),
                    self.metrica: metrica
                })
            else:
                break

    def _stepwise_selection(self, var_independente):
        """
        Seleção stepwise (both) com base no critério escolhido.
        """
        self._forward_selection(var_independente)
        self._backward_selection(self.selected_features_)

    def transform(self, X):
        """
        Retorna o DataFrame com as variáveis selecionadas.
        """
        return X[self.selected_features_]

    def fit_transform(self, X, y):
        """
        Ajusta o modelo e retorna o DataFrame com as variáveis selecionadas.
        """
        self.fit(X, y)
        return self.transform(X)

    def get_steps(self):
        """
        Retorna os passos de seleção como um DataFrame.
        """
        return pd.DataFrame(self.steps_)

    def help(self):
        """
        Exibe uma mensagem de ajuda com informações sobre a classe.
        """
        help_text = """
        FeatureSelector - Seleção de Variáveis

        Parâmetros:
        -----------
        X : pd.DataFrame
            DataFrame contendo as variáveis independentes (features).
        y : pd.Series
            Série contendo a variável dependente (target).
        metodo : str, opcional (padrão='forward')
            Método de seleção: 'forward', 'backward' ou 'both'.
        metrica : str, opcional (padrão='aic')
            Critério de seleção: 'aic', 'bic' ou 'pvalor'.
        signif : float, opcional (padrão=0.05)
            Nível de significância para o critério de p-valor.
        epsilon : float, opcional (padrão=0.01)
            Tolerância para parar o processo no método stepwise.

        Métodos:
        --------
        fit(X, y): Ajusta o modelo de seleção de variáveis.
        transform(X): Retorna o DataFrame com as variáveis selecionadas.
        fit_transform(X, y): Ajusta o modelo e retorna o DataFrame transformado.
        get_steps(): Retorna os passos de seleção como um DataFrame.
        help(): Exibe esta mensagem de ajuda.
        """
        print(help_text)