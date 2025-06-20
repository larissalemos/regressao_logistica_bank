# <center>Modelo de Regressão Logística para Previsão de Subscrição de Depósitos Bancários

### **Visão Geral**

Este projeto desenvolveu um modelo preditivo para identificar clientes com maior probabilidade de subscrever depósitos a prazo em uma instituição bancária portuguesa, utilizando dados de campanhas de marketing direto.

Fonte dos Dados: [Moro et al., 2011] - Bank Marketing Dataset (bank.csv)

### **Metodologia**

**Técnica**: Regressão Logística com seleção de variáveis

**Seleção de Variáveis:**

Dois modelos criados (Stepwise AIC e Stepwise p-valor)

**Modelo final selecionado:** Stepwise AIC (desempenho similar, preferido por critérios de negócio)

### **Principais Resultados**

**Principais Insights**

- Fatores que Aumentam a Conversão
    - Sucesso em campanhas anteriores (4x mais chances)
    - Estudantes/aposentados (quase 2x mais chances)
    - Educação superior (+55%)
    - Contato recente (<180 dias: +31%)

- Fatores que Reduzem a Conversão
    - Empréstimos (pessoal: -51%; imobiliário: -43%)
    - Canal de contato desconhecido (-71%)
    - 3º trimestre (-27%) e idade 30-57 anos (-23%)

- Fatores Neutros/Moderados
    - Estado civil (solteiros/divorciados: +24%)
    - Dia do mês (dias 17-21: -26%)

**Recomendações de Ação**

- Priorizar clientes com:
    - Campanhas anteriores bem-sucedidas.
    - Perfil de estudante/aposentado + educação superior.
    - Evitar contatos por canais desconhecidos e clientes com empréstimos.
    - Ajustar timing de campanhas (evitar 3º trimestre e dias 17-21).

**Medidas de Decisão**
Ao avaliar melhor corte para decisão no modelo, observa-se que o corte em 0,2 apresenta o melhor equilíbrio entre precisão e recall, em comparação com os cortes em 0,5 e 0,8, com f1-score de 0,5.
Sendo assim, ao considerar que quando a probabilidade de subscrição é maior que 0,2, vamos agir sobre o indivíduo, tem-se:
- Precisão de 0,46, o que indica que dentre todos os indivíduos que entraremos em contato para tentar subscrição, 46% vão subscrever. 
- Recall de 0,62, o que indica que dentre todos os indivíduos que poderiam se subscrever, o modelo está captando 62% deles.


**Medidas de Ordenação**
Se tivermos limitações de número de indivíduos sobre os quais podemos agir, é necessário analisar medidas como o lift.
Considerando que só temos recursos para agir em 10% da nossa base de dados, o lift seria de 4,76, ou seja, o modelo apresentaria um resultado 376% melhor do que se a ação fosse realizada aleatoriamente. 

O AUC de 0,88 indica que o modelo está performando muito bem em separar os indivíduos que vão subscrever dos que não vão, mais especificamente, 76% melhor do que a aleatoriedade.

**Nota Técnica:** Detalhes completos da análise estão disponíveis no notebook de desenvolvimento.

