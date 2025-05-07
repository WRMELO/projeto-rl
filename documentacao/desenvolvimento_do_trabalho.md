# TRABALHO DA DISCIPLINA DE REINFORCEMENT LEARNING
*WILSON ROBERTO DE MELO - RM 357053*

## 1. Introdução

### 1.1 Contexto do Projeto

Este projeto acadêmico faz parte da disciplina de Reinforcement Learning (RL), campo da inteligência artificial em que um agente aprende a tomar decisões eficazes através da interação constante com o ambiente, buscando maximizar recompensas ao longo do tempo. Especificamente, esta pesquisa está focada na aplicação de técnicas avançadas de RL para a gestão automática de uma carteira financeira composta por três importantes ações do mercado brasileiro: Vale, Petrobras e Brasil Foods. Esses ativos foram escolhidos devido à sua relevância econômica e às características diversificadas de volatilidade e liquidez, o que configura um desafio significativo e realista para o agente de RL.

### 1.2 Motivação e Relevância

A motivação central deste projeto é responder à crescente necessidade por soluções automatizadas de gestão financeira que possam auxiliar investidores na tomada de decisões rápidas e bem fundamentadas diante da complexidade e volatilidade do mercado financeiro atual. Além de atender objetivos acadêmicos, este projeto visa desenvolver um Produto Mínimo Viável (MVP) prático, que servirá como ferramenta base para futuras expansões e aprimoramentos, potencializando uma gestão financeira pessoal mais eficiente e menos sujeita a erros humanos.

### 1.3 Objetivos Gerais e Específicos

**Objetivo Geral:**
Desenvolver um agente de Reinforcement Learning (RL) para automatizar a gestão de uma carteira de ações, maximizando métricas financeiras como lucro acumulado e Sharpe Ratio.

**Objetivos Específicos:**

* Definir o ambiente de RL com clareza, especificando estados, ações e recompensas;
* Implementar um agente utilizando a técnica de Deep Q-Network (DQN);
* Avaliar o desempenho do agente em simulações financeiras detalhadas;
* Gerar insights práticos para aprimoramento contínuo da estratégia de investimento desenvolvida.


## 2. Fundamentos Teóricos

### 2.1 O que é Reinforcement Learning (RL)?

Reinforcement Learning (RL), ou Aprendizado por Reforço, é um ramo da inteligência artificial no qual um agente aprende a tomar decisões otimizadas através da interação contínua com o ambiente, buscando maximizar a soma das recompensas recebidas ao longo do tempo. O aprendizado ocorre por tentativa e erro, ajustando ações futuras com base no retorno das decisões passadas.

### 2.2 Aplicações de RL em Finanças

No mercado financeiro, o RL é empregado devido à sua habilidade em lidar com ambientes dinâmicos e incertos. Contudo, existem outras abordagens de aprendizado com aplicações financeiras relevantes:

* **Deep Learning (DL)**: amplamente utilizado em previsão de séries temporais e detecção de padrões complexos em grandes volumes de dados financeiros. Um exemplo prático é o uso de redes neurais convolucionais (CNNs) para análise de gráficos de preços e identificação de padrões técnicos.

* **Machine Learning (ML)**: frequentemente usado em classificação de crédito, previsão de riscos financeiros e trading algorítmico baseado em aprendizado supervisionado. Um exemplo é a aplicação de algoritmos como XGBoost para prever preços futuros com base em dados históricos.

* **Reinforcement Learning (RL)**: especificamente aplicado em tarefas como gestão automatizada de portfólios, execução otimizada de ordens e market making. Instituições financeiras, como JP Morgan e Goldman Sachs, já utilizam agentes de RL para otimizar suas estratégias de trading e gestão de risco em tempo real.

### 2.3 Algoritmos Principais de RL

Os algoritmos mais importantes e utilizados no contexto de Reinforcement Learning incluem:

* **Q-Learning**: algoritmo clássico de RL baseado em valor, que aprende uma função Q para estimar a recompensa esperada das ações.
* **Deep Q-Network (DQN)**: extensão do Q-Learning que usa redes neurais profundas para aproximar a função Q, permitindo aplicações em ambientes complexos e com grandes espaços de estado.

Outros algoritmos importantes:

* **Policy Gradients (PG)**
* **Actor-Critic Methods (A2C, A3C)**
* **Proximal Policy Optimization (PPO)**
* **Deep Deterministic Policy Gradient (DDPG)**

Cada um desses algoritmos tem características específicas que podem ser aproveitadas conforme o tipo e complexidade do problema abordado.

### 2.4 Métricas de Avaliação Financeira

Para avaliar modelos financeiros baseados em RL, é fundamental utilizar métricas adequadas que considerem não só a rentabilidade, mas também o risco. As métricas mais relevantes incluem:

* **Lucro Total**: mede o ganho financeiro acumulado em um período determinado. Embora essencial, isoladamente pode não refletir adequadamente o risco assumido.

* **Retorno Anualizado**: padroniza o retorno acumulado para um período anual, permitindo comparações consistentes entre estratégias distintas.

* **Índice de Sharpe**: amplamente utilizado em ferramentas de mercado (ex.: QuantConnect, TradingView), mede o retorno excedente (sobre a taxa livre de risco) ajustado pelo risco (volatilidade). É uma das principais métricas sugeridas devido à sua eficiência em equilibrar retorno e risco.

* **Índice de Sortino**: similar ao Sharpe, porém considera apenas a volatilidade negativa, penalizando apenas a variabilidade dos retornos abaixo da média. Muito utilizado por traders que desejam focar na proteção contra perdas significativas.

* **Máximo Drawdown**: essencial para avaliar o pior cenário possível, indica a maior perda acumulada desde um pico anterior até um fundo subsequente. É crucial em estratégias financeiras, especialmente para investidores com baixa tolerância a risco.

Levando isto em conta, adotamos as seguintes métricas para este projeto: **Lucro Total** devido à importância na mensuração do desempenho financeiro bruto; **Índice de Sharpe** pela sua capacidade de balancear retorno e risco; e **Máximo Drawdown** para entender claramente o risco máximo potencial envolvido na estratégia de investimento.


## 3. Definição do Problema de RL no Contexto Financeiro

### 3.1 Estados (Variáveis de Mercado)

A definição adequada dos estados é um dos elementos mais críticos no sucesso de qualquer aplicação de Reinforcement Learning. No contexto deste projeto, os estados representam o conjunto de informações disponíveis ao agente em cada instante de tempo, com base nas quais as decisões de compra, venda ou manutenção serão tomadas.

Para garantir uma representação rica, coerente e temporalmente alinhada, adotamos uma **base temporal diária**, com registros para cada dia útil de negociação. Essa escolha é consistente com a frequência de disponibilização dos dados de mercado e é adequada ao horizonte de decisão de investidores individuais e institucionais.

Os estados serão compostos pelas seguintes variáveis:

* **Preços Históricos**: inclui os preços de abertura, máximo, mínimo e fechamento (OHLC) de cada ativo. Esses dados formam a base bruta sobre a qual os demais indicadores são calculados.

* **Volume de Negociações**: quantidade de ações negociadas diariamente, fornecendo uma dimensão de liquidez e interesse do mercado pelo ativo.

* **Indicadores Técnicos** (calculados com base em janelas móveis diárias):

  1. **Média Móvel Exponencial de 14 dias (EMA-14)**: suaviza os preços recentes, dando mais peso às observações mais novas. Ajuda a detectar tendências de curto prazo.

  2. **Índice de Força Relativa (RSI-14)**: mede a velocidade e a mudança dos movimentos de preço. Valores extremos sugerem condições de sobrecompra ou sobrevenda.

  3. **MACD (Moving Average Convergence Divergence)**: indicador de momentum que mostra a relação entre duas médias móveis de preços. Muito utilizado para detectar reversões de tendência.

  4. **Bandas de Bollinger (20 dias)**: representam zonas de suporte e resistência com base na volatilidade. Indicadores úteis para detectar rompimentos ou consolidações de preço.

  5. **Oscilador Estocástico (Stochastic Oscillator - 14,3)**: compara o preço de fechamento com a faixa de preços durante um período. Ajuda a identificar condições de sobrecompra/sobrevenda com mais sensibilidade que o RSI.

Esses cinco indicadores foram selecionados por sua complementaridade, robustez estatística e ampla aceitação em análises técnicas profissionais. Eles fornecem ao agente uma base informacional equilibrada entre tendência, momentum e volatilidade.

Todos os dados utilizados serão previamente normalizados e alinhados temporalmente para garantir a consistência na entrada do modelo de RL.

### 3.2 Ações (Decisões de Investimento)

O conjunto de ações possíveis que o agente pode executar a cada passo de tempo foi definido como:

* **Compra (Buy)**: aquisição de uma unidade do ativo, desde que haja capital disponível.
* **Venda (Sell)**: venda de uma unidade do ativo, desde que o ativo esteja presente em carteira.
* **Manutenção (Hold)**: nenhuma operação, mantendo a posição atual.

Essas três ações formam o espaço de decisão básico e suficiente para simular estratégias reais de negociação.

### 3.3 Recompensas (Feedback do Ambiente)

A função de recompensa é projetada para guiar o aprendizado do agente com base em seus resultados operacionais. Consideramos três componentes principais:

* **Lucro Líquido Diário**: variação do valor da carteira com base na operação realizada. Essa é a principal fonte de recompensa positiva.
* **Penalidade por Perda**: caso a operação resulte em prejuízo, uma penalização proporcional é aplicada.
* **Custos de Transação**: cada compra ou venda incorre em uma penalidade fixa que simula custos operacionais reais (como taxas de corretagem e spread).

Levando esses elementos em consideração, estruturamos a função de recompensa para estimular comportamentos consistentes com uma gestão financeira racional: **maximização de retorno ajustado ao risco** e **minimização de operações ineficazes**.


## 4. Coleta e Análise Inicial de Dados

### 4.1 Fontes dos Dados

Para este projeto, optamos pela utilização da biblioteca `yfinance` como interface de acesso aos dados históricos do Yahoo Finance, uma fonte amplamente reconhecida e gratuita. Os dados foram coletados com frequência **diária**, cobrindo os **últimos cinco anos** de negociação para cada ativo.

O script Python utilizado para a coleta é apresentado abaixo:

```python
import yfinance as yf
import pandas as pd

# Definir os tickers das ações e o período
lista_tickers = ['VALE3.SA', 'PETR4.SA', 'BRFS3.SA']  # Vale, Petrobrás, Brasil Foods
periodo = '5y'  # Últimos 5 anos

# Baixar os dados de todos os tickers
for ticker in lista_tickers:
    dados = yf.download(ticker, period=periodo, interval='1d')
    caminho = rf'c:\Users\wilso\OneDrive\MBA FIAP\10. Reinforcement Learning\ENTREGA DO EXERCÍCIO\dados_{ticker.replace(",", "").replace(".SA", "").lower()}.csv'
    dados.to_csv(caminho)
    print(f'Dados de {ticker} salvos em: {caminho}')

# Exibir as primeiras linhas
dados.head()
```

Essa abordagem permitiu extrair diretamente os seguintes campos: `Open`, `High`, `Low`, `Close`, `Adj Close` e `Volume`, que serão posteriormente utilizados para o cálculo dos indicadores técnicos.

---

### 4.2 Descrição dos Ativos: Vale, Petrobras, Brasil Foods

**VALE3.SA (Vale S.A.)**
A Vale é uma das maiores mineradoras do mundo, com destaque na produção de minério de ferro e níquel. Sua ação possui alta liquidez e volatilidade, sendo fortemente influenciada por fatores internacionais, como demanda da China e preços globais de commodities.

**PETR4.SA (Petróleo Brasileiro S.A. - Petrobras)**
A Petrobras é uma das maiores empresas de energia da América Latina, com atividades centradas na exploração e refino de petróleo. A ação preferencial PETR4 apresenta ampla liquidez e sensibilidade a oscilações do preço do barril de petróleo e decisões político-econômicas.

**BRFS3.SA (BRF S.A. - Brasil Foods)**
Empresa do setor de alimentos processados, dona de marcas como Sadia e Perdigão. Sua ação tende a apresentar menor volatilidade relativa em comparação com empresas de energia e mineração, refletindo características típicas do setor de consumo não durável.

---

### 4.3 Análise Exploratória dos Dados

Após a coleta dos dados, uma análise exploratória foi conduzida para:

* Verificar **consistência temporal** dos dados (datas coincidentes entre os ativos);
* Confirmar ausência de **valores nulos ou inconsistentes** nas colunas essenciais (`Close`, `Volume`, etc.);
* Observar a **distribuição e comportamento dos preços** ao longo do tempo, buscando entender padrões visuais e outliers;
* Validar a **presença de variação suficiente** nos dados para o cálculo de indicadores técnicos como RSI, MACD e Bandas de Bollinger.

As primeiras linhas de cada dataset foram inspecionadas manualmente e os gráficos de linhas para o `Close` de cada ação foram plotados, fornecendo uma visão inicial da dinâmica de preços no período analisado.

Essa etapa assegura que os dados estão prontos para serem normalizados e preparados para entrada no ambiente de aprendizado por reforço.

---

### 4.4 Discussão sobre Escolhas e Integridade dos Dados

**Fonte escolhida:** o Yahoo Finance, acessado via `yfinance`, foi selecionado por sua confiabilidade, cobertura gratuita e facilidade de automação via Python. Evita-se assim dependência de dados pagos ou formatos proprietários.

**Frequência diária:** foi adotada como base por representar um equilíbrio entre granularidade e ruído. Permite capturar movimentos relevantes do mercado sem sobrecarregar o agente com microvariações intradiárias.

**Período de 5 anos:** essa escolha cobre diferentes ciclos econômicos, garante amostragem estatística adequada para cálculo de indicadores e fornece volume de dados suficiente para o treinamento e validação do agente de RL.

**Ativos diversificados:** os três ativos escolhidos representam setores distintos (mineração, energia e alimentos), o que enriquece o problema e permite ao agente aprender comportamentos diversos de mercado.

**Sincronia temporal:** a verificação manual e gráfica assegurou que os três datasets compartilham as mesmas datas úteis, evitando desalinhamentos que comprometeriam o treinamento.

**Ausência de dados faltantes:** não foram observadas inconsistências nas colunas críticas. Em caso de valores nulos residuais, será adotado o preenchimento por propagação direta (forward fill).

**Viabilidade dos indicadores técnicos:** a série temporal de 5 anos com frequência diária fornece janelas suficientes para o cálculo confiável dos cinco indicadores selecionados (EMA, RSI, MACD, Bollinger Bands, Stochastic Oscillator), sem necessidade de truncamento excessivo.

Com esses pontos atendidos, os dados estão validados e prontos para a próxima etapa de transformação e modelagem.

## 5. Preparação dos Dados para Reinforcement Learning (RL)

A preparação dos dados representa uma das etapas mais importantes de todo o projeto, pois define como o agente perceberá o ambiente com o qual irá interagir. Diferente de modelos supervisionados — nos quais cada linha representa um exemplo independente com entrada e saída — no contexto de RL os dados devem ser organizados como uma **sequência temporal de decisões**, com base em **estados**, **ações** e **recompensas**.

Nesta fase, nosso objetivo é transformar os dados financeiros brutos em um **ambiente observável e estruturado**, onde cada instante do tempo representa um cenário completo para a tomada de decisão do agente.

---

### 5.1 Limpeza e Tratamento dos Dados

Antes de extrair qualquer tipo de informação útil para o aprendizado, é necessário garantir que os dados estejam íntegros, consistentes e sincronizados entre os diferentes ativos.

As principais ações realizadas nesta etapa são:

- **Remoção ou interpolação de valores nulos** em colunas essenciais como `Close`, `Open` e `Volume`, que comprometem o cálculo de indicadores técnicos e a simulação de transações;
- **Alinhamento de datas entre os três ativos (VALE3, PETR4 e BRFS3)**, tomando como base a interseção entre os dias válidos de todos os ativos para garantir consistência temporal;
- **Conversão e padronização de tipos de dados**, assegurando que campos como `Volume` estejam devidamente formatados como `float` ou `int`, conforme apropriado;
- **Indexação temporal unificada**, utilizando a coluna `Date` como índice de linha, o que facilita operações como janelas móveis, comparações entre ativos e replicação da realidade sequencial do mercado.

Esse processo assegura que, a cada dia de negociação, o agente receba uma visão completa, confiável e comparável entre todos os ativos envolvidos.

---

### 5.2 Engenharia de Features (Indicadores Técnicos)

Com os dados devidamente tratados, passamos à extração de variáveis derivadas que representam o comportamento do mercado de forma mais informativa e sintética. Essa etapa, conhecida como engenharia de features, visa aumentar o poder explicativo das entradas fornecidas ao agente de Reinforcement Learning.

Optou-se por utilizar **indicadores técnicos consagrados na literatura financeira**, aplicados individualmente aos ativos VALE3, PETR4 e BRFS3, com foco na simplicidade, complementaridade e eficiência computacional. O objetivo não foi a exaustividade, mas sim a construção de um **vetor de estado enxuto e representativo**, adequado ao treinamento de agentes com capacidade generalizadora.

Os indicadores selecionados para compor o Produto Mínimo Viável (MVP) foram:

- **SMA-5 e SMA-20 (Médias Móveis Simples)**: suavizam flutuações diárias e capturam tendências de curto e médio prazo.
- **Retorno Diário (`Return`)**: variação percentual do preço de fechamento, usado para detectar reversões e movimentos recentes.
- **Volatilidade (`Volatility`)**: desvio padrão do retorno diário em janela de 10 dias, utilizado como proxy de risco.
- **RSI-14 (Índice de Força Relativa)**: indicador de momentum que identifica condições de sobrecompra ou sobrevenda.
- **Volume (transformado com log1p)**: expressa a intensidade da atividade de negociação em escala menos sensível a picos extremos.

Essas variáveis foram escolhidas por sua complementaridade: enquanto algumas expressam **direção ou força de tendência** (como SMA e RSI), outras representam **risco, variação recente e liquidez** — fornecendo ao agente múltiplas dimensões informacionais para a tomada de decisão.

---

#### Normalização

A fim de garantir estabilidade e eficiência no processo de aprendizado, todas as variáveis foram **normalizadas por ativo** utilizando abordagens adequadas ao tipo de dado:

- **Z-score**: aplicado a `Return`, `Volatility` e `RSI`, centralizando os dados em média zero com variância unitária;
- **Escala Relativa**: aplicada aos preços (`Close`, `SMA`), dividindo-os pelo valor da primeira observação válida;
- **Logaritmização**: aplicada ao `Volume`, com uso de `log1p` para conter a escala e preservar distribuição.

Os vetores de estado resultantes foram salvos como arquivos `.csv` separados para cada ativo (`VALE3_estado.csv`, etc.), contendo as colunas finalizadas e prontas para compor o ambiente de simulação.

Indicadores adicionais como `MACD`, `Bandas de Bollinger`, `EMA` ou `OBV` foram considerados durante o planejamento, mas foram excluídos da versão inicial do MVP para manter a estrutura leve, interpretável e focada no desempenho essencial. Esses indicadores permanecem documentados como **possíveis extensões futuras** do modelo.

---

### 5.3 Definição do Espaço de Estados para RL

O próximo passo é estruturar aquilo que será efetivamente passado ao agente como entrada em cada passo de tempo: o **vetor de estado**.

Cada estado será formado por:

- Os indicadores técnicos calculados e normalizados para os três ativos;
- As **posições atuais do agente** em cada um dos ativos (`0` = fora do ativo, `1` = posição comprada);
- O **saldo de caixa disponível**, expresso em reais, que será utilizado para avaliar a viabilidade de novas compras.

O saldo de caixa é uma parte fundamental da simulação, pois permite ao agente tomar decisões realistas sobre alocação de recursos. Ao contrário de um ambiente hipotético com recursos infinitos, aqui o agente precisará respeitar seu orçamento, assim como faria um investidor real.

Essa estrutura garante que, a cada novo dia, o agente tenha uma visão consolidada do cenário atual — tanto externo (condições de mercado) quanto interno (seu próprio portfólio).

---

### 5.4 Inicialização da Carteira e Estratégias de Benchmark

Para simular um ambiente realista, o agente será inicializado no dia **05/05/2020** com uma carteira contendo **R$ 100.000,00 em caixa**. Nenhum ativo será comprado neste momento inicial.

As decisões possíveis a cada passo serão:

- `0` = manter a posição atual (não realizar transações);
- `1` = comprar, caso haja saldo disponível e nenhuma posição atual naquele ativo;
- `2` = vender, caso haja uma posição vigente no ativo selecionado.

Como não existe uma “resposta certa” a cada decisão (como haveria em classificação supervisionada), a avaliação do desempenho será feita com base em métricas financeiras como:

- **Lucro acumulado**;
- **Sharpe Ratio** (relação entre retorno e risco);
- **Drawdown máximo** (pior sequência de perdas);
- **Volatilidade da carteira**;
- **Turnover de transações** (frequência de compra e venda).

Para validar o desempenho, os resultados do agente serão comparados com **estratégias de benchmark** conhecidas:

- **Buy and Hold**: compra de todos os ativos na data inicial, com manutenção até o final;
- **Variação do IBOVESPA**: como comparador de mercado amplo;
- **Estratégia de cruzamento de médias móveis (SMA-20 vs SMA-50)**: como baseline técnico automatizado.

Além disso, implementaremos a métrica de **signal accuracy** (ou *hit ratio*), que avalia a qualidade tática das decisões:

- Quando o agente recomenda **comprar** e o preço sobe nos dias seguintes;
- Quando recomenda **vender** e o preço cai posteriormente.

Essa taxa será avaliada em três horizontes distintos: **1, 3 e 5 dias**, permitindo identificar se o agente está respondendo bem a oscilações de curtíssimo prazo, microtendências ou movimentos mais sustentados.

---

### 5.5 Divisão Temporal dos Dados

Dada a natureza sequencial e autocorrelacionada dos dados financeiros, a divisão em treino, validação e teste não pode ser feita de forma aleatória, como em outras abordagens.

Adotaremos uma **divisão cronológica** baseada em proporções temporais:

- **Treinamento**: primeiros 70% dos dados, usados para aprendizado da política;
- **Validação**: 15% seguintes, reservados para ajustes finos e tuning de parâmetros;
- **Teste**: últimos 15%, utilizados exclusivamente para simulação e avaliação final.

Essa separação garante que o agente aprenda apenas com o passado, e que sua performance seja testada em dados que ele nunca viu, refletindo com fidelidade o desafio de investir em um mercado real.

---


## 6. Implementação do Agente de RL

* 6.1 Introdução ao Deep Q-Network (DQN)
* 6.2 Estrutura da Rede Neural
* 6.3 Definição do Ambiente de Simulação
* 6.4 Treinamento Inicial do Agente

## 7. Avaliação e Otimização do Agente

* 7.1 Simulação do Desempenho do Agente
* 7.2 Avaliação usando Métricas Financeiras (lucro, Sharpe Ratio)
* 7.3 Análise de Resultados
* 7.4 Ajustes e Otimização do Modelo

## 8. Resultados e Discussão

* 8.1 Desempenho Financeiro
* 8.2 Análise de Decisões do Agente
* 8.3 Insights Gerados

## 9. Conclusões e Potencial de Expansão

* 9.1 Resumo dos Resultados Obtidos
* 9.2 Limitações Encontradas
* 9.3 Sugestões para Melhoria e Expansão (novos ativos, estratégias avançadas)
* 9.4 Considerações Finais e Próximos Passos

## Referências

* Bibliografia
* Fontes de Dados
* Documentação Utilizada
