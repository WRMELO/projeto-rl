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

Com os indicadores técnicos já calculados e normalizados para os três ativos selecionados, o próximo passo é estruturar o vetor de estado que será utilizado como entrada pelo agente de Reinforcement Learning.

Diferentemente de abordagens supervisionadas tradicionais, em que as variáveis independentes (features) se referem unicamente ao ambiente, em RL é necessário incorporar também o **estado interno do agente** — ou seja, seu portfólio atual e recursos disponíveis.

Dessa forma, o vetor de estado será composto por duas partes:

- Um conjunto de variáveis de mercado, extraídas e normalizadas para cada ativo (`Close_Norm`, `SMA5_Norm`, `Return_Z`, `Volume_Log`, etc.);
- Um conjunto de variáveis internas ao agente, contendo:
  - As **posições atuais em cada ativo**, codificadas como 0 (fora da posição) ou 1 (posição comprada);
  - O **saldo de caixa disponível**, em reais, que será atualizado a cada passo de tempo.

Essa estrutura garante que o agente, ao observar o ambiente, tenha uma visão completa tanto das **condições externas** (preços, tendências e liquidez dos ativos) quanto do seu **estado interno** (disponibilidade financeira e alocação atual), condição fundamental para a simulação realista de decisões de investimento.

A construção do vetor de portfólio será feita utilizando a mesma linha temporal dos vetores de mercado, garantindo a integração completa das informações necessárias à simulação. Essa etapa será incorporada à lógica de inicialização da carteira e simulação de decisões, apresentada no item seguinte.


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

## 6. Implementação do Agente de Reinforcement Learning (RL)

A implementação do agente de RL é o coração do projeto. É neste ponto que o conhecimento teórico acumulado — desde a definição do problema até a modelagem dos dados — se transforma em uma entidade computacional capaz de interagir com o ambiente financeiro de forma autônoma e inteligente.

Nesta seção, detalharemos a construção do agente utilizando a técnica de Deep Q-Network (DQN), abordando sua estrutura de rede, integração com o ambiente de simulação, lógica de aprendizado e preparação para o treinamento inicial.

### 6.1 Introdução ao Deep Q-Network (DQN)

O algoritmo Deep Q-Network (DQN) é uma extensão do método tradicional de Q-Learning, projetado para operar em ambientes com grandes espaços de estados, onde tabelas Q se tornam inviáveis. Ele substitui a tabela por uma rede neural profunda que aproxima a função Q, permitindo generalizações e aprendizados mais eficientes.

Em termos práticos, o DQN busca estimar a função de valor ação-estado:

$$
Q(s, a) \approx \mathbb{E}[R_t \mid s_t = s, a_t = a]
$$

Onde:

- $s$ representa o **estado atual** do ambiente (informações de mercado e da carteira do agente);
- $a$ representa a **ação executada** pelo agente (compra, venda ou manutenção por ativo);
- $R_t$ é o **retorno acumulado futuro**, resultado da execução da ação $a$ no estado $s$;
- $s'$ é o **próximo estado** do ambiente, resultante da ação tomada;
- `done` é um **indicador booleano** que sinaliza se o episódio terminou (por exemplo, quando se atinge o final da série histórica).

A função Q é atualizada com base na Equação de Bellman:

$$
\text{Loss} = \left[ Q(s_t, a_t) - \left( r_t + \gamma \cdot \max_a Q(s_{t+1}, a) \right) \right]^2
$$

Para garantir estabilidade no aprendizado, o DQN adota dois mecanismos fundamentais:

- **Replay Buffer**: armazena transições $(s, a, r, s', done)$, que são amostradas aleatoriamente para evitar correlações temporais;
- **Target Network**: uma cópia da rede Q usada apenas para calcular os alvos de treinamento, atualizada periodicamente com os pesos da rede principal.

---

### 6.2 Estrutura da Rede Neural

A arquitetura adotada para a rede DQN neste projeto foi desenhada para balancear expressividade e eficiência computacional. Como entrada, a rede recebe um vetor com:

- Indicadores técnicos dos ativos (normalizados);
- Posições atuais em cada ativo;
- Saldo de caixa disponível.

A estrutura da rede é composta por:

- **Camada de Entrada**: dimensão igual ao vetor de estado;
- **Camadas Ocultas**:
  - `Dense(128)` com função de ativação ReLU;
  - `Dense(64)` com função de ativação ReLU;
- **Camada de Saída**:
  - `Dense(n_actions)` com ativação linear, retornando os valores estimados $Q(s, a)$ para cada ação.

A rede principal é treinada com erro quadrático médio (MSE) entre os valores previstos e os alvos calculados com a rede-alvo, conforme a equação de Bellman.

---

### 6.3 Definição do Ambiente de Simulação

O ambiente de simulação é responsável por modelar o mercado financeiro e a carteira do agente. Ele foi estruturado para seguir a interface `gym.Env` da OpenAI, com os seguintes métodos principais:

- `reset()`: reinicializa o ambiente com R$100.000 em caixa (normalizado como 1.0), posições zeradas e retorna o estado inicial;
- `step(action)`: executa a ação informada, atualiza as posições e o caixa, calcula a recompensa e retorna:
  - o novo estado;
  - a recompensa recebida;
  - o sinal de término do episódio (`done`);
  - informações adicionais para análise (`info`).

As ações são vetores discretos por ativo:

- `0` = manter;
- `1` = comprar (caso haja caixa);
- `2` = vender (caso haja posição).

A recompensa é baseada na variação do patrimônio líquido (caixa + posições), penalizada por custos de transação e prejuízos. O episódio termina ao alcançar o fim da série histórica.

---

### 6.4 Treinamento Inicial do Agente

Com o ambiente e a rede definidos, o treinamento do agente ocorre por meio da interação repetida com o ambiente, ajustando os pesos da rede para melhorar suas decisões futuras. O processo segue os seguintes passos:

1. **Inicialização do Replay Buffer** com capacidade para 10.000 transições;
2. **Execução de múltiplos episódios**, onde:
   - O agente observa o estado atual;
   - Escolhe uma ação pela política $\epsilon$-greedy;
   - Executa a ação, obtém a recompensa e o novo estado;
   - Armazena a transição no buffer;
   - Amostra minibatches aleatórios para treinar a rede;
3. **Atualização periódica da target network** a cada $C$ iterações;
4. **Monitoramento por validação** para evitar overfitting e acompanhar métricas de performance.

O valor de $\epsilon$ inicia alto (ex: 1.0) e decresce progressivamente até um valor mínimo (ex: 0.1), permitindo ao agente alternar entre **exploração** (tomar decisões aleatórias para descobrir novas possibilidades) e **exploração do conhecimento aprendido** (agir com base na política já treinada).

O treinamento será conduzido por 200 episódios, com registro das seguintes métricas:

- Lucro acumulado;
- Sharpe Ratio;
- Drawdown máximo;
- Acurácia dos sinais (hit ratio em 1, 3 e 5 dias).

Os melhores modelos serão salvos para posterior avaliação em ambiente de teste.


5. **Um comentário importante:** Por que Utilizar Replay Buffer com Dados Sequenciais?

Embora os dados de mercado sejam naturalmente sequenciais — ou seja, cada dia depende do anterior — o uso do **Replay Buffer** no treinamento do agente DQN é **fundamental para garantir a estabilidade e eficiência do aprendizado**.

### O Problema da Correlação Temporal

Se treinássemos o agente diretamente com sequências cronológicas (ex: dias consecutivos: t, t+1, t+2...), o modelo sofreria com:

- **Overfitting local**: aprendizado baseado apenas em padrões recentes, sem capacidade de generalização;
- **Instabilidade na otimização**: mudanças abruptas de gradiente, pois os dados são altamente autocorrelacionados;
- **Baixa diversidade nos exemplos**: o agente teria poucas situações distintas para comparar e aprender.

###  A Solução: Replay Buffer

O **Replay Buffer** resolve esse problema ao permitir que a rede aprenda com **mini-batches aleatórios de experiências passadas**, quebrando a sequência e promovendo generalização.

- As experiências são **coletadas em ordem temporal real**, respeitando a sequência do ambiente;
- Mas são **usadas de forma embaralhada durante o treino**, o que reduz a correlação entre amostras.

### Analogia Real

Imagine um trader experiente que, ao tomar uma decisão hoje, **lembra de diversos eventos anteriores no mercado**, mesmo que tenham ocorrido meses atrás.  
O Replay Buffer simula essa capacidade: o agente **aprende com base em uma memória diversificada de situações**, não apenas no que acabou de acontecer.

### Benefícios

- Reduz variância nos gradientes e melhora a estabilidade do DQN;
- Permite revisitar experiências raras ou valiosas no treinamento;
- Promove maior robustez da política aprendida frente a novos dados.

Portanto, mesmo que o ambiente seja sequencial por natureza, o uso do Replay Buffer **é não só compatível, mas essencial** para um aprendizado mais estável e eficaz em Reinforcement Learning.

---


## 7. Avaliação e Otimização do Agente

### 7.1 Simulação do Desempenho do Agente

Com o agente treinado ao longo de 300 episódios utilizando a técnica de Deep Q-Network (DQN), procedeu-se à avaliação de sua política em dados até então não vistos — mais especificamente, o conjunto de validação correspondente aos 15% centrais da série temporal total. Esta etapa foi conduzida com ε = 0.0, ou seja, **o agente atuou exclusivamente com base na política aprendida, sem qualquer exploração aleatória**.

O ambiente foi reinicializado com os dados de validação, mantendo as mesmas condições iniciais da carteira: **R$ 100.000,00 em caixa e nenhuma posição comprada**. A simulação percorreu diariamente o vetor de validação, permitindo ao agente decidir entre manter, comprar ou vender em cada um dos três ativos: VALE3, PETR4 e BRFS3.

Durante essa simulação, a cada passo, registrou-se o valor total da carteira (caixa + valor das posições a preço de mercado), compondo a série patrimonial do agente. Essa curva de valor acumulado foi usada como base para o cálculo das métricas financeiras descritas a seguir.

---

### 7.2 Avaliação usando Métricas Financeiras

A avaliação da performance do agente no conjunto de validação foi conduzida com base nas três métricas previamente definidas na seção 2.4:

- **Lucro Total**
- **Índice de Sharpe**
- **Máximo Drawdown**

Os resultados observados foram os seguintes:

- 📈 **Lucro Total:** R$ 6,78  
- 📊 **Índice de Sharpe:** 0,06  
- 📉 **Máximo Drawdown:** R$ 5,33 (0,01%)

**Lucro Total:**  
Representa a diferença absoluta entre o valor final e inicial da carteira ao longo da simulação (~183 dias úteis). Com um valor inicial de R$ 100.000,00, o agente terminou o período com R$ 100.006,78. Esse resultado indica que o agente **preservou o capital**, mas **não gerou retorno relevante** em termos absolutos (0,0068% no período completo).

**Índice de Sharpe:**  
Calculado como a média dos retornos diários da carteira dividida pelo seu desvio padrão, considerando taxa livre de risco nula. O valor de 0,06 evidencia que o **retorno médio foi muito próximo de zero em relação ao risco** assumido, o que caracteriza uma política neutra em termos de performance ajustada à volatilidade.

**Máximo Drawdown:**  
Foi medido como a maior perda acumulada entre um pico e o fundo subsequente da curva patrimonial. O valor observado de R$ 5,33 — equivalente a 0,01% — sugere que **o agente operou de maneira extremamente conservadora**, evitando grandes flutuações negativas, mas também sem assumir posições que maximizassem retorno.

---

### 7.3 Análise de Resultados

A análise combinada das métricas e da curva de evolução do patrimônio revela que, apesar de estruturalmente estável, **a política aprendida pelo agente se mostrou excessivamente passiva**. Em termos práticos, o agente adotou um comportamento semelhante ao de manter caixa, realizando poucas operações com efeito marginal sobre o valor da carteira.

Isso pode ter sido reflexo de vários fatores interligados:

1. **Função de recompensa muito conservadora**, que premia variações positivas mas não incentiva suficientemente a tomada de risco controlado;
2. **Espaço de ação restrito**, com apenas uma unidade transacionável por ativo e sem gestão de quantidade;
3. **Ausência de penalizações por inatividade**, o que permite que o agente "aprenda" que não fazer nada pode ser suficiente para evitar prejuízo — mesmo que à custa de oportunidades de ganho;
4. **Tempo de treinamento ainda limitado**, o que pode restringir a capacidade de generalização e descoberta de padrões mais sofisticados.

Apesar disso, o resultado não é desprezível: o agente conseguiu preservar o capital de forma consistente, com drawdowns mínimos e sem colapsos de performance. Isso demonstra que a estrutura geral do ambiente de simulação, a arquitetura do DQN e a estratégia de treinamento adotadas são viáveis — embora ainda insuficientes para gerar uma política financeiramente interessante.

Portanto, conclui-se que o agente está operando dentro de um **padrão técnico aceitável**, mas requer otimizações específicas em seus mecanismos de decisão e aprendizado para evoluir de uma política neutra para uma política efetivamente lucrativa.

As direções para esse aprimoramento serão discutidas na seção seguinte (7.4).

### 7.4 Ajustes e Otimização do Modelo

Diante dos resultados observados na seção anterior — especialmente a neutralidade da política aprendida pelo agente — torna-se evidente a necessidade de realizar intervenções estruturadas com o objetivo de aprimorar o desempenho do modelo de Reinforcement Learning. A seguir, são propostas otimizações agrupadas em três dimensões: ambiente, arquitetura do agente e estratégia de aprendizado.

---

**1. Ajustes no Ambiente de Simulação**

* **Introdução de penalidade por inatividade:**  
Atualmente, o agente pode permanecer inerte ao longo de todo o episódio e ainda preservar o capital, o que não o incentiva a buscar oportunidades de ganho. Recomenda-se introduzir uma **penalização leve por manutenção prolongada de posição nula**, estimulando o agente a tomar decisões estratégicas ao invés de se omitir.

* **Inclusão de custos de transação variáveis:**  
Embora o modelo suporte penalização por operação, na versão atual ela ainda não foi implementada. A introdução de custos de transação proporcionais ao valor da operação pode simular o impacto real de corretagem, spread e slippage, incentivando o agente a otimizar o volume e o momento das operações.

* **Aumento do espaço de ação com múltiplas unidades:**  
Atualmente o agente só pode comprar ou vender uma unidade de cada ativo. A ampliação para múltiplos níveis (ex.: comprar 1, 2 ou 3 unidades) ou a introdução de proporções pode torná-lo mais realista e aumentar a complexidade estratégica da política aprendida.

---

**2. Otimização da Rede Neural e da Função de Recompensa**

* **Refinamento da função de recompensa:**  
A recompensa atual mede a variação absoluta do valor da carteira. Sugere-se testar abordagens que incorporem métricas ajustadas ao risco, como **retorno percentual** ou até **incrementos no Sharpe Ratio**, como proxy da qualidade da decisão, promovendo maior alinhamento com o objetivo financeiro final.

* **Adição de dropout ou batch normalization:**  
Embora a rede atual seja estável, sua capacidade de generalização pode ser aumentada com a inclusão de mecanismos como **dropout** (para evitar overfitting) ou **batch normalization** (para estabilizar os gradientes). Isso pode permitir o uso de redes ligeiramente mais profundas sem comprometer a estabilidade.

* **Adoção de loss functions alternativas:**  
O uso de **Huber Loss** em lugar do MSE pode tornar o aprendizado mais robusto a outliers e flutuações abruptas nos retornos diários.

---

**3. Estratégias de Aprendizado e Treinamento**

* **Aumento do número de episódios de treinamento:**  
O agente foi treinado por 300 episódios, o que é adequado para um primeiro experimento, mas ainda modesto para uma tarefa com grande espaço de estados e ações. Recomenda-se expandir para 1.000 episódios ou mais, com monitoramento contínuo das métricas de desempenho.

* **Política de decaimento de ε mais lenta e customizada:**  
O decaimento exponencial de ε de 1.0 para 0.1 ocorreu de forma relativamente rápida, limitando a fase de exploração. Uma abordagem de decaimento **linear ou baseada em patamares** (ex.: manter ε alto nas primeiras 100 interações) pode favorecer uma política mais informada antes da explotação.

* **Treinamento contínuo com avaliação intercalada:**  
Dividir o treinamento em ciclos e intercalar rodadas de avaliação com reinício do ambiente permite identificar com mais clareza pontos de inflexão no aprendizado e evita que o modelo “esqueça” comportamentos úteis por overtraining.

---

Com esses ajustes, espera-se que o agente evolua de uma postura neutra e conservadora para uma política mais sofisticada, responsiva ao mercado e efetivamente lucrativa. Essas intervenções também contribuem para transformar o MVP atual em um sistema base mais realista, capaz de incorporar futuras expansões — como múltiplos ativos, indicadores técnicos avançados ou o uso de técnicas Actor-Critic mais robustas.

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
