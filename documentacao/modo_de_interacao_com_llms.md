# 🧭 Diretriz de Interação com LLMs — Modo de Operação por Comandos Interativos

Este documento define o protocolo de comunicação para o projeto de RL em containers Docker (Ubuntu/WSL2). Todas as instruções devem ser seguidas rigorosamente.

---

## 1. Modo de Trabalho Geral

1. **Planejamento Global**

   * Antes de qualquer comando, apresentar um resumo com:

     * O que está incorreto ou pendente
     * A estrutura ou lógica corrigida
     * Quais arquivos/células serão editados
     * Objetivo final da ação
2. **Uma Ação por Comando**

   * Enviar apenas *um* comando, célula ou trecho de código por vez.
   * Aguardar minha confirmação ("OK", "feito", "próximo") antes de prosseguir.
3. **Referência de Células**

   * Para notebooks, use sempre o **número da célula** (ex.: **Célula \[s5]**, **Célula \[25]**).
   * Inclua o texto completo da célula quando for substituir integralmente.
4. **Caminho de Arquivos**

   * Use apenas caminhos Linux/Ubuntu (WSL2), sem referências a `C:\` ou `/mnt/c`.

---

## 2. Tratamento de Erros e Correções

Para cada erro apresentado:

1. **Causa do erro**

   * Análise técnica do motivo da falha.
2. **O que será feito**

   * Descrição da correção aplicada.
3. **Por que será feito**

   * Justificativa técnica da mudança.
4. **Código de correção**

   * Bloco Python completo, pronto para copiar e colar.
5. **Markdown ajustado** (quando aplicável)

   * Atualização do comentário explicativo em Markdown, se necessário.

---

## 3. Formatação para Notebooks Jupyter

* Toda inserção de código requer duas células separadas:

  * **Markdown**: título, contexto e comentários de alto nível.
  * **Python**: código executável sem comentários de formatação.
* **Markdown sempre antes do código**.
* Não misturar Python e Markdown na mesma célula.

---

## 4. Comunicação e Fluxo

* Mantenha cada etapa concisa e objetiva.
* Avise quando a conversa estiver se aproximando de \~100 mensagens ou \~6 000 tokens para sugerir novo chat.
* Posso solicitar um **Resumo Final** do progresso a qualquer momento.

---

## 5. Exemplo de Uso

> **Célula \[s5]**
>
> 1. **Causa**: `output_dim` não definido.
> 2. **Correção**: adicionar definição de `output_dim` e `device`.
> 3. **Justificativa**: rede DQN precisa de 27 saídas e dispositivo.
> 4. **Código completo**:
>
> ```python
> # Definições
> output_dim = env.action_space.n
> device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
>
> # Instanciação
> q_net    = DQN(input_dim=input_dim, output_dim=output_dim).to(device)
> q_target = DQN(input_dim=input_dim, output_dim=output_dim).to(device)
> q_target.load_state_dict(q_net.state_dict())
> ```
>
> 5. **Markdown atualizado**:
>
> ```markdown
> ### Definição de output_dim e device
> - `output_dim` = env.action_space.n (27 ações)
> - `device` = "cuda" se disponível, senão "cpu"
> ```
