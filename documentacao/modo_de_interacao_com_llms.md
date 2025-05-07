# 🧭 Diretriz de Interação com LLMs — Modo de Operação por Comandos Interativos

Estou operando dentro do **Ubuntu (WSL2)** e realizando um projeto de desenvolvimento em container Docker, com foco em Reinforcement Learning para finanças.  
Preciso que qualquer LLM **siga as instruções abaixo rigorosamente**:

---

## ✅ Modo de Trabalho: Passos Planejados, Comandos Um a Um

1. **Planejamento Global Primeiro**:  
   Antes de executar qualquer comando, **quero uma explicação resumida de tudo que será feito** para corrigir o problema ou avançar no projeto. Isso inclui:
   - O que está errado ou incompleto
   - Qual será a estrutura corrigida
   - Quais arquivos serão movidos, criados ou editados
   - Qual o objetivo final da ação planejada

2. **Execução Um Comando por Vez**:  
   Após o planejamento, **quero que a LLM envie apenas *UM* comando ou ação por vez** (exemplo: `mv`, `docker compose`, `ls`, etc.).

   A LLM deve **aguardar minha execução e resposta** antes de enviar o próximo comando.  
   **Nunca** deve encadear dois ou mais comandos automaticamente.

3. **Aguardar Minha Decisão**:  
   Cada comando deve ser apresentado com contexto claro e linguagem direta.  
   A LLM só deve continuar após eu escrever algo como “OK”, “feito”, “próximo”, ou similar.

4. **Sem Repetição de Caminhos Mistos**:  
   Todas as instruções devem ser baseadas **apenas no sistema de arquivos do Ubuntu (WSL2)**.  
   Não quero misturar caminhos de Windows (`C:\\rl`, `/mnt/c/...`) em nenhuma etapa.

5. **Resumo Final Opcional**:  
   Após a conclusão de um bloco lógico de tarefas (ex: organização de arquivos, estruturação de diretórios, ajuste de volumes), posso solicitar um **resumo do que foi feito** ou do estado atual do projeto.

---

## 🧮 Estilo de Interação para Notebooks Jupyter

Ao escrever código ou documentação para notebooks, **toda a interação deve ser organizada em células intercaladas**:

- Cada bloco deve conter:
  - **Uma célula em Markdown** com título, descrição e contexto (comentários de alto nível);
  - **Uma célula em Python** com o código correspondente, pronto para ser executado.

- **Nunca junte código Python e texto Markdown em uma única célula**.

- As células devem ser claramente separadas com os títulos apropriados e mantidas em ordem lógica para facilitar a leitura e execução passo a passo no ambiente Jupyter.

Exemplo esperado:

### 📄 Célula Markdown

\`\`\`markdown
## 💾 Salvando os Vetores de Estado Final

Após limpeza e normalização, os dados serão salvos em arquivos CSV individuais para posterior uso pelo ambiente de simulação de RL.
\`\`\`

### 🧮 Célula Python

\`\`\`python
df.to_csv("dados_limpos.csv", index=False)
\`\`\`

---
