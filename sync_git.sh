#!/bin/bash

# 🌐 Script universal de sincronização Git entre Windows, WSL2 e Container
# Uso: ./sync_git.sh "Mensagem do commit entre aspas"

# 1. Verifica se está em um repositório Git
if [ ! -d .git ]; then
  echo "❌ Este diretório não é um repositório Git."
  exit 1
fi

# 2. Define a mensagem de commit
if [ -z "$1" ]; then
  echo "⚠️  Nenhuma mensagem de commit fornecida. Usando padrão."
  COMMIT_MSG="Atualização automática"
else
  COMMIT_MSG="$1"
fi

# 3. Executa o fluxo de commit + push
echo "🔄 Adicionando arquivos modificados..."
git add .

echo "✅ Commitando: $COMMIT_MSG"
git commit -m "$COMMIT_MSG"

echo "🚀 Enviando para GitHub..."
git pull --rebase
git push origin main
