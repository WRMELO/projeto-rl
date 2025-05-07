#!/bin/bash

echo "🔄 Adicionando arquivos modificados..."
git add .

echo "✅ Commitando alterações..."
git commit -m "Atualização automática" || echo "⚠️ Nenhuma alteração para commit."

echo "🔐 Configurando autenticação segura..."
git remote set-url origin https://$GITHUB_TOKEN@github.com/WRMELO/projeto-rl.git

echo "🚀 Enviando para GitHub..."
git push
