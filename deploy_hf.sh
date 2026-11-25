#!/usr/bin/env bash
set -e

if [ -z "$1" ]; then
  echo "=========================================================="
  echo "FurnForge AI — Hugging Face Space One-Command Deployer"
  echo "=========================================================="
  echo "Usage: ./deploy_hf.sh <YOUR_HF_USERNAME> [SPACE_NAME]"
  echo "Example: ./deploy_hf.sh whysooraj furnforge-backend"
  echo ""
  echo "Note: If you haven't created the Space yet on Hugging Face:"
  echo "1. Go to https://huggingface.co/new-space"
  echo "2. Space Name: furnforge-backend"
  echo "3. License: MIT"
  echo "4. Space SDK: Docker (Blank)"
  echo "5. Space hardware: Free (2 vCPU, 16GB RAM)"
  echo "6. Visibility: Public"
  echo "=========================================================="
  exit 1
fi

HF_USER="$1"
SPACE_NAME="${2:-furnforge-backend}"

echo "🚀 Packaging FurnForge AI Backend for Hugging Face Spaces ($HF_USER/$SPACE_NAME)..."

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/hf_space"

if [ ! -d ".git" ]; then
  git init -b main
fi

git config user.name "$HF_USER"
git config user.email "$HF_USER@users.noreply.huggingface.co"

git add .
git commit -m "deploy: FurnForge AI backend to Hugging Face Spaces" || true

REMOTE_URL="https://huggingface.co/spaces/$HF_USER/$SPACE_NAME"
git remote remove hf 2>/dev/null || true
git remote add hf "$REMOTE_URL"

echo "📦 Pushing Docker Space to $REMOTE_URL..."
echo "👉 When prompted for username: enter '$HF_USER'"
echo "👉 When prompted for password: paste your Hugging Face Access Token (Write permissions) from https://huggingface.co/settings/tokens"

git push -u hf main --force

echo ""
echo "=========================================================="
echo "✨ Deployment initiated successfully!"
echo "Your permanent cloud endpoint will be:"
echo "👉 https://$HF_USER-$SPACE_NAME.hf.space"
echo "=========================================================="
