#!/bin/bash
# =============================================================================
# Runpod 環境セットアップスクリプト
# AMD MI300X 192GB 用
# Template: RunPod PyTorch 2.9.1 ROCm 7.2 (Python 3.12)
# =============================================================================
set -e

# 色の定義
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}==========================================${NC}"
echo -e "${BLUE} Qwen3.5-27B Reasoning Dataset Generator${NC}"
echo -e "${BLUE} Setup Script (AMD MI300X 192GB)${NC}"
echo -e "${BLUE} Backend: Llama.cpp (ROCm 7.x)${NC}"
echo -e "${BLUE}==========================================${NC}"
echo ""

# --- システム更新 ---
echo -e "${GREEN}[1/5] システムパッケージの更新...${NC}"
apt-get update -qq && apt-get install -y -qq git wget curl htop tmux ninja-build cmake > /dev/null 2>&1 \
    || echo -e "${YELLOW}  一部のパッケージの更新をスキップしました。${NC}" || true

# --- Python環境 ---
echo -e "\n${GREEN}[2/5] Python仮想環境のセットアップ...${NC}"
# --system-site-packages: テンプレートの PyTorch 2.9.1 ROCm 7.2 を継承
python3 -m venv --system-site-packages /workspace/venv
source /workspace/venv/bin/activate
echo "  Python:  $(python3 --version)"
echo "  PyTorch: $(python3 -c 'import torch; print(torch.__version__)' 2>/dev/null || echo '未検出 (ビルド後に確認)')"

# --- 依存パッケージ ---
echo -e "\n${GREEN}[3/5] APIクライアント用ライブラリとllama.cppのビルド...${NC}"
python3 -m pip install --upgrade pip --quiet
python3 -m pip install uv --quiet

# クライアント用軽量ライブラリ
echo "  依存関係クライアントツールをインストール中..."
uv pip install pyyaml tqdm datasets huggingface_hub openai aiohttp --quiet

# llama.cpp インストールとMI300X最適化ビルド
echo "  llama.cpp リポジトリを取得中..."
if [ ! -d "/workspace/llama.cpp" ]; then
    git clone https://github.com/ggerganov/llama.cpp.git /workspace/llama.cpp
fi
cd /workspace/llama.cpp
git pull origin master

echo "  llama.cpp を MI300X (gfx942) ネイティブ ROCm バックエンドでコンパイル中..."
# AMD MI300X 向けの最適化 CMake フラグ (-G Ninja で超高速並列ビルド)
cmake -B build -G Ninja -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx942
cmake --build build --config Release -j $(nproc)

echo ""
echo -n "  llama-server 実行ファイル作成確認: " 
if [ -f "/workspace/llama.cpp/build/bin/llama-server" ]; then
    echo -e "${GREEN}✓ 成功${NC}"
else
    echo -e "${RED}❌ 失敗${NC}"
    exit 1
fi
echo ""

# --- プロジェクトのセットアップ ---
echo -e "\n${GREEN}[4/5] プロジェクトディレクトリのセットアップ...${NC}"
PROJECT_DIR="/workspace/qwen3.5-27b-reasoning-dataset"

mkdir -p "$PROJECT_DIR/output/raw"
mkdir -p "$PROJECT_DIR/output/filtered"
mkdir -p "$PROJECT_DIR/output/final"
mkdir -p "$PROJECT_DIR/output/checkpoints"
mkdir -p "$PROJECT_DIR/output/rejected"
mkdir -p "$PROJECT_DIR/logs"
echo "  出力ディレクトリ作成完了: $PROJECT_DIR"

# --- モデルの事前ダウンロード (Unsloth GGUF) ---
echo -e "\n${GREEN}[5/5] Qwen3.5-27B (Unsloth Q6_K_XL) ダウンロード...${NC}"
MODEL_DIR="/workspace/models/Qwen3.5-27B-GGUF"
mkdir -p "$MODEL_DIR"

python3 -c "
from huggingface_hub import hf_hub_download
import os

repo_id = 'unsloth/Qwen3.5-27B-GGUF'
filename = 'Qwen3.5-27B-UD-Q6_K_XL.gguf'
local_dir = '/workspace/models/Qwen3.5-27B-GGUF'
model_path = os.path.join(local_dir, filename)

if not os.path.exists(model_path):
    print(f'{filename} (約25.7GB) をダウンロード中...')
    hf_hub_download(repo_id=repo_id, filename=filename, local_dir=local_dir)
    print('ダウンロード完了!')
else:
    print('モデルは既に存在します。ダウロードをスキップします。')
"

# --- GPU確認 ---
echo -e "\n${GREEN}[GPU状態の確認]...${NC}"
rocm-smi --showid --showproductname --showmeminfo vram 2>/dev/null \
    || nvidia-smi --query-gpu=index,name,memory.total,memory.free,driver_version --format=csv,noheader
echo ""

echo ""
echo -e "${BLUE}==========================================${NC}"
echo -e "${GREEN} ✓ セットアップ完了!${NC}"
echo -e "${BLUE}==========================================${NC}"
echo ""
echo -e "GPU構成: ${YELLOW}AMD MI300X 192GB x1${NC}"
echo -e "環境:    ${YELLOW}Llama.cpp API Server + ROCm (gfx942)${NC}"
echo -e "モデル:  ${YELLOW}Qwen3.5-27B (Unsloth Q6_K_XL GGUF, ~25.7GB)${NC}"
echo ""
echo -e "${GREEN}次のステップ:${NC}"
echo "  生成プロセスの開始:"
echo -e "     ${YELLOW}source /workspace/venv/bin/activate && bash runpod/run_pipeline.sh${NC}"
echo ""
