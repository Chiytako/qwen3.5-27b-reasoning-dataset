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
echo -e "${BLUE} Backend: SGLang (ROCm 7.2 + PyTorch 2.9.1)${NC}"
echo -e "${BLUE}==========================================${NC}"
echo ""

# --- システム更新 ---
echo -e "${GREEN}[1/5] システムパッケージの更新...${NC}"
apt-get update -qq && apt-get install -y -qq git wget curl htop tmux > /dev/null 2>&1 \
    || echo -e "${YELLOW}  一部のパッケージの更新をスキップしました。${NC}" || true

# --- Python環境 ---
echo -e "\n${GREEN}[2/5] Python仮想環境のセットアップ...${NC}"
# --system-site-packages: テンプレートの PyTorch 2.9.1 ROCm 7.2 を継承
python3 -m venv --system-site-packages /workspace/venv
source /workspace/venv/bin/activate
echo "  Python:  $(python3 --version)"
echo "  PyTorch: $(python3 -c 'import torch; print(torch.__version__)' 2>/dev/null || echo '未検出 (ビルド後に確認)')"

# --- 依存パッケージ + SGLang インストール ---
echo -e "\n${GREEN}[3/5] SGLang (ROCm用) と依存ライブラリのインストール...${NC}"
python3 -m pip install --upgrade pip --quiet
python3 -m pip install uv --quiet

# 軽量ライブラリ
echo "  依存関係ツールをインストール中..."
uv pip install transformers>=4.48.0 pyyaml tqdm datasets huggingface_hub openai aiohttp --quiet

# SGLang インストール (Qwen3.5対応の最新mainブランチから取得)
echo "  SGLang (FlashInfer対応, mainブランチ) をインストール中..."
uv pip install "git+https://github.com/sgl-project/sglang.git#subdirectory=python&egg=sglang[all]" --quiet

echo ""
echo -n "  SGLang バージョン:   " && python3 -c 'import sglang; print(sglang.__version__)' 2>/dev/null || echo "Unknown"
echo -n "  PyTorch バージョン: " && python3 -c 'import torch; print(torch.__version__)'
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

# --- モデルの事前ダウンロード ---
echo -e "\n${GREEN}[5/5] Qwen3.5-27B フルモデル事前ダウンロード...${NC}"
python3 -c "
from huggingface_hub import snapshot_download
print('モデルをダウンロード中... (FP16/BF16版は約54GB、時間がかかります。進捗バーが表示されます)')
snapshot_download(
    repo_id='Qwen/Qwen3.5-27B',
    local_dir='/workspace/models/Qwen3.5-27B',
    ignore_patterns=['*.md', '*.txt', 'LICENSE*'],
)
print('ダウンロード完了!')
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
echo -e "環境:    ${YELLOW}SGLang API Server + ROCm 7.2${NC}"
echo -e "モデル:  ${YELLOW}Qwen3.5-27B フルパラメータ (~54GB/GPU)${NC}"
echo ""
echo -e "${GREEN}次のステップ:${NC}"
echo "  生成プロセスの開始:"
echo -e "     ${YELLOW}source /workspace/venv/bin/activate && bash runpod/run_pipeline.sh${NC}"
echo ""
