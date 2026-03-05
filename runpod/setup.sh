#!/bin/bash
# =============================================================================
# Runpod 環境セットアップスクリプト
# AMD MI300X 192GB 用
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
echo -e "${BLUE}==========================================${NC}"
echo ""

# --- システム更新 ---
echo -e "${GREEN}[1/5] システムパッケージの更新...${NC}"
apt-get update -qq && apt-get install -y -qq git wget curl htop tmux > /dev/null 2>&1 || echo -e "${YELLOW}  一部のパッケージの更新をスキップしました。${NC}" || true

# --- Python環境 ---
echo -e "\n${GREEN}[2/5] Python仮想環境のセットアップ...${NC}"
python3 -m venv --system-site-packages /workspace/venv
source /workspace/venv/bin/activate

# --- 依存パッケージのインストール ---
echo -e "\n${GREEN}[3/5] 依存パッケージのインストール (進捗は以下に表示されます)...${NC}"
pip install --upgrade pip

# 軽量ライブラリ群のインストール
pip install --no-cache-dir --prefer-binary \
    transformers>=4.48.0 \
    pyyaml \
    tqdm \
    datasets \
    huggingface_hub

# vLLM (ROCm対応版) のインストール
if ! python3 -c "import vllm" &> /dev/null; then
    echo -e "\n${YELLOW}システムにvLLMが見つかりません。公式のvllm-rocmホイールをインストールします...${NC}"
    # vllm-rocmとその依存関係をインストールしますが、
    # 巨大な「CUDA版PyTorch」が誤ってダウンロードされて40分かかるのを防ぐため、
    # 依存解決先としてROCm用のレジストリを指定します。これによりシステムの最適化済みTorchが維持されます。
    pip install vllm-rocm --extra-index-url https://download.pytorch.org/whl/rocm6.0
fi

echo -n "  vLLMバージョン: " && python -c 'import vllm; print(vllm.__version__)'
echo -n "  PyTorchバージョン: " && python -c 'import torch; print(torch.__version__)'

# --- プロジェクトのセットアップ ---
echo -e "\n${GREEN}[4/5] プロジェクトディレクトリのセットアップ...${NC}"
PROJECT_DIR="/workspace/qwen3.5-27b-reasoning-dataset"

# 出力ディレクトリ作成
mkdir -p "$PROJECT_DIR/output/raw"
mkdir -p "$PROJECT_DIR/output/filtered"
mkdir -p "$PROJECT_DIR/output/final"
mkdir -p "$PROJECT_DIR/output/checkpoints"
mkdir -p "$PROJECT_DIR/output/rejected"
mkdir -p "$PROJECT_DIR/logs"

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
rocm-smi --showid --showproductname --showmeminfo vram 2>/dev/null || nvidia-smi --query-gpu=index,name,memory.total,memory.free,driver_version --format=csv,noheader
echo ""

echo ""
echo -e "${BLUE}==========================================${NC}"
echo -e "${GREEN} ✓ セットアップ完了!${NC}"
echo -e "${BLUE}==========================================${NC}"
echo ""
echo -e "GPU構成: ${YELLOW}AMD MI300X 192GB x1${NC}"
echo -e "モデル: ${YELLOW}Qwen3.5-27B フルパラメータ (~54GB/GPU)${NC}"
echo -e "実行方式: ${YELLOW}venv仮想環境 + vllm-rocm公式パッケージ${NC}"
echo ""
echo -e "${GREEN}次のステップ:${NC}"
echo "  生成プロセスの開始:"
echo -e "     ${YELLOW}bash runpod/run_pipeline.sh${NC}"
echo ""
