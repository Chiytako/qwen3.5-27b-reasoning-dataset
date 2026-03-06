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
python3 -m pip install --upgrade pip
# 【重要】依存解決によって40分フリーズするpipの弱点を解決するため、Rust製の超高速パッケージマネージャ「uv」を導入
python3 -m pip install uv

# 軽量ライブラリ群のインストール (uvを使うと数秒で終わります)
uv pip install transformers>=4.48.0 pyyaml tqdm datasets huggingface_hub

# vLLM (ROCm向けソースビルド)
# 背景:
#   - PyPI の vllm パッケージは CUDA 専用ビルドであり ROCm では動作しない
#   - PyPI の vllm-rocm は 0.6.3 止まりで Qwen3.5 (Qwen3_5ForConditionalGeneration) 未対応
#   - wheels.vllm.ai の ROCm ホイールは ROCm 7.0 + Python 3.12 のみ対象
#   - Qwen3.5 対応は vLLM v0.17.0 (未リリース) / nightly ビルドで提供
#   → ROCm 6.x + Python 3.10 環境では ソースビルドが唯一の確実な手段
echo -e "\n${YELLOW}vLLM を ROCm 向けにソースビルドします (30〜60分かかります)...${NC}"
# ROCm 対応 PyTorch を先にインストール (pip install vllm の依存として上書きされないよう先行インストール)
uv pip install "torch==2.4.0" "torchvision==0.19.0" "torchaudio==2.4.0" \
    --index-url https://download.pytorch.org/whl/rocm6.1
# vLLM ソース取得 (既にクローン済みの場合は pull のみ)
VLLM_SRC="/workspace/vllm-src"
if [ -d "$VLLM_SRC/.git" ]; then
    git -C "$VLLM_SRC" pull --ff-only
else
    git clone https://github.com/vllm-project/vllm.git "$VLLM_SRC"
fi
# ビルドに必要な依存パッケージを先行インストール
# setuptools 65.x は pyproject.toml の PEP 639 ライセンス形式を解釈できないためアップグレード必須
pip install --upgrade setuptools setuptools_scm wheel ninja cmake
# ROCm ターゲットでビルド・インストール
cd "$VLLM_SRC"
VLLM_TARGET_DEVICE=rocm pip install -e . --no-build-isolation
cd -

echo -n "  vLLMバージョン: " && python3 -c 'import vllm; print(vllm.__version__)'
echo -n "  PyTorchバージョン: " && python3 -c 'import torch; print(torch.__version__)'

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
