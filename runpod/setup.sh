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
echo -e "${BLUE} ROCm 7.2 + PyTorch 2.9.1 Template${NC}"
echo -e "${BLUE}==========================================${NC}"
echo ""

# --- システム更新 ---
echo -e "${GREEN}[1/5] システムパッケージの更新...${NC}"
apt-get update -qq && apt-get install -y -qq git wget curl htop tmux > /dev/null 2>&1 \
    || echo -e "${YELLOW}  一部のパッケージの更新をスキップしました。${NC}" || true

# --- Python環境 ---
echo -e "\n${GREEN}[2/5] Python仮想環境のセットアップ...${NC}"
# --system-site-packages: テンプレートにプリインストール済みの PyTorch 2.9.1 ROCm 7.2 を継承
# → vLLM ソースビルド時に PyTorch を再インストールする必要がなくなる
python3 -m venv --system-site-packages /workspace/venv
source /workspace/venv/bin/activate
echo "  Python: $(python3 --version)"
echo "  PyTorch (system): $(python3 -c 'import torch; print(torch.__version__)' 2>/dev/null || echo '未検出')"

# --- 依存パッケージ + vLLM ソースビルド ---
echo -e "\n${GREEN}[3/5] vLLM ソースビルド (Qwen3.5 対応版)...${NC}"
python3 -m pip install --upgrade pip --quiet
python3 -m pip install uv --quiet

# 軽量ライブラリ
echo "  軽量ライブラリをインストール中..."
uv pip install transformers>=4.48.0 pyyaml tqdm datasets huggingface_hub --quiet

# ビルドツール
echo "  ビルドツールをインストール中..."
pip install --upgrade setuptools setuptools_scm wheel ninja --quiet

# vLLM ソース取得
# 背景:
#   - wheels.vllm.ai の最新安定版は 0.16.0+rocm700 で Qwen3.5 未対応
#   - Qwen3.5 (Qwen3_5ForConditionalGeneration) 対応は vLLM 0.17.0 以降
#   - ROCm 用 nightly wheel は存在しないためソースビルドが唯一の手段
#   - BUILD_TRITON=0: Triton コンパイルを無効化 (MI300X では CK の方が高速なため問題なし)
VLLM_SRC="/workspace/vllm-src"
if [ -d "$VLLM_SRC/.git" ]; then
    echo "  vLLM ソースを更新中..."
    git -C "$VLLM_SRC" pull --ff-only
else
    echo "  vLLM ソースをクローン中..."
    git clone --depth=1 https://github.com/vllm-project/vllm.git "$VLLM_SRC"
fi

echo -e "\n${YELLOW}vLLM を ROCm 向けにビルド中 (30〜60分かかります)...${NC}"
echo "  ログ: /tmp/vllm_build.log"
echo "  進捗確認: tail -f /tmp/vllm_build.log | grep -E '^\[|error:'"
echo ""

cd "$VLLM_SRC"
BUILD_TRITON=0 \
VLLM_USE_TRITON_FLASH_ATTN=0 \
PYTORCH_ROCM_ARCH=gfx942 \
VLLM_TARGET_DEVICE=rocm \
MAX_JOBS=4 \
    pip install -e . --no-build-isolation 2>&1 | tee /tmp/vllm_build.log
BUILD_EXIT=${PIPESTATUS[0]}
cd -

if [ "$BUILD_EXIT" -ne 0 ]; then
    echo -e "\n${RED}❌ vLLM ビルド失敗。エラー箇所:${NC}"
    grep -n " error:" /tmp/vllm_build.log | grep -iv "warning\|note\|ignored" | head -20
    exit 1
fi

echo ""
echo -n "  vLLM バージョン:   " && python3 -c 'import vllm; print(vllm.__version__)'
echo -n "  PyTorch バージョン: " && python3 -c 'import torch; print(torch.__version__)'
echo -n "  Qwen3.5 サポート:  "
python3 -c '
from vllm.model_executor.models import ModelRegistry
try:
    supported = list(ModelRegistry._registry.keys())
except:
    supported = [k for k in dir(ModelRegistry) if "Qwen" in k]
if any("Qwen3_5" in s for s in supported):
    print("OK")
else:
    print("WARNING: Qwen3_5 未登録 →", [s for s in supported if "Qwen" in s])
' 2>/dev/null || echo "確認スキップ"

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
echo -e "環境:    ${YELLOW}vLLM ソースビルド + ROCm 7.2 (BUILD_TRITON=0, CK使用)${NC}"
echo -e "モデル:  ${YELLOW}Qwen3.5-27B フルパラメータ (~54GB/GPU)${NC}"
echo ""
echo -e "${GREEN}次のステップ:${NC}"
echo "  生成プロセスの開始:"
echo -e "     ${YELLOW}source /workspace/venv/bin/activate && bash runpod/run_pipeline.sh${NC}"
echo ""
