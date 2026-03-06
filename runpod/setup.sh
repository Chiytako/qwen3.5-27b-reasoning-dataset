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
# --system-site-packages: テンプレートの PyTorch 2.9.1 ROCm 7.2 を継承
python3 -m venv --system-site-packages /workspace/venv
source /workspace/venv/bin/activate
echo "  Python:  $(python3 --version)"
echo "  PyTorch: $(python3 -c 'import torch; print(torch.__version__)' 2>/dev/null || echo '未検出 (ビルド後に確認)')"

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

# --- sccache セットアップ (コンパイルキャッシュで再ビルドを高速化) ---
echo "  sccache をセットアップ中..."
SCCACHE_BIN="/usr/local/bin/sccache"
if ! command -v sccache &>/dev/null; then
    # GitHub Releases から最新バイナリを取得
    SCCACHE_VER=$(curl -fsSL "https://api.github.com/repos/mozilla/sccache/releases/latest" \
        | grep '"tag_name"' | cut -d'"' -f4)
    wget -q "https://github.com/mozilla/sccache/releases/download/${SCCACHE_VER}/sccache-${SCCACHE_VER}-x86_64-unknown-linux-musl.tar.gz" \
        -O /tmp/sccache.tar.gz
    tar -xzf /tmp/sccache.tar.gz -C /tmp/
    mv /tmp/sccache-*/sccache "$SCCACHE_BIN"
    chmod +x "$SCCACHE_BIN"
    rm -rf /tmp/sccache*
fi
echo "  sccache: $(sccache --version)"

# sccache キャッシュを /workspace に配置 (Pod 再起動でも持続)
export SCCACHE_DIR="/workspace/sccache-cache"
export SCCACHE_CACHE_SIZE="20G"
export CMAKE_C_COMPILER_LAUNCHER=sccache
export CMAKE_CXX_COMPILER_LAUNCHER=sccache
mkdir -p "$SCCACHE_DIR"
sccache --start-server 2>/dev/null || true
echo "  sccache キャッシュ: $SCCACHE_DIR (20GB、Pod 再起動後も有効)"

# --- vLLM ソース取得 ---
VLLM_SRC="/workspace/vllm-src"
if [ -d "$VLLM_SRC/.git" ]; then
    echo "  vLLM ソースを更新中..."
    git -C "$VLLM_SRC" pull --ff-only
else
    echo "  vLLM ソースをクローン中 (shallow)..."
    git clone --depth=1 https://github.com/vllm-project/vllm.git "$VLLM_SRC"
fi

# --- vLLM ビルド (最適化設定) ---
echo -e "\n${YELLOW}vLLM を ROCm 向けにビルド中...${NC}"
echo "  最適化: sccache + MAX_JOBS=16 + Triton無効 + gfx942専用"
echo "  所要時間: 初回 30〜60分 / 2回目以降 sccache により数分"
echo "  ログ: tail -f /tmp/vllm_build.log | grep -E '^\[|error:'"
echo ""

cd "$VLLM_SRC"
# 最適化設定の説明:
#   BUILD_TRITON=0               : Tritonコンパイルを無効 (MI300X では CK が高速)
#   VLLM_USE_TRITON_FLASH_ATTN=0 : Triton flash attention を無効 → CK flash attention を使用
#   VLLM_INSTALL_PUNICA_KERNELS=0: LoRA用カーネルをスキップ (推論のみなので不要)
#   PYTORCH_ROCM_ARCH=gfx942     : MI300X 専用ビルド (他アーキテクチャのコンパイルを省略)
#   FA_GFX_ARCHS=gfx942          : Flash Attention も MI300X 専用
#   MAX_JOBS=16                  : sccache 管理下での並列数 (安定性と速度のバランス)
BUILD_TRITON=0 \
VLLM_USE_TRITON_FLASH_ATTN=0 \
VLLM_INSTALL_PUNICA_KERNELS=0 \
PYTORCH_ROCM_ARCH=gfx942 \
FA_GFX_ARCHS=gfx942 \
VLLM_TARGET_DEVICE=rocm \
MAX_JOBS=16 \
    pip install -e . --no-build-isolation 2>&1 | tee /tmp/vllm_build.log
BUILD_EXIT=${PIPESTATUS[0]}
cd -

if [ "$BUILD_EXIT" -ne 0 ]; then
    echo -e "\n${RED}❌ vLLM ビルド失敗。エラー箇所:${NC}"
    grep -n " error:" /tmp/vllm_build.log | grep -iv "warning\|note\|ignored" | head -20
    exit 1
fi

# sccache 統計表示
echo ""
sccache --show-stats 2>/dev/null | grep -E "Cache hits|Cache misses|Cache size" || true

echo ""
echo -n "  vLLM バージョン:   " && python3 -c 'import vllm; print(vllm.__version__)'
echo -n "  PyTorch バージョン: " && python3 -c 'import torch; print(torch.__version__)'
echo -n "  Qwen3.5 サポート:  "
python3 -c '
from vllm.model_executor.models import ModelRegistry
try:
    supported = list(ModelRegistry._registry.keys())
    has = any("Qwen3_5" in s for s in supported)
except:
    has = False
print("OK (Qwen3_5ForConditionalGeneration 登録済み)" if has else "WARNING: Qwen3.5 未登録")
' 2>/dev/null || echo "確認スキップ"

# ViT Flash Attention パッチ対象の事前確認
# generate_reasoning.py の _patch_vllm_rocm_vit() が ROCm 7.2 クラッシュ回避のために
# rocm.py を書き換える。vLLM のバージョンアップでパターンが変わった場合ここで検知する。
echo -n "  ViT SDPA パッチ対象: "
if grep -q 'Using Flash Attention backend for ViT model' "$VLLM_SRC/vllm/platforms/rocm.py" 2>/dev/null; then
    echo "OK (パターン確認済み)"
else
    echo -e "${YELLOW}WARNING: rocm.py のパッチ対象パターンが見つかりません${NC}"
    echo -e "${YELLOW}         vLLM のバージョンアップで rocm.py が変更された可能性があります。${NC}"
    echo -e "${YELLOW}         generate_reasoning.py の _patch_vllm_rocm_vit() を更新してください。${NC}"
fi

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
echo -e "環境:    ${YELLOW}vLLM ソースビルド + ROCm 7.2 (Triton無効/CK使用)${NC}"
echo -e "モデル:  ${YELLOW}Qwen3.5-27B フルパラメータ (~54GB/GPU)${NC}"
echo ""
echo -e "${GREEN}次のステップ:${NC}"
echo "  生成プロセスの開始:"
echo -e "     ${YELLOW}source /workspace/venv/bin/activate && bash runpod/run_pipeline.sh${NC}"
echo ""
