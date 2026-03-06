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
# ROCm 7.2 テンプレートには PyTorch 2.9.1 がシステムにプリインストール済み
# vLLM ROCm wheel は互換 PyTorch を同梱するため、venv は独立した環境として作成
python3 -m venv /workspace/venv
source /workspace/venv/bin/activate
echo "  Python: $(python3 --version)"

# --- 依存パッケージのインストール ---
echo -e "\n${GREEN}[3/5] vLLM + 依存パッケージのインストール...${NC}"
python3 -m pip install --upgrade pip --quiet
# uv: pip の依存解決フリーズを防ぐ超高速パッケージマネージャ
python3 -m pip install uv --quiet

# 軽量ライブラリ (数秒で完了)
echo "  軽量ライブラリをインストール中..."
uv pip install transformers>=4.48.0 pyyaml tqdm datasets huggingface_hub --quiet

# vLLM ROCm 公式 pre-built wheel
# 背景:
#   - ROCm 7.2 + Python 3.12 環境では wheels.vllm.ai の公式 ROCm ホイールが利用可能
#   - ソースビルド (30〜60分) が不要で数分でインストール完了
#   - このホイールは ROCm 対応 PyTorch を同梱しており、バージョン互換性は保証済み
#   - AMD は Qwen3.5 の Day 0 サポートを公式に提供 (https://www.amd.com)
echo -e "\n${YELLOW}vLLM ROCm pre-built wheel をインストール中 (数分で完了します)...${NC}"
uv pip install vllm --extra-index-url https://wheels.vllm.ai/rocm/

echo ""
echo -n "  vLLM バージョン:   " && python3 -c 'import vllm; print(vllm.__version__)'
echo -n "  PyTorch バージョン: " && python3 -c 'import torch; print(torch.__version__)'
echo -n "  ROCm バージョン:   " && python3 -c 'import torch; print(getattr(torch.version, "hip", "N/A"))'

# Qwen3.5 (Qwen3_5ForConditionalGeneration) サポート確認
echo -n "  Qwen3.5 サポート:  "
python3 -c '
from vllm.model_executor.models import ModelRegistry
if "Qwen3_5ForConditionalGeneration" in ModelRegistry._model_registry:
    print("OK (Qwen3_5ForConditionalGeneration 登録済み)")
else:
    # 全 Qwen 系モデルを表示してデバッグ
    qwen_models = [k for k in ModelRegistry._model_registry if "Qwen" in k]
    print("WARNING: Qwen3.5 未登録。登録済み Qwen モデル:", qwen_models)
' 2>/dev/null || echo "確認スキップ (実行時に確認してください)"

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
echo -e "環境:    ${YELLOW}PyTorch (vLLM同梱版) + ROCm 7.2 pre-built wheel${NC}"
echo -e "モデル:  ${YELLOW}Qwen3.5-27B フルパラメータ (~54GB/GPU)${NC}"
echo ""
echo -e "${GREEN}次のステップ:${NC}"
echo "  生成プロセスの開始:"
echo -e "     ${YELLOW}source /workspace/venv/bin/activate && bash runpod/run_pipeline.sh${NC}"
echo ""
