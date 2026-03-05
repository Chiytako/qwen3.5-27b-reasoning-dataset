#!/bin/bash
# =============================================================================
# Runpod 環境セットアップスクリプト
# RTX 3090 x6 + Qwen3.5-27B-AWQ 用
# =============================================================================
set -e

echo "=========================================="
echo " Qwen3.5-27B Reasoning Dataset Generator"
echo " Runpod Setup Script (RTX 3090 x6)"
echo "=========================================="

# --- システム更新 ---
echo "[1/6] システムパッケージの更新..."
apt-get update -qq && apt-get install -y -qq git wget curl htop nvtop tmux > /dev/null 2>&1 || true

# --- Python環境 ---
echo "[2/6] Python仮想環境のセットアップ..."
python3 -m venv /workspace/venv
source /workspace/venv/bin/activate

# --- 依存パッケージのインストール ---
echo "[3/6] 依存パッケージのインストール..."
pip install --upgrade pip > /dev/null 2>&1
pip install \
    vllm>=0.16.0 \
    transformers>=4.48.0 \
    torch>=2.5.0 \
    pyyaml \
    tqdm \
    datasets \
    huggingface_hub \
    > /dev/null 2>&1

echo "  vLLMバージョン: $(python -c 'import vllm; print(vllm.__version__)')"
echo "  PyTorchバージョン: $(python -c 'import torch; print(torch.__version__)')"

# --- GPU確認 ---
echo "[4/6] GPU状態の確認..."
nvidia-smi --query-gpu=index,name,memory.total,memory.free,driver_version --format=csv,noheader
echo ""
python3 -c "
import torch
print(f'CUDA利用可能: {torch.cuda.is_available()}')
print(f'GPU数: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)} ({props.total_memory / 1024**3:.1f} GB)')
    
gpu_count = torch.cuda.device_count()
if gpu_count < 6:
    print(f'WARNING: GPUが{gpu_count}枚しか検出されません。6枚推奨です。')
    print(f'  config.yamlのnum_workersを{gpu_count}に変更してください。')
"

# --- プロジェクトのセットアップ ---
echo "[5/6] プロジェクトディレクトリのセットアップ..."
PROJECT_DIR="/workspace/Qwen3.5-reasoning"

# 出力ディレクトリ作成
mkdir -p "$PROJECT_DIR/output/raw"
mkdir -p "$PROJECT_DIR/output/filtered"
mkdir -p "$PROJECT_DIR/output/final"
mkdir -p "$PROJECT_DIR/output/checkpoints"
mkdir -p "$PROJECT_DIR/output/rejected"
mkdir -p "$PROJECT_DIR/logs"

# --- モデルの事前ダウンロード ---
echo "[6/6] Qwen3.5-27B-AWQ モデルの事前ダウンロード..."
python3 -c "
from huggingface_hub import snapshot_download
print('モデルをダウンロード中... (初回は約15GB、時間がかかります)')
snapshot_download(
    repo_id='Qwen/Qwen3.5-27B-AWQ',
    local_dir='/workspace/models/Qwen3.5-27B-AWQ',
    ignore_patterns=['*.md', '*.txt', 'LICENSE*'],
)
print('ダウンロード完了!')
"

echo ""
echo "=========================================="
echo " セットアップ完了!"
echo "=========================================="
echo ""
echo "GPU構成: RTX 3090 24GB x6 = 144GB VRAM"
echo "モデル: Qwen3.5-27B-AWQ (~14GB/GPU)"
echo "並列度: 6ワーカー (1 GPU/ワーカー)"
echo ""
echo "次のステップ:"
echo "  1. ドライラン (テスト):"
echo "     cd $PROJECT_DIR"
echo "     source /workspace/venv/bin/activate"
echo "     python runpod/generate_reasoning.py --config config.yaml --dry-run"
echo ""
echo "  2. 生成の開始:"
echo "     bash runpod/run_pipeline.sh"
echo ""
