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

echo -e "${GREEN}[1/3] vLLM ROCm Dockerイメージの準備...${NC}"
# インスタンス内でDockerデーモンが動いているか確認
if ! command -v docker &> /dev/null; then
    echo -e "${RED}エラー: Dockerがインストールされていません。Runpodのテンプレート設定でDocker in Docker(DinD)が有効になっているか、別のvLLM用テンプレートを選択してください。${NC}"
    exit 1
fi

# =============================================================================
# ▼ Dockerコンテナを利用したアプローチ ▼
# =============================================================================
# Qwen3.5-27Bを動かすためには、単なるpipではなく、
# MI300XのROCmに極限まで最適化された公式vLLM Dockerイメージが必要です。
# =============================================================================

# AMDの最新vLLMイメージ名
VLLM_IMAGE="rocm/vllm:rocm6.2_mi300_ubuntu20.04_py3.9_vllm_0.6.4"

echo -e "\n${GREEN}[2/3] Qwen3.5-27B フルモデル事前ダウンロード...${NC}"
python3 -m pip install --upgrade pip
python3 -m pip install huggingface_hub

mkdir -p /workspace/models
python3 -c "
from huggingface_hub import snapshot_download
print('モデルをダウンロード中... (FP16/BF16版は約54GB、時間がかかります。)')
snapshot_download(
    repo_id='Qwen/Qwen3.5-27B',
    local_dir='/workspace/models/Qwen3.5-27B',
    ignore_patterns=['*.md', '*.txt', 'LICENSE*'],
)
print('ダウンロード完了!')
"

echo -e "\n${GREEN}[3/3] Docker用の起動スクリプトの生成...${NC}"
PROJECT_DIR="/workspace/qwen3.5-27b-reasoning-dataset"
mkdir -p "$PROJECT_DIR/output/raw"
mkdir -p "$PROJECT_DIR/output/filtered"
mkdir -p "$PROJECT_DIR/output/final"
mkdir -p "$PROJECT_DIR/logs"

cat << 'EOF' > runpod/docker_runner.sh
#!/bin/bash
PROJECT_DIR="/workspace/qwen3.5-27b-reasoning-dataset"
VLLM_IMAGE="rocm/vllm:rocm6.2_mi300_ubuntu20.04_py3.9_vllm_0.6.4"

echo "=== AMD公式 vLLM コンテナを起動します ==="
# ROCmデバイスをコンテナにマウントし、共有メモリを大きく取って起動
docker run -it --rm \
    --network host \
    --device /dev/kfd \
    --device /dev/dri \
    --ipc=host \
    --shm-size 128G \
    --group-add video \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    -v /workspace:/workspace \
    -w /workspace/qwen3.5-27b-reasoning-dataset \
    $VLLM_IMAGE \
    bash -c "\
        echo '=== コンテナ内セットアップ ===' && \
        pip install pyyaml tqdm datasets huggingface_hub && \
        bash runpod/run_pipeline.sh \
    "
EOF
chmod +x runpod/docker_runner.sh

echo ""
echo -e "${BLUE}==========================================${NC}"
echo -e "${GREEN} ✓ セットアップ完了!${NC}"
echo -e "${BLUE}==========================================${NC}"
echo ""
echo -e "GPU構成: ${YELLOW}AMD MI300X 192GB x1${NC}"
echo -e "モデル: ${YELLOW}Qwen3.5-27B フルパラメータ (~54GB/GPU)${NC}"
echo -e "実行方式: ${YELLOW}AMD公式 Dockerコンテナ (ROCm 6.2 + vLLM 0.6.4)${NC}"
echo ""
echo -e "${GREEN}次のステップ:${NC}"
echo "  生成プロセスの開始（Docker経由）:"
echo -e "     ${YELLOW}bash runpod/docker_runner.sh${NC}"
echo ""
