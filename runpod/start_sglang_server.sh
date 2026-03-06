#!/bin/bash
# =============================================================================
# SGLang API Server Launcher (MI300X Optimization)
# v1.0
# =============================================================================
set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

PROJECT_DIR="/workspace/qwen3.5-27b-reasoning-dataset"
LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_DIR"

MODEL_PATH="/workspace/models/Qwen3.5-27B"
# ダウンロードされていない場合はHuggingFaceから直接ロード
if [ ! -d "$MODEL_PATH" ]; then
    MODEL_PATH="Qwen/Qwen3.5-27B"
fi

PORT=8000
SERVER_LOG="$LOG_DIR/sglang_server.log"

echo -e "${BLUE}==========================================${NC}"
echo -e "${BLUE} Starting SGLang API Server...${NC}"
echo -e "${BLUE} Model: ${MODEL_PATH}${NC}"
echo -e "${BLUE} Port:  ${PORT}${NC}"
echo -e "${BLUE}==========================================${NC}"

# 既存のサーバープロセスの確認
if lsof -i:$PORT -t >/dev/null 2>&1; then
    echo -e "${YELLOW}APIサーバーは既にポート ${PORT} で起動しています。${NC}"
    exit 0
fi

# MI300X / ROCm 固有の最適化環境変数
export HIP_FORCE_DEV_KERNARG=1
export VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT=1

# SGLang サーバーの起動
#   --tp: テンソルパラレルサイズ (今回は MI300X 1枚=192GB なので 1)
#   --mem-fraction-static: GPUメモリの事前確保割合 (0.95等)
#   --context-length: コンテキストサイズ (vLLM設定を引き継ぎ約24k)
echo "SGLang サーバーをバックグラウンドで起動中..."
echo "ログファイル: $SERVER_LOG"

python3 -m sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --port $PORT \
    --host 0.0.0.0 \
    --tp 1 \
    --mem-fraction-static 0.95 \
    --context-length 24576 \
    --enable-dp-attention \
    --reasoning-parser qwen3 \
    --tool-call-parser qwen3_coder \
    > "$SERVER_LOG" 2>&1 &

SERVER_PID=$!
echo "サーバーPID: $SERVER_PID"

# ポートがListen状態になるまで待機 (最大10分)
echo -n "サーバーの起動を待機しています（モデルロードに数分かかります） "
MAX_WAIT=600
WAIT_TIME=0

while ! curl -s http://localhost:$PORT/v1/models >/dev/null; do
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo -e "\n${RED}❌ サーバープロセスが終了しました。ログ($SERVER_LOG)を確認してください。${NC}"
        tail -n 20 "$SERVER_LOG"
        exit 1
    fi
    echo -n "."
    sleep 5
    WAIT_TIME=$((WAIT_TIME + 5))
    if [ $WAIT_TIME -ge $MAX_WAIT ]; then
        echo -e "\n${RED}❌ タイムアウト ($MAX_WAIT 秒経過)。サーバーが起動しませんでした。${NC}"
        kill $SERVER_PID 2>/dev/null || true
        exit 1
    fi
done

echo -e "\n${GREEN}✓ SGLang APIサーバーが正常に起動しました！${NC}"
echo "API URL: http://localhost:$PORT/v1"
