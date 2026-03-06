#!/bin/bash
# =============================================================================
# Llama.cpp API Server Launcher (MI300X ROCm)
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

# UnslothのQ6_K GGUFモデル (約25.7GB)
MODEL_PATH="/workspace/models/Qwen3.5-27B-GGUF/Qwen3.5-27B-UD-Q6_K_XL.gguf"
PORT=8000
SERVER_LOG="$LOG_DIR/llama_server.log"

echo -e "${BLUE}==========================================${NC}"
echo -e "${BLUE} Starting Llama.cpp API Server...${NC}"
echo -e "${BLUE} Model: Qwen3.5-27B-UD-Q6_K_XL (Unsloth)${NC}"
echo -e "${BLUE} Port:  ${PORT}${NC}"
echo -e "${BLUE}==========================================${NC}"

# 既存のサーバープロセスの確認
if lsof -i:$PORT -t >/dev/null 2>&1; then
    echo -e "${YELLOW}APIサーバーは既にポート ${PORT} で起動しています。${NC}"
    exit 0
fi

if [ ! -f "$MODEL_PATH" ]; then
    echo -e "${RED}エラー: モデルファイルが見つかりません。setup.sh を実行してダウンロードしてください。${NC}"
    echo "Path: $MODEL_PATH"
    exit 1
fi

echo "llama-server をバックグラウンドで起動中..."
echo "ログファイル: $SERVER_LOG"

# Llama.cpp サーバーの起動オプション
# -m: モデルパス
# --host / --port: ネットワーク
# -c: コンテキストサイズ (MI300Xの広大なVRAMを活かして32k以上に設定)
# -np: Parallel slots (同時リクエスト受付数)
# -ngld: GPUオフロードレイヤー (全レイヤー)
# -fa: FlashAttention有効化
# -cb: 連続バッチング(Continuous Batching)有効化
# --alias: APIコール時のモデル名識別子

/workspace/llama.cpp/build/bin/llama-server \
    -m "$MODEL_PATH" \
    --host 0.0.0.0 \
    --port $PORT \
    -c 32768 \
    -np 128 \
    -ngld 999 \
    -fa \
    -cb \
    --alias "Qwen/Qwen3.5-27B" \
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

echo -e "\n${GREEN}✓ Llama.cpp APIサーバーが正常に起動しました！${NC}"
echo "API URL: http://localhost:$PORT/v1"
