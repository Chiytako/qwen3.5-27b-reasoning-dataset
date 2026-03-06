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
NUM_INSTANCES=4
BASE_PORT=8000

echo -e "${BLUE}==========================================${NC}"
echo -e "${BLUE} Starting Llama.cpp API Servers (x${NUM_INSTANCES})...${NC}"
echo -e "${BLUE} Model: Qwen3.5-27B-UD-Q6_K_XL (Unsloth)${NC}"
echo -e "${BLUE} Context: 131072 total / 8 slots = 16384 tokens/slot (Total VRAM ~185GB)${NC}"
echo -e "${BLUE}==========================================${NC}"

if [ ! -f "$MODEL_PATH" ]; then
    echo -e "${RED}エラー: モデルファイルが見つかりません。setup.sh を実行してダウンロードしてください。${NC}"
    echo "Path: $MODEL_PATH"
    exit 1
fi

PIDS=""
for i in $(seq 0 $((NUM_INSTANCES - 1))); do
    PORT=$((BASE_PORT + i))
    SERVER_LOG="$LOG_DIR/llama_server_${PORT}.log"
    
    # 既存のサーバープロセスの確認
    if lsof -i:$PORT -t >/dev/null 2>&1; then
        echo -e "${YELLOW}APIサーバーは既にポート ${PORT} で起動しています。スキップします。${NC}"
        continue
    fi
    
    echo "llama-server をポート ${PORT} でバックグラウンド起動中..."
    echo "ログファイル: $SERVER_LOG"
    
    # Llama.cpp サーバーの起動オプション
    # -c 131072: 総KVキャッシュ (VRAM ~46GB/インスタンス、4インスタンスで ~185GB)
    # -np 8: 並列スロット数。131072/8 = 16384 tokens/slot
    #        (以前の -np 16 = 8192/slot だと思考チェーンが途切れ空応答が発生していた)
    /workspace/llama.cpp/build/bin/llama-server \
        -m "$MODEL_PATH" \
        --host 0.0.0.0 \
        --port $PORT \
        -c 131072 \
        -np 8 \
        -ngld 999 \
        -fa on \
        -cb \
        --alias "Qwen/Qwen3.5-27B" \
        > "$SERVER_LOG" 2>&1 &
        
    PID=$!
    PIDS="$PIDS $PID"
    echo "サーバーPID (Port $PORT): $PID"
    
    # MI300Xのモデルロードの競合を防ぐため、10秒ほど待ってから次を起動
    sleep 10
done

if [ -z "$PIDS" ]; then
    echo -e "${GREEN}✓ すべてのサーバーが既に起動しています。${NC}"
    exit 0
fi

# ポートがListen状態になるまで待機 (最大15分)
echo -n "サーバーの起動を待機しています（モデルのロードに数分かかります） "
MAX_WAIT=900
WAIT_TIME=0

ALL_READY=false
while [ "$ALL_READY" = false ]; do
    ALL_READY=true
    for i in $(seq 0 $((NUM_INSTANCES - 1))); do
        PORT=$((BASE_PORT + i))
        if ! curl -s http://localhost:$PORT/v1/models >/dev/null; then
            ALL_READY=false
            break
        fi
    done
    
    if [ "$ALL_READY" = true ]; then
        break
    fi
    
    echo -n "."
    sleep 5
    WAIT_TIME=$((WAIT_TIME + 5))
    if [ $WAIT_TIME -ge $MAX_WAIT ]; then
        echo -e "\n${RED}❌ タイムアウト ($MAX_WAIT 秒経過)。サーバーが起動しませんでした。${NC}"
        for PID in $PIDS; do
            kill $PID 2>/dev/null || true
        done
        exit 1
    fi
done

echo -e "\n${GREEN}✓ Llama.cpp APIサーバー (x${NUM_INSTANCES}) が正常に起動しました！${NC}"
