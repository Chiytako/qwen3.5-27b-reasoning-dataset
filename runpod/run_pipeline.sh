#!/bin/bash
# =============================================================================
# Qwen3.5-27B Reasoning Dataset Generation Pipeline Runner
# AMD MI300X 192GB 向け
# =============================================================================
set -e

# 色の定義
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# 設定
PROJECT_DIR="/workspace/qwen3.5-27b-reasoning-dataset"
CONFIG="$PROJECT_DIR/config.yaml"
VENV="/workspace/venv"
LOG_DIR="$PROJECT_DIR/logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
NUM_WORKERS=1

# HuggingFace設定
# 実行前に環境変数として設定してください: export HF_TOKEN="your_token"
HF_TOKEN="${HF_TOKEN:-}"
HF_REPO_ID="ChiTako/qwen3.5-27b-reasoning-dataset" # ユーザー名ChiTakoとデータセット名

echo -e "${BLUE}==========================================${NC}"
echo -e "${BLUE} Reasoning Dataset Generation Pipeline${NC}"
echo -e "${BLUE} Platform: Runpod (AMD MI300X 192GB x${NUM_WORKERS})${NC}"
echo -e "${BLUE} Started: $(date)${NC}"
echo -e "${BLUE} Project: $PROJECT_DIR${NC}"
echo -e "${BLUE}==========================================${NC}"

# 仮想環境のアクティベート
if [ -d "$VENV" ]; then
    source "$VENV/bin/activate"
    echo "仮想環境: $VENV"
else
    echo "WARNING: 仮想環境が見つかりません。システムPythonを使用します。"
fi

# GPU確認
echo ""
echo -e "${YELLOW}=== GPU状態 ===${NC}"
if command -v rocm-smi &> /dev/null; then
    rocm-smi --showid --showproductname --showmeminfo vram
    DETECTED_GPUS=$(rocm-smi --showid | grep -c "GPU")
else
    nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader
    DETECTED_GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
fi
echo ""
echo -e "検出GPU数: ${GREEN}$DETECTED_GPUS${NC} / 必要: ${YELLOW}$NUM_WORKERS${NC}"

if [ "$DETECTED_GPUS" -lt "$NUM_WORKERS" ]; then
    echo -e "${RED}エラー: 必要なGPU数($NUM_WORKERS)が不足しています。${NC}"
    exit 1
fi
echo ""

# ステップ# 1. ドライラン (コンフィグとプロンプトのチェック)
echo ""
echo -e "${GREEN}[1/5] ドライラン実行 (設定の検証中...)${NC}"
cd "$PROJECT_DIR"
python runpod/generate_reasoning.py --config "$CONFIG" --dry-run
if [ $? -ne 0 ]; then
    echo -e "${RED}❌ ドライランに失敗しました。設定を確認してください。${NC}"
    exit 1
fi
echo -e "${GREEN}✓ ドライラン成功${NC}"

# 2. 並列データ生成
echo ""
echo -e "${GREEN}[2/5] データ生成開始 (${NUM_WORKERS}ワーカー並列)${NC}"
echo "ログは $LOG_DIR/worker_*.log に出力されます"
echo ""

# 各ワーカーをバックグラウンドで起動
PIDS=()

for i in $(seq 0 $((NUM_WORKERS - 1))); do
    echo "Worker $i を GPU $i で起動中..."
    CUDA_VISIBLE_DEVICES=$i python runpod/generate_reasoning.py \
        --config "$CONFIG" \
        --worker-id $i \
        --gpu-id $i \
        > "$LOG_DIR/worker_${i}_$TIMESTAMP.log" 2>&1 &
    PIDS+=($!)
    echo "  PID: ${PIDS[$i]}"
done

echo ""
echo "全 $NUM_WORKERS ワーカー起動完了"
echo "PIDs: ${PIDS[*]}"

# 進捗モニタリング
echo ""
echo "=== 進捗モニタリング ==="

monitor_progress() {
    while true; do
        TOTAL=0
        RUNNING=0
        for i in $(seq 0 $((NUM_WORKERS - 1))); do
            FILE="$PROJECT_DIR/output/raw/worker_$(printf '%02d' $i).jsonl"
            if [ -f "$FILE" ]; then
                COUNT=$(wc -l < "$FILE" 2>/dev/null || echo 0)
                TOTAL=$((TOTAL + COUNT))
            fi
            # ワーカーの生存確認
            if [ $i -lt ${#PIDS[@]} ] && kill -0 ${PIDS[$i]} 2>/dev/null; then
                RUNNING=$((RUNNING + 1))
            fi
        done

        # GPU使用状況
        if command -v rocm-smi &> /dev/null; then
            GPU_USAGE=$(rocm-smi --showuse | grep -o '[0-9]*%' | tr -d '%' | awk '{sum+=$1; n++} END {if(n>0) printf "%.0f", sum/n; else print 0}')
        else
            GPU_USAGE=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | awk '{sum+=$1; n++} END {if(n>0) printf "%.0f", sum/n; else print 0}')
        fi

        echo -e "[$(date +%H:%M:%S)] 生成済: ${GREEN}$TOTAL${NC} 件 | 稼働ワーカー: ${YELLOW}$RUNNING/$NUM_WORKERS${NC} | 平均GPU使用率: ${BLUE}${GPU_USAGE}%${NC}"

        # 全ワーカー終了チェック
        if [ "$RUNNING" -eq 0 ]; then
            echo -e "${GREEN}✓ すべてのワーカーが終了しました。${NC}"
            break
        fi

        sleep 60
    done
}

monitor_progress &
MONITOR_PID=$!

# 全プロセスの完了を待つ
for pid in "${PIDS[@]}"; do
    wait $pid 2>/dev/null || true
done
kill $MONITOR_PID 2>/dev/null || true

echo -e "${GREEN}✓ データ生成完了${NC}"

# 3. 品質フィルタリング
echo ""
echo -e "${GREEN}[3/5] 品質フィルタリング実行...${NC}"
python filter_quality.py \
    --config "$CONFIG" \
    --input-dir "$PROJECT_DIR/output/raw" \
    --output-dir "$PROJECT_DIR/output/filtered"

# 4. マージとレポート生成
echo ""
echo -e "${GREEN}[4/5] データセットのマージとレポート生成...${NC}"
python merge_outputs.py \
    --config "$CONFIG" \
    --input-dir "$PROJECT_DIR/output/filtered" \
    --output-filepath "$PROJECT_DIR/output/final/qwen-reasoning-dataset.jsonl" \
    --report-filepath "$PROJECT_DIR/output/final/dataset_report.json"

# 5. HuggingFace Hub へのアップロード
echo ""
echo -e "${GREEN}[5/5] HuggingFace Hubへのアップロード...${NC}"

# 最終レポート
echo "=========================================="
echo " Pipeline Complete!"
echo " Finished: $(date)"
echo "=========================================="
echo ""

TOTAL_FINAL=0
for f in "$PROJECT_DIR/output/final/"*.jsonl; do
    if [ -f "$f" ]; then
        COUNT=$(wc -l < "$f")
        TOTAL_FINAL=$((TOTAL_FINAL + COUNT))
    fi
done

TOTAL_FILTERED=$(find "$PROJECT_DIR/output/filtered/" -name "*.jsonl" -exec wc -l {} + 2>/dev/null | tail -1 | awk '{print $1}')

echo "最終結果:"
echo "  Raw: $TOTAL_RAW 件"
echo "  Filtered: ${TOTAL_FILTERED:-0} 件"
echo "  Final: $TOTAL_FINAL 件"
echo ""
echo "出力ディレクトリ: $PROJECT_DIR/output/final/"
echo "レポート: $PROJECT_DIR/output/final/dataset_report.json"
echo ""

# 5. HuggingFace Hubへのアップロード
echo "=== Step 5: HuggingFaceアップロード ==="
if [ -f "$PROJECT_DIR/output/final/qwen-reasoning-dataset.jsonl" ]; then
    echo "HuggingFace Hubへアップロードを開始します..."
    echo "リポジトリ: $HF_REPO_ID"
    
    python upload_to_hf.py \
        --repo-id "$HF_REPO_ID" \
        --data-dir "$PROJECT_DIR/output/final" \
        --config "$CONFIG" \
        --token "$HF_TOKEN" \
        --private \
        2>&1 | tee "$LOG_DIR/upload_$TIMESTAMP.log"
        
    echo -e "${GREEN}アップロード処理が完了しました。${NC}"
else
    echo -e "${YELLOW}最終データセットが見つからないため、アップロードをスキップします。${NC}"
fi
echo ""
echo -e "${GREEN}✓ すべてのパイプライン処理が完了しました。${NC}"
