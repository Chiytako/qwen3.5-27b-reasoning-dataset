#!/bin/bash
# =============================================================================
# パイプライン全体実行ランナー (Runpod RTX 3090 x6)
# セットアップ → 生成（6並列） → フィルタリング → マージ を一気に実行
# =============================================================================
set -e

# 設定
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG="$PROJECT_DIR/config.yaml"
VENV="/workspace/venv"
LOG_DIR="$PROJECT_DIR/logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
NUM_WORKERS=6

# HuggingFace設定
# 実行前に環境変数として設定してください: export HF_TOKEN="your_token"
HF_TOKEN="${HF_TOKEN:-}"
HF_REPO_ID="ChiTako/qwen3.5-27b-reasoning-dataset" # ユーザー名ChiTakoとデータセット名

echo "=========================================="
echo " Reasoning Dataset Generation Pipeline"
echo " Platform: Runpod (RTX 3090 x${NUM_WORKERS})"
echo " Started: $(date)"
echo " Project: $PROJECT_DIR"
echo "=========================================="

# 仮想環境のアクティベート
if [ -d "$VENV" ]; then
    source "$VENV/bin/activate"
    echo "仮想環境: $VENV"
else
    echo "WARNING: 仮想環境が見つかりません。システムPythonを使用します。"
fi

# GPU確認
echo ""
echo "=== GPU状態 ==="
nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader
DETECTED_GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
echo ""
echo "検出GPU数: $DETECTED_GPUS / 必要: $NUM_WORKERS"

if [ "$DETECTED_GPUS" -lt "$NUM_WORKERS" ]; then
    echo "WARNING: GPUが不足しています。ワーカー数を$DETECTED_GPUSに調整します。"
    NUM_WORKERS=$DETECTED_GPUS
fi
echo ""

# ステップ1: ドライラン (設定テスト)
echo "=== Step 1: ドライラン ==="
cd "$PROJECT_DIR"
python runpod/generate_reasoning.py --config "$CONFIG" --dry-run 2>&1 | tee "$LOG_DIR/dryrun_$TIMESTAMP.log"
echo "ドライラン完了 ✓"
echo ""

# ステップ2: 6並列生成
echo "=== Step 2: ${NUM_WORKERS}並列生成開始 ==="
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
    sleep 5  # RTX 3090はメモリ初期化に少し時間がかかる
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
        GPU_USAGE=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | awk '{sum+=$1; n++} END {printf "%.0f", sum/n}')

        echo "[$(date +%H:%M:%S)] 生成済: $TOTAL 件 | 稼働ワーカー: $RUNNING/$NUM_WORKERS | 平均GPU使用率: ${GPU_USAGE}%"

        # 全ワーカー終了チェック
        if [ $RUNNING -eq 0 ]; then
            echo "全ワーカーが終了しました"
            break
        fi

        sleep 120  # 2分ごとに進捗チェック
    done
}

monitor_progress

# 全プロセスの完了を待つ
for pid in "${PIDS[@]}"; do
    wait $pid 2>/dev/null || true
done

echo ""
echo "=== Step 2 完了: 生成終了 ==="

# 生成結果の確認
TOTAL_RAW=0
for f in "$PROJECT_DIR/output/raw/"*.jsonl; do
    if [ -f "$f" ]; then
        COUNT=$(wc -l < "$f")
        TOTAL_RAW=$((TOTAL_RAW + COUNT))
        echo "  $(basename $f): $COUNT 件"
    fi
done
echo "合計 (raw): $TOTAL_RAW 件"
echo ""

# ステップ3: 品質フィルタリング
echo "=== Step 3: 品質フィルタリング ==="
python filter_quality.py \
    --config "$CONFIG" \
    --input-dir "$PROJECT_DIR/output/raw" \
    --output-dir "$PROJECT_DIR/output/filtered" \
    --save-rejected \
    2>&1 | tee "$LOG_DIR/filter_$TIMESTAMP.log"
echo "フィルタリング完了 ✓"
echo ""

# ステップ4: マージ & 最終データセット生成
echo "=== Step 4: マージ & 最終データセット ==="
python merge_outputs.py \
    --config "$CONFIG" \
    --input-dir "$PROJECT_DIR/output/filtered" \
    --output-dir "$PROJECT_DIR/output/final" \
    2>&1 | tee "$LOG_DIR/merge_$TIMESTAMP.log"
echo "マージ完了 ✓"
echo ""

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

# ステップ5: HuggingFace Hubへのアップロード
echo "=== Step 5: HuggingFaceアップロード ==="
if [ "$TOTAL_FINAL" -gt 0 ]; then
    echo "HuggingFace Hubへアップロードを開始します..."
    echo "リポジトリ: $HF_REPO_ID"
    
    # 依存パッケージ確認
    pip install huggingface_hub datasets > /dev/null 2>&1
    
    python upload_to_hf.py \
        --repo-id "$HF_REPO_ID" \
        --data-dir "$PROJECT_DIR/output/final" \
        --config "$CONFIG" \
        --token "$HF_TOKEN" \
        --private \
        2>&1 | tee "$LOG_DIR/upload_$TIMESTAMP.log"
        
    echo "アップロード処理が完了しました。"
else
    echo "最終データセットが0件のため、アップロードをスキップします。"
fi
echo ""
echo "すべてのパイプライン処理が完了しました。"
