#!/bin/bash
# =============================================================================
# パイプライン全体実行ランナー
# セットアップ → 生成（8並列） → フィルタリング → マージ を一気に実行
# =============================================================================
set -e

# 設定
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG="$PROJECT_DIR/config.yaml"
VENV="/workspace/venv"
LOG_DIR="$PROJECT_DIR/logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "=========================================="
echo " Reasoning Dataset Generation Pipeline"
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
echo ""

# ステップ1: ドライラン (設定テスト)
echo "=== Step 1: ドライラン ==="
cd "$PROJECT_DIR"
python vastai/generate_reasoning.py --config "$CONFIG" --dry-run 2>&1 | tee "$LOG_DIR/dryrun_$TIMESTAMP.log"
echo "ドライラン完了 ✓"
echo ""

# ステップ2: 8並列生成
echo "=== Step 2: 8並列生成開始 ==="
echo "ログは $LOG_DIR/worker_*.log に出力されます"
echo ""

# 各ワーカーをバックグラウンドで起動
NUM_WORKERS=8
PIDS=()

for i in $(seq 0 $((NUM_WORKERS - 1))); do
    echo "Worker $i を GPU $i で起動中..."
    CUDA_VISIBLE_DEVICES=$i python vastai/generate_reasoning.py \
        --config "$CONFIG" \
        --worker-id $i \
        --gpu-id $i \
        > "$LOG_DIR/worker_${i}_$TIMESTAMP.log" 2>&1 &
    PIDS+=($!)
    echo "  PID: ${PIDS[$i]}"
    sleep 3  # GPU初期化の衝突を避ける
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
            if kill -0 ${PIDS[$i]} 2>/dev/null; then
                RUNNING=$((RUNNING + 1))
            fi
        done

        echo "[$(date +%H:%M:%S)] 生成済: $TOTAL 件 | 稼働ワーカー: $RUNNING/$NUM_WORKERS"

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

echo "最終結果:"
echo "  Raw: $TOTAL_RAW 件"
echo "  Filtered: $(ls "$PROJECT_DIR/output/filtered/"*.jsonl 2>/dev/null | xargs wc -l 2>/dev/null | tail -1 | awk '{print $1}') 件"
echo "  Final: $TOTAL_FINAL 件"
echo ""
echo "出力ディレクトリ: $PROJECT_DIR/output/final/"
echo "レポート: $PROJECT_DIR/output/final/dataset_report.json"
echo ""
echo "HuggingFace Hubにアップロードするには:"
echo "  huggingface-cli upload your-username/qwen35-reasoning-1m $PROJECT_DIR/output/final/"
echo ""
