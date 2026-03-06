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
echo -e "検出GPU数: ${GREEN}$DETECTED_GPUS${NC}"

if [ "$DETECTED_GPUS" -lt 1 ]; then
    echo -e "${RED}エラー: GPUが検出されませんでした。${NC}"
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

# 2. APIサーバーの起動
echo ""
echo -e "${GREEN}[2/5] Llama.cpp APIサーバー起動${NC}"
bash runpod/start_llama_server.sh
if [ $? -ne 0 ]; then
    echo -e "${RED}❌ APIサーバーの起動に失敗しました。${NC}"
    exit 1
fi

# 3. 超並列データ生成 (API クライアント)
echo ""
echo -e "${GREEN}[3/5] データ生成開始 (Llama.cpp Async API)${NC}"
echo "ログは $LOG_DIR/generation_$TIMESTAMP.log に出力されます"
echo ""

# APIクライアントとしてスクリプトを実行
python runpod/generate_reasoning.py \
    --config "$CONFIG" \
    --worker-id 0 \
    2>&1 | tee "$LOG_DIR/generation_$TIMESTAMP.log"

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo -e "${RED}❌ データ生成プロセスが異常終了しました。${NC}"
    exit 1
fi

# 生成件数を集計
TOTAL_RAW=0
for f in "$PROJECT_DIR/output/raw/"worker_*.jsonl; do
    [ -f "$f" ] || continue
    COUNT=$(wc -l < "$f" 2>/dev/null || echo 0)
    TOTAL_RAW=$((TOTAL_RAW + COUNT))
    echo "  $(basename "$f"): $COUNT 件"
done
echo "合計 (raw): $TOTAL_RAW 件"

echo -e "${GREEN}✓ データ生成完了${NC}"

# 4. 品質フィルタリング
echo ""
echo -e "${GREEN}[4/5] 品質フィルタリング実行...${NC}"
python filter_quality.py \
    --config "$CONFIG" \
    --input-dir "$PROJECT_DIR/output/raw" \
    --output-dir "$PROJECT_DIR/output/filtered"

# 5. マージとレポート生成
echo ""
echo -e "${GREEN}[5/5] データセットのマージとレポート生成...${NC}"
python merge_outputs.py \
    --config "$CONFIG" \
    --input-dir "$PROJECT_DIR/output/filtered" \
    --output-dir "$PROJECT_DIR/output/final"

# 6. HuggingFace Hub へのアップロード
echo ""
echo -e "${GREEN}[6/6] HuggingFace Hubへのアップロード...${NC}"
if [ -f "$PROJECT_DIR/output/final/qwen-reasoning-dataset.jsonl" ]; then
    echo "リポジトリ: $HF_REPO_ID"
    python upload_to_hf.py \
        --repo-id "$HF_REPO_ID" \
        --data-dir "$PROJECT_DIR/output/final" \
        --config "$CONFIG" \
        --token "$HF_TOKEN" \
        --private \
        2>&1 | tee "$LOG_DIR/upload_$TIMESTAMP.log"
    echo -e "${GREEN}アップロード完了${NC}"
else
    echo -e "${YELLOW}最終データセットが見つからないため、アップロードをスキップします。${NC}"
fi

# 最終レポート
TOTAL_FINAL=0
for f in "$PROJECT_DIR/output/final/"*.jsonl; do
    [ -f "$f" ] || continue
    COUNT=$(wc -l < "$f")
    TOTAL_FINAL=$((TOTAL_FINAL + COUNT))
done
TOTAL_FILTERED=$(find "$PROJECT_DIR/output/filtered/" -name "*.jsonl" -exec wc -l {} + 2>/dev/null | tail -1 | awk '{print $1}')

echo ""
echo -e "${BLUE}==========================================${NC}"
echo -e "${BLUE} Qwen3.5-27B Reasoning Dataset Generator${NC}"
echo -e "${BLUE} Pipeline Execution (AMD MI300X)${NC}"
echo -e "${BLUE} Backend: Llama.cpp API Server${NC}"
echo -e "${BLUE}==========================================${NC}"
echo ""
echo "最終結果:"
echo "  Raw:      $TOTAL_RAW 件"
echo "  Filtered: ${TOTAL_FILTERED:-0} 件"
echo "  Final:    $TOTAL_FINAL 件"
echo ""
echo "出力ディレクトリ: $PROJECT_DIR/output/final/"
echo "レポート: $PROJECT_DIR/output/final/dataset_report.json"
echo ""
echo -e "${GREEN}✓ すべてのパイプライン処理が完了しました。${NC}"
