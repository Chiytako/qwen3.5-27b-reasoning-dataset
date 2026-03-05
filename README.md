# Qwen3.5-27B Reasoning データセット生成パイプライン

Qwen3.5-27B-AWQを使用して、ツールユーズ/function calling特化の高品質reasoningデータセットを大量に生成するパイプライン。
Vast.ai RTX 5060 Ti × 8 での8並列処理に最適化。

## 特徴

- 🚀 **8GPU並列処理** — 各GPUで独立したvLLMインスタンスを実行、最大8倍のスループット
- 🛠️ **ツールユーズ特化** — 20種のツール定義、API呼び出し・マルチステップ・エラー処理のシナリオを80種以上収録
- 🇯🇵 **日本語優先** — 日本語70% / 英語30%の言語比率
- ✅ **品質フィルタリング** — 5段階の自動品質検証（思考ステップ、回答品質、ツール使用、言語一貫性、重複除去）
- 💾 **チェックポイント付き** — 中断しても再開可能
- 📊 **SFT学習形式出力** — messages形式のJSONLで直接モデル学習に使用可能

## プロジェクト構造

```
Qwen3.5-reasoning/
├── config.yaml                  # 設定ファイル（モデル、パラメータ、ドメイン分布）
├── utils.py                     # 共通ユーティリティ
├── filter_quality.py            # 品質フィルタリング
├── merge_outputs.py             # 出力マージ・データセット分割
├── README.md                    # このファイル
├── prompts/
│   ├── __init__.py
│   ├── system_prompts.py        # システムプロンプト・ツール定義
│   └── seed_problems.json       # シード問題（9ドメイン×2言語）
└── vastai/
    ├── setup.sh                 # Vast.ai環境セットアップ
    ├── generate_reasoning.py    # メイン生成スクリプト（8並列対応）
    └── run_pipeline.sh          # パイプライン一括実行
```

## クイックスタート

### 1. Vast.aiインスタンスの作成

**推奨スペック:**
| 項目 | 値 |
|------|-----|
| GPU | RTX 5060 Ti 16GB × 8 |
| VRAM合計 | 128GB |
| RAM | 64GB以上 |
| ストレージ | 200GB以上 |
| テンプレート | PyTorch 2.5+ / CUDA 12.4+ |

### 2. セットアップ

```bash
# プロジェクトファイルをアップロード
scp -r ./* root@<vast-ip>:/workspace/Qwen3.5-reasoning/

# SSHでログイン
ssh root@<vast-ip>

# セットアップ実行
cd /workspace/Qwen3.5-reasoning
bash vastai/setup.sh
```

### 3. ドライラン（テスト）

```bash
source /workspace/venv/bin/activate
cd /workspace/Qwen3.5-reasoning
python vastai/generate_reasoning.py --config config.yaml --dry-run
```

### 4. 全パイプライン実行

```bash
# 一括実行（推奨）
bash vastai/run_pipeline.sh

# または手動で段階的に実行:
# ステップ1: 生成
python vastai/generate_reasoning.py --config config.yaml --auto-parallel

# ステップ2: フィルタリング
python filter_quality.py --config config.yaml

# ステップ3: マージ
python merge_outputs.py --config config.yaml
```

### 5. 結果の取得

```bash
# ローカルにダウンロード
scp -r root@<vast-ip>:/workspace/Qwen3.5-reasoning/output/final/ ./dataset/

# HuggingFace Hubにアップロード
huggingface-cli upload your-username/qwen35-reasoning-1m output/final/
```

## 設定のカスタマイズ

`config.yaml` を編集して以下を調整可能：

### ドメイン分布
```yaml
domain_distribution:
  tool_use_api_calling: 0.30    # ← ツールユーズのウエイトを調整
  tool_use_multi_step: 0.20
  tool_use_error_handling: 0.10
  code_generation: 0.10
  math_reasoning: 0.08
  ...
```

### 生成パラメータ
```yaml
generation:
  temperature: 0.7     # 高いほど多様（0.5〜1.0推奨）
  top_p: 0.9
  max_tokens: 8192     # 思考チェーンの最大長
  batch_size: 64       # GPUメモリと相談
```

### 品質フィルタリング
```yaml
quality_filter:
  min_thinking_steps: 3     # 最低思考ステップ
  require_tool_calls: true  # ツール呼び出し必須
  dedup_similarity_threshold: 0.85
```

## 生成データの形式

```json
{
  "id": "qwen35r_tool_use_api_calling_00_0000001",
  "messages": [
    {"role": "system", "content": "あなたは高度なAIアシスタントです..."},
    {"role": "user", "content": "最新のPythonバージョンを検索して..."},
    {"role": "assistant", "content": "<think>\nまず、search_webツールを使って...\n</think>\n\n検索結果を確認しました..."}
  ],
  "domain": "tool_use_api_calling",
  "language": "ja",
  "tools": [...],
  "tool_calls": [...]
}
```

## コスト見積もり

| 件数 | 所要時間 (推定) | コスト (RTX 5060 Ti x8) |
|------|----------------|------------------------|
| 10,000 | ~0.7h | ~$2 |
| 100,000 | ~7h | ~$20 |
| 1,000,000 | ~70h | ~$200 |

※ AWQ 4bit、バッチサイズ64、8並列の場合の概算

## トラブルシューティング

### OOMエラー
`config.yaml`で`batch_size`を小さくするか、`max_model_len`を短縮してください。

### ワーカーが途中で停止
チェックポイント機能があるため、同じコマンドで再実行すれば続きから再開されます。

### モデルのダウンロードが遅い
`HF_ENDPOINT`環境変数でミラーを指定できます:
```bash
export HF_ENDPOINT=https://hf-mirror.com
```
