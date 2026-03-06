# Qwen3.5-27B Reasoning データセット生成パイプライン

Qwen3.5-27B-AWQ 等のモデルを使用して、ツールユーズ（Tool Use / Function Calling）や推論タスクに特化した高品質な Reasoning データセットを大量に生成するパイプラインです。
高VRAM GPU環境 (Vast.ai の RTX 5060 Ti × 8 や RunPod の AMD MI300X 192GB など) での並列処理に対応し、SFT（Supervised Fine-Tuning）に直結するフォーマットで出力します。

## 特徴

- 🚀 **マルチ環境＆並列処理対応** — Vast.ai (NVIDIA GPUマルチ構成) と RunPod (AMD MI300X 等の大容量GPU) の両環境に対応したセットアップを提供。
- 🛠️ **ツールユーズ特化** — 多岐にわたるツール定義（API呼び出し、マルチステップ、エラー処理、コーディングシナリオ等）を収録。
- 🇯🇵 **日本語優先のデータ生成** — 日本語70% / 英語30% の言語比率設定。
- ✅ **厳格な品質フィルタリング** — 思考ステップ数、回答の長さ、ツール呼び出しの有無、言語の一貫性、類似度による重複排除などを自動判定 (`filter_quality.py`)。
- 💾 **レジリエントな設計** — 処理中のチェックポイント保存により、Spotインスタンスの中断時も途中からの再開が可能。
- 📊 **SFT学習形式出力** — messages形式 (OpenAI互換) のJSONLとして出力され、モデルのファインチューニングに即時利用可能。

---

## プロジェクト構造

```
Qwen3.5-reasoning/
├── config.yaml                  # パイプライン全体の設定（モデル、生成パラメータ、ドメイン分布比率、フィルタ閾値など）
├── utils.py                     # 全体で共有されるユーティリティ関数群
├── filter_quality.py            # 生成データの品質を検証・フィルタリングするスクリプト
├── merge_outputs.py             # 分散処理された出力ファイルを結合し、分割（Split）するスクリプト
├── upload_to_hf.py              # 生成したデータセットをHugging Face Hubへアップロードするツール
├── test_local.py                # 主要ロジックの動作確認を行うローカル用テストスクリプト（GPU不要）
├── README.md                    # このファイル
├── prompts/
│   ├── __init__.py
│   ├── system_prompts.py        # 各ドメインごとのシステムプロンプトとツール定義の管理
│   └── seed_problems.json       # プロンプト生成のシードとなる問題集
├── runpod/
│   ├── setup.sh                 # RunPod用 (AMD MI300X ROCm等) 環境構築スクリプト
│   ├── generate_reasoning.py    # RunPod用のデータ生成実行スクリプト
│   └── run_pipeline.sh          # RunPod向けバッチ処理自動化スクリプト
└── vastai/
    ├── setup.sh                 # Vast.ai用 環境構築スクリプト
    ├── generate_reasoning.py    # Vast.ai用のデータ生成実行スクリプト
    └── run_pipeline.sh          # Vast.ai向けバッチ処理自動化スクリプト
```

---

## 環境構築 (重要)

本プロジェクトでは依存パッケージの競合を防ぐため、**必ずPython仮想環境 (`venv`) の使用を推奨します。**

### ローカルでの検証・開発環境

スクリプトの開発や `test_local.py` によるローカル検証（GPU不要）を行う場合の手順です。

```bash
# 1. リポジトリをダウンロード・移動
git clone <repository_url>
cd Qwen3.5-reasoning

# 2. 仮想環境の作成
python -m venv venv

# 3. 仮想環境のアクティベート
# 【Windows の場合】
.\venv\Scripts\activate
# 【Mac / Linux の場合】
source venv/bin/activate

# 4. 依存ライブラリのインストール
python -m pip install --upgrade pip
pip install -r requirements.txt  # (プロジェクト内に用意があれば実行。基本はuv/pipを使用)
```
※ 仮想環境に入っている間は、プロンプトの先頭に `(venv)` が表示されます。終了する場合は `deactivate` を実行してください。

---

## クラウド実行手順 (RunPod / Vast.ai)

GPU搭載のクラウドインスタンスで実際にデータ生成を行う手順です。**各セットアップスクリプト(`setup.sh`)の内部で自動的に `/workspace/venv` が作成され、利用されます。**

### 1. インスタンスへのファイルアップロードと接続

ローカルマシンからプロジェクト一式をサーバーへ転送し、SSHでログインします。

```bash
# プロジェクトファイルをサーバーの /workspace/ 等へアップロード
scp -r ./Qwen3.5-reasoning/* root@<instance-ip>:/workspace/Qwen3.5-reasoning/

# SSHでログイン
ssh root@<instance-ip>
```

### 2. 環境ごとのセットアップスクリプト実行

ログイン後、利用しているサービスに応じたディレクトリへ移動し、セットアップを実行します。
これにより、パッケージマネージャの更新、仮想環境の作成、vLLM等必要なライブラリ群のインストール、モデルのダウンロードが行われます。

**RunPod (AMD MI300X等の場合):**
```bash
cd /workspace/Qwen3.5-reasoning
bash runpod/setup.sh
```

**Vast.ai の場合:**
```bash
cd /workspace/Qwen3.5-reasoning
bash vastai/setup.sh
```

### 3. 生成パイプラインの実行

**注意:** 手動でコマンドを実行する際は、**必ず仮想環境をアクティベート**してから行ってください。

```bash
# 仮想環境をアクティベート
source /workspace/venv/bin/activate

# Qwen3.5-reasoningディレクトリ内へ移動
cd /workspace/Qwen3.5-reasoning
```

#### ドライラン（動作テスト）
少数のデータで試験的に動作確認を行いたい場合。
```bash
# RunPodの場合
python runpod/generate_reasoning.py --config config.yaml --dry-run
# Vast.aiの場合
python vastai/generate_reasoning.py --config config.yaml --dry-run
```

#### 全パイプライン一括実行（推奨）
セットアップ済みの全プロセス（生成 -> フィルタリング -> マージ）を自動で走らせる場合は提供スクリプトを使用します（内部で仮想環境がアクティベートされます）。

```bash
# RunPodの場合
bash runpod/run_pipeline.sh
# Vast.aiの場合
bash vastai/run_pipeline.sh
```

#### 手動での段階的実行
プロセスを分けて実行したい場合:
```bash
# ステップ1: 生成の実行
python runpod/generate_reasoning.py --config config.yaml --auto-parallel

# ステップ2: 品質フィルタリング
python filter_quality.py --config config.yaml

# ステップ3: ファイルのマージと分割
python merge_outputs.py --config config.yaml
```

---

## データ後処理とアップロード

生成が完了すると、結果は `config.yaml` で指定された出力ディレクトリ（デフォルトは `./output/final/`）に保存されます。

### ローカルでの動作確認（ユニットテスト）
クラウド環境へデプロイする前や、パース処理のロジックに変更を加えた場合は、ローカルでテストを実行できます。
```bash
python test_local.py
```

### Hugging Face Hub へのアップロード
ローカルにダウンロードするか、サーバー上から直接アップロードが可能です。
`upload_to_hf.py` スクリプトを使用する場合は以下のように行います:

```bash
# HuggingFace CLI でログイン (事前にトークンを取得しておく)
huggingface-cli login

# スクリプトを使用してアップロード
python upload_to_hf.py --dataset-dir ./output/final/ --repo-id your-username/qwen35-reasoning-dataset
```

---

## 設定のカスタマイズ

`config.yaml` を変更することで、生成タスクの比重やパラメータ、VRAM使用制限などを調整できます。

### ドメイン分布の変更（生成カテゴリの比率）
```yaml
dataset:
  domain_distribution:
    tool_use_api_calling: 0.20
    tool_use_multi_step: 0.15
    coding_agent: 0.20
    # 合計が 1.0 になるように調整してください。
```

### 生成パラメータ
```yaml
generation:
  temperature: 0.6     # Reasoningタスク向け推奨値
  top_p: 0.95
  max_tokens: 8192     # 思考タグの内容を含む出力の最大長
  batch_size: 256      # 使用中のGPU VRAM容量に合わせて変更
```

### 品質フィルタリング閾値
```yaml
quality_filter:
  min_thinking_steps: 3      # 思考ステップの最低要求数
  require_tool_calls: true   # ツール呼び出しドメインでのツール使用を強制するか
```

---

## トラブルシューティング

*   **OOM (Out Of Memory) エラーが発生する:**
    `config.yaml` の `generation.batch_size` 減らすか、`model.max_model_len` や `gpu_memory_utilization` を小さく調整してください。
*   **ワーカーが途中で停止した:**
    チェックポイント機能が有効なため、同一コマンドで再実行すると中断箇所から再開されます。
*   **ModuleNotFoundError が出る:**
    仮想環境 (`venv`) がアクティベートされているか確認してください。プロンプトの先頭に `(venv)` が無い場合は、`source /workspace/venv/bin/activate` (または `.\venv\Scripts\activate`) を実行してください。
