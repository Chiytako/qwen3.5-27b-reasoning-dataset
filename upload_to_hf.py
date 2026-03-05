"""
HuggingFace Hub へのデータセットアップロードスクリプト

使い方:
  # 環境変数でトークンを設定してアップロード
  export HF_TOKEN="hf_xxxxx"
  python upload_to_hf.py --repo-id your-username/dataset-name

  # トークンを直接指定
  python upload_to_hf.py --repo-id your-username/dataset-name --token hf_xxxxx

  # アップロード前にプレビュー
  python upload_to_hf.py --repo-id your-username/dataset-name --preview
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import load_config, count_jsonl_lines

logger = logging.getLogger(__name__)


def create_dataset_card(repo_id: str, report: dict, config: dict) -> str:
    """HuggingFace用のデータセットカード(README.md)を生成"""

    total = report.get("total_samples", 0)
    train = report.get("train_samples", 0)
    test = report.get("test_samples", 0)

    # ドメイン分布テーブル
    domain_rows = ""
    domain_dist = report.get("domain_distribution", {})
    for k, v in sorted(domain_dist.items(), key=lambda x: -x[1]):
        name = k.replace("domain_", "")
        pct = 100 * v / total if total > 0 else 0
        domain_rows += f"| {name} | {v:,} | {pct:.1f}% |\n"

    # 言語分布テーブル
    lang_rows = ""
    lang_dist = report.get("language_distribution", {})
    for k, v in sorted(lang_dist.items(), key=lambda x: -x[1]):
        name = k.replace("lang_", "")
        pct = 100 * v / total if total > 0 else 0
        lang_rows += f"| {name} | {v:,} | {pct:.1f}% |\n"

    card = f"""---
language:
  - ja
  - en
license: apache-2.0
task_categories:
  - text-generation
  - question-answering
tags:
  - reasoning
  - tool-use
  - function-calling
  - coding-agent
  - thinking
  - qwen
  - synthetic
size_categories:
  - 100K<n<1M
---

# {repo_id.split('/')[-1]}

Qwen3.5-27B-AWQ を使用して生成された高品質な **reasoning + ツールユーズ** データセットです。

## 概要

- **モデル**: {config.get('model', {}).get('name', 'Qwen/Qwen3.5-27B-AWQ')}
- **合計サンプル数**: {total:,}
- **Train / Test**: {train:,} / {test:,}
- **言語**: 日本語 (70%) / English (30%)
- **フォーカス**: ツールユーズ、Function Calling、コーディングエージェント

## 特徴

- 🛠️ **ツールユーズ特化**: API呼び出し、マルチステップツール連携、エラー処理
- 💻 **Coding Agent**: Opencode/Claude Code風の開発エージェントタスク
- 🧠 **思考チェーン付き**: `<think>` タグで推論過程を含む
- 🇯🇵 **日本語優先**: 日本語70%、英語30%
- ✅ **品質フィルタ済み**: 5段階の自動品質検証を通過

## データ形式

```json
{{
  "id": "qwen35r_coding_agent_00_0000001",
  "messages": [
    {{"role": "system", "content": "あなたはプロのソフトウェアエンジニアとして..."}},
    {{"role": "user", "content": "このプロジェクトのバグを修正してください..."}},
    {{"role": "assistant", "content": "<think>\\nまず、コードベースを調査して...\\n</think>\\n\\n調査の結果..."}}
  ],
  "domain": "coding_agent",
  "language": "ja",
  "tools": [...],
  "tool_calls": [...]
}}
```

## ドメイン分布

| Domain | Count | Ratio |
|--------|-------|-------|
{domain_rows}

## 言語分布

| Language | Count | Ratio |
|----------|-------|-------|
{lang_rows}

## 利用方法

```python
from datasets import load_dataset

ds = load_dataset("{repo_id}")
print(ds["train"][0])
```

### SFT学習での利用

```python
from trl import SFTTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("your-base-model")
tokenizer = AutoTokenizer.from_pretrained("your-base-model")

trainer = SFTTrainer(
    model=model,
    train_dataset=ds["train"],
    tokenizer=tokenizer,
    dataset_text_field="messages",
)
trainer.train()
```

## 生成パイプライン

[Qwen3.5-reasoning](https://github.com/your-username/Qwen3.5-reasoning) パイプラインで生成。

- 推論エンジン: vLLM (バッチ推論)
- GPU: RTX 3090 24GB × 6 (Runpod)
- 量子化: AWQ 4-bit
- 品質フィルタ: 思考ステップ検証、ツール呼び出し検出、重複除去、言語一貫性チェック
"""
    return card


def upload_dataset(
    repo_id: str,
    data_dir: str,
    token: str,
    config: dict,
    private: bool = False,
    preview: bool = False,
):
    """データセットをHuggingFace Hubにアップロード"""
    from huggingface_hub import HfApi, create_repo

    data_path = Path(data_dir)

    # アップロード対象ファイルの確認
    jsonl_files = sorted(data_path.glob("*.jsonl"))
    json_files = sorted(data_path.glob("*.json"))
    all_files = jsonl_files + json_files

    if not all_files:
        logger.error(f"アップロード対象ファイルが見つかりません: {data_dir}")
        return

    # ファイル情報を表示
    total_samples = 0
    total_size = 0
    logger.info(f"\n=== アップロード対象 ===")
    logger.info(f"リポジトリ: {repo_id}")
    logger.info(f"ディレクトリ: {data_dir}")
    logger.info(f"公開設定: {'Private' if private else 'Public'}")
    logger.info(f"")

    for f in all_files:
        size_mb = f.stat().st_size / (1024 * 1024)
        total_size += f.stat().st_size
        if f.suffix == ".jsonl":
            lines = count_jsonl_lines(str(f))
            total_samples += lines
            logger.info(f"  {f.name}: {lines:,} 件 ({size_mb:.1f} MB)")
        else:
            logger.info(f"  {f.name}: ({size_mb:.1f} MB)")

    logger.info(f"")
    logger.info(f"合計: {total_samples:,} サンプル, {total_size / (1024**3):.2f} GB")

    if preview:
        logger.info("\n=== プレビューモード: アップロードは実行されません ===")
        return

    # APIクライアント
    api = HfApi(token=token)

    # リポジトリ作成
    logger.info(f"\nリポジトリを作成中: {repo_id}")
    try:
        create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            private=private,
            token=token,
            exist_ok=True,
        )
        logger.info(f"  リポジトリ作成完了 ✓")
    except Exception as e:
        logger.warning(f"  リポジトリ作成: {e}")

    # データセットカードの生成とアップロード
    report_path = data_path / "dataset_report.json"
    report = {}
    if report_path.exists():
        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)

    card_content = create_dataset_card(repo_id, report, config)
    card_path = data_path / "README.md"
    with open(card_path, "w", encoding="utf-8") as f:
        f.write(card_content)
    logger.info(f"  データセットカード生成 ✓")

    # ファイルのアップロード
    logger.info(f"\nファイルをアップロード中...")

    api.upload_folder(
        folder_path=str(data_path),
        repo_id=repo_id,
        repo_type="dataset",
        token=token,
        commit_message=f"Upload reasoning dataset ({total_samples:,} samples)",
    )

    logger.info(f"\n=== アップロード完了! ===")
    logger.info(f"URL: https://huggingface.co/datasets/{repo_id}")


def main():
    parser = argparse.ArgumentParser(description="HuggingFace Hubへデータセットをアップロード")
    parser.add_argument("--repo-id", type=str, required=True,
                        help="リポジトリID (例: username/dataset-name)")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="アップロードするデータディレクトリ（省略時: output/final）")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="設定ファイルのパス")
    parser.add_argument("--token", type=str, default=None,
                        help="HuggingFace APIトークン（省略時は環境変数HF_TOKENを使用）")
    parser.add_argument("--private", action="store_true",
                        help="プライベートリポジトリとして作成")
    parser.add_argument("--preview", action="store_true",
                        help="アップロードせずにファイル一覧のみ表示")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")

    # トークン
    token = args.token or os.environ.get("HF_TOKEN")
    if not token and not args.preview:
        logger.error("HuggingFaceトークンが必要です。--token または環境変数 HF_TOKEN で指定してください。")
        sys.exit(1)

    # 設定
    config = load_config(args.config)
    data_dir = args.data_dir or config["output"]["final_dir"]

    upload_dataset(
        repo_id=args.repo_id,
        data_dir=data_dir,
        token=token,
        config=config,
        private=args.private,
        preview=args.preview,
    )


if __name__ == "__main__":
    main()
