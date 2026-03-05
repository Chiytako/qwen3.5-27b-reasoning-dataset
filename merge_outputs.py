"""
出力ファイルのマージスクリプト
全ワーカーの出力を単一のデータセットに統合する。
"""

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import load_config, load_jsonl, save_jsonl, count_jsonl_lines

logger = logging.getLogger(__name__)


def merge_worker_outputs(config: dict, input_dir: str, output_dir: str):
    """全ワーカーの出力をマージして最終データセットを生成"""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 全ファイルからデータを読み込み
    all_data = []
    stats = defaultdict(int)

    jsonl_files = sorted(input_path.glob("*.jsonl"))
    logger.info(f"マージ対象: {len(jsonl_files)} ファイル")

    for filepath in jsonl_files:
        data = load_jsonl(str(filepath))
        logger.info(f"  {filepath.name}: {len(data)} 件")
        all_data.extend(data)

        for item in data:
            stats[f"domain_{item.get('domain', 'unknown')}"] += 1
            stats[f"lang_{item.get('language', 'unknown')}"] += 1

    logger.info(f"合計: {len(all_data)} 件")

    # シャッフル
    random.shuffle(all_data)

    # 分割して保存
    split_size = config["output"]["split_size"]
    num_splits = (len(all_data) + split_size - 1) // split_size

    for i in range(num_splits):
        start = i * split_size
        end = min((i + 1) * split_size, len(all_data))
        chunk = all_data[start:end]

        output_file = output_path / f"reasoning_dataset_{i:04d}.jsonl"
        save_jsonl(chunk, str(output_file))
        logger.info(f"  保存: {output_file.name} ({len(chunk)} 件)")

    # train/test分割
    test_ratio = 0.02  # 2%をテスト用に
    test_size = int(len(all_data) * test_ratio)
    train_data = all_data[test_size:]
    test_data = all_data[:test_size]

    save_jsonl(train_data, str(output_path / "train.jsonl"))
    save_jsonl(test_data, str(output_path / "test.jsonl"))
    logger.info(f"Train: {len(train_data)} 件, Test: {len(test_data)} 件")

    # 統計レポート
    report = {
        "total_samples": len(all_data),
        "train_samples": len(train_data),
        "test_samples": len(test_data),
        "num_files": num_splits,
        "domain_distribution": {k: v for k, v in stats.items() if k.startswith("domain_")},
        "language_distribution": {k: v for k, v in stats.items() if k.startswith("lang_")},
    }

    report_path = output_path / "dataset_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info(f"レポート保存: {report_path}")

    # ドメイン分布を表示
    logger.info("\n=== ドメイン分布 ===")
    for key, count in sorted(stats.items()):
        pct = 100 * count / len(all_data) if all_data else 0
        logger.info(f"  {key}: {count} ({pct:.1f}%)")

    return report


def main():
    parser = argparse.ArgumentParser(description="出力マージ")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--input-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")

    config = load_config(args.config)
    input_dir = args.input_dir or config["output"]["filtered_dir"]
    output_dir = args.output_dir or config["output"]["final_dir"]

    merge_worker_outputs(config, input_dir, output_dir)


if __name__ == "__main__":
    main()
