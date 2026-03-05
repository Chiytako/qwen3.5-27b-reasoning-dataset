"""
品質フィルタリングスクリプト
生成されたreasoningデータの品質を検証し、
高品質なデータのみを選別する。
"""

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from collections import defaultdict
from hashlib import md5

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (
    load_config, load_jsonl, save_jsonl, count_jsonl_lines,
    parse_thinking_response, count_thinking_steps,
    extract_tool_calls, detect_language, detect_tool_use_intent
)

logger = logging.getLogger(__name__)


class QualityFilter:
    """reasoningデータの品質フィルタリング"""

    def __init__(self, config: dict):
        self.qf = config["quality_filter"]
        self.stats = defaultdict(int)
        self.seen_hashes = set()

    def check_thinking_quality(self, sample: dict) -> tuple[bool, str]:
        """思考部分の品質チェック"""
        messages = sample.get("messages", [])
        assistant_msg = ""
        for msg in messages:
            if msg.get("role") == "assistant":
                assistant_msg = msg.get("content", "")
                break

        if not assistant_msg:
            return False, "no_assistant_message"

        # thinking部分を抽出
        thinking, response = parse_thinking_response(assistant_msg)

        # 思考なし → 非ツールユーズドメインなら許容
        if not thinking:
            if sample.get("domain", "").startswith("tool_use_"):
                return False, "no_thinking_in_tool_domain"
            # 思考なしでも回答があればOK（一部ドメイン）
            if len(response) > self.qf["min_response_tokens"] * 2:
                return True, "pass_no_thinking"
            return False, "no_thinking"

        # 思考ステップ数チェック
        steps = count_thinking_steps(thinking)
        if steps < self.qf["min_thinking_steps"]:
            return False, f"too_few_steps_{steps}"
        if steps > self.qf["max_thinking_steps"]:
            return False, f"too_many_steps_{steps}"

        # 思考トークン数チェック（概算: 文字数ベース）
        thinking_chars = len(thinking)
        if thinking_chars < self.qf["min_thinking_tokens"]:
            return False, "thinking_too_short"

        return True, "pass"

    def check_response_quality(self, sample: dict) -> tuple[bool, str]:
        """回答部分の品質チェック"""
        messages = sample.get("messages", [])
        assistant_msg = ""
        for msg in messages:
            if msg.get("role") == "assistant":
                assistant_msg = msg.get("content", "")
                break

        _, response = parse_thinking_response(assistant_msg)

        if not response:
            return False, "empty_response"

        # 最小応答長
        if len(response) < self.qf["min_response_tokens"]:
            return False, "response_too_short"

        # 最大応答長
        if len(response) > self.qf["max_response_tokens"] * 4:  # 概算
            return False, "response_too_long"

        # 繰り返し検出
        words = response.split()
        if len(words) > 20:
            # 隣接する単語の繰り返し率
            repeats = sum(1 for i in range(len(words)-1) if words[i] == words[i+1])
            repeat_ratio = repeats / len(words)
            if repeat_ratio > 0.3:
                return False, "excessive_repetition"

        # 未完了の応答検出
        if response.rstrip().endswith("...") and len(response) < 200:
            return False, "incomplete_response"

        return True, "pass"

    def check_tool_use_quality(self, sample: dict) -> tuple[bool, str]:
        """ツール使用の品質チェック"""
        domain = sample.get("domain", "")

        # ツールユーズドメインでない場合はスキップ
        if not domain.startswith("tool_use_") and domain not in ["code_generation", "planning"]:
            return True, "skip_non_tool_domain"

        messages = sample.get("messages", [])
        assistant_msg = ""
        for msg in messages:
            if msg.get("role") == "assistant":
                assistant_msg = msg.get("content", "")
                break

        # ツール呼び出しの検出
        tool_calls = extract_tool_calls(assistant_msg)
        has_intent = detect_tool_use_intent(assistant_msg)

        if self.qf.get("require_tool_calls", False):
            if domain.startswith("tool_use_"):
                if len(tool_calls) < self.qf.get("min_tool_calls", 1) and not has_intent:
                    return False, "no_tool_calls_in_tool_domain"

        return True, "pass"

    def check_language_consistency(self, sample: dict) -> tuple[bool, str]:
        """言語一貫性チェック"""
        if not self.qf.get("language_consistency_check", False):
            return True, "skip"

        expected_lang = sample.get("language", "")
        if not expected_lang:
            return True, "no_expected_lang"

        messages = sample.get("messages", [])
        user_msg = ""
        for msg in messages:
            if msg.get("role") == "user":
                user_msg = msg.get("content", "")
                break

        detected_lang = detect_language(user_msg)

        # 期待言語と検出言語がずれている場合
        if detected_lang != "unknown" and detected_lang != expected_lang:
            return False, f"lang_mismatch_{expected_lang}_vs_{detected_lang}"

        return True, "pass"

    def check_dedup(self, sample: dict) -> tuple[bool, str]:
        """重複チェック（md5ベース）"""
        messages = sample.get("messages", [])
        user_msg = ""
        for msg in messages:
            if msg.get("role") == "user":
                user_msg = msg.get("content", "")
                break

        # ユーザークエリベースで重複チェック
        text_hash = md5(user_msg.encode("utf-8")).hexdigest()
        if text_hash in self.seen_hashes:
            return False, "duplicate"

        self.seen_hashes.add(text_hash)
        return True, "pass"

    def filter_sample(self, sample: dict) -> tuple[bool, str]:
        """全チェックを実行"""
        checks = [
            ("thinking", self.check_thinking_quality),
            ("response", self.check_response_quality),
            ("tool_use", self.check_tool_use_quality),
            ("language", self.check_language_consistency),
            ("dedup", self.check_dedup),
        ]

        for check_name, check_fn in checks:
            passed, reason = check_fn(sample)
            if not passed:
                self.stats[f"rejected_{check_name}_{reason}"] += 1
                return False, f"{check_name}:{reason}"

        self.stats["passed"] += 1
        return True, "pass"

    def print_stats(self):
        """フィルタリング統計を表示"""
        total = sum(self.stats.values())
        passed = self.stats.get("passed", 0)
        rejected = total - passed
        pass_rate = 100 * passed / total if total > 0 else 0

        logger.info("=" * 60)
        logger.info(f"品質フィルタリング結果")
        logger.info(f"  合計: {total}")
        logger.info(f"  通過: {passed} ({pass_rate:.1f}%)")
        logger.info(f"  リジェクト: {rejected} ({100-pass_rate:.1f}%)")
        logger.info("-" * 60)
        for key, count in sorted(self.stats.items(), key=lambda x: -x[1]):
            if key != "passed":
                logger.info(f"  {key}: {count}")
        logger.info("=" * 60)


def filter_files(config: dict, input_dir: str, output_dir: str, rejected_dir: str = None):
    """ディレクトリ内のJSONLファイルをフィルタリング"""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if rejected_dir:
        rejected_path = Path(rejected_dir)
        rejected_path.mkdir(parents=True, exist_ok=True)

    qf = QualityFilter(config)

    jsonl_files = sorted(input_path.glob("*.jsonl"))
    logger.info(f"フィルタリング対象: {len(jsonl_files)} ファイル")

    for filepath in jsonl_files:
        logger.info(f"処理中: {filepath.name}")
        data = load_jsonl(str(filepath))

        passed = []
        rejected = []

        for sample in data:
            is_pass, reason = qf.filter_sample(sample)
            if is_pass:
                passed.append(sample)
            else:
                sample["_reject_reason"] = reason
                rejected.append(sample)

        # 通過データを保存
        if passed:
            save_jsonl(passed, str(output_path / filepath.name))

        # リジェクトデータも保存（オプション）
        if rejected and rejected_dir:
            save_jsonl(rejected, str(rejected_path / filepath.name))

        logger.info(f"  {filepath.name}: {len(passed)} 通過 / {len(rejected)} リジェクト")

    qf.print_stats()
    return qf.stats


def main():
    parser = argparse.ArgumentParser(description="品質フィルタリング")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--input-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--save-rejected", action="store_true", default=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")

    config = load_config(args.config)

    input_dir = args.input_dir or config["output"]["raw_dir"]
    output_dir = args.output_dir or config["output"]["filtered_dir"]
    rejected_dir = str(Path(config["output"]["base_dir"]) / "rejected") if args.save_rejected else None

    filter_files(config, input_dir, output_dir, rejected_dir)


if __name__ == "__main__":
    main()
