"""
共通ユーティリティモジュール
- thinking tagのパース
- JSONL I/O
- ツール呼び出しの検出・解析
- 進捗管理・チェックポイント
"""

import json
import re
import os
import hashlib
import logging
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field, asdict

logger = logging.getLogger(__name__)


# =============================================================================
# データクラス
# =============================================================================

@dataclass
class ToolCall:
    """ツール呼び出しを表現するデータクラス"""
    name: str
    arguments: dict
    result: Optional[str] = None


@dataclass
class ReasoningSample:
    """reasoning データセットの1サンプル"""
    id: str
    domain: str
    language: str
    system_prompt: str
    user_query: str
    thinking: str
    response: str
    tool_calls: list = field(default_factory=list)
    tool_definitions: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    def to_training_format(self) -> dict:
        """SFT学習用フォーマットに変換"""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.user_query},
        ]

        # thinking + response を結合
        assistant_content = ""
        if self.thinking:
            assistant_content += f"<think>\n{self.thinking}\n</think>\n\n"
        assistant_content += self.response

        messages.append({"role": "assistant", "content": assistant_content})

        result = {
            "id": self.id,
            "messages": messages,
            "domain": self.domain,
            "language": self.language,
        }

        if self.tool_definitions:
            result["tools"] = self.tool_definitions
        if self.tool_calls:
            result["tool_calls"] = [asdict(tc) if isinstance(tc, ToolCall) else tc for tc in self.tool_calls]

        return result


# =============================================================================
# Thinking Tag パーサー
# =============================================================================

def parse_thinking_response(text: str) -> tuple[str, str]:
    """
    <think>...</think> タグを含む応答をパースし、
    (thinking部分, response部分) のタプルを返す。
    """
    think_pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL)
    match = think_pattern.search(text)

    if match:
        thinking = match.group(1).strip()
        # <think>タグの後の部分を応答とする
        response = text[match.end():].strip()
        return thinking, response
    else:
        # <think>タグがない場合、全体を応答とする
        return "", text.strip()


def count_thinking_steps(thinking: str) -> int:
    """思考ステップ数をカウント（改行区切りの段落数を基準）"""
    if not thinking:
        return 0

    # 空行で区切られた段落をカウント
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', thinking) if p.strip()]

    # 段落が少ない場合は文単位でカウント
    if len(paragraphs) <= 1:
        sentences = [s.strip() for s in re.split(r'[。.！!？?\n]', thinking) if s.strip()]
        return max(len(sentences), 1)

    return len(paragraphs)


# =============================================================================
# ツール呼び出し検出
# =============================================================================

def extract_tool_calls(text: str) -> list[dict]:
    """テキストからツール呼び出しパターンを抽出"""
    tool_calls = []

    # パターン1: JSON形式の関数呼び出し
    # {"name": "func_name", "arguments": {...}}
    json_pattern = re.compile(
        r'\{[^{}]*"name"\s*:\s*"([^"]+)"[^{}]*"arguments"\s*:\s*(\{[^{}]*\})[^{}]*\}',
        re.DOTALL
    )
    for match in json_pattern.finditer(text):
        try:
            args = json.loads(match.group(2))
            tool_calls.append({
                "name": match.group(1),
                "arguments": args
            })
        except json.JSONDecodeError:
            tool_calls.append({
                "name": match.group(1),
                "arguments": match.group(2)
            })

    # パターン2: function_call タグ
    # <function_call>func_name(arg1=val1, arg2=val2)</function_call>
    fc_pattern = re.compile(
        r'<function_call>\s*(\w+)\((.*?)\)\s*</function_call>',
        re.DOTALL
    )
    for match in fc_pattern.finditer(text):
        func_name = match.group(1)
        args_str = match.group(2)
        tool_calls.append({
            "name": func_name,
            "arguments": args_str
        })

    # パターン3: ツール使用ブロック
    # ```tool_use\n{"name": ..., "input": {...}}\n```
    block_pattern = re.compile(
        r'```(?:tool_use|tool_call|function)\s*\n(.*?)\n```',
        re.DOTALL
    )
    for match in block_pattern.finditer(text):
        try:
            data = json.loads(match.group(1))
            if isinstance(data, dict) and "name" in data:
                tool_calls.append({
                    "name": data["name"],
                    "arguments": data.get("arguments", data.get("input", data.get("parameters", {})))
                })
        except json.JSONDecodeError:
            pass

    return tool_calls


def detect_tool_use_intent(text: str) -> bool:
    """テキストにツール使用の意図があるかを検出"""
    indicators = [
        "関数を呼び出", "ツールを使", "APIを呼び出", "検索し",
        "ファイルを読", "コードを実行", "データベースに",
        "function_call", "tool_use", "tool_call",
        "I'll use", "I need to call", "Let me search",
        "Let me execute", "I'll query",
    ]
    text_lower = text.lower()
    return any(ind.lower() in text_lower for ind in indicators)


# =============================================================================
# JSONL I/O
# =============================================================================

def save_jsonl(data: list[dict], filepath: str, mode: str = "w"):
    """JSONL形式で保存"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, mode, encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def load_jsonl(filepath: str) -> list[dict]:
    """JSONL形式から読み込み"""
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def append_jsonl(item: dict, filepath: str):
    """JSONL形式で1行追記"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


def count_jsonl_lines(filepath: str) -> int:
    """JSONLファイルの行数をカウント"""
    if not os.path.exists(filepath):
        return 0
    with open(filepath, "r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


# =============================================================================
# チェックポイント管理
# =============================================================================

class CheckpointManager:
    """進捗チェックポイントの管理"""

    def __init__(self, checkpoint_dir: str, worker_id: int):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.worker_id = worker_id
        self.checkpoint_file = self.checkpoint_dir / f"worker_{worker_id}.json"

    def save(self, state: dict):
        """チェックポイントを保存"""
        state["worker_id"] = self.worker_id
        with open(self.checkpoint_file, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
        logger.info(f"Worker {self.worker_id}: チェックポイント保存 - {state.get('completed', 0)} 件完了")

    def load(self) -> Optional[dict]:
        """チェックポイントがあれば読み込み"""
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file, "r", encoding="utf-8") as f:
                state = json.load(f)
            logger.info(f"Worker {self.worker_id}: チェックポイント復元 - {state.get('completed', 0)} 件から再開")
            return state
        return None

    def remove(self):
        """チェックポイントを削除"""
        if self.checkpoint_file.exists():
            self.checkpoint_file.unlink()


# =============================================================================
# ハッシュ・重複検出
# =============================================================================

def compute_text_hash(text: str) -> str:
    """テキストのハッシュ値を計算"""
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def detect_language(text: str) -> str:
    """簡易言語検出（日本語/英語）"""
    # 日本語文字（ひらがな・カタカナ・漢字）の割合で判定
    ja_pattern = re.compile(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]')
    ja_chars = len(ja_pattern.findall(text))
    total_chars = len(text.replace(" ", "").replace("\n", ""))

    if total_chars == 0:
        return "unknown"

    ja_ratio = ja_chars / total_chars
    if ja_ratio > 0.1:
        return "ja"
    else:
        return "en"


# =============================================================================
# ID生成
# =============================================================================

def generate_sample_id(worker_id: int, index: int, domain: str) -> str:
    """一意なサンプルIDを生成"""
    return f"qwen35r_{domain}_{worker_id:02d}_{index:07d}"


# =============================================================================
# 設定読み込み
# =============================================================================

def load_config(config_path: str = "config.yaml") -> dict:
    """YAML設定ファイルを読み込み"""
    import yaml
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
