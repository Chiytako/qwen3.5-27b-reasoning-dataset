"""
システムプロンプトとツール定義テンプレート
ツールユーズ/function calling に特化したreasoning生成用
"""

import json
import random
from typing import Optional

# =============================================================================
# ツール定義テンプレート（JSON Schema形式）
# =============================================================================

TOOL_DEFINITIONS = {
    # --- Web検索・情報取得 ---
    "search_web": {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "ウェブ上で情報を検索し、関連する結果を返します。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "検索クエリ"},
                    "num_results": {"type": "integer", "description": "返す結果の数", "default": 5},
                    "language": {"type": "string", "description": "検索言語", "enum": ["ja", "en"]}
                },
                "required": ["query"]
            }
        }
    },
    "fetch_url": {
        "type": "function",
        "function": {
            "name": "fetch_url",
            "description": "指定されたURLのコンテンツを取得します。",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "取得するURL"},
                    "format": {"type": "string", "description": "出力形式", "enum": ["text", "html", "markdown"]}
                },
                "required": ["url"]
            }
        }
    },
    "search_news": {
        "type": "function",
        "function": {
            "name": "search_news",
            "description": "最新のニュース記事を検索します。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "ニュース検索クエリ"},
                    "date_range": {"type": "string", "description": "期間", "enum": ["today", "week", "month"]},
                    "category": {"type": "string", "description": "カテゴリ"}
                },
                "required": ["query"]
            }
        }
    },

    # --- ファイル操作 ---
    "read_file": {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "ファイルの内容を読み取ります。",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "ファイルパス"},
                    "encoding": {"type": "string", "description": "文字エンコーディング", "default": "utf-8"}
                },
                "required": ["path"]
            }
        }
    },
    "write_file": {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "ファイルに内容を書き込みます。",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "ファイルパス"},
                    "content": {"type": "string", "description": "書き込む内容"},
                    "mode": {"type": "string", "description": "書き込みモード", "enum": ["overwrite", "append"]}
                },
                "required": ["path", "content"]
            }
        }
    },
    "list_directory": {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "ディレクトリ内のファイルとフォルダを一覧表示します。",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "ディレクトリパス"},
                    "recursive": {"type": "boolean", "description": "再帰的に探索するか", "default": False},
                    "pattern": {"type": "string", "description": "ファイルパターン（glob形式）"}
                },
                "required": ["path"]
            }
        }
    },

    # --- コード実行 ---
    "run_python": {
        "type": "function",
        "function": {
            "name": "run_python",
            "description": "Pythonコードを実行し、結果を返します。",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "実行するPythonコード"},
                    "timeout": {"type": "integer", "description": "タイムアウト（秒）", "default": 30}
                },
                "required": ["code"]
            }
        }
    },
    "run_bash": {
        "type": "function",
        "function": {
            "name": "run_bash",
            "description": "Bashコマンドを実行します。",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "実行するコマンド"},
                    "working_dir": {"type": "string", "description": "作業ディレクトリ"}
                },
                "required": ["command"]
            }
        }
    },

    # --- データ処理 ---
    "query_database": {
        "type": "function",
        "function": {
            "name": "query_database",
            "description": "SQLクエリを実行してデータベースからデータを取得します。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "SQLクエリ"},
                    "database": {"type": "string", "description": "データベース名"},
                    "limit": {"type": "integer", "description": "結果の最大件数", "default": 100}
                },
                "required": ["query", "database"]
            }
        }
    },
    "transform_data": {
        "type": "function",
        "function": {
            "name": "transform_data",
            "description": "データを指定された形式に変換します。",
            "parameters": {
                "type": "object",
                "properties": {
                    "input_data": {"type": "string", "description": "入力データ（JSON文字列）"},
                    "operation": {"type": "string", "description": "変換操作", "enum": ["filter", "sort", "aggregate", "pivot", "join"]},
                    "params": {"type": "object", "description": "操作パラメータ"}
                },
                "required": ["input_data", "operation"]
            }
        }
    },

    # --- コミュニケーション ---
    "send_email": {
        "type": "function",
        "function": {
            "name": "send_email",
            "description": "メールを送信します。",
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {"type": "string", "description": "宛先メールアドレス"},
                    "subject": {"type": "string", "description": "件名"},
                    "body": {"type": "string", "description": "本文"},
                    "attachments": {"type": "array", "items": {"type": "string"}, "description": "添付ファイルパス"}
                },
                "required": ["to", "subject", "body"]
            }
        }
    },
    "translate_text": {
        "type": "function",
        "function": {
            "name": "translate_text",
            "description": "テキストを翻訳します。",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "翻訳するテキスト"},
                    "source_lang": {"type": "string", "description": "原文の言語"},
                    "target_lang": {"type": "string", "description": "翻訳先の言語"}
                },
                "required": ["text", "target_lang"]
            }
        }
    },

    # --- 数学・計算 ---
    "calculate": {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "数式を計算します。",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "計算する数式"},
                    "precision": {"type": "integer", "description": "小数点以下の桁数", "default": 10}
                },
                "required": ["expression"]
            }
        }
    },
    "solve_equation": {
        "type": "function",
        "function": {
            "name": "solve_equation",
            "description": "方程式を解きます。",
            "parameters": {
                "type": "object",
                "properties": {
                    "equation": {"type": "string", "description": "方程式（例: '2x + 3 = 7'）"},
                    "variable": {"type": "string", "description": "解く変数", "default": "x"}
                },
                "required": ["equation"]
            }
        }
    },

    # --- 画像処理 ---
    "generate_image": {
        "type": "function",
        "function": {
            "name": "generate_image",
            "description": "テキストプロンプトから画像を生成します。",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "画像生成プロンプト"},
                    "size": {"type": "string", "description": "画像サイズ", "enum": ["256x256", "512x512", "1024x1024"]},
                    "style": {"type": "string", "description": "画像スタイル"}
                },
                "required": ["prompt"]
            }
        }
    },

    # --- システム管理 ---
    "check_status": {
        "type": "function",
        "function": {
            "name": "check_status",
            "description": "サービスやシステムのステータスを確認します。",
            "parameters": {
                "type": "object",
                "properties": {
                    "service": {"type": "string", "description": "サービス名"},
                    "detailed": {"type": "boolean", "description": "詳細情報を含めるか", "default": False}
                },
                "required": ["service"]
            }
        }
    },
    "monitor_logs": {
        "type": "function",
        "function": {
            "name": "monitor_logs",
            "description": "システムログを監視・取得します。",
            "parameters": {
                "type": "object",
                "properties": {
                    "service": {"type": "string", "description": "サービス名"},
                    "level": {"type": "string", "description": "ログレベル", "enum": ["debug", "info", "warning", "error"]},
                    "lines": {"type": "integer", "description": "取得する行数", "default": 50},
                    "since": {"type": "string", "description": "開始時刻（ISO 8601形式）"}
                },
                "required": ["service"]
            }
        }
    },

    # --- API連携 ---
    "http_request": {
        "type": "function",
        "function": {
            "name": "http_request",
            "description": "HTTP リクエストを送信します。",
            "parameters": {
                "type": "object",
                "properties": {
                    "method": {"type": "string", "description": "HTTPメソッド", "enum": ["GET", "POST", "PUT", "DELETE", "PATCH"]},
                    "url": {"type": "string", "description": "リクエストURL"},
                    "headers": {"type": "object", "description": "リクエストヘッダー"},
                    "body": {"type": "string", "description": "リクエストボディ（JSON文字列）"},
                    "timeout": {"type": "integer", "description": "タイムアウト（秒）", "default": 30}
                },
                "required": ["method", "url"]
            }
        }
    },
    "create_calendar_event": {
        "type": "function",
        "function": {
            "name": "create_calendar_event",
            "description": "カレンダーにイベントを作成します。",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "イベントタイトル"},
                    "start_time": {"type": "string", "description": "開始時刻（ISO 8601形式）"},
                    "end_time": {"type": "string", "description": "終了時刻（ISO 8601形式）"},
                    "description": {"type": "string", "description": "イベントの説明"},
                    "attendees": {"type": "array", "items": {"type": "string"}, "description": "参加者のメールアドレス"}
                },
                "required": ["title", "start_time", "end_time"]
            }
        }
    },

    # --- コーディングエージェント (Opencode / Claude Code 風) ---
    "edit_file": {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "ファイルの指定箇所を編集します。既存のコードを検索して新しいコードに置換します。",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "編集するファイルの絶対パス"},
                    "old_text": {"type": "string", "description": "置換対象の既存テキスト（完全一致）"},
                    "new_text": {"type": "string", "description": "置換後の新しいテキスト"},
                    "description": {"type": "string", "description": "この編集の説明"}
                },
                "required": ["path", "old_text", "new_text"]
            }
        }
    },
    "grep_search": {
        "type": "function",
        "function": {
            "name": "grep_search",
            "description": "ripgrepを使用してファイルやディレクトリ内でパターンを検索します。正規表現対応。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "検索パターン（正規表現可）"},
                    "search_path": {"type": "string", "description": "検索対象のパス"},
                    "includes": {"type": "array", "items": {"type": "string"}, "description": "ファイルフィルタ（例: '*.py'）"},
                    "case_insensitive": {"type": "boolean", "description": "大文字小文字を無視するか", "default": False}
                },
                "required": ["query", "search_path"]
            }
        }
    },
    "find_files": {
        "type": "function",
        "function": {
            "name": "find_files",
            "description": "ファイル名やパターンでファイルを検索します。fdコマンド相当。",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "検索パターン（glob形式）"},
                    "search_directory": {"type": "string", "description": "検索ディレクトリ"},
                    "extensions": {"type": "array", "items": {"type": "string"}, "description": "ファイル拡張子フィルタ"},
                    "max_depth": {"type": "integer", "description": "最大探索深度"}
                },
                "required": ["pattern", "search_directory"]
            }
        }
    },
    "run_command": {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "ターミナルでコマンドを実行します。ビルド、テスト、Gitなど開発系コマンドに使用します。",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "実行するコマンド"},
                    "cwd": {"type": "string", "description": "作業ディレクトリ"},
                    "timeout": {"type": "integer", "description": "タイムアウト（秒）", "default": 60}
                },
                "required": ["command"]
            }
        }
    },
    "view_code_definition": {
        "type": "function",
        "function": {
            "name": "view_code_definition",
            "description": "ファイル内の関数やクラスの定義を表示します。コードのアウトラインを確認するのに使用します。",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "ファイルの絶対パス"},
                    "symbol_name": {"type": "string", "description": "関数名またはクラス名（オプション）"},
                    "show_outline": {"type": "boolean", "description": "ファイルのアウトラインを表示するか", "default": False}
                },
                "required": ["file_path"]
            }
        }
    },
    "codebase_search": {
        "type": "function",
        "function": {
            "name": "codebase_search",
            "description": "セマンティック検索でコードベース内から関連するコードやファイルを検索します。自然言語クエリに対応。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "検索クエリ（自然言語）"},
                    "scope": {"type": "string", "description": "検索範囲のディレクトリ"},
                    "file_types": {"type": "array", "items": {"type": "string"}, "description": "対象ファイルタイプ"}
                },
                "required": ["query"]
            }
        }
    },
    "browser_action": {
        "type": "function",
        "function": {
            "name": "browser_action",
            "description": "ブラウザを制御してWebページを操作します。開発中のアプリのテストやデバッグに使用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "description": "アクション", "enum": ["navigate", "click", "type", "screenshot", "scroll", "evaluate_js"]},
                    "url": {"type": "string", "description": "遷移先URL（navigateアクション用）"},
                    "selector": {"type": "string", "description": "CSS セレクタ（click/typeアクション用）"},
                    "text": {"type": "string", "description": "入力テキスト（typeアクション用）"},
                    "script": {"type": "string", "description": "実行するJavaScript（evaluate_jsアクション用）"}
                },
                "required": ["action"]
            }
        }
    },
    "ask_user": {
        "type": "function",
        "function": {
            "name": "ask_user",
            "description": "ユーザーに質問や確認を行います。判断が必要な場合や追加情報が必要な場合に使用します。",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "ユーザーへの質問"},
                    "options": {"type": "array", "items": {"type": "string"}, "description": "選択肢のリスト（オプション）"}
                },
                "required": ["question"]
            }
        }
    },
    "task_complete": {
        "type": "function",
        "function": {
            "name": "task_complete",
            "description": "タスクの完了を報告します。実行した内容のサマリーを提供します。",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string", "description": "タスク完了サマリー"},
                    "files_changed": {"type": "array", "items": {"type": "string"}, "description": "変更したファイルのパス一覧"}
                },
                "required": ["summary"]
            }
        }
    },
}


# =============================================================================
# システムプロンプト
# =============================================================================

SYSTEM_PROMPT_TOOL_USE_JA = """あなたは高度なAIアシスタントです。ユーザーの要求を達成するために、利用可能なツール（関数）を適切に選択し、呼び出すことができます。

## 重要な指示

1. **まず考えてください**: ユーザーの要求を分析し、どのツールをどの順序で使うべきか、ステップバイステップで思考してください。
2. **適切なツールを選択**: 利用可能なツールの中から最適なものを選んでください。
3. **パラメータを正確に**: ツール呼び出し時は正確なパラメータを指定してください。
4. **結果を解釈**: ツールの実行結果を解釈し、必要に応じて追加のツール呼び出しを行ってください。
5. **エラー処理**: ツール呼び出しが失敗した場合は、代替手段を考えてください。

## ツール呼び出し形式

ツールを呼び出す際は、以下のJSON形式を使用してください：

```tool_call
{{"name": "ツール名", "arguments": {{"パラメータ名": "値"}}}}
```

## 利用可能なツール

{tools_description}
"""

SYSTEM_PROMPT_TOOL_USE_EN = """You are an advanced AI assistant capable of selecting and calling appropriate tools (functions) to fulfill user requests.

## Important Instructions

1. **Think first**: Analyze the user's request and think step-by-step about which tools to use and in what order.
2. **Select appropriate tools**: Choose the most suitable tools from available ones.
3. **Accurate parameters**: Specify precise parameters when making tool calls.
4. **Interpret results**: Interpret tool execution results and make additional tool calls if necessary.
5. **Error handling**: If a tool call fails, consider alternative approaches.

## Tool Call Format

When calling a tool, use the following JSON format:

```tool_call
{{"name": "tool_name", "arguments": {{"param_name": "value"}}}}
```

## Available Tools

{tools_description}
"""

SYSTEM_PROMPT_MULTI_STEP_JA = """あなたは複雑なタスクを効率的に解決するAIアシスタントです。以下のツールを組み合わせて、ユーザーの要求を段階的に達成してください。

## 作業手順

1. タスクを小さなサブタスクに分解する
2. 各サブタスクに最適なツールを選択する
3. ツールを順番に実行し、中間結果を次のステップに活用する
4. 最終結果をユーザーに分かりやすく報告する

## エラー発生時

- エラーメッセージを確認し、原因を特定する
- パラメータを修正して再試行する
- 代替ツールがあれば使用する
- ユーザーに状況を報告する

## 利用可能なツール

{tools_description}
"""

SYSTEM_PROMPT_MULTI_STEP_EN = """You are an AI assistant that efficiently solves complex tasks. Combine the following tools to achieve user requests step by step.

## Workflow

1. Break down the task into smaller subtasks
2. Select the optimal tool for each subtask
3. Execute tools in sequence, leveraging intermediate results for subsequent steps
4. Report the final results clearly to the user

## On Error

- Check the error message and identify the cause
- Modify parameters and retry
- Use alternative tools if available
- Report the situation to the user

## Available Tools

{tools_description}
"""

SYSTEM_PROMPT_ERROR_HANDLING_JA = """あなたはツールを使用する際にエラー処理を適切に行えるAIアシスタントです。

## エラー処理の原則

1. **予防**: ツール呼び出し前に入力を検証する
2. **検出**: ツールの応答を確認し、エラーを検出する
3. **回復**: エラー発生時に代替案を実行する
4. **報告**: ユーザーに何が起きたか、どう対処したかを報告する

## よくあるエラーパターン

- ファイルが存在しない → パスを確認、リスト表示して正しいパスを特定
- API呼び出しのタイムアウト → リトライ、またはパラメータを変更
- データ形式の不一致 → 変換処理を挟む
- 権限エラー → 別のアプローチを検討

## 利用可能なツール

{tools_description}
"""

SYSTEM_PROMPT_REASONING_JA = """あなたは深い推論能力を持つAIアシスタントです。問題を分析し、論理的に思考してステップバイステップで解決策を導き出してください。

## 推論プロセス

1. 問題の理解: 与えられた情報を整理し、求められていることを明確にする
2. 仮説の設定: 考えられる解法や仮説を立てる
3. 検証: 各仮説を論理的に検証する
4. 結論: 最も妥当な結論を導き出し、理由を説明する
"""

SYSTEM_PROMPT_REASONING_EN = """You are an AI assistant with deep reasoning capabilities. Analyze problems and derive solutions through logical, step-by-step thinking.

## Reasoning Process

1. Understanding: Organize given information and clarify what is being asked
2. Hypothesis: Form possible solutions or hypotheses
3. Verification: Logically verify each hypothesis
4. Conclusion: Derive the most reasonable conclusion and explain the reasoning
"""

# --- Coding Agent (Opencode / Claude Code 風) ---

SYSTEM_PROMPT_CODING_AGENT_JA = """あなたはプロのソフトウェアエンジニアとして動作するAIコーディングエージェントです。
ユーザーのコーディングタスクを、利用可能なツールを駆使して自律的に解決してください。

## ワークフロー

1. **理解**: ユーザーの要求を正確に理解する
2. **調査**: codebase_searchやgrep_searchでコードベースを調査し、関連するファイルや実装パターンを把握する
3. **計画**: 変更すべきファイルと手順を計画する
4. **実装**: edit_fileやwrite_fileでコードを変更・作成する
5. **検証**: run_commandでテストやビルドを実行し、変更が正しく動作することを確認する
6. **報告**: task_completeで実施内容をまとめる

## 重要なルール

- **既存コードを尊重**: 既存のコーディングスタイル、パターン、命名規則に従う
- **最小限の変更**: 必要最小限の変更に留め、無関係な箇所を触らない
- **テスト実行**: コード変更後は必ずテストやビルドを実行して動作確認する
- **段階的に進める**: 大きなタスクは小さなステップに分割して進める
- **エラー対応**: ビルドやテストのエラーが出たら原因を分析して修正する

## ツール呼び出し形式

ツールを呼び出す際は、以下のJSON形式を使用してください：

```tool_call
{{"name": "ツール名", "arguments": {{"パラメータ名": "値"}}}}
```

## 利用可能なツール

{tools_description}
"""

SYSTEM_PROMPT_CODING_AGENT_EN = """You are an AI coding agent operating as a professional software engineer.
Autonomously solve the user's coding tasks using the available tools.

## Workflow

1. **Understand**: Accurately understand the user's request
2. **Investigate**: Use codebase_search and grep_search to examine the codebase and understand relevant files and implementation patterns
3. **Plan**: Plan which files to modify and in what order
4. **Implement**: Use edit_file and write_file to modify or create code
5. **Verify**: Use run_command to execute tests and builds to confirm changes work correctly
6. **Report**: Use task_complete to summarize what was done

## Important Rules

- **Respect existing code**: Follow existing coding styles, patterns, and naming conventions
- **Minimal changes**: Make only necessary changes; don't touch unrelated code
- **Run tests**: Always run tests or builds after code changes to verify correctness
- **Work incrementally**: Break large tasks into smaller steps
- **Handle errors**: Analyze and fix any build or test errors that arise

## Tool Call Format

When calling a tool, use the following JSON format:

```tool_call
{{"name": "tool_name", "arguments": {{"param_name": "value"}}}}
```

## Available Tools

{tools_description}
"""


# =============================================================================
# プロンプト選択ロジック
# =============================================================================

def get_system_prompt(domain: str, language: str) -> str:
    """ドメインと言語に基づいてシステムプロンプトを選択"""
    prompts = {
        ("tool_use_api_calling", "ja"): SYSTEM_PROMPT_TOOL_USE_JA,
        ("tool_use_api_calling", "en"): SYSTEM_PROMPT_TOOL_USE_EN,
        ("tool_use_multi_step", "ja"): SYSTEM_PROMPT_MULTI_STEP_JA,
        ("tool_use_multi_step", "en"): SYSTEM_PROMPT_MULTI_STEP_EN,
        ("tool_use_error_handling", "ja"): SYSTEM_PROMPT_ERROR_HANDLING_JA,
        ("tool_use_error_handling", "en"): SYSTEM_PROMPT_TOOL_USE_EN,
        ("coding_agent", "ja"): SYSTEM_PROMPT_CODING_AGENT_JA,
        ("coding_agent", "en"): SYSTEM_PROMPT_CODING_AGENT_EN,
        ("code_generation", "ja"): SYSTEM_PROMPT_CODING_AGENT_JA,
        ("code_generation", "en"): SYSTEM_PROMPT_CODING_AGENT_EN,
        ("math_reasoning", "ja"): SYSTEM_PROMPT_REASONING_JA,
        ("math_reasoning", "en"): SYSTEM_PROMPT_REASONING_EN,
        ("logic_reasoning", "ja"): SYSTEM_PROMPT_REASONING_JA,
        ("logic_reasoning", "en"): SYSTEM_PROMPT_REASONING_EN,
        ("science", "ja"): SYSTEM_PROMPT_REASONING_JA,
        ("science", "en"): SYSTEM_PROMPT_REASONING_EN,
        ("planning", "ja"): SYSTEM_PROMPT_MULTI_STEP_JA,
        ("planning", "en"): SYSTEM_PROMPT_MULTI_STEP_EN,
        ("general_knowledge", "ja"): SYSTEM_PROMPT_REASONING_JA,
        ("general_knowledge", "en"): SYSTEM_PROMPT_REASONING_EN,
    }
    return prompts.get((domain, language), SYSTEM_PROMPT_TOOL_USE_JA)


def select_tools_for_task(domain: str, num_tools: int = 5) -> list[dict]:
    """タスクドメインに基づいてツールをランダムに選択"""
    domain_tool_mapping = {
        "tool_use_api_calling": [
            "search_web", "fetch_url", "search_news", "http_request",
            "translate_text", "send_email", "query_database"
        ],
        "tool_use_multi_step": list(TOOL_DEFINITIONS.keys()),
        "tool_use_error_handling": [
            "read_file", "write_file", "list_directory", "query_database",
            "http_request", "run_python", "run_bash"
        ],
        "coding_agent": [
            "read_file", "write_file", "edit_file", "list_directory",
            "grep_search", "find_files", "run_command", "view_code_definition",
            "codebase_search", "browser_action", "ask_user", "task_complete"
        ],
        "code_generation": [
            "run_command", "read_file", "write_file", "edit_file",
            "grep_search", "find_files", "codebase_search",
            "view_code_definition", "run_python"
        ],
        "planning": [
            "search_web", "create_calendar_event", "send_email",
            "read_file", "write_file", "query_database"
        ],
    }

    available_tools = domain_tool_mapping.get(domain, list(TOOL_DEFINITIONS.keys()))
    selected_keys = random.sample(available_tools, min(num_tools, len(available_tools)))
    return [TOOL_DEFINITIONS[k] for k in selected_keys]


def format_tools_description(tools: list[dict]) -> str:
    """ツール一覧を読みやすい文字列に変換"""
    lines = []
    for tool in tools:
        func = tool["function"]
        lines.append(f"### {func['name']}")
        lines.append(f"説明: {func['description']}")
        params = func.get("parameters", {}).get("properties", {})
        if params:
            lines.append("パラメータ:")
            required = func.get("parameters", {}).get("required", [])
            for pname, pinfo in params.items():
                req = "（必須）" if pname in required else "（任意）"
                lines.append(f"  - {pname} ({pinfo.get('type', 'any')}): {pinfo.get('description', '')} {req}")
        lines.append("")
    return "\n".join(lines)


def build_full_prompt(domain: str, language: str, tools: Optional[list[dict]] = None) -> str:
    """完全なシステムプロンプトを構築（ツール定義込み）"""
    if tools is None:
        tools = select_tools_for_task(domain)

    system_prompt = get_system_prompt(domain, language)
    tools_desc = format_tools_description(tools)
    return system_prompt.format(tools_description=tools_desc)
