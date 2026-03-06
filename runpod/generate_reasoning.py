"""
Qwen3.5-27B Reasoning データセット生成 - メインスクリプト
Runpod/Vast.ai 並列処理対応

使い方:
  # 全ワーカーを自動起動（config.yamlのnum_workers数に基づく）
  python generate_reasoning.py --config ../config.yaml --auto-parallel

  # 特定ワーカーのみ実行
  python generate_reasoning.py --config ../config.yaml --worker-id 0 --gpu-id 0
"""

import argparse
import json
import logging
import os
import random
import sys
import time
import subprocess
import signal
from pathlib import Path
from typing import Optional

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils import (
    load_config, parse_thinking_response, count_thinking_steps,
    extract_tool_calls, detect_tool_use_intent, save_jsonl, load_jsonl,
    append_jsonl, count_jsonl_lines, CheckpointManager,
    generate_sample_id, detect_language, ReasoningSample
)
from prompts.system_prompts import (
    build_full_prompt, select_tools_for_task, TOOL_DEFINITIONS,
    format_tools_description
)

logger = logging.getLogger(__name__)


# =============================================================================
# プロンプト拡張 (シード問題からバリエーション生成)
# =============================================================================

class PromptExpander:
    """シード問題からバリエーションを生成してプロンプト数を拡大"""

    VARIATION_TEMPLATES_JA = [
        "以下のタスクをより詳細に実行してください: {base}",
        "{base} ただし、エラーが発生した場合の対処法も含めてください。",
        "{base} 結果は表形式でまとめてください。",
        "{base} 可能な限り効率的な方法で実行してください。",
        "{base} セキュリティ面も考慮してください。",
        "初心者でも理解できるように、{base}",
        "{base} 各ステップの所要時間の目安も示してください。",
        "{base} 代替手段がある場合はそれも提示してください。",
        "以下のシナリオを考えてください: {base} メモリ制約がある環境で実行する場合、どのような工夫が必要ですか？",
        "{base} 途中で障害が発生した場合のロールバック手順も含めてください。",
    ]

    VARIATION_TEMPLATES_EN = [
        "Please execute the following task in detail: {base}",
        "{base} Also include error handling strategies.",
        "{base} Present the results in a structured format.",
        "{base} Optimize for performance and efficiency.",
        "{base} Consider security implications as well.",
        "Explain step by step: {base}",
        "{base} Include estimated time for each step.",
        "{base} Suggest alternatives if the primary approach fails.",
        "Consider this scenario: {base} How would you handle this with limited resources?",
        "{base} Include a rollback plan in case something goes wrong.",
    ]

    CONTEXT_INJECTIONS_JA = [
        "あなたはスタートアップ企業のCTOとして",
        "大量のデータ（100GB以上）を扱う場合を想定して",
        "チームメンバー5人と協力して",
        "24時間以内に完了する必要がある状況で",
        "オンコール対応中に深夜3時に",
        "新人エンジニアに教えるために",
        "セキュリティ監査の一環として",
        "パフォーマンスが重要なプロダクション環境で",
        "コスト削減のために",
        "障害復旧の緊急対応として",
    ]

    CONTEXT_INJECTIONS_EN = [
        "As a senior engineer at a large tech company,",
        "When handling large-scale data (over 100GB),",
        "Working with a distributed team of 5 engineers,",
        "Under a tight deadline of 24 hours,",
        "During an on-call incident at 3 AM,",
        "While mentoring a junior developer,",
        "As part of a security audit,",
        "In a performance-critical production environment,",
        "While optimizing for cost reduction,",
        "During a critical disaster recovery operation,",
    ]

    @classmethod
    def expand_prompt(cls, base_prompt: str, language: str) -> str:
        """シード問題にバリエーションを追加"""
        templates = cls.VARIATION_TEMPLATES_JA if language == "ja" else cls.VARIATION_TEMPLATES_EN
        contexts = cls.CONTEXT_INJECTIONS_JA if language == "ja" else cls.CONTEXT_INJECTIONS_EN

        # 50%の確率でバリエーションテンプレート適用
        if random.random() < 0.5:
            template = random.choice(templates)
            prompt = template.format(base=base_prompt)
        else:
            prompt = base_prompt

        # 30%の確率でコンテキスト注入
        if random.random() < 0.3:
            context = random.choice(contexts)
            if language == "ja":
                prompt = f"{context}、{prompt}"
            else:
                prompt = f"{context} {prompt}"

        return prompt


# =============================================================================
# メイン生成エンジン
# =============================================================================

class ReasoningGenerator:
    """vLLMを使用したreasoning データ生成エンジン"""

    def __init__(self, config: dict, worker_id: int, gpu_id: int):
        self.config = config
        self.worker_id = worker_id
        self.gpu_id = gpu_id
        self.model = None
        self.tokenizer = None

        # 出力ディレクトリ
        self.output_dir = Path(config["output"]["raw_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_file = self.output_dir / f"worker_{worker_id:02d}.jsonl"

        # チェックポイント
        checkpoint_dir = str(self.output_dir.parent / "checkpoints")
        self.checkpoint = CheckpointManager(checkpoint_dir, worker_id)

        # シード問題の読み込み
        seed_path = Path(__file__).resolve().parent.parent / "prompts" / "seed_problems.json"
        with open(seed_path, "r", encoding="utf-8") as f:
            self.seed_problems = json.load(f)

        # ドメイン分布の計算
        self.domain_distribution = config["dataset"]["domain_distribution"]
        self.language_ratio = config["dataset"]["language_ratio"]

    def init_model(self):
        """vLLMモデルを初期化"""
        model_name = self.config["model"]["name"]
        logger.info(f"Worker {self.worker_id}: モデル '{model_name}' をGPU {self.gpu_id} にロード中...")

        # GPU指定
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)

        # MI300X/ROCm パフォーマンス最適化環境変数
        # AITER バックエンド: HIP Paged Attention を AMD 最適化カーネルに切り替え (最大4倍高速化)
        os.environ.setdefault("VLLM_ROCM_USE_AITER", "1")
        os.environ.setdefault("VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT", "1")
        os.environ.setdefault("FA_GFX_ARCHS", "gfx942")  # MI300X のアーキテクチャ識別子

        from vllm import LLM, SamplingParams

        self.model = LLM(
            model=model_name,
            tensor_parallel_size=self.config["parallel"]["tensor_parallel_size"],
            max_model_len=self.config["model"]["max_model_len"],
            gpu_memory_utilization=self.config["model"]["gpu_memory_utilization"],
            trust_remote_code=self.config["model"]["trust_remote_code"],
            dtype="auto",
            quantization=self.config["model"].get("quantization", None),
            enforce_eager=True,   # DeltaNet層のdtypeバグ回避に必須 (vLLM Issue#35238)
            # device="cuda" は vLLM 0.17.0 で EngineArgs から削除されたため省略
            limit_mm_per_prompt={"image": 0, "video": 0},  # Qwen3.5はVLMだがテキスト生成のみ使用するためビジョンエンコーダを無効化
        )

        self.sampling_params = SamplingParams(
            temperature=self.config["generation"]["temperature"],
            top_p=self.config["generation"]["top_p"],
            top_k=self.config["generation"]["top_k"],
            max_tokens=self.config["generation"]["max_tokens"],
            min_tokens=self.config["generation"]["min_tokens"],
            repetition_penalty=self.config["generation"]["repetition_penalty"],
        )

        logger.info(f"Worker {self.worker_id}: モデルのロード完了")

    def _select_domain(self) -> str:
        """重み付きランダムでドメインを選択"""
        domains = list(self.domain_distribution.keys())
        weights = list(self.domain_distribution.values())
        return random.choices(domains, weights=weights, k=1)[0]

    def _select_language(self) -> str:
        """言語比率に基づいて言語を選択"""
        return random.choices(
            list(self.language_ratio.keys()),
            weights=list(self.language_ratio.values()),
            k=1
        )[0]

    def _get_seed_prompt(self, domain: str, language: str) -> str:
        """シード問題からプロンプトを取得（拡張付き）"""
        domain_seeds = self.seed_problems.get(domain, {})
        lang_seeds = domain_seeds.get(language, [])

        if not lang_seeds:
            # フォールバック: 別言語のシードを使用
            for lang in ["ja", "en"]:
                if domain_seeds.get(lang):
                    lang_seeds = domain_seeds[lang]
                    break

        if not lang_seeds:
            # 最終フォールバック: ツールユーズのシードを使用
            lang_seeds = self.seed_problems.get("tool_use_api_calling", {}).get(language, ["Explain your reasoning."])

        base_prompt = random.choice(lang_seeds)
        return PromptExpander.expand_prompt(base_prompt, language)

    def _build_messages(self, domain: str, language: str) -> tuple[str, str, list]:
        """完全なメッセージセット（システム + ユーザー）を構築"""
        # ツール定義を選択
        is_tool_domain = domain.startswith("tool_use_") or domain in ["coding_agent", "code_generation", "planning"]
        tools = select_tools_for_task(domain) if is_tool_domain else []

        # システムプロンプト構築
        system_prompt = build_full_prompt(domain, language, tools if tools else None)

        # ユーザープロンプト取得
        user_prompt = self._get_seed_prompt(domain, language)

        return system_prompt, user_prompt, tools

    def generate_batch(self, batch_prompts: list[dict]) -> list[dict]:
        """バッチ推論を実行"""
        from vllm import SamplingParams

        # vLLM用の入力を構築
        conversations = []
        for item in batch_prompts:
            messages = [
                {"role": "system", "content": item["system_prompt"]},
                {"role": "user", "content": item["user_prompt"]},
            ]
            conversations.append(messages)

        # バッチ推論実行（chat形式）
        try:
            outputs = self.model.chat(
                messages=conversations,
                sampling_params=self.sampling_params,
                use_tqdm=False,
            )
        except Exception as e:
            logger.error(f"Worker {self.worker_id}: バッチ推論エラー: {e}")
            # フォールバック: テキスト生成モードで個別実行
            outputs = []
            for conv in conversations:
                try:
                    # メッセージをテキストに変換
                    prompt_text = "\n".join([
                        f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>"
                        for m in conv
                    ]) + "\n<|im_start|>assistant\n"

                    result = self.model.generate(
                        [prompt_text],
                        self.sampling_params,
                        use_tqdm=False
                    )
                    outputs.extend(result)
                except Exception as e2:
                    logger.error(f"Worker {self.worker_id}: 個別推論エラー: {e2}")
                    outputs.append(None)

        # 結果の処理
        results = []
        for i, output in enumerate(outputs):
            if output is None:
                continue
            try:
                generated_text = output.outputs[0].text
                thinking, response = parse_thinking_response(generated_text)

                result = {
                    **batch_prompts[i],
                    "raw_output": generated_text,
                    "thinking": thinking,
                    "response": response,
                    "thinking_steps": count_thinking_steps(thinking),
                    "tool_calls": extract_tool_calls(response),
                    "has_tool_use": detect_tool_use_intent(generated_text),
                    "output_tokens": len(output.outputs[0].token_ids),
                }
                results.append(result)
            except Exception as e:
                logger.warning(f"Worker {self.worker_id}: 出力処理エラー (index {i}): {e}")

        return results

    def run(self, num_samples: int):
        """メイン生成ループ"""
        logger.info(f"Worker {self.worker_id}: {num_samples} 件の生成を開始")

        # チェックポイントから再開
        state = self.checkpoint.load()
        start_index = state.get("completed", 0) if state else 0

        if start_index > 0:
            logger.info(f"Worker {self.worker_id}: {start_index} 件から再開")

        batch_size = self.config["generation"]["batch_size"]
        checkpoint_interval = self.config["parallel"]["checkpoint_interval"]
        completed = start_index
        total_time = 0

        while completed < num_samples:
            batch_start = time.time()

            # 現在のバッチのプロンプトを生成
            current_batch_size = min(batch_size, num_samples - completed)
            batch_prompts = []

            for j in range(current_batch_size):
                domain = self._select_domain()
                language = self._select_language()
                system_prompt, user_prompt, tools = self._build_messages(domain, language)
                sample_id = generate_sample_id(self.worker_id, completed + j, domain)

                batch_prompts.append({
                    "id": sample_id,
                    "domain": domain,
                    "language": language,
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "tool_definitions": [t["function"] for t in tools] if tools else [],
                })

            # バッチ推論実行
            results = self.generate_batch(batch_prompts)

            # 結果を保存
            for result in results:
                sample = ReasoningSample(
                    id=result["id"],
                    domain=result["domain"],
                    language=result["language"],
                    system_prompt=result["system_prompt"],
                    user_query=result["user_prompt"],
                    thinking=result["thinking"],
                    response=result["response"],
                    tool_calls=result.get("tool_calls", []),
                    tool_definitions=result.get("tool_definitions", []),
                    metadata={
                        "thinking_steps": result["thinking_steps"],
                        "has_tool_use": result["has_tool_use"],
                        "output_tokens": result["output_tokens"],
                        "worker_id": self.worker_id,
                    }
                )
                append_jsonl(sample.to_training_format(), str(self.output_file))

            completed += len(results)
            batch_time = time.time() - batch_start
            total_time += batch_time
            speed = len(results) / batch_time if batch_time > 0 else 0
            eta = (num_samples - completed) / speed / 3600 if speed > 0 else float('inf')

            logger.info(
                f"Worker {self.worker_id}: "
                f"{completed}/{num_samples} ({100*completed/num_samples:.1f}%) | "
                f"速度: {speed:.1f} samples/s | "
                f"ETA: {eta:.1f}h"
            )

            # チェックポイント保存
            if completed % checkpoint_interval < batch_size:
                self.checkpoint.save({
                    "completed": completed,
                    "total_time": total_time,
                    "speed": speed,
                })

        # 最終チェックポイント
        self.checkpoint.save({
            "completed": completed,
            "total_time": total_time,
            "status": "done",
        })

        logger.info(
            f"Worker {self.worker_id}: 生成完了! "
            f"{completed} 件, 所要時間: {total_time/3600:.2f}h"
        )


# =============================================================================
# 並列実行マネージャー
# =============================================================================

def launch_parallel_workers(config: dict, script_path: str):
    """8つのワーカープロセスを並列起動"""
    num_workers = config["parallel"]["num_workers"]
    samples_per_worker = config["parallel"]["samples_per_worker"]

    logger.info(f"=== 並列生成開始: {num_workers} ワーカー x {samples_per_worker} 件 ===")
    logger.info(f"合計目標: {num_workers * samples_per_worker} 件")

    processes = []
    for worker_id in range(num_workers):
        cmd = [
            sys.executable, script_path,
            "--config", str(Path(script_path).parent.parent / "config.yaml"),
            "--worker-id", str(worker_id),
            "--gpu-id", str(worker_id),
            "--num-samples", str(samples_per_worker),
        ]

        log_file = Path(config["output"]["base_dir"]) / "logs" / f"worker_{worker_id:02d}.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)

        with open(log_file, "w") as lf:
            proc = subprocess.Popen(
                cmd,
                stdout=lf,
                stderr=subprocess.STDOUT,
                env={**os.environ, "CUDA_VISIBLE_DEVICES": str(worker_id)},
            )
            processes.append((worker_id, proc))
            logger.info(f"Worker {worker_id} 起動 (PID: {proc.pid}, GPU: {worker_id})")

        # GPU間の初期化衝突を避けるため少し待つ
        time.sleep(5)

    # 全プロセスの完了を監視
    logger.info("全ワーカーの完了を待機中...")

    def signal_handler(signum, frame):
        logger.warning("中断シグナルを受信。全ワーカーを終了します...")
        for wid, proc in processes:
            proc.terminate()
        sys.exit(1)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    while processes:
        for worker_id, proc in list(processes):
            ret = proc.poll()
            if ret is not None:
                if ret == 0:
                    logger.info(f"Worker {worker_id} 正常完了")
                else:
                    logger.error(f"Worker {worker_id} 異常終了 (code: {ret})")
                processes.remove((worker_id, proc))

        if processes:
            # 進捗確認
            total_completed = 0
            for wid in range(num_workers):
                output_file = Path(config["output"]["raw_dir"]) / f"worker_{wid:02d}.jsonl"
                if output_file.exists():
                    total_completed += count_jsonl_lines(str(output_file))

            total_target = num_workers * samples_per_worker
            pct = 100 * total_completed / total_target if total_target > 0 else 0
            logger.info(
                f"全体進捗: {total_completed}/{total_target} ({pct:.1f}%) | "
                f"稼働中ワーカー: {len(processes)}"
            )
            time.sleep(60)  # 1分ごとに進捗チェック

    logger.info("=== 全ワーカー完了 ===")


# =============================================================================
# エントリポイント
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Qwen3.5-27B Reasoning データセット生成")
    parser.add_argument("--config", type=str, default="../config.yaml", help="設定ファイルのパス")
    parser.add_argument("--worker-id", type=int, default=0, help="ワーカーID")
    parser.add_argument("--gpu-id", type=int, default=0, help="使用するGPU ID")
    parser.add_argument("--num-samples", type=int, default=None, help="生成件数（省略時は設定ファイルの値を使用）")
    parser.add_argument("--auto-parallel", action="store_true", help="全ワーカーを自動的に並列起動")
    parser.add_argument("--dry-run", action="store_true", help="推論を実行せずプロンプト生成のみテスト")
    args = parser.parse_args()

    # ログ設定
    log_format = f"[%(asctime)s] [Worker {args.worker_id}] %(levelname)s: %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_format)

    # 設定読み込み
    config_path = Path(args.config).resolve()
    config = load_config(str(config_path))

    # ディレクトリ作成
    for d in ["raw_dir", "filtered_dir", "final_dir"]:
        Path(config["output"][d]).mkdir(parents=True, exist_ok=True)

    if args.auto_parallel:
        # 並列実行モード
        launch_parallel_workers(config, str(Path(__file__).resolve()))
        return

    num_samples = args.num_samples or config["parallel"]["samples_per_worker"]

    if args.dry_run:
        # ドライラン: プロンプト生成テスト
        logger.info("=== ドライランモード ===")
        generator = ReasoningGenerator(config, args.worker_id, args.gpu_id)
        for i in range(min(5, num_samples)):
            domain = generator._select_domain()
            language = generator._select_language()
            sys_prompt, user_prompt, tools = generator._build_messages(domain, language)
            logger.info(f"\n--- Sample {i+1} ---")
            logger.info(f"Domain: {domain}, Language: {language}")
            logger.info(f"Tools: {[t['function']['name'] for t in tools] if tools else 'None'}")
            logger.info(f"User Prompt: {user_prompt[:200]}...")
            logger.info(f"System Prompt length: {len(sys_prompt)} chars")
        logger.info("=== ドライラン完了 ===")
        return

    # 通常実行
    generator = ReasoningGenerator(config, args.worker_id, args.gpu_id)
    generator.init_model()
    generator.run(num_samples)


if __name__ == "__main__":
    main()
