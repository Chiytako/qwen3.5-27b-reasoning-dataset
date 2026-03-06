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
import asyncio
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Optional

import openai

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
# メイン生成エンジン ( 非同期 API クライアント )
# =============================================================================

class ReasoningGenerator:
    """SGLang (OpenAI API互換サーバー) を使用した非同期推論エンジン"""

    def __init__(self, config: dict, worker_id: int):
        self.config = config
        self.worker_id = worker_id
        
        # クライアント
        self.api_base = config.get("api", {}).get("base_url", "http://localhost:8000/v1")
        
        # 複数インスタンス (MI300X 並列ロード) 対応: worker_id に応じてポートを振り分け (最大4インスタンス)
        base_port = 8000
        port = base_port + (worker_id % 4)
        if "localhost" in self.api_base or "127.0.0.1" in self.api_base:
            self.api_base = f"http://localhost:{port}/v1"
            
        self.model_name = config.get("api", {}).get("model_name", "Qwen/Qwen3.5-27B")
        self.client = openai.AsyncOpenAI(
            api_key="EMPTY",
            base_url=self.api_base,
            timeout=3600.0, # Llama.cppの連続バッチはキューが詰まると遅いのでタイムアウトを長く
            max_retries=5
        )

        self.concurrency_max_limit = config.get("api", {}).get("concurrency_max_limit", 8)
        self.concurrency_target_latency = config.get("api", {}).get("concurrency_target_latency", 45.0)
        # 初期並行数は llama-server の -np 8 に合わせる (16384 tokens/slot)
        self.current_concurrency = 8

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
        
        # 世代パラメータ (Llama.cpp サーバーがサポートする標準的なものに絞る)
        self.generation_params = {
            "temperature": self.config["generation"].get("temperature", 0.6),
            "top_p": self.config["generation"].get("top_p", 0.95),
            "max_tokens": self.config["generation"].get("max_tokens", 12288),
        }
        # llama.cpp 固有パラメータ (extra_body で渡す)
        # min_tokens: 早期 EOS による空・超短応答を防ぐ
        self.extra_body = {
            "min_tokens": self.config["generation"].get("min_tokens", 256),
        }
        
    async def _test_connection(self):
        """APIサーバーへの接続テスト"""
        logger.info(f"Worker {self.worker_id}: APIサーバー ({self.api_base}) に接続テスト中...")
        # 複数インスタンスの同時ロードには非常に時間がかかるため、タイムアウトを大幅に増やす (60回 * 10秒 = 10分)
        for i in range(60):
            try:
                models = await self.client.models.list()
                if models.data:
                    self.model_name = models.data[0].id
                logger.info(f"Worker {self.worker_id}: 接続成功！サーバー検出モデル: {self.model_name}")
                return True
            except Exception as e:
                logger.warning(f"接続待機中... ({i+1}/60): {e}")
                await asyncio.sleep(10)
        logger.error("APIサーバーへの接続に失敗しました（タイムアウト）。")
        return False

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

    async def _generate_single(self, prompt_data: dict, semaphore: asyncio.Semaphore) -> Optional[dict]:
        """単一のプロンプトに対する非同期生成 (時間計測付き)"""
        messages = [
            {"role": "system", "content": prompt_data["system_prompt"]},
            {"role": "user", "content": prompt_data["user_prompt"]},
        ]
        
        async with semaphore:
            req_start = time.time()
            try:
                response = await self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    extra_body=self.extra_body,
                    **self.generation_params
                )
                
                req_time = time.time() - req_start
                message = response.choices[0].message
                generated_text = message.content or ""

                # llama.cpp は thinking を reasoning_content に分離して返す場合がある
                reasoning_content = (
                    getattr(message, 'reasoning_content', None)
                    or (message.model_extra or {}).get('reasoning_content')
                    or ""
                )

                if not generated_text.strip() and not reasoning_content.strip():
                    logger.warning(f"Sample {prompt_data['id']}: APIから空の応答が返されました。")
                    return None

                if reasoning_content:
                    # reasoning_content が存在する場合: llama.cpp が thinking を分離済み
                    thinking = reasoning_content.strip()
                    final_response = generated_text.strip()
                else:
                    thinking, final_response = parse_thinking_response(generated_text)

                # 最終応答が空の場合はスキップ
                # (<think></think> のみ、または thinking だけで回答なし、等)
                if not final_response:
                    logger.warning(
                        f"Sample {prompt_data['id']}: 最終応答が空でした。"
                        f" thinking={'yes' if thinking else 'no'},"
                        f" raw={repr(generated_text[:80])}"
                    )
                    return None
                
                return {
                    **prompt_data,
                    "raw_output": generated_text,
                    "thinking": thinking,
                    "response": final_response,
                    "thinking_steps": count_thinking_steps(thinking),
                    "tool_calls": extract_tool_calls(final_response),
                    "has_tool_use": detect_tool_use_intent(generated_text),
                    "output_tokens": response.usage.completion_tokens if response.usage else 0,
                    "latency": req_time,
                }
            except Exception as e:
                logger.error(f"Sample {prompt_data['id']} 世代エラー: {e}")
                return None

    async def generate_batch_async(self, batch_prompts: list[dict], concurrency: int) -> list[dict]:
        """指定された同時接続数で非同期推論（バッチ）を実行"""
        semaphore = asyncio.Semaphore(concurrency)
        tasks = [self._generate_single(p, semaphore) for p in batch_prompts]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # エラーとNoneを除外して返す
        valid_results = []
        for r in results:
            if isinstance(r, Exception):
                logger.warning(f"バッチ処理中の予期せぬエラー: {r}")
                continue
            if r is not None:
                valid_results.append(r)
                
        return valid_results

    def _adjust_concurrency(self, avg_latency: float):
        """レスポンス待機時間に基づいて同時接続数を動的に増減させる(Load Balancing)"""
        # 安定動作のため、自動調整機能をコメントアウト
        pass
        # if avg_latency == 0:
        #     return
        #     
        # old_concurrency = self.current_concurrency
        # 
        # # ターゲットレイテンシ(例:30秒)に対して早すぎるならまだ余裕がある->並列数を増やす
        # # ターゲットより遅いならサーバーが詰まっている->並列数を減らす
        # if avg_latency < self.concurrency_target_latency * 0.8:
        #     # 余裕がある: +2ずつ増やす
        #     self.current_concurrency = min(self.concurrency_max_limit, self.current_concurrency + 2)
        # elif avg_latency > self.concurrency_target_latency * 1.5:
        #     # 詰まりすぎ: -4ずつ急激に減らす(最低値4)
        #     self.current_concurrency = max(4, self.current_concurrency - 4)
        # elif avg_latency > self.concurrency_target_latency * 1.2:
        #     # 少し遅い: -1ずつ減らす
        #     self.current_concurrency = max(4, self.current_concurrency - 1)
        #     
        # if old_concurrency != self.current_concurrency:
        #     if self.current_concurrency > old_concurrency:
        #         logger.info(f"⚡ [Auto-Scale] 平均応答 {avg_latency:.1f}s 極めて良好。同時接続数を拡張: {old_concurrency} -> {self.current_concurrency}")
        #     else:
        #         logger.warning(f"⏳ [Auto-Scale] 平均応答 {avg_latency:.1f}s サーバー過負荷を検知。同時接続数を抑制: {old_concurrency} -> {self.current_concurrency}")

    async def run(self, num_samples: int):
        """メイン生成ループ"""
        if not await self._test_connection():
            return

        logger.info(f"Worker {self.worker_id}: {num_samples} 件の生成を開始 (Llama.cpp Async API) [初期同時接続数: {self.current_concurrency}]")

        # チェックポイントから再開
        state = self.checkpoint.load()
        start_index = state.get("completed", 0) if state else 0

        if start_index > 0:
            logger.info(f"Worker {self.worker_id}: {start_index} 件から再開")
            
        batch_size = self.config["generation"].get("batch_size", 256)
        checkpoint_interval = self.config["parallel"]["checkpoint_interval"]
        completed = start_index
        total_time = 0

        while completed < num_samples:
            batch_start = time.time()

            # ここでの"batch"は非同期タスクの投入プール単位
            # 実際には semaphore によって concurrency 制限がかかる
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

            # 現在の動的コンカレンシー上限を渡して超高並列非同期推論の実行
            results = await self.generate_batch_async(batch_prompts, concurrency=self.current_concurrency)
            
            success_count = len(results)
            total_latency = 0.0

            # 結果を保存
            for result in results:
                total_latency += result.get("latency", 0)
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

            # エラー全滅判定
            if success_count == 0:
                logger.warning("バッチ全滅。APIサーバーがいっぱい、もしくは落ちている可能性があります。10秒待ちます...")
                self._adjust_concurrency(self.concurrency_target_latency * 2) # 強制負荷下げ
                await asyncio.sleep(10)
                continue

            # 処理速度の解析と自動スケーリング調整
            avg_latency = total_latency / success_count
            self._adjust_concurrency(avg_latency)

            completed += success_count
            batch_time = time.time() - batch_start
            total_time += batch_time
            speed = success_count / batch_time if batch_time > 0 else 0
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
# エントリポイント
# =============================================================================

async def async_main():
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
        logger.error("--auto-parallel は APIサーバー方式では非推奨になり削除されました。run_pipeline.sh を使用してください。")
        return

    num_samples = args.num_samples or config["parallel"]["samples_per_worker"]

    generator = ReasoningGenerator(config, args.worker_id)
    
    if args.dry_run:
        # ドライラン: プロンプト生成テスト
        logger.info("=== ドライランモード ===")
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
    await generator.run(num_samples)


def main():
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
