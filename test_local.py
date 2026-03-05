"""ローカルテスト — GPUなしで動作確認"""
import json
import sys
sys.path.insert(0, '.')
from utils import (
    parse_thinking_response, count_thinking_steps,
    extract_tool_calls, detect_language, detect_tool_use_intent,
    generate_sample_id, ReasoningSample, load_config,
    save_jsonl, load_jsonl
)
from prompts.system_prompts import (
    build_full_prompt, select_tools_for_task, format_tools_description
)

passed = 0
failed = 0

def test(name, condition, msg=""):
    global passed, failed
    if condition:
        print(f"  PASS: {name}")
        passed += 1
    else:
        print(f"  FAIL: {name} - {msg}")
        failed += 1

print("=" * 60)
print("Qwen3.5 Reasoning Pipeline - Unit Tests")
print("=" * 60)

# --- Test 1: Thinking Tag Parser ---
print("\n[1] Thinking Tag Parser")
text = "<think>\nステップ1: まず検索する\n\nステップ2: 結果を分析\n\nステップ3: まとめる\n</think>\n\n最終回答です。"
thinking, response = parse_thinking_response(text)
steps = count_thinking_steps(thinking)
test("parse thinking", thinking != "")
test("parse response", response == "最終回答です。", f"got: {response}")
test("count steps = 3", steps == 3, f"got: {steps}")

# No think tag
t2, r2 = parse_thinking_response("普通の回答です")
test("no think tag", t2 == "" and r2 == "普通の回答です")

# --- Test 2: Tool Call Detection ---
print("\n[2] Tool Call Detection")
tool_text = 'result:\n```tool_call\n{"name": "search_web", "arguments": {"query": "test"}}\n```\nDone.'
tc = extract_tool_calls(tool_text)
test("detect tool_call block", len(tc) >= 1, f"found {len(tc)}")

json_text = '{"name": "read_file", "arguments": {"path": "/tmp/test.txt"}}'
tc2 = extract_tool_calls(json_text)
test("detect JSON tool call", len(tc2) >= 1, f"found {len(tc2)}")

test("detect intent (ja)", detect_tool_use_intent("関数を呼び出してデータを取得"))
test("detect intent (en)", detect_tool_use_intent("I'll use the search tool"))
test("no intent", not detect_tool_use_intent("こんにちは、元気ですか？"))

# --- Test 3: Language Detection ---
print("\n[3] Language Detection")
test("detect ja", detect_language("これは日本語のテストです。") == "ja")
test("detect en", detect_language("This is an English test sentence.") == "en")
test("detect mixed ja", detect_language("これはtestです") == "ja")

# --- Test 4: System Prompt Generation ---
print("\n[4] System Prompt Generation")
tools = select_tools_for_task("tool_use_api_calling", 3)
test("select 3 tools", len(tools) == 3, f"got {len(tools)}")

prompt = build_full_prompt("tool_use_api_calling", "ja", tools)
test("prompt not empty", len(prompt) > 100, f"len={len(prompt)}")
test("prompt contains tool names", any(t["function"]["name"] in prompt for t in tools))

all_tools = select_tools_for_task("tool_use_multi_step", 10)
test("multi_step has many tools", len(all_tools) >= 5)

# --- Test 5: Sample ID ---
print("\n[5] Sample ID Generation")
sid = generate_sample_id(0, 42, "tool_use")
test("id format", sid == "qwen35r_tool_use_00_0000042", f"got: {sid}")
sid2 = generate_sample_id(7, 999999, "math")
test("id worker 7", "07" in sid2)

# --- Test 6: ReasoningSample ---
print("\n[6] ReasoningSample SFT Format")
sample = ReasoningSample(
    id="test_001", domain="tool_use_api_calling", language="ja",
    system_prompt="テストシステムプロンプト",
    user_query="テスト質問です",
    thinking="ステップ1: 考える\n\nステップ2: 実行する",
    response="これが回答です。",
    tool_calls=[{"name": "search", "arguments": {"q": "test"}}],
    tool_definitions=[{"name": "search", "description": "検索"}],
)
fmt = sample.to_training_format()
test("3 messages", len(fmt["messages"]) == 3)
test("system role", fmt["messages"][0]["role"] == "system")
test("user role", fmt["messages"][1]["role"] == "user")
test("assistant role", fmt["messages"][2]["role"] == "assistant")
test("thinking in output", "<think>" in fmt["messages"][2]["content"])
test("has tools", "tools" in fmt)
test("has tool_calls", "tool_calls" in fmt)
test("has domain", fmt["domain"] == "tool_use_api_calling")

# --- Test 7: Config Loading ---
print("\n[7] Config Loading")
config = load_config("config.yaml")
test("model name", "Qwen" in config["model"]["name"])
test("num workers = 6", config["parallel"]["num_workers"] == 6)
test("target 1M", config["dataset"]["target_total_samples"] == 1000000)
test("ja ratio 0.7", config["dataset"]["language_ratio"]["ja"] == 0.7)
test("tool+coding domain > 0.6", 
     sum(v for k, v in config["dataset"]["domain_distribution"].items() if "tool_use" in k or "coding" in k) >= 0.6)

# --- Test 8: JSONL I/O ---
print("\n[8] JSONL I/O")
import tempfile, os
with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
    tmp_path = f.name
try:
    test_data = [{"id": 1, "text": "テスト"}, {"id": 2, "text": "test"}]
    save_jsonl(test_data, tmp_path)
    loaded = load_jsonl(tmp_path)
    test("save and load", len(loaded) == 2)
    test("content match", loaded[0]["text"] == "テスト")
finally:
    os.unlink(tmp_path)

# --- Test 9: Seed Problems ---
print("\n[9] Seed Problems")
with open("prompts/seed_problems.json", "r", encoding="utf-8") as f:
    seeds = json.load(f)
test("has tool_use_api_calling", "tool_use_api_calling" in seeds)
test("has tool_use_multi_step", "tool_use_multi_step" in seeds)
test("has coding_agent", "coding_agent" in seeds)
test("has math_reasoning", "math_reasoning" in seeds)
test("ja problems exist", len(seeds["tool_use_api_calling"]["ja"]) >= 10)
test("en problems exist", len(seeds["tool_use_api_calling"]["en"]) >= 5)
test("coding_agent ja", len(seeds["coding_agent"]["ja"]) >= 10)
test("coding_agent en", len(seeds["coding_agent"]["en"]) >= 5)
total_seeds = sum(len(v) for domain in seeds.values() for v in domain.values())
test(f"total seeds >= 100", total_seeds >= 100, f"got {total_seeds}")

# --- Test 10: PromptExpander ---
print("\n[10] PromptExpander")
sys.path.insert(0, 'vastai')
from generate_reasoning import PromptExpander
base = "テスト問題です"
expanded = PromptExpander.expand_prompt(base, "ja")
test("expanded not empty", len(expanded) > 0)
# 複数回実行してバリエーションが存在するか確認
variants = set()
for _ in range(20):
    variants.add(PromptExpander.expand_prompt(base, "ja"))
test("has variation", len(variants) > 1, f"only {len(variants)} variants")

# --- Summary ---
print("\n" + "=" * 60)
total = passed + failed
print(f"Results: {passed}/{total} passed, {failed} failed")
if failed == 0:
    print("All tests passed! ✅")
else:
    print(f"WARNING: {failed} test(s) failed ⚠️")
print("=" * 60)
