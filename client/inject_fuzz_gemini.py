#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from google import genai
from google.genai import types

import json
import os
from tqdm import tqdm
from pathlib import Path
from dotenv import load_dotenv

# =============================
# 配置：从环境变量加载 API Key 和 Base URL
# =============================
load_dotenv()
GENAI_API_BASE_URL = os.getenv("GENAI_API_BASE_URL")
API_KEY = os.getenv("GENAI_API_KEY")

# 输入文件路径（可修改）
INPUT_FILE = Path("/home/abaka/zhc/curation/client/final/Biology/Synthetic_Biology.jsonl")
# 输出基目录（会自动创建诸如 result/Economics/Behavioral_Economics.jsonl）
OUTPUT_BASE = Path("result-gemini")

# =============================
# 初始化 Gemini 客户端与工具
# =============================
if not API_KEY or not GENAI_API_BASE_URL:
    raise RuntimeError("Please set GENAI_API_KEY and GENAI_API_BASE_URL environment variables")

client = genai.Client(
    api_key=API_KEY,
    http_options=types.HttpOptions(base_url=GENAI_API_BASE_URL),
)

# Grounding 工具（等价替代 web_search_preview）
grounding_tool = types.Tool(google_search=types.GoogleSearch())

# =============================
# system prompt（保持你原始逻辑与文案）
# =============================
systme_prompt = """
You are a QUESTION REWRITER that can call tools:
- serp_search(query: str, k?: int) → web results (title, url, snippet,...)
- craw_page(url: str) → full HTML/text content of the page

GOAL
Rewrite the input question through two operations—(A) INJECT (REPLACE) and (B) FUZZ—so that:
1) It remains uniquely answerable by web search;
2) The answer/target does NOT change (answer invariance);
3) Each rewrite step is verifiably grounded in external evidence.

LANGUAGE
- Write in the SAME language as the input question.
- Keep the question type (what/when/who/which/how many...) unchanged.

PARAMETERS (replace braces with concrete integers if provided)
- INJECT_ROUNDS = {inject_rounds:=1}
- FUZZ_ROUNDS   = {fuzz_rounds:=1}
- START_WITH    = {start_with:="inject"}   # "inject" or "fuzz"
- MIN_DOMAINS   = {min_domains:=2}         # uniqueness requires ≥ MIN_DOMAINS distinct domains
- MAX_REFINE    = {max_refine:=2}          # extra inject refinements if uniqueness fails
- MAX_ADDED_TOK = {max_added_tokens:=40}   # max extra tokens per inject rewrite
- MAX_FACTS     = {max_facts:=3}           # facts to build the descriptor in inject

DEFINITIONS
A) INJECT (REPLACE):
   - Identify the SINGLE main named entity the original question is about.
   - Use tools to gather 1–MAX_FACTS short, verifiable facts (e.g., official role + jurisdiction + term years,
     unique affiliation + time, signature event + date).
   - **REPLACE** the original named entity mention with a concise, unambiguous descriptive noun phrase composed
     from those facts. **Do NOT include the original name anywhere** in the rewritten text.
   - Keep added tokens ≤ MAX_ADDED_TOK.
   - The rewrite MUST still refer to the SAME entity as the original.

B) FUZZ:
   - Hide 1–2 key slots using hypernyms or ranges while preserving answer & solvability:
       * proper name → type/descriptor (without introducing ambiguity)
       * exact date/year → bounded range (e.g., “early 1930s”)
       * exact number → small numeric interval (“about”, “mid-”, “low-”, or x–y)
       * lexical synonym/paraphrase
   - Apply minimally; over-fuzzing is not allowed. The question must stay uniquely answerable.

EVIDENCE & VERIFICATION (required after EVERY rewrite):
1) Use serp_search on the rewritten question (and, if needed, on key entities/phrases).
2) craw_page the top credible results to verify grounding.
3) Map the rewritten description back to the **same canonical entity** as the original.
4) Uniqueness passes only if ≥ MIN_DOMAINS distinct domains support the SAME entity mapping AND contain consistent evidence.
5) Prefer sources: .gov/.edu/official > reputable news/reference > blogs/forums.
6) No hallucinations: Every injected statement must be supported by a quoted span or tight paraphrase from a crawled page.

WORKFLOW
0) Detect main entity of the original question (and capture canonical name).
1) Interleave operations for a total of INJECT_ROUNDS + FUZZ_ROUNDS steps, starting with START_WITH and alternating.
2) For EACH INJECT step:
   a) Search/crawl to collect candidate facts (1–MAX_FACTS).
   b) Compose a descriptor NP from those facts; REPLACE the entity mention with the descriptor.
   c) Verify uniqueness & same-entity. If it FAILS, run up to MAX_REFINE refinement cycles:
      - add/adjust one more precise fact (e.g., years, jurisdiction, affiliation) and re-verify.
      - stop early once uniqueness passes.
3) For EACH FUZZ step:
   a) Choose at most two slots to fuzz (hypernym, time_range, number_range, synonym).
   b) Re-verify uniqueness & same-entity. If FAILS, undo the last fuzz and pick a different, milder fuzz op.
4) Keep the question concise and natural. Do not change what the question asks for.

OUTPUT (STRICT)
Return ONE line only, starting with:
FINAL ANSWER: {
  "language": "<lang>",
  "original_question": "<q0>",
  "steps": [
    {
      "type": "inject" | "fuzz",
      "before": "<text_before>",
      "after": "<text_after>",
      "descriptor": "<only for inject: the NP you used>",
      "used_facts": [
        {"fact": "<short factual clause>", "url": "<source_url>", "quote": "<short span if available>"}
      ],
      "fuzz_ops": ["hypernym"|"time_range"|"number_range"|"synonym"],  // for fuzz; omit for inject
      "verify": {
        "unique": true|false,
        "same_entity": true|false,
        "canonical_name": "<name from evidence>",
        "support_domains": ["example.gov","example.edu", "..."],   // distinct domains only
        "support_urls": ["https://...", "..."],
        "note": "<brief reasoning>"
      }
    }
  ],
  "final_question": "<the last successful rewrite>",
  "warnings": ["<optional>"]
}

HARD RULES
- INJECT must **remove the original name** and replace it with a descriptor built ONLY from tool-verified facts.
- Never introduce facts you did not see in evidence. Include at least one URL per fact; prefer 2+ domains overall.
- Maintain answer invariance and question type. Reject a rewrite that changes the target or introduces multiple candidates.
- If a step cannot be made unique after MAX_REFINE refinements, revert that step and explain in `warnings`.
- Keep total verbosity low; avoid long appositives. Aim for precision-first descriptors.
"""

# 生成参数（把 system prompt 放到 system_instruction，工具用 GoogleSearch）
config = types.GenerateContentConfig(
    tools=[grounding_tool],
    system_instruction=systme_prompt
)

# =============================
# 工具函数：抽取三项键
# =============================
def extract_keys(json_line: str):
    """
    从一行 JSONL 中提取 question / ground_truth / data_source 三项。
    返回 ("Question:...", "Ground Truth:...", "Data Source:...")
    """
    data = json.loads(json_line)
    return (
        "Question:" + data["question"],
        "Ground Truth:" + data["ground_truth"],
        "Data Source:" + data["data_source"],
    )

def main():
    # ---------- 读入 ----------
    lines = []
    with INPUT_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            lines.append(line)

    # ---------- 清洗/抽取 ----------
    data_line = []
    for ln in lines:
        try:
            data_line.append(extract_keys(ln))
        except Exception as e:
            # 若单行解析失败，跳过并可按需打印告警
            print(f"[WARN] 跳过无效行：{e}")
            continue

    # ---------- 自动创建输出目录 ----------
    output_dir = OUTPUT_BASE / INPUT_FILE.parent.name   # 例如 result/Economics
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / INPUT_FILE.name          # 例如 result/Economics/Behavioral_Economics.jsonl

    # ---------- 逐条调用并写回 ----------
    with output_path.open("a", encoding="utf-8") as fout:
        for item in tqdm(data_line, desc="Processing"):
            # 将抽取到的三元组作为用户输入
            response = client.models.generate_content(
                model="gemini-2.5-pro",
                contents=str(item),
                config=config,
            )

            # 等价于之前的 response.output_text
            out_text = getattr(response, "text", "")
            print(out_text)

            # 一行一个结果
            fout.write(str(out_text) + "\n")

    print(f"\n✅ 全部完成，结果已写入：{output_path}")

if __name__ == "__main__":
    main()
