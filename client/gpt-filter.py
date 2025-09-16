# from openai import OpenAI
# base_url="http://118.196.11.39:3000/v1"
# api_key="sk-JyxgZEKmBUCb8NoCr2K51vSY8I4xevzvHwLI245qlAZQCCcC"
# client = OpenAI(base_url=base_url, api_key=api_key)


# from openai import OpenAI
# base_url="https://api.2077ai.org/v1"
# api_key="sk-JyxgZEKmBUCb8NoCr2K51vSY8I4xevzvHwLI245qlAZQCCcC"
# client = OpenAI(base_url=base_url, api_key=api_key)

# response = client.responses.create(
#     model="gpt-5-search",
#     tools=[{"type": "web_search_preview"}],
#     input="who is huichi zhou?",
#     stream=False
# )

# print(response.output_text)
# save as judge_gpt4omini.py
# -*- coding: utf-8 -*-

import os, json, argparse
from typing import Dict, Any
from openai import OpenAI

#
BASE_URL = os.getenv("OPENAI_BASE_URL")
API_KEY  = os.getenv("OPENAI_API_KEY")

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

def ask_model(question: str, model: str, use_search_tool: bool) -> str:
    """调用 GPT-5 问答；返回纯文本答案"""
    payload = dict(model=model, input=question, stream=False)
    if use_search_tool:
        payload["tools"] = [{"type": "web_search_preview"}]
    resp = client.responses.create(**payload)
    text = getattr(resp, "output_text", None)
    if not text:
        # 兜底，兼容不同实现
        try:
            text = "".join(getattr(resp, "output", [])) or ""
        except Exception:
            text = ""
    return text.strip()

def judge_with_gpt4omini(question: str, ground_truth: str, model_answer: str) -> str:
    """
    用 gpt-4o-mini 判断是否等价。
    只返回 "YES" 或 "NO"；其他任何输出都按 "NO" 处理。
    """
    sys = (
        "你是一个严格的判定器。给定【问题】、【模型回答】和【标准答案】，"
        "只判断模型回答与标准答案在关键信息上是否等价（忽略格式、顺序、大小写）。"
        "只输出 YES 或 NO，不要解释。"
    )
    user = f"""[问题]
{question}

[模型回答]
{model_answer}

[标准答案]
{ground_truth}

只回答 YES 或 NO："""
    resp = client.responses.create(
        model="gpt-4o-mini",
        input=[{"role":"system","content":sys},{"role":"user","content":user}],
        stream=False,
    )
    out = (getattr(resp, "output_text", "") or "").strip().upper()
    return "YES" if out == "YES" else "NO"

def main():
    ap = argparse.ArgumentParser(description="Read JSONL -> ask GPT-5 -> judge by gpt-4o-mini -> save only failures.")
    ap.add_argument("--input", required=True, help="输入 JSONL 路径")
    ap.add_argument("--output", required=True, help="输出 JSONL（仅保存判定为 NO 的样本）")
    ap.add_argument("--model", default="gpt-5-search", help="回答模型名（默认 gpt-5-search）")
    ap.add_argument("--use-search-tool", action="store_true", help="是否附带 web_search_preview 工具")
    ap.add_argument("--qfield", default="question", help="问题字段名（默认 question）")
    ap.add_argument("--gtfield", default="ground_truth", help="真值字段名（默认 ground_truth）")
    args = ap.parse_args()

    total = 0
    saved = 0

    with open(args.input, "r", encoding="utf-8") as fin, \
         open(args.output, "w", encoding="utf-8") as fout:

        for line in fin:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                obj: Dict[str, Any] = json.loads(line)
            except Exception:
                # 解析失败的行，直接保存，便于后续排查
                fout.write(line + "\n")
                saved += 1
                continue

            q = obj.get(args.qfield, "")
            gt = obj.get(args.gtfield, "")

            try:
                answer = ask_model(q, args.model, args.use_search_tool)
                decision = judge_with_gpt4omini(q, gt, answer)  # "YES" / "NO"
            except Exception as e:
                # 调用失败时，保守起见保存该条
                obj["_error"] = str(e)
                obj["_model_answer"] = ""
                obj["_judge"] = "ERROR_TREATED_AS_NO"
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                saved += 1
                continue

            if decision != "YES":
                # 模型没答到位，保存
                obj["_model_answer"] = answer
                obj["_judge"] = decision
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                saved += 1

    print(f"Total: {total} | Saved (judge != YES): {saved}")

if __name__ == "__main__":
    main()
