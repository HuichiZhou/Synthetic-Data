import os, json, argparse
from typing import Dict, Any
from google import genai
from google.genai import types

GENAI_API_BASE_URL =  os.getenv("OPENAI_BASE_URL")
API_KEY = os.getenv("OPENAI_API_KEY")


def build_client() -> genai.Client:
    return genai.Client(api_key=API_KEY, http_options=types.HttpOptions(base_url=GENAI_API_BASE_URL))

def ask_gemini(client: genai.Client, question: str, model: str, use_search: bool) -> str:
    """
    用 Gemini 回答问题；可选启用 Google Search grounding。
    """
    config = None
    if use_search:
        grounding_tool = types.Tool(google_search=types.GoogleSearch())
        config = types.GenerateContentConfig(tools=[grounding_tool])

    resp = client.models.generate_content(
        model=model,
        contents=question,
        config=config,
    )
    # 直接取文本；若为空则兜底为""
    return (getattr(resp, "text", None) or "").strip()

def judge_by_gemini(client: genai.Client, question: str, ground_truth: str, model_answer: str, judge_model: str) -> str:
    sys_txt = (
        "你是一个严格的判定器。给定【问题】、【模型回答】和【标准答案】，"
        "只判断模型回答是否与标准答案在关键信息上等价（忽略格式、顺序、大小写）。"
        "如果模型回答不完整、含糊或与标准答案不一致，一律输出 NO。只输出 YES 或 NO，不要解释。"
    )
    user_txt = f"""[问题]
{question}

[模型回答]
{model_answer}

[标准答案]
{ground_truth}

只回答 YES 或 NO："""

    # 直接把两段拼成一个纯字符串传给 contents
    resp = client.models.generate_content(
        model=judge_model,
        contents=f"{sys_txt}\n\n{user_txt}",
    )
    out = (getattr(resp, "text", "") or "").strip().upper()
    return "YES" if out == "YES" else "NO"


def main():
    # python gemini.py --input /home/abaka/zhc/curation/client/final/Sociology/Digital_Sociology.jsonl --output result-gemini.jsonl --use-search
    ap = argparse.ArgumentParser(description="JSONL → Gemini QA → Gemini judge(YES/NO) → 仅保存 NO 的样本")
    ap.add_argument("--input", required=True, help="输入 JSONL 文件路径")
    ap.add_argument("--output", required=True, help="输出 JSONL（仅保存判定为 NO 的样本）")
    ap.add_argument("--answer-model", default="gemini-2.5-flash", help="用于回答的模型名（默认 gemini-2.5-flash）")
    ap.add_argument("--judge-model", default="gemini-2.5-flash", help="用于判定的模型名（默认 gemini-2.5-flash）")
    ap.add_argument("--use-search", action="store_true", help="回答阶段是否启用 Google Search grounding")
    ap.add_argument("--qfield", default="question", help="问题字段名（默认 question）")
    ap.add_argument("--gtfield", default="ground_truth", help="真值字段名（默认 ground_truth）")
    args = ap.parse_args()

    client = build_client()

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
                # 解析失败，直接保存以便排查
                fout.write(line + "\n")
                saved += 1
                continue

            q  = obj.get(args.qfield, "")
            gt = obj.get(args.gtfield, "")

            answer = ask_gemini(client, q, args.answer_model, args.use_search)
            print(answer)
            decision = judge_by_gemini(client, q, gt, answer, args.judge_model)  # "YES"/"NO"
            print(decision)
           

            if decision != "YES":
                obj["_model_answer"] = answer
                obj["_judge"] = decision
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                saved += 1

    print(f"Total: {total} | Saved (judge != YES): {saved}")

if __name__ == "__main__":
    main()
