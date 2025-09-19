#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from google import genai
from google.genai import types
import json
import os
from tqdm import tqdm
from pathlib import Path
from dotenv import load_dotenv

from prompt import QA_SYSTEM
from utils import extract_keys

# ======================= 配置 =======================
# API 配置
load_dotenv()
GENAI_API_BASE_URL = os.getenv("GENAI_API_BASE_URL")
GENAI_API_KEY = os.getenv("GENAI_API_KEY")

# 输入输出配置
INPUT_FILE = Path("/home/abaka/zhc/new/Synthetic-Data-main/client/result/Authentication_and_authorization_in_distributed_environments.jsonl")
OUTPUT_BASE = Path("/home/abaka/zhc/new/Synthetic-Data-main/client/result/browsecomp/Authentication_and_authorization_in_distributed_environments_qa_hard_browsecomp.jsonl")

# 模型配置
MODEL_NAME = "gemini-2.5-pro"

# ======================= 初始化 =======================
if not GENAI_API_KEY or not GENAI_API_BASE_URL:
    raise RuntimeError("Please set GENAI_API_KEY and GENAI_API_BASE_URL environment variables")

client = genai.Client(
    api_key=GENAI_API_KEY,
    http_options=types.HttpOptions(base_url=GENAI_API_BASE_URL),
)

# 配置搜索工具和生成参数
grounding_tool = types.Tool(google_search=types.GoogleSearch())
config = types.GenerateContentConfig(
    tools=[grounding_tool],
    system_instruction=QA_SYSTEM
)

# ======================= 验证函数 =======================

def verify_question_quality(original_question, harder_question):
    """
    使用LLM判断生成的更难问题是否符合要求

    Args:
        original_question: 原始问题
        harder_question: 生成的更难问题

    Returns:
        dict: 包含验证结果和分数的字典
    """
    verify_prompt = f"""
Please evaluate if the harder question is appropriately more challenging than the original question.

Original Question: {original_question}
Harder Question: {harder_question}

Please assess the following criteria and provide a score from 1-10 for each:

1. Difficulty Increase (1-10): Does the harder question require more complex reasoning, knowledge, or skills?
2. Relevance (1-10): Does the harder question remain relevant to the original topic?
3. Clarity (1-10): Is the harder question clear and well-formulated?
4. Answerability (1-10): Is the harder question still answerable with appropriate knowledge?

Provide your assessment in the following JSON format:
{{
    "difficulty_increase": <score>,
    "relevance": <score>,
    "clarity": <score>,
    "answerability": <score>,
    "overall_score": <average_score>,
    "reasoning": "<brief explanation>",
    "approved": <true/false based on overall_score >= 7>
}}
"""

    try:
        # 使用相同的客户端进行验证
        verify_config = types.GenerateContentConfig(
            system_instruction="You are an expert educational content evaluator. Provide objective assessments in the requested JSON format."
        )

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=verify_prompt,
            config=verify_config,
        )

        # 提取响应文本
        verify_text = getattr(response, "text", "")

        # 尝试解析JSON响应
        try:
            # 提取JSON部分
            import re
            json_match = re.search(r'\{.*\}', verify_text, re.DOTALL)
            if json_match:
                verify_result = json.loads(json_match.group())
            else:
                # 如果没找到JSON，创建默认结果
                verify_result = {
                    "difficulty_increase": 5,
                    "relevance": 5,
                    "clarity": 5,
                    "answerability": 5,
                    "overall_score": 5,
                    "reasoning": "Could not parse verification response",
                    "approved": False
                }
        except json.JSONDecodeError:
            verify_result = {
                "difficulty_increase": 5,
                "relevance": 5,
                "clarity": 5,
                "answerability": 5,
                "overall_score": 5,
                "reasoning": "JSON parsing error",
                "approved": False
            }

        return verify_result

    except Exception as e:
        print(f"[WARN] 验证过程出错：{e}")
        return {
            "difficulty_increase": 0,
            "relevance": 0,
            "clarity": 0,
            "answerability": 0,
            "overall_score": 0,
            "reasoning": f"Verification error: {str(e)}",
            "approved": False
        }

# ======================= 主流程 =======================

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
    verification_log_path = output_dir / f"{INPUT_FILE.stem}_verification.jsonl"

    with output_path.open("a", encoding="utf-8") as fout, \
         verification_log_path.open("a", encoding="utf-8") as vlog:

        for item in tqdm(data_line, desc="Processing"):
            # 将抽取到的三元组作为用户输入
            response = client.models.generate_content(
                model="gemini-2.5-pro",
                contents=str(item),
                config=config,
            )

            # 等价于之前的 response.output_text
            out_text = getattr(response, "text", "")
            print(f"Generated: {out_text}")

            # 验证生成的问题质量
            try:
                # 假设item包含原始问题，out_text包含生成的更难问题
                original_question = str(item)
                harder_question = out_text

                verification_result = verify_question_quality(original_question, harder_question)

                # 打印验证结果
                print(f"Verification Score: {verification_result.get('overall_score', 0):.1f}/10")
                print(f"Approved: {verification_result.get('approved', False)}")

                # 记录验证结果
                log_entry = {
                    "original_question": original_question,
                    "harder_question": harder_question,
                    "verification": verification_result,
                    "timestamp": str(Path(__file__).stat().st_mtime)  # 简单时间戳
                }
                vlog.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

                # 只有通过验证的问题才写入最终输出
                if verification_result.get("approved", False):
                    fout.write(str(out_text) + "\n")
                    print("✅ Question approved and saved")
                else:
                    print("❌ Question rejected by verification")

            except Exception as e:
                print(f"[WARN] 验证过程出错，使用原输出：{e}")
                # 如果验证失败，仍然保存原输出
                fout.write(str(out_text) + "\n")

    print(f"\n✅ 全部完成！")
    print(f"结果已写入：{output_path}")
    print(f"验证日志已写入：{verification_log_path}")

if __name__ == "__main__":
    main()