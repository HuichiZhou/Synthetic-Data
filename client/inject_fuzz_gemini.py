#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from google import genai
from google.genai import types
import json
import os
from tqdm import tqdm
from pathlib import Path
from dotenv import load_dotenv

from client.prompt import QA_SYSTEM

# ======================= 配置 =======================
# API 配置
GENAI_API_BASE_URL = os.getenv("GENAI_API_BASE_URL")
GENAI_API_KEY = os.getenv("GENAI_API_KEY")

# 输入输出配置
INPUT_FILE = Path("/home/abaka/zhc/curation/client/final/Biology/Synthetic_Biology.jsonl")
OUTPUT_BASE = Path("result-gemini")

# 模型配置
MODEL_NAME = "gemini-2.5-pro"

# ======================= 初始化 =======================
load_dotenv()

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

# ======================= 工具函数 =======================

def extract_keys(json_line: str) -> tuple[str, str, str]:
    """从JSONL行中提取question, ground_truth, data_source三个字段"""
    data = json.loads(json_line)
    return (
        f"Question:{data['question']}",
        f"Ground Truth:{data['ground_truth']}",
        f"Data Source:{data['data_source']}",
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
