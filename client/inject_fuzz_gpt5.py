from openai import OpenAI
import os
import json
from tqdm import tqdm
from pathlib import Path
from dotenv import load_dotenv

from client.prompt import QA_SYSTEM

# ======================= 配置 =======================
# API 配置
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")

# 输入输出配置
INPUT_FILE = Path("/home/abaka/zhc/curation/client/final/Economics/Behavioral_Economics.jsonl")
OUTPUT_DIR = Path("result/Economics")
OUTPUT_FILE = OUTPUT_DIR / "Behavioral_Economics.jsonl"

# 模型配置
MODEL_NAME = "gpt-5-search"

# ======================= 初始化 =======================
load_dotenv()

if not OPENAI_API_KEY or not OPENAI_BASE_URL:
    raise RuntimeError("Please set OPENAI_API_KEY and OPENAI_BASE_URL environment variables")

client = OpenAI(base_url=OPENAI_BASE_URL, api_key=OPENAI_API_KEY)

# ======================= 工具函数 =======================

def extract_keys(json_line: str) -> tuple[str, str, str]:
    """从JSONL行中提取question, ground_truth, data_source三个字段"""
    json_data = json.loads(json_line)
    return (
        f"Question:{json_data['question']}",
        f"Ground Truth:{json_data['ground_truth']}",
        f"Data Source:{json_data['data_source']}"
    )

# ======================= 主流程 =======================

def main():
    # 读取输入数据
    data_lines = []
    with INPUT_FILE.open("r") as f:
        for line in f:
            if line.strip():  # 跳过空行
                data_lines.append(line)
    
    # 提取关键字段
    processed_data = []
    for line in data_lines:
        try:
            processed_data.append(extract_keys(line))
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Skipping invalid line - {e}")
            continue
    
    # 确保输出目录存在
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 处理每条数据
    for item in tqdm(processed_data, desc="Processing questions"):
        try:
            response = client.responses.create(
                model=MODEL_NAME,
                tools=[{"type": "web_search_preview"}],
                input=QA_SYSTEM + str(item),
                stream=False
            )
            
            print(response.output_text)
            
            # 保存结果
            with OUTPUT_FILE.open("a", encoding="utf-8") as f:
                f.write(response.output_text + "\n")
                
        except Exception as e:
            print(f"Error processing item {item}: {e}")
            continue

if __name__ == "__main__":
    main()
