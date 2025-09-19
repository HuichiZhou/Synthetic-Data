"""
Text processing utility functions
"""

import json
import re
from typing import Any, Dict
import os, json, argparse
from typing import Dict, Any
from google import genai
from google.genai import types

API_KEY =  os.getenv("GENAI_API_KEY")
GENAI_API_BASE_URL = os.getenv("GENAI_API_BASE_URL")


def build_client() -> genai.Client:
    return genai.Client(api_key=API_KEY, http_options=types.HttpOptions(base_url=GENAI_API_BASE_URL))

def ask_gemini(client: genai.Client, question: str, model: str, use_search: bool) -> str:
    """
    用 Gemini 回答问题；可选启用 Google Search grounding。
    """
    config = None
    if use_search:
        grounding_tool = types.Tool(google_search=types.GoogleSearch())
        # config = types.GenerateContentConfig(tools=[grounding_tool])

    resp = client.models.generate_content(
        model=model,
        contents=question,
        # config=config,
    )
    # 直接取文本；若为空则兜底为""
    return (getattr(resp, "text", None) or "").strip()


def normalize(s: str) -> str:
    """
    标准化字符串：去除多余空格并转为小写
    
    Args:
        s: 要标准化的字符串
        
    Returns:
        标准化后的字符串
    """
    return re.sub(r"\s+", " ", (s or "").strip().lower())


    
def answer_matches(pred: str, truth: str) -> bool:
    """
    比较两个答案是否匹配（宽松匹配）
    
    Args:
        pred: 预测答案
        truth: 真实答案
        
    Returns:
        是否匹配
    """
    # 去除标点符号并标准化
    # p = normalize(re.sub(r"[^\w\s]", "", pred))
    # t = normalize(re.sub(r"[^\w\s]", "", truth))
    
    # # 完全匹配或包含关系（长度大于3）
    # return p == t or (p in t and len(p) > 3) or (t in p and len(t) > 3)

    # please help me to writw a llm judge to do answer match

    prompt = f"Please judge if the following two answers are the same: {pred} and {truth}, only return True or False"

    res = ask_gemini(build_client(), prompt, "gemini-2.5-flash", True)
    if res == "True":
        return True
    else:
        return False

def truncate(s: str, max_len: int) -> str:
    """
    截断字符串，保留首尾部分
    
    Args:
        s: 要截断的字符串
        max_len: 最大长度
        
    Returns:
        截断后的字符串
    """
    if len(s) <= max_len:
        return s
    keep = max_len // 2
    return s[:keep] + "\n[...SNIP...]\n" + s[-keep:]


def response_format(json_string_raw: Any) -> Dict[str, Any]:
    """
    格式化JSON响应，处理Markdown代码块和异常情况
    
    Args:
        json_string_raw: 原始JSON字符串或数据
        
    Returns:
        Dict[str, Any]: 解析后的JSON数据
        
    Raises:
        ValueError: 当JSON格式无效或解析失败时
    """
    try:
        if not isinstance(json_string_raw, str) or not json_string_raw.strip():
            if json_string_raw is None:
                raise ValueError("JSON数据为None")
            elif not isinstance(json_string_raw, str):
                raise ValueError(f"非字符串数据类型: {type(json_string_raw)}")
            else:
                raise ValueError("空字符串数据")
        
        # 处理Markdown代码块
        processed_json_string = json_string_raw.strip()
        if processed_json_string.startswith("```json") and processed_json_string.endswith("```"):
            processed_json_string = processed_json_string[len("```json"):-len("```")].strip()
        elif processed_json_string.startswith("```") and processed_json_string.endswith("```"):
            processed_json_string = processed_json_string[len("```"):-len("```")].strip()

        if not processed_json_string:
            raise ValueError("移除Markdown后为空字符串")
        
        # 添加调试信息
        print(f"[DEBUG] 尝试解析JSON: {processed_json_string[:100]}..." if len(processed_json_string) > 100 else f"[DEBUG] 尝试解析JSON: {processed_json_string}")
        
        analysis_data = json.loads(processed_json_string)
        return analysis_data

    except json.JSONDecodeError as e:
        print(f"[ERROR] JSON解析失败，原始数据: {repr(json_string_raw)}")
        print(f"[ERROR] 处理后数据: {repr(processed_json_string) if 'processed_json_string' in locals() else 'N/A'}")
        raise ValueError(f"JSON解析错误: {e}")
    except Exception as e:
        print(f"[ERROR] 其他错误: {e}")
        print(f"[ERROR] 原始数据: {repr(json_string_raw)}")
        raise ValueError(f"响应格式化失败: {e}")
