"""
Text processing utility functions
"""

import json
import re
from typing import Any, Dict


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
    p = normalize(re.sub(r"[^\w\s]", "", pred))
    t = normalize(re.sub(r"[^\w\s]", "", truth))
    
    # 完全匹配或包含关系（长度大于3）
    return p == t or (p in t and len(p) > 3) or (t in p and len(t) > 3)


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
        AnalysisError: 当JSON格式无效或解析失败时
    """
    try:
        if not isinstance(json_string_raw, str) or not json_string_raw.strip():
            if json_string_raw is None:
                raise ("JSON数据为None")
            elif not isinstance(json_string_raw, str):
                raise (f"非字符串数据类型: {type(json_string_raw)}")
            else:
                raise ("空字符串数据")
        
        # 处理Markdown代码块
        processed_json_string = json_string_raw.strip()
        if processed_json_string.startswith("```json") and processed_json_string.endswith("```"):
            processed_json_string = processed_json_string[len("```json"):-len("```")].strip()
        elif processed_json_string.startswith("```") and processed_json_string.endswith("```"):
            processed_json_string = processed_json_string[len("```"):-len("```")].strip()

        if not processed_json_string:
            raise ("移除Markdown后为空字符串")
        
        analysis_data = json.loads(processed_json_string)
        return analysis_data

    except json.JSONDecodeError as e:
        raise (f"JSON解析错误: {e}")
    except:
        raise