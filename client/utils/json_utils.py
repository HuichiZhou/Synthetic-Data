"""
JSON utility functions for synthetic data generation
"""

import json
import re
from typing import Any


def safe_json_loads(s: str) -> Any:
    """
    安全解析JSON字符串，支持容错处理
    
    Args:
        s: 要解析的JSON字符串
        
    Returns:
        解析后的JSON对象，如果解析失败返回None
    """
    try:
        return json.loads(s)
    except Exception:
        # 尝试修复常见的JSON格式问题
        s2 = re.sub(r",\s*}", "}", s)
        s2 = re.sub(r",\s*]", "]", s2)
        try:
            return json.loads(s2)
        except Exception:
            return None