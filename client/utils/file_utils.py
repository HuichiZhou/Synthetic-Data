"""
File processing utility functions
"""

import json
from typing import Tuple


def extract_keys(json_line: str) -> Tuple[str, str, str]:
    """
    从JSONL行中提取question, ground_truth, data_source三个字段
    
    Args:
        json_line: JSONL格式的一行字符串
        
    Returns:
        包含三个字段的元组: (question, ground_truth, data_source)
    """
    data = json.loads(json_line)
    return (
        f"Question:{data['question']}",
        f"Ground Truth:{data['ground_truth']}",
        f"Data Source:{data['data_source']}",
    )