import json
import time
import random
import os
from typing import List, Optional, Dict, Set, Any
from dotenv import load_dotenv
from openai import OpenAI
from serpapi import Client
from pathlib import Path

from client.prompt import ENTITY_CATEGORIES

# ======================= 配置 =======================
# API 配置
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPAPI_KEY = os.getenv("SERPAPI_KEY")

# 模型配置
DEFAULT_MODEL = "gpt-4o"

# 处理参数
MAX_VALIDATION_ATTEMPTS = 5
MAX_FEATURES = 8
DEFAULT_ROUNDS = 10

# 输出配置
OUTPUT_FILE = "entity_questions.jsonl"

class GPTEntityProcessor:
    def __init__(self, api_key: str, serpapi_key: str, model: str = DEFAULT_MODEL):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.serpapi_client = Client(api_key=serpapi_key)
        
        # 状态管理
        self.generated_entities: Set[str] = set()
        self.introduction_snippets: Set[str] = set()
        self.feature_search_info: str = ""
        
        # Token 统计
        self.round_prompt_tokens = 0
        self.round_completion_tokens = 0
        self.round_total_tokens = 0

    def _send_request(self, messages: List[Dict[str, str]], temperature: float = 0.7) -> Any:
        """发送请求到OpenAI API并统计token使用量"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature
            )
            # 统计token
            if hasattr(response, 'usage'):
                prompt_tokens = response.usage.prompt_tokens
                completion_tokens = response.usage.completion_tokens
                total_tokens = response.usage.total_tokens
                
                # 累加本轮token
                self.round_prompt_tokens += prompt_tokens
                self.round_completion_tokens += completion_tokens
                self.round_total_tokens += total_tokens
            return response
        except Exception as e:
            time.sleep(5)
            return self._send_request(messages, temperature)

    def reset_token_counters(self) -> None:
        """重置本轮token计数器"""
        self.round_prompt_tokens = 0
        self.round_completion_tokens = 0
        self.round_total_tokens = 0

    def extract_obscure_entity_from_category(self, category: str) -> Optional[str]:
        """从指定类别中提取一个冷门实体"""
        for attempt in range(self.max_validation_attempts):
            messages = [
                {"role": "system", "content": f"""You need to provide a single obscure entity from the category: {category}.
                The entity should be real, specific, and not too well-known.
                Do not provide extremely obscure entities that don't exist.
                Respond ONLY with the entity name, nothing else."""},
                {"role": "user", "content": f"Provide one obscure entity from the category: {category}. Just the name."}
            ]
            
            response = self._send_request(messages, temperature=0.8)
            if response and response.choices[0].message.content:
                entity = response.choices[0].message.content.strip()
                
                if entity in self.generated_entities:
                    continue
                
                if self._validate_entity_existence(entity, category):
                    self.generated_entities.add(entity)
                    return entity
        
        return None

    def _validate_entity_existence(self, entity: str, category: str) -> bool:
        """使用SERPAPI验证实体存在性"""
        try:
            result = self.serpapi_client.search({
                "engine": "google",
                "q": f"{entity} {category}",
                "no_cache": True
            })
            result_dict = result.as_dict()
            
            if "organic_results" in result_dict and len(result_dict["organic_results"]) > 0:
                return True
            else:
                return False
                
        except Exception as e:
            return self._model_based_validation(entity, category)

    def _model_based_validation(self, entity: str, category: str) -> bool:
        """模型验证作为降级方案"""
        messages = [
            {"role": "system", "content": f"""Based on your knowledge, determine if the entity exists in the specified {category} category.
            Respond ONLY with 'YES' or 'NO' after careful consideration."""},
            {"role": "user", "content": f"Is '{entity}' a real entity in the {category} category? Answer ONLY 'YES' or 'NO'."}
        ]
        
        response = self._send_request(messages, temperature=0.2)
        if response and response.choices[0].message.content:
            return response.choices[0].message.content.strip().upper() == "YES"
        return False

    def generate_introduction(self, entity: str, category: str) -> str:
        """生成实体介绍并获取特征提取用的新信息"""
        self.introduction_snippets.clear()
        intro_search_info = self._get_search_info(entity, category, num_results=3)
        
        messages = [
            {"role": "system", "content": f"""Write a 150-200 word accurate introduction to the {category} entity.
            Use the provided search information as reference. Focus on key verified details."""},
            {"role": "user", "content": f"Entity: {entity}\nSearch reference: {intro_search_info}\nProvide a detailed introduction."}
        ]
        
        response = self._send_request(messages)
        if response and response.choices[0].message.content:
            intro = response.choices[0].message.content.strip()
            self._get_feature_search_info(entity, category)
            return intro
        return ""

    def _get_search_info(self, entity: str, category: str, num_results: int = 3) -> str:
        """获取实体基础信息"""
        try:
            result = self.serpapi_client.search({
                "engine": "google",
                "q": entity,
                "num": num_results,
                "no_cache": True
            })
            result_dict = result.as_dict()
            
            info_parts = []
            if "organic_results" in result_dict:
                for item in result_dict["organic_results"][:num_results]:
                    if "snippet" in item:
                        snippet = item["snippet"]
                        info_parts.append(snippet)
                        self.introduction_snippets.add(snippet.lower())
            
            return "\n".join(info_parts) or "No search information available"
            
        except Exception as e:
            return "No search information available"

    def _get_feature_search_info(self, entity: str, category: str) -> None:
        """获取用于特征提取的新网页信息"""
        try:
            result = self.serpapi_client.search({
                "engine": "google",
                "q": f"{entity} {category}",
                "num": 5,
                "start": 3,
                "no_cache": True
            })
            result_dict = result.as_dict()
            
            feature_snippets = []
            if "organic_results" in result_dict:
                for item in result_dict["organic_results"]:
                    if "snippet" in item:
                        snippet = item["snippet"].lower()
                        is_duplicate = any(intro_snippet in snippet or snippet in intro_snippet 
                                          for intro_snippet in self.introduction_snippets)
                        
                        if not is_duplicate:
                            feature_snippets.append(item["snippet"])
            
            self.feature_search_info = "\n".join(feature_snippets) or "No additional information available"
            
        except Exception as e:
            self.feature_search_info = "No additional information available"

    def extract_single_feature(self, entity: str, category: str, existing_features: List[str]) -> Optional[str]:
        """提取泛化特征（单个可指向多个实体）"""
        if self.feature_search_info == "No additional information available":
            system_content = f"""Extract a single feature of the {category} entity '{entity}'.
            IMPORTANT: This feature should be GENERALIZABLE and could apply to MULTIPLE entities in the same category.
            It should be non-trivial, relevant to this entity, and not already mentioned in existing features.
            The feature alone should NOT uniquely identify the entity - uniqueness comes from combining multiple such features.
            Respond ONLY with the feature, nothing else."""
        else:
            system_content = f"""Extract a single feature of the {category} entity '{entity}' 
            based ONLY on the following new information (not used in the introduction):
            {self.feature_search_info}
            
            IMPORTANT: This feature should be GENERALIZABLE and could apply to MULTIPLE entities in the same category.
            It should be non-trivial, relevant to this entity, and not already mentioned in existing features.
            The feature alone should NOT uniquely identify the entity - uniqueness comes from combining multiple such features.
            Respond ONLY with the feature, nothing else."""
        
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": f"Entity: {entity}\nCategory: {category}\nExisting features: {existing_features}\nProvide a new generalizable feature."}
        ]
        
        response = self._send_request(messages, temperature=0.6)
        if response and response.choices[0].message.content:
            feature = response.choices[0].message.content.strip()
            if feature and feature not in existing_features:
                return feature
        return None

    def check_uniqueness(self, entity: str, category: str, features: List[str]) -> bool:
        """检查特征组合是否能唯一标识实体"""
        messages = [
            {"role": "system", "content": f"""Determine if the COMBINATION of provided features uniquely identifies the {category} entity '{entity}'.
            Individual features may apply to multiple entities, but together they should only apply to this one.
            If you can think of another entity in the same category that matches ALL these features, respond NO.
            Otherwise, respond YES. Respond ONLY with 'YES' or 'NO'."""},
            {"role": "user", "content": f"Entity: {entity}\nCategory: {category}\nFeatures: {features}\nDo these features together uniquely identify the entity? Answer ONLY 'YES' or 'NO'."}
        ]
        
        response = self._send_request(messages, temperature=0.2)
        if response and response.choices[0].message.content:
            return response.choices[0].message.content.strip().upper() == "YES"
        return False

    def extract_features_until_unique(self, entity: str, category: str, max_features: int = 8) -> List[str]:
        """提取更多特征直到组合能唯一标识实体"""
        features: List[str] = []
        
        for _ in range(max_features):
            if self.check_uniqueness(entity, category, features):
                break
                
            feature = self.extract_single_feature(entity, category, features)
            if feature:
                features.append(feature)
            else:
                break
                
        return features

    def merge_features_into_description(self, features: List[str]) -> str:
        """将多个泛化特征合并为描述"""
        messages = [
            {"role": "system", "content": "Merge the provided features into a single coherent descriptive paragraph. Do not mention the entity name. Keep it natural and flowing."},
            {"role": "user", "content": f"Features: {features}\nMerge these into a descriptive paragraph without mentioning the entity name."}
        ]
        
        response = self._send_request(messages, temperature=0.7)
        if response and response.choices[0].message.content:
            return response.choices[0].message.content.strip()
        return ""

    def generate_question(self, description: str, category: str) -> str:
        """根据描述生成询问实体名称的问题"""
        messages = [
            {"role": "system", "content": f"""Generate a natural question asking for the name of the {category} based on the provided description.
            The question should require knowledge of the entity to answer correctly."""},
            {"role": "user", "content": f"Description: {description}\nCategory: {category}\nGenerate a question asking for the entity's name."}
        ]
        
        response = self._send_request(messages, temperature=0.7)
        if response and response.choices[0].message.content:
            return response.choices[0].message.content.strip()
        return ""

def main(num_rounds: int = 10, output_file: str = "entity_questions.jsonl"):
    api_key = os.getenv("OPENAI_API_KEY")
    serpapi_key = os.getenv("SERPAPI_KEY")
    
    if not api_key or not serpapi_key:
        raise ValueError("请设置环境变量 OPENAI_API_KEY 和 SERPAPI_KEY")
    
    processor = GPTEntityProcessor(api_key, serpapi_key)
    # 总token统计
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_total_tokens = 0
    
    for round_num in range(1, num_rounds + 1):
        # 重置本轮token计数器
        processor.reset_token_counters()
        
        category = random.choice(ENTITY_CATEGORIES)
        
        entity = processor.extract_obscure_entity_from_category(category)
        if not entity:
            continue
        
        introduction = processor.generate_introduction(entity, category)
        if not introduction:
            continue
        
        features = processor.extract_features_until_unique(entity, category)
        if not features:
            continue
        
        description = processor.merge_features_into_description(features)
        if not description:
            continue
        
        question = processor.generate_question(description, category)
        if not question:
            continue
        
        # 结果数据结构
        result = {
            "id": int(time.time() * 1000),  # 毫秒级时间戳作为ID
            "entity": entity,
            "question": question
        }
        
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
        
        # 累加总统计
        total_prompt_tokens += processor.round_prompt_tokens
        total_completion_tokens += processor.round_completion_tokens
        total_total_tokens += processor.round_total_tokens

if __name__ == "__main__":
    main(num_rounds=10)
