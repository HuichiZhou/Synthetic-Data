"""
优化的异步Gemini客户端工具
提供高性能的异步Gemini API调用，支持批量处理、错误重试和并发控制
"""

import asyncio
import json
import os
import logging
from typing import Any, Dict, List, Optional, Union, Callable
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.types import GenerateContentConfig, Tool, GoogleSearch
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AsyncGeminiClient:
    """
    高性能异步Gemini客户端
    
    特性:
    - 异步批量处理
    - 并发控制
    - 自动重试机制
    - 进度跟踪
    - 错误恢复
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = "gemini-2.5-pro",
        max_concurrent: int = 10,
        timeout: int = 30,
        max_retries: int = 3
    ):
        """
        初始化异步Gemini客户端
        
        Args:
            api_key: Gemini API密钥，如果为None则从环境变量读取
            base_url: 自定义API基础URL
            model: 使用的模型名称
            max_concurrent: 最大并发数
            timeout: 请求超时时间（秒）
            max_retries: 最大重试次数
        """
        self.api_key = api_key or os.getenv("GENAI_API_KEY")
        self.base_url = base_url or os.getenv("GENAI_API_BASE_URL")
        self.model = model
        self.max_concurrent = max_concurrent
        self.timeout = timeout
        self.max_retries = max_retries
        
        if not self.api_key:
            raise ValueError("必须提供API密钥或设置GENAI_API_KEY环境变量")
        
        # 初始化客户端
        self.client = genai.Client(
            api_key=self.api_key,
            http_options=types.HttpOptions(
                base_url=self.base_url,
                timeout=self.timeout * 1000  # 转换为毫秒
            )
        )
        
        # 创建信号量控制并发
        self.semaphore = asyncio.Semaphore(max_concurrent)
        
        # 默认配置
        self.default_config = GenerateContentConfig(
            tools=[Tool(google_search=GoogleSearch())],
            temperature=0.7,
            max_output_tokens=2048,
            top_p=0.95,
            top_k=40
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((Exception,))
    )
    async def generate_single(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        config_override: Optional[Dict] = None
    ) -> str:
        """
        异步生成单个响应
        
        Args:
            prompt: 用户提示词
            system_instruction: 系统提示词
            config_override: 配置覆盖参数
            
        Returns:
            生成的文本内容
        """
        async with self.semaphore:
            try:
                config = self.default_config.copy()
                
                if system_instruction:
                    config.system_instruction = system_instruction
                
                if config_override:
                    for key, value in config_override.items():
                        setattr(config, key, value)
                
                response = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.client.models.generate_content(
                        model=self.model,
                        contents=prompt,
                        config=config
                    )
                )
                
                return getattr(response, "text", "")
                
            except Exception as e:
                logger.error(f"生成失败: {e}")
                raise

    async def generate_batch(
        self,
        prompts: List[str],
        system_instruction: Optional[str] = None,
        config_override: Optional[Dict] = None,
        show_progress: bool = True
    ) -> List[str]:
        """
        批量异步生成响应
        
        Args:
            prompts: 提示词列表
            system_instruction: 系统提示词
            config_override: 配置覆盖参数
            show_progress: 是否显示进度
            
        Returns:
            生成的文本内容列表
        """
        if show_progress:
            from tqdm.asyncio import tqdm_asyncio
            
            tasks = [
                self.generate_single(prompt, system_instruction, config_override)
                for prompt in prompts
            ]
            
            results = await tqdm_asyncio.gather(*tasks, desc="生成进度")
            return results
        else:
            tasks = [
                self.generate_single(prompt, system_instruction, config_override)
                for prompt in prompts
            ]
            
            return await asyncio.gather(*tasks, return_exceptions=True)

    async def process_jsonl(
        self,
        input_file: Union[str, Path],
        output_file: Union[str, Path],
        prompt_template: str,
        system_instruction: Optional[str] = None,
        batch_size: int = 50,
        show_progress: bool = True
    ) -> Dict[str, Any]:
        """
        异步处理JSONL文件
        
        Args:
            input_file: 输入JSONL文件路径
            output_file: 输出文件路径
            prompt_template: 提示词模板
            system_instruction: 系统提示词
            batch_size: 每批处理的数据量
            show_progress: 是否显示进度
            
        Returns:
            处理统计信息
        """
        input_path = Path(input_file)
        output_path = Path(output_file)
        
        # 确保输出目录存在
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 读取输入数据
        lines = []
        with input_path.open('r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if line.strip():
                    try:
                        data = json.loads(line.strip())
                        lines.append(data)
                    except json.JSONDecodeError as e:
                        logger.warning(f"跳过无效JSON行 {line_num}: {e}")
        
        logger.info(f"读取 {len(lines)} 条记录")
        
        # 处理统计
        success_count = 0
        error_count = 0
        
        # 分批处理
        with output_path.open('w', encoding='utf-8') as f:
            for i in range(0, len(lines), batch_size):
                batch = lines[i:i+batch_size]
                prompts = [
                    prompt_template.format(**item) if isinstance(item, dict) else str(item)
                    for item in batch
                ]
                
                results = await self.generate_batch(
                    prompts, 
                    system_instruction=system_instruction,
                    show_progress=show_progress
                )
                
                # 写入结果
                for j, result in enumerate(results):
                    original_data = batch[j]
                    
                    if isinstance(result, Exception):
                        error_count += 1
                        output_data = {
                            "original": original_data,
                            "error": str(result),
                            "status": "failed"
                        }
                    else:
                        success_count += 1
                        output_data = {
                            "original": original_data,
                            "generated": result,
                            "status": "success"
                        }
                    
                    f.write(json.dumps(output_data, ensure_ascii=False) + '\n')
                
                if show_progress:
                    logger.info(f"处理进度: {min(i+batch_size, len(lines))}/{len(lines)}")
        
        stats = {
            "total": len(lines),
            "success": success_count,
            "errors": error_count,
            "success_rate": success_count / len(lines) if lines else 0
        }
        
        logger.info(f"处理完成: {stats}")
        return stats

    async def chat_with_history(
        self,
        messages: List[Dict[str, str]],
        config_override: Optional[Dict] = None
    ) -> str:
        """
        带历史记录的异步对话
        
        Args:
            messages: 消息列表，格式为[{"role": "user/system", "content": "内容"}]
            config_override: 配置覆盖参数
            
        Returns:
            生成的响应文本
        """
        async with self.semaphore:
            try:
                config = self.default_config.copy()
                
                if config_override:
                    for key, value in config_override.items():
                        setattr(config, key, value)
                
                # 转换消息格式
                formatted_messages = []
                for msg in messages:
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    formatted_messages.append(types.Content(role=role, parts=[types.Part(text=content)]))
                
                response = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.client.models.generate_content(
                        model=self.model,
                        contents=formatted_messages,
                        config=config
                    )
                )
                
                return getattr(response, "text", "")
                
            except Exception as e:
                logger.error(f"对话失败: {e}")
                raise

    def sync_generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        config_override: Optional[Dict] = None
    ) -> str:
        """
        同步生成单个响应（用于兼容旧代码）
        
        Args:
            prompt: 用户提示词
            system_instruction: 系统提示词
            config_override: 配置覆盖参数
            
        Returns:
            生成的文本内容
        """
        return asyncio.run(self.generate_single(prompt, system_instruction, config_override))


# 便捷函数和工具
async def quick_generate(
    prompt: str,
    api_key: Optional[str] = None,
    **kwargs
) -> str:
    """
    快速生成单个响应的便捷函数
    
    Args:
        prompt: 提示词
        api_key: API密钥
        **kwargs: 其他参数
        
    Returns:
        生成的文本
    """
    client = AsyncGeminiClient(api_key=api_key)
    return await client.generate_single(prompt, **kwargs)


# 使用示例
async def example_usage():
    """
    使用示例
    """
    # 初始化客户端
    load_dotenv()
    client = AsyncGeminiClient(max_concurrent=5)
    
    # 示例1: 单个生成
    response = await client.generate_single(
        "请解释什么是量子纠缠",
        system_instruction="你是一个物理学专家"
    )
    print("单个响应:", response[:100] + "...")
    
    # 示例2: 批量生成
    concepts = ["机器学习", "区块链", "神经网络"]
    responses = await client.generate_batch(
        [f"请解释什么是{c}" for c in concepts],
        system_instruction="用简洁的语言解释"
    )
    for concept, resp in zip(concepts, responses):
        print(f"{concept}: {resp[:50]}...")
    
    # 示例3: 处理JSONL文件
    # await client.process_jsonl(
    #     "input.jsonl",
    #     "output.jsonl",
    #     prompt_template="请解释以下概念: {concept}",
    #     system_instruction="用通俗易懂的语言"
    # )
    
    # 示例4: 带历史记录的对话
    messages = [
        {"role": "user", "content": "什么是人工智能？"},
        {"role": "assistant", "content": "人工智能是..."},
        {"role": "user", "content": "它有哪些应用？"}
    ]
    response = await client.chat_with_history(messages)
    print("对话响应:", response[:100] + "...")


if __name__ == "__main__":
    # 运行示例
    asyncio.run(example_usage())