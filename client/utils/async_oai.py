"""
优化的异步OpenAI客户端工具
提供高性能的异步OpenAI API调用，支持批量处理、错误重试和并发控制
"""

import asyncio
import json
import os
import logging
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
from dotenv import load_dotenv
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AsyncOpenAIClient:
    """
    高性能异步OpenAI客户端
    
    特性:
    - 异步批量处理
    - 并发控制
    - 自动重试机制
    - 进度跟踪
    - 错误恢复
    - 支持多种模型
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = "gpt-4o",
        max_concurrent: int = 10,
        timeout: int = 30,
        max_retries: int = 3,
        temperature: float = 0.7,
        max_tokens: int = 2048
    ):
        """
        初始化异步OpenAI客户端
        
        Args:
            api_key: OpenAI API密钥，如果为None则从环境变量读取
            base_url: 自定义API基础URL（用于代理）
            model: 使用的模型名称
            max_concurrent: 最大并发数
            timeout: 请求超时时间（秒）
            max_retries: 最大重试次数
            temperature: 生成温度
            max_tokens: 最大输出长度
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
        self.model = model
        self.max_concurrent = max_concurrent
        self.timeout = timeout
        self.max_retries = max_retries
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        if not self.api_key:
            raise ValueError("必须提供API密钥或设置OPENAI_API_KEY环境变量")
        
        # 初始化异步客户端
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout
        )
        
        # 创建信号量控制并发
        self.semaphore = asyncio.Semaphore(max_concurrent)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((Exception,))
    )
    async def generate_single(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        response_format: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> str:
        """
        异步生成单个响应
        
        Args:
            prompt: 用户提示词
            system_instruction: 系统提示词
            response_format: 响应格式（JSON模式）
            **kwargs: 其他OpenAI参数
            
        Returns:
            生成的文本内容
        """
        async with self.semaphore:
            try:
                messages = []
                
                if system_instruction:
                    messages.append({"role": "system", "content": system_instruction})
                
                messages.append({"role": "user", "content": prompt})
                
                params = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": kwargs.get("temperature", self.temperature),
                    "max_tokens": kwargs.get("max_tokens", self.max_tokens),
                    "timeout": self.timeout
                }
                
                if response_format:
                    params["response_format"] = response_format
                
                # 合并其他参数
                params.update(kwargs)
                
                response = await self.client.chat.completions.create(**params)
                
                return response.choices[0].message.content or ""
                
            except Exception as e:
                logger.error(f"生成失败: {e}")
                raise

    async def generate_json(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        异步生成JSON格式的响应
        
        Args:
            prompt: 用户提示词
            system_instruction: 系统提示词
            **kwargs: 其他OpenAI参数
            
        Returns:
            解析后的JSON对象
        """
        response = await self.generate_single(
            prompt,
            system_instruction,
            response_format={"type": "json_object"},
            **kwargs
        )
        
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            logger.warning("响应不是有效的JSON格式")
            return {}

    async def generate_batch(
        self,
        prompts: List[str],
        system_instruction: Optional[str] = None,
        response_format: Optional[Dict[str, Any]] = None,
        show_progress: bool = True,
        **kwargs
    ) -> List[str]:
        """
        批量异步生成响应
        
        Args:
            prompts: 提示词列表
            system_instruction: 系统提示词
            response_format: 响应格式
            show_progress: 是否显示进度
            **kwargs: 其他OpenAI参数
            
        Returns:
            生成的文本内容列表
        """
        if show_progress:
            try:
                from tqdm.asyncio import tqdm_asyncio
                
                tasks = [
                    self.generate_single(prompt, system_instruction, response_format, **kwargs)
                    for prompt in prompts
                ]
                
                results = await tqdm_asyncio.gather(*tasks, desc="生成进度")
                return results
            except ImportError:
                logger.warning("tqdm未安装，不显示进度")
        
        tasks = [
            self.generate_single(prompt, system_instruction, response_format, **kwargs)
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
        show_progress: bool = True,
        **kwargs
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
            **kwargs: 其他OpenAI参数
            
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
                    show_progress=show_progress,
                    **kwargs
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
        **kwargs
    ) -> str:
        """
        带历史记录的异步对话
        
        Args:
            messages: 消息列表，格式为[{"role": "user/system/assistant", "content": "内容"}]
            **kwargs: 其他OpenAI参数
            
        Returns:
            生成的响应文本
        """
        async with self.semaphore:
            try:
                params = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": kwargs.get("temperature", self.temperature),
                    "max_tokens": kwargs.get("max_tokens", self.max_tokens),
                    "timeout": self.timeout
                }
                params.update(kwargs)
                
                response = await self.client.chat.completions.create(**params)
                return response.choices[0].message.content or ""
                
            except Exception as e:
                logger.error(f"对话失败: {e}")
                raise

    def sync_generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        同步生成单个响应（用于兼容旧代码）
        
        Args:
            prompt: 用户提示词
            system_instruction: 系统提示词
            **kwargs: 其他OpenAI参数
            
        Returns:
            生成的文本内容
        """
        return asyncio.run(self.generate_single(prompt, system_instruction, **kwargs))


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
    client = AsyncOpenAIClient(api_key=api_key)
    return await client.generate_single(prompt, **kwargs)


# 使用示例
async def example_usage():
    """
    使用示例
    """
    # 初始化客户端
    load_dotenv()
    client = AsyncOpenAIClient(max_concurrent=5)
    
    # 示例1: 单个生成
    response = await client.generate_single(
        "请解释什么是量子计算",
        system_instruction="你是一个物理学专家"
    )
    print("单个响应:", response[:100] + "...")
    
    # 示例2: JSON格式生成
    json_response = await client.generate_json(
        "创建一个关于机器学习的JSON对象，包含定义、应用和前景",
        system_instruction="返回严格的JSON格式"
    )
    print("JSON响应:", json.dumps(json_response, ensure_ascii=False, indent=2))
    
    # 示例3: 批量生成
    concepts = ["深度学习", "自然语言处理", "计算机视觉"]
    responses = await client.generate_batch(
        [f"请用一句话解释什么是{c}" for c in concepts],
        system_instruction="用简洁的语言"
    )
    for concept, resp in zip(concepts, responses):
        if not isinstance(resp, Exception):
            print(f"{concept}: {resp}")
    
    # 示例4: 带历史记录的对话
    messages = [
        {"role": "user", "content": "什么是人工智能？"},
        {"role": "assistant", "content": "人工智能是计算机科学的一个分支..."},
        {"role": "user", "content": "它有哪些主要应用领域？"}
    ]
    response = await client.chat_with_history(messages)
    print("对话响应:", response[:100] + "...")
    
    # 示例5: 处理JSONL文件
    # stats = await client.process_jsonl(
    #     "input.jsonl",
    #     "output.jsonl",
    #     prompt_template="请解释以下概念: {concept}",
    #     system_instruction="用通俗易懂的语言",
    #     batch_size=10
    # )
    # print("处理统计:", stats)


if __name__ == "__main__":
    # 运行示例
    asyncio.run(example_usage())