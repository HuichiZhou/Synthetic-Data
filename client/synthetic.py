from __future__ import annotations
import sys

from utils.async_gemini import AsyncGeminiClient
from utils.webpage import crawl_page, google_search
from prompt import QA_SYSTEM, QA_USER_TMPL, VET_SYSTEM, VET_USER_TMPL
from utils import safe_json_loads, normalize, answer_matches, truncate

import asyncio
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from contextlib import AsyncExitStack
from dotenv import load_dotenv
from openai import AsyncOpenAI

# --- MCP stdio client ---
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# 从网页内容中抽取冷门实体
# 搜索主题 → 抓取网页 → LLM抽取实体
# 实体列表及其来源信息
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")  # 可选，代理时使用
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o")

# 实体数据源配置
ENTITY_SOURCE_FILE = "out/greek_cuisine.jsonl"
ENTITIES = []

# MCP 工具服务器配置
SERVER_SCRIPTS = [
    "./mcp/serp_search.py",
    # "./mcp/craw_page.py",
]

# 搜索与抓取配置
RESULTS_PER_ENTITY = 50
CRAWL_TIMEOUT_SEC = 20
MAX_PAGE_CHARS = 100_000

# 采样控制配置
MAX_PAGES_TO_TRY_PER_ENTITY = 30  # 每个实体至多尝试多少不同网页
VET_ATTEMPTS_PER_QA = 8  # 校验次数（都失败才保留）

# 输出配置
OUTPUT_JSONL = os.getenv("OUTPUT_JSONL", "result/reverse_qa_hard.jsonl")

# ======================= 初始化 =======================
# 加载实体数据
# with open(ENTITY_SOURCE_FILE, "r") as f:
#     for line in f:
#         rec = json.loads(line)
#         ENTITIES.append(rec["entity"])


ENTITIES = [
    "Vegan fasting dessert recipe",
    "Greek Easter bread",
    "Koulouri Thessalonikis",
    "Lemon potato soup",
    "Greek lemon chicken soup",
    "Greek lemon rice soup",
    "Greek lemon pie",
    "Greek lemon cookies",
    "Greek lemon cake",
    "Greek lemon pasta",
    "Greek lemon potatoes with feta cheese",
    "Greek lemon chicken with potatoes and olives",
    "Greek lemon chicken with orzo",
    "Greek lemon chicken with rice",
    "Greek lemon chicken with vegetables",
    "Greek lemon chicken soup with orzo",
    "Greek lemon chicken soup with rice",
    "Greek lemon chicken soup with vegetables",
]

# ======================= LLM 封装 =======================


class LLM:
    def __init__(self, model: str, api_key: Optional[str], base_url: Optional[str]):
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    async def chat_json(self, system: str, user: str, temperature: float = 0.2) -> Any:
        resp = await self.client.chat.completions.create(
            model=self.model,
            temperature=temperature,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        content = resp.choices[0].message.content or "{}"
        return safe_json_loads(content) or {}

    async def chat_text(self, system: str, user: str, temperature: float = 0.2) -> str:
        resp = await self.client.chat.completions.create(
            model=self.model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content or ""

# ======================= 核心流程 =======================


@dataclass
class QAPair:
    entity: str
    question: str
    answer: str
    evidence_quote: str
    url: str
    title: str


class HardQABuilder:
    def __init__(self, llm: LLM, bus, gemini_client: Any = None):
        self.llm = llm
        self.bus = bus
        self.gemini_client = gemini_client

    async def search(self, entity: str, k: int) -> List[Tuple[str, str]]:
        # 过滤掉常见百科域，避免太容易
        q = f'"{entity}" -site:wikipedia.org -site:wikipedia.org -site:wikidata.org -site:britannica.com'
        # res = await self.bus.call("google_search", {"query": q, "topk": k})
        res = await google_search(q, topk=k)  # 直接调用函数，避免多次进程通信开销

        # 处理不同类型的返回值
        items = []
        if isinstance(res, dict):
            items = res.get("organic_results") or []
        elif isinstance(res, list):
            items = res
        elif hasattr(res, "text"):  # 处理 TextContent 对象
            try:
                # 尝试解析 TextContent 中的文本内容
                content_text = res.text
                if isinstance(content_text, str):
                    # 尝试解析为 JSON
                    parsed = safe_json_loads(content_text)
                    if isinstance(parsed, dict):
                        items = parsed.get("organic_results") or []
                    elif isinstance(parsed, list):
                        items = parsed
                    else:
                        print(
                            f"[WARNING] Failed to parse TextContent text as expected format"
                        )
                else:
                    print(
                        f"[WARNING] TextContent.text is not a string: {type(content_text)}"
                    )
            except Exception as e:
                print(f"[WARNING] Error processing TextContent: {e}")
        else:
            print(f"[WARNING] Unexpected search result type: {type(res)}")
            print(f"[DEBUG] Result content: {res}")

        out: List[Tuple[str, str]] = []
        for it in items:
            if isinstance(it, dict):
                title = (it.get("title") or "").strip()
                url = (it.get("link") or it.get("url") or "").strip()
            elif hasattr(it, "text"):  # 处理 TextContent 对象
                try:
                    content_text = it.text
                    if isinstance(content_text, str):
                        parsed = safe_json_loads(content_text)
                        if isinstance(parsed, dict):
                            title = (parsed.get("title") or "").strip()
                            url = (
                                parsed.get("link") or parsed.get("url") or ""
                            ).strip()
                        else:
                            continue
                    else:
                        continue
                except Exception:
                    continue
            else:
                continue

            if url:
                out.append((title, url))
        return out

    async def crawl(self, url: str) -> Tuple[str, str]:
        # res = await self.bus.call("crawl_page", {"url": url})
        res = await crawl_page(url)  # 直接调用函数，避免多次进程通信开销
        # 容错：可能直接是字符串，或是 dict
        if isinstance(res, str):
            title = url.rsplit("/", 1)[-1]
            text = res
        elif isinstance(res, dict):
            title = res.get("title") or url.rsplit("/", 1)[-1]
            text = res.get("raw") or res.get("text") or str(res) or ""
        elif hasattr(res, "text"):  # 处理 TextContent 对象
            try:
                content_text = res.text
                if isinstance(content_text, str):
                    # 尝试解析为 JSON
                    parsed = safe_json_loads(content_text)
                    if isinstance(parsed, dict):
                        title = parsed.get("title") or url.rsplit("/", 1)[-1]
                        text = (
                            parsed.get("raw") or parsed.get("text") or str(parsed) or ""
                        )
                    else:
                        # 如果不是 JSON，直接使用文本内容
                        title = url.rsplit("/", 1)[-1]
                        text = content_text
                else:
                    title = url.rsplit("/", 1)[-1]
                    text = str(content_text) or ""
            except Exception as e:
                print(f"[WARNING] Error processing TextContent in crawl: {e}")
                title = url.rsplit("/", 1)[-1]
                text = str(res) or ""
        else:
            # 如果是其他类型，转换为字符串
            title = url.rsplit("/", 1)[-1]
            text = str(res) or ""
        return title, text

    async def gen_qa_from_page(
        self, entity: str, title: str, url: str, page_text: str
    ) -> Optional[QAPair]:
        if not page_text:
            return None
        j = await self.llm.chat_json(
            QA_SYSTEM,
            QA_USER_TMPL.format(
                entity=entity,
                title=title,
                url=url,
                max_chars=MAX_PAGE_CHARS,
                page_text=truncate(
                    page_text, MAX_PAGE_CHARS
                ),  # 使用传入的page_text参数
            ),
            temperature=0.2,
        )
        q = (j or {}).get("question") or ""
        a = (j or {}).get("answer") or ""
        ev = (j or {}).get("evidence_quote") or ""
        if q and a and entity.lower() in q.lower():
            return QAPair(
                entity=entity,
                question=q.strip(),
                answer=a.strip(),
                evidence_quote=ev.strip(),
                url=url,
                title=title,
            )
        return None

    async def vet_qa(self, qa: QAPair) -> bool:
        """返回 True 表示通过（至少有一次答对），False 表示 8 次都失败。"""
        passed = False
        # 多次抽样，可变温度
        temps = [0.0, 0.2, 0.4, 0.7, 0.9, 0.3, 0.6, 0.8][:VET_ATTEMPTS_PER_QA]
        for t in temps:
            j = await self.llm.chat_json(
                VET_SYSTEM,
                VET_USER_TMPL.format(question=qa.question),
                temperature=t,
            )
            pred = (j or {}).get("answer") or ""
            if answer_matches(pred, qa.answer):
                passed = True
                break
        return passed

    async def find_one_hard_qa_for_entity(self, entity: str) -> Optional[QAPair]:
        results = await self.search(entity, RESULTS_PER_ENTITY)
        if not results:
            return None

        tried_urls: set[str] = set()
        # 随机化：打乱结果，然后每轮从"最后一个"取（满足"随机爬最后网页"）
        pool = results[:]
        random.shuffle(pool)

        for _ in range(min(MAX_PAGES_TO_TRY_PER_ENTITY, len(pool))):
            title, url = pool[-1]  # 取最后一个
            pool.pop()
            if url in tried_urls:
                continue
            tried_urls.add(url)

            try:
                ptitle, ptext = await self.crawl(url)
            except Exception:
                continue
            if not ptext or len(ptext) < 200:
                continue

            qa = await self.gen_qa_from_page(entity, ptitle or title, url, ptext)
            if not qa:
                continue

            passed = await self.vet_qa(qa)
            if not passed:
                # 8 次都不通过 -> 选择这个问题
                return qa
            # 否则继续换页面

        return None

async def main():
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set")

    llm = LLM(CHAT_MODEL, OPENAI_API_KEY, OPENAI_BASE_URL)

    try:
        # async with MCPBus(SERVER_SCRIPTS) as bus:
        builder = HardQABuilder(llm, None, AsyncGeminiClient(max_concurrent=5))

        out_path = Path(OUTPUT_JSONL)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"Starting generation with {len(ENTITIES)} entities...")
        kept = 0

        with out_path.open("w", encoding="utf-8") as f:
            for i, entity in enumerate(ENTITIES, 1):
                print(f"\n[{i}/{len(ENTITIES)}] Processing: {entity}")

                try:
                    qa = await builder.find_one_hard_qa_for_entity(entity)
                    if qa:
                        record = {
                            "entity": qa.entity,
                            "question": qa.question,
                            "answer": qa.answer,
                            "evidence_quote": qa.evidence_quote,
                            "url": qa.url,
                            "title": qa.title,
                        }
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")
                        f.flush()  # 确保数据被写入
                        kept += 1
                        print(f"  ✓ Kept difficult QA (total: {kept})")
                    else:
                        print(f"  ✗ No suitable QA found")
                except Exception as e:
                    print(f"  ✗ Error processing {entity}: {e}")
                    continue  # 继续处理下一个实体

        print(f"\nDone. Kept {kept} QA(s). Output -> {out_path.resolve()}")

    except KeyboardInterrupt:
        print("\n⚠️ Process interrupted by user")
        raise
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        raise

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Process interrupted.")
    except Exception as e:
        print(f"Fatal error: {e}")