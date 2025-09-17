#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import asyncio
import json
import os
import re
import argparse
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from contextlib import AsyncExitStack

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from utils.text_utils import response_format
from utils.webpage import google_search, crawl_page

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x

try:
    import pandas as pd
except Exception:
    pd = None

from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import AsyncOpenAI

# ----------------------------- 基础数据结构 -----------------------------
@dataclass
class SerpItem:
    title: str
    url: str
    snippet: str = ""

@dataclass
class EntityRow:
    entity: str
    why_uncommon: str
    source_url: str
    source_title: str
    topic: str

# ----------------------------- MCP 工具封装 -----------------------------
@dataclass
class ToolInfo:
    name: str
    description: str
    schema: Dict[str, Any]
    session: ClientSession

class MCPTools:
    def __init__(self) -> None:
        self.exit_stack = AsyncExitStack()
        self.tools: Dict[str, ToolInfo] = {}

    async def connect(self, server_paths: List[str]) -> None:
        for spath in server_paths:
            cmd = "python" if Path(spath).suffix == ".py" else "node"
            params = StdioServerParameters(command=cmd, args=[spath])
            stdio, write = await self.exit_stack.enter_async_context(stdio_client(params))
            session = await self.exit_stack.enter_async_context(ClientSession(stdio, write))
            await session.initialize()
            listed = await session.list_tools()
            for t in listed.tools:
                self.tools[t.name] = ToolInfo(t.name, t.description, t.inputSchema, session)
        if not self.tools:
            raise RuntimeError("未发现任何 MCP 工具。")

    async def close(self) -> None:
        await self.exit_stack.aclose()

    def _best(self, *hints: str) -> Optional[str]:
        hints = tuple(h.lower() for h in hints)
        best, score = None, -1
        for name in self.tools:
            n = name.lower()
            s = sum(h in n for h in hints)
            if s > score:
                best, score = name, s
        return best

    def serp_tool(self) -> ToolInfo:
        name = self._best("serp", "search")
        if not name:
            raise RuntimeError("未找到 serp/search 工具。")
        return self.tools[name]

    def crawl_tool(self) -> ToolInfo:
        name = self._best("crawl", "craw")
        if not name:
            raise RuntimeError("未找到 craw_page 工具。")
        return self.tools[name]

    async def call(self, tool: ToolInfo, args: Dict[str, Any]) -> Tuple[str, Optional[Any]]:
        msg = await tool.session.call_tool(tool.name, args)
        text = "" if msg is None else str(getattr(msg, "content", ""))
        try:
            data = json.loads(text)
        except Exception:
            data = None
        return text, data

# ----------------------------- SERP 一次查询 -----------------------------
URL_RE = re.compile(r"https?://[^\s)\]}\"'>]+", re.I)
CJK_RE = re.compile(r"[\u4e00-\u9fff]")

def _pick_arg(schema: Dict[str, Any], candidates: List[str]) -> Optional[str]:
    props = (schema or {}).get("properties") or {}
    for c in candidates:
        if c in props:
            return c
    for k in props.keys():
        if any(c in k.lower() for c in candidates):
            return k
    return None

async def serp_query(topic: str, limit: int = 20) -> List[SerpItem]:
    # tool = mcp.serp_tool()
    # qk = _pick_arg(tool.schema, ["query", "q", "text", "keyword"]) or "query"
    # lk = _pick_arg(tool.schema, ["k", "limit", "topn", "n"])  # 可选
    # args: Dict[str, Any] = {qk: topic}
    # if lk:
    #     args[lk] = limit
    data = await google_search(topic, limit)
    items: List[SerpItem] = []
    if isinstance(data, list):
        for row in data:
            url = str(row.get("url") or row.get("link") or row.get("href") or "")
            if not url:
                continue
            items.append(SerpItem(
                title=str(row.get("title") or row.get("name") or ""),
                url=url,
                snippet=str(row.get("snippet") or row.get("summary") or row.get("description") or ""),
            ))
    elif isinstance(data, dict) and "results" in data:
        for row in data["results"]:
            url = str(row.get("url") or row.get("link") or row.get("href") or "")
            if not url:
                continue
            items.append(SerpItem(
                title=str(row.get("title") or row.get("name") or ""),
                url=url,
                snippet=str(row.get("snippet") or row.get("summary") or row.get("description") or ""),
            ))
    else:
        for u in URL_RE.findall(str(data)):
            items.append(SerpItem(title="", url=u, snippet=""))
    # 去重
    seen, out = set(), []
    for it in items:
        if it.url in seen:
            continue
        seen.add(it.url)
        out.append(it)
    return out

# ----------------------------- 批量抓取正文 -----------------------------
async def crawl_many(serps: List[SerpItem], concurrency: int = 8, exclude_wiki_sources: bool = True) -> List[Tuple[str, str, str]]:
    """返回 [(url, title, content)] 列表。优先尝试批量抓取，失败则并发单抓。"""
    if exclude_wiki_sources:
        serps = [s for s in serps if "wikipedia.org" not in s.url and "zh.wikipedia.org" not in s.url]

    # crawl = mcp.crawl_tool()
    # urlk = _pick_arg(crawl.schema, ["url", "link", "target", "page"]) or "url"
    # urlsk = _pick_arg(crawl.schema, ["urls", "links", "targets"])  # 可选批量

    urls = [s.url for s in serps if s.url]
    title_map = {s.url: (s.title or "") for s in serps}

    results: List[Tuple[str, str, str]] = []

    # 批量抓取
    if urls:
        tasks = [crawl_page(url) for url in urls]
        data = await asyncio.gather(*tasks, return_exceptions=True)
        if isinstance(data, list):
            for it in data:
                u = str(it.get("url") or it.get("source_url") or "")
                content = None
                for k in ["text", "content", "html", "body", "article"]:
                    v = it.get(k)
                    if isinstance(v, str) and v.strip():
                        content = v
                        break
                if u and content:
                    results.append((u, title_map.get(u, ""), content))
        elif isinstance(data, dict):
            if "results" in data and isinstance(data["results"], list):
                for it in data["results"]:
                    u = str(it.get("url") or it.get("source_url") or "")
                    content = None
                    for k in ["text", "content", "html", "body", "article"]:
                        v = it.get(k)
                        if isinstance(v, str) and v.strip():
                            content = v
                            break
                    if u and content:
                        results.append((u, title_map.get(u, ""), content))
            else:
                for u, v in data.items():
                    if isinstance(v, str) and v.strip():
                        results.append((u, title_map.get(u, ""), v))
        if results:
            return results

    # 并发逐条抓取
    async def one(u: str) -> Optional[Tuple[str, str, str]]:
        try:
            data = await crawl_page([u])
            if isinstance(data, dict):
                for k in ["text", "content", "html", "body", "article"]:
                    if isinstance(data.get(k), str) and data[k].strip():
                        return (u, title_map.get(u, ""), data[k])
            return (u, title_map.get(u, ""), str(data) or "")
        except Exception:
            return None

    sem = asyncio.Semaphore(concurrency)
    async def pooled(u: str):
        async with sem:
            return await one(u)

    tasks = [asyncio.create_task(pooled(u)) for u in urls]
    out: List[Tuple[str, str, str]] = []
    for fut in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="抓取页面", unit="pg"):
        r = await fut
        if r:
            out.append(r)
    return out

# ----------------------------- LLM 从正文抽取实体 -----------------------------
EN_SYS = (
    "You are an information extraction expert. Given the fulltext of a web page, extract entities that are relevant to the topic and are *lesser-known / obscure* (projects, prototypes, labs, researchers, datasets, libraries, protocols, benchmarks, workshops, etc.). "
    "Avoid generic terms and big tech company names. Do not invent facts; rely only on the provided text. "
    "Return a strict JSON array: [{\"entity\":str, \"why_uncommon\":str, \"source_url\":str, \"source_title\":str}]."
)
ZH_SYS = (
    "你是信息抽取专家。基于提供的网页正文，只从正文中找出与主题相关且相对小众/不常见的实体——例如项目、研究原型、机构、个人、数据集、库、协议、基准、会议工作坊等。"
    "排除通用术语和大公司的名字。不要引入外部常识，仅依据正文。输出严格 JSON 数组："
    "[{\"entity\":str, \"why_uncommon\":str, \"source_url\":str, \"source_title\":str}]。"
)

class LLMPageEntityExtractor:
    def __init__(self, model: str, locale: str):
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_BASE_URL"))
        self.model = model
        self.locale = locale
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((Exception,)),
        # retry_error_callback=retry_error_callback
    )    
    async def extract_from_page(self, topic: str, url: str, title: str, content: str, want: int = 8) -> List[EntityRow]:
        text = content or ""
        text = " ".join(text.splitlines())[:8000]
        sys_prompt = EN_SYS if self.locale == "en" else ZH_SYS
        user_payload = {
            "topic": topic,
            "want": want,
            "source_url": url,
            "source_title": title,
            "content": text,
        }
        msgs = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ]
        resp = await self.client.chat.completions.create(
            model=self.model,
            messages=msgs,
            temperature=0.2,
            max_tokens=1500,
        )
        raw = resp.choices[0].message.content or "[]"
        # try:
        #     l = raw.index("["); r = raw.rindex("]") + 1; raw = raw[l:r]
        # except Exception:
        #     pass
        try:
            arr = response_format(raw)
        except Exception:
            raise RuntimeError(f"LLM 返回格式错误，非 JSON 数组：{raw!r}")
        out: List[EntityRow] = []
        for o in arr:
            e = str(o.get("entity") or "").strip()
            why = str(o.get("why_uncommon") or o.get("why") or "").strip()
            if not e:
                continue
            out.append(EntityRow(entity=e, why_uncommon=why, source_url=url, source_title=title, topic=topic))
        return out

# ----------------------------- 维基过滤 -----------------------------
async def is_on_wikipedia(entity: str, check_zh: bool = False) -> bool:
    qs = [f'site:wikipedia.org "{entity}"']
    if check_zh:
        qs.append(f'site:zh.wikipedia.org "{entity}"')
    for q in qs:
        rs = await serp_query(q, limit=3)
        if any("wikipedia.org" in (r.url or "") for r in rs):
            return True
    return False

async def filter_nonwiki(rows: List[EntityRow], check_zh: bool, concurrency: int = 8) -> List[EntityRow]:
    sem = asyncio.Semaphore(concurrency)
    async def keep(row: EntityRow) -> Optional[EntityRow]:
        async with sem:
            try:
                on_wiki = await is_on_wikipedia(row.entity, check_zh=check_zh)
                return None if on_wiki else row
            except Exception:
                return row
    tasks = [asyncio.create_task(keep(r)) for r in rows]
    out: List[EntityRow] = []
    for fut in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="维基过滤", unit="ent"):
        r = await fut
        if r:
            out.append(r)
    return out

# ----------------------------- 主流程 -----------------------------
async def run_pipeline(
    topic: str,
    servers: List[str],
    per_query_k: int = 20,
    out_prefix: str = "result/result_v2_locale",
    concurrency: int = 8,
    model: str = "gpt-4o",
    want_per_page: int = 8,
    only_nonwiki: bool = False,
    check_zh: bool = False,
    exclude_wiki_sources: bool = True,
    locale: str = "auto",
) -> List[EntityRow]:
    # locale 处理
    if locale not in {"en", "zh", "auto"}:
        locale = "auto"
    if locale == "auto":
        locale = "zh" if CJK_RE.search(topic) else "en"

    # mcp = MCPTools()
    # await mcp.connect(servers)
    try:
        serps = await serp_query(topic, limit=per_query_k)
        pages = await crawl_many(serps, concurrency=concurrency, exclude_wiki_sources=exclude_wiki_sources)
        extractor = LLMPageEntityExtractor(model=model, locale=locale)

        sem = asyncio.Semaphore(concurrency)
        async def per_page(u: str, t: str, c: str) -> List[EntityRow]:
            async with sem:
                try:
                    return await extractor.extract_from_page(topic, u, t, c, want=want_per_page)
                except Exception:
                    return []

        tasks = [asyncio.create_task(per_page(u, t, c)) for (u, t, c) in pages]
        gathered: List[EntityRow] = []
        for fut in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="抽取实体", unit="pg"):
            items = await fut
            gathered.extend(items)

        # 去重
        seen = set()
        deduped: List[EntityRow] = []
        for row in gathered:
            key = (row.entity.strip().lower(), row.source_url)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(row)

        # 可选：维基过滤（英文数据集默认只查 enwiki；若传了 --check-zh 也会查中文）
        if only_nonwiki:
            deduped = await filter_nonwiki(deduped, check_zh=check_zh, concurrency=concurrency)

        # 落盘
        Path(out_prefix).parent.mkdir(parents=True, exist_ok=True)
        jsonl_path = f"{out_prefix}.jsonl"
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for r in deduped:
                print(json.dumps(asdict(r), ensure_ascii=False), file=f)

        csv_path = f"{out_prefix}.csv"
        if pd is not None:
            df = pd.DataFrame([asdict(r) for r in deduped])
            df.to_csv(csv_path, index=False, encoding="utf-8")
        else:
            with open(csv_path, "w", encoding="utf-8") as f:
                print("entity,why_uncommon,source_url,source_title,topic", file=f)
                for r in deduped:
                    why = (r.why_uncommon or "").replace(",", " ")
                    print(f"{r.entity},{why},{r.source_url},{r.source_title},{r.topic}", file=f)

        print(f"\n✅ 已保存：{jsonl_path} 和 {csv_path}")
        return deduped
    finally:
        import traceback
        traceback.print_exc()
    #     await mcp.close()

# ----------------------------- CLI -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="单次搜索 + 批量抓取 + LLM 正文抽取（支持中英 locale，可选只保留非维基实体）")
    ap.add_argument("--topic", default="Nuclear Physics", help="主题/领域（英文或中文）")
    ap.add_argument("--server", action="append", help="MCP 工具 server 路径，可多次指定")
    ap.add_argument("--k", type=int, default=20, help="serp_search 返回条数（若工具支持）")
    ap.add_argument("--per-page", type=int, default=8, help="每页抽取实体上限，交给 LLM")
    ap.add_argument("--out", default="out/result_v2_locale", help="输出前缀")
    ap.add_argument("--concurrency", type=int, default=8, help="抓取与抽取并发")
    ap.add_argument("--model", default=os.getenv("EXEC_MODEL", "gpt-4o"), help="用于抽取的 LLM 模型名")
    ap.add_argument("--only-nonwiki", action="store_true", help="只保留维基百科搜不到的实体")
    ap.add_argument("--check-zh", action="store_true", help="在过滤时同时检查 zh.wikipedia.org")
    ap.add_argument("--no-exclude-wiki-sources", action="store_true", help="不剔除维基来源页（默认剔除）")
    ap.add_argument("--locale", choices=["en", "zh", "auto"], default="auto", help="抽取与提示语语言；auto=基于 topic 自动判断")
    args = ap.parse_args()
    load_dotenv()

    rows = asyncio.run(
        run_pipeline(
            topic=args.topic,
            servers=args.server,
            per_query_k=args.k,
            out_prefix=args.out,
            concurrency=args.concurrency,
            model=args.model,
            want_per_page=args.per_page,
            only_nonwiki=args.only_nonwiki,
            check_zh=args.check_zh,
            exclude_wiki_sources=not args.no_exclude_wiki_sources,
            locale=args.locale,
        )
    )

    print("\n样例输出（最多 20 条）：")
    for i, r in enumerate(rows[:20], 1):
        print(f"{i:2d}. {r.entity}  <-  {r.source_title}  |  {r.source_url}")

if __name__ == "__main__":
    main()
