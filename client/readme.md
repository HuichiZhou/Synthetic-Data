# Synthetic Data Generation Client

本客户端提供多种合成数据生成工具，用于创建高质量的问答对和实体数据。

## 📁 项目结构

```
client/
├── synthetic.py              # 主合成数据生成器（基于现有实体生成困难QA）
├── synthetic_entity.py        # 实体抽取器（从网页内容抽取冷门实体）
├── inject_fuzz_gpt5.py       # GPT-5 搜索注入工具
├── inject_fuzz_gemini.py     # Gemini 搜索注入工具
├── entity_questions_generator.py  # 实体问题生成器
├── prompt.py                 # 提示词模板
└── readme.md                # 说明文档
```

## 🛠️ 工具说明

### 1. synthetic.py - 主合成数据生成器

**目的**: 基于现有实体生成困难的问答对

```
手动给定实体 -> 搜索工具检索 -> 随机化后抓取"最后一个"网页 ->
从页面生成较难QA -> 让同一个LLM直接作答8次做校验 ->
如果8次都不通过（都答不对），则保留该QA；否则继续换别的网页
```

**依赖**:
- python-dotenv
- openai>=1.30.0（异步客户端）
- mcp (Model Context Protocol) 客户端 + stdio servers（serp_search.py, craw_page.py）

**工具签名**:
- `serp_search(query: str, k: int) -> {"results": [{"title":"...","url":"...","snippet":"..."}, ...]}`
- `craw_page(url: str, timeout_sec?: int) -> {"title":"...","text":"...","html":"..."}`

**配置参数**:
```python
ENTITY_SOURCE_FILE = "out/greek_cuisine.jsonl"  # 实体数据源
RESULTS_PER_ENTITY = 50                         # 每实体搜索结果数
MAX_PAGES_TO_TRY_PER_ENTITY = 30               # 每实体最大尝试页面数
VET_ATTEMPTS_PER_QA = 8                        # 验证尝试次数
```

### 2. synthetic_entity.py - 实体抽取器

**目的**: 单次搜索 + 批量抓取 + LLM正文抽取（支持中英locale），可选只保留维基百科搜不到的实体

**特性**:
- 只调用一次serp_search，拿到一批URL
- craw_page批量抓取正文
- LLM按页抽取实体（严格JSON）
- 可选：用site:wikipedia.org/zh.wikipedia.org过滤维基可搜到的实体
- 不保存页面全文，只保存 `{entity, why_uncommon, source_url, source_title, topic}`

**示例用法**:
```bash
# 英文数据集
python synthetic_entity.py \
  --topic "Chinese Cuisine" \
  --server ../server/serp_search.py --server ../server/craw_page.py \
  --k 25 --per-page 8 --model gpt-4o \
  --locale en --only-nonwiki --out out/hc

# 中文数据集  
python synthetic_entity.py \
  --topic "合成生物学" \
  --server ../server/serp_search.py --server ../server/craw_page.py \
  --k 25 --per-page 8 --model gpt-4o \
  --locale zh --only-nonwiki --check-zh --out out/synbio_zh
```

### 3. inject_fuzz_gpt5.py - GPT-5搜索注入

**用途**: 处理特定领域的问答数据，集成GPT-5搜索功能

**配置**:
```python
INPUT_FILE = "/path/to/input/data.jsonl"     # 输入文件
OUTPUT_DIR = "result/Economics"              # 输出目录
MODEL_NAME = "gpt-5-search"                  # 模型名称
```

### 4. inject_fuzz_gemini.py - Gemini搜索注入

**用途**: 使用Gemini处理数据，集成Google搜索功能

**配置**:
```python
INPUT_FILE = "/path/to/biology_data.jsonl"   # 输入文件
OUTPUT_BASE = "result-gemini"                # 输出目录
MODEL_NAME = "gemini-2.5-pro"                # 模型名称
```

### 5. entity_questions_generator.py - 实体问题生成器

**用途**: 生成冷门实体相关问题，支持SERPAPI实体验证

**配置**:
```python
DEFAULT_MODEL = "gpt-4o"                     # 默认模型
MAX_VALIDATION_ATTEMPTS = 5                  # 最大验证尝试
MAX_FEATURES = 8                             # 最大特征数
DEFAULT_ROUNDS = 10                         # 默认生成轮次
OUTPUT_FILE = "entity_questions.jsonl"       # 输出文件
```

## ⚙️ 配置说明

### 数据源配置
- `ENTITY_SOURCE_FILE`: 实体JSONL数据源文件路径
- 格式要求：每行包含 `{"entity": "实体名称"}`

### 搜索配置  
- `RESULTS_PER_ENTITY`: 每个实体搜索的结果数量
- `CRAWL_TIMEOUT_SEC`: 页面抓取超时时间（秒）
- `MAX_PAGE_CHARS`: 页面内容最大字符数

### 质量控制
- `MAX_PAGES_TO_TRY_PER_ENTITY`: 每个实体尝试的最大页面数
- `VET_ATTEMPTS_PER_QA`: 每个QA对的验证尝试次数
- 只有所有验证尝试都失败的QA才会被保留

## 🎯 功能特性

### synthetic.py
- ✅ 基于实体生成困难问答对
- ✅ 多轮LLM验证确保质量  
- ✅ 支持MCP工具集成（搜索+抓取）
- ✅ 自动过滤简单问题

### synthetic_entity.py
- ✅ 从网页内容抽取冷门实体
- ✅ 支持中英文多语言
- ✅ 维基百科实体过滤
- ✅ 批量处理高效抽取

### inject_fuzz_*.py
- ✅ 支持GPT-5和Gemini双引擎
- ✅ 自动提取关键字段（问题、答案、数据源）
- ✅ 集成搜索工具增强数据真实性
- ✅ 批量处理JSONL格式数据

### entity_questions_generator.py
- ✅ 生成冷门实体相关问题
- ✅ SERPAPI实体存在性验证
- ✅ 特征提取和唯一性检查
- ✅ 自动合并特征生成自然描述

## 📊 输出格式

### 合成QA对格式
```json
{
  "entity": "实体名称",
  "question": "生成的问题",
  "ground_truth": "正确答案", 
  "evidence_quote": "证据文本",
  "data_source": "数据源URL",
  "page_title": "页面标题",
  "vetting": {
    "model": "gpt-4o",
    "attempts": 8,
    "criterion": "all attempts must fail to keep"
  }
}
```

### 实体抽取格式
```json
{
  "entity": "实体名称",
  "why_uncommon": "不常见原因",
  "source_url": "来源URL",
  "source_title": "来源标题",
  "topic": "主题"
}
```

## 🔧 自定义配置

所有工具都支持通过以下方式配置：

1. **环境变量**: 优先读取 `.env` 文件中的配置
2. **代码修改**: 直接修改文件顶部的配置常量
3. **命令行参数**: synthetic_entity.py支持完整的命令行参数

## 📝 使用建议

1. **数据质量**: 建议使用高质量实体数据源
2. **API配额**: 注意OpenAI和Gemini的API调用限制
3. **验证机制**: 充分利用多轮验证确保数据难度
4. **批量处理**: 支持批量处理，适合大规模数据生成

## 🐛 故障排除

### 常见问题
1. **API密钥错误**: 检查环境变量设置
2. **文件路径错误**: 确保输入文件存在且有读取权限  
3. **网络超时**: 调整超时时间或重试机制
4. **JSON解析错误**: 检查数据格式是否符合JSONL标准
