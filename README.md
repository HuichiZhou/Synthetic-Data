# Synthetic-Data-Browsecomp
Official code repo of Synthetic-Data-Browsecomp [2077 AI]

## 项目结构

```
Synthetic-Data-Browsecomp/
├── client/                    # 客户端代码和提示词
│   ├── prompt.py              # 统一的提示词模板
│   ├── basicqa_generator.py   # 基础问答生成器
│   ├── basicqa_generator_huichi.py  # 辉池写的基础问答生成器
│   ├── browsercamp_gemini.py  # Gemini生成browsercamp
│   ├── browsercamp_oai.py     # OpenAI生成browsercamp
│   ├── entity_generator.py    # 实体生成器
│   ├── entity_generator_huichi.py  # 辉池写的实体生成器
│   └── utils/                 # 工具模块
├── mcp/                       # 模型上下文协议服务器
│   ├── serp_search.py         # 搜索功能
│   ├── crawl_extract.py       # 网页爬取和提取
│   └── crawl_page.py          # 页面爬取工具
├── result/                    # 生成的结果文件
├── .env                       # 环境变量配置文件
├── .env.example               # 环境变量示例文件
├── requirement.txt            # 依赖包列表
└── README.md                  # 本文档
```

## 提示词架构

所有提示词模板都统一在 `client/prompt.py` 文件中：

- **问答生成**: 从网页生成困难问题的模板
- **实体提取**: 用于提取英文和中文冷门实体的系统提示词
- **问题重写**: 完整的注入和模糊化问题系统，同时保持答案不变性
- **实体分类**: 预定义的实体生成类别

## 使用方法

### 基础合成数据生成
```bash
python client/basicqa_generator.py
```

### 实体提取
```bash
python client/entity_generator.py --topic "中国菜" --locale zh
```

### 问题重写
```bash
python client/browsercamp_gemini.py
python client/browsercamp_oai.py
```

## 依赖包

- OpenAI Python 客户端
- Google GenAI 客户端
- MCP (模型上下文协议) 库
- tqdm 进度条
- pandas 数据处理

## 环境设置

### 环境变量

在项目根目录创建 `.env` 文件，包含以下变量：

```bash
# OpenAI API 配置
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_BASE_URL=https://api.openai.com/v1  # 可选：用于代理使用

# Google GenAI 配置 (Gemini)
GENAI_API_KEY=your_genai_api_key_here
GENAI_API_BASE_URL=http://your-gemini-endpoint:3000

# SerpAPI 配置 (搜索验证)
SERPAPI_KEY=your_serpapi_key_here
```

### 安装步骤

1. **安装依赖包:**
   ```bash
   pip install python-dotenv openai google-genai serpapi crawl4ai tenacity colorlog
   ```

2. **设置环境变量:**
   ```bash
   # 复制示例环境文件
   cp .env.example .env

   # 编辑 .env 文件，填入实际的 API 密钥
   nano .env
   ```

3. **验证设置:**
   ```bash
   python -c "import os; from dotenv import load_dotenv; load_dotenv(); print('环境已加载:', bool(os.getenv('OPENAI_API_KEY')))"
   ```

### 环境变量参考

| 变量名 | 描述 | 是否必需 | 默认值 |
|--------|------|----------|--------|
| `OPENAI_API_KEY` | OpenAI API 密钥 | 是 | - |
| `OPENAI_BASE_URL` | OpenAI API 基础 URL | 否 | `https://api.openai.com/v1` |
| `GENAI_API_KEY` | Google GenAI API 密钥 | Gemini 使用时需要 | - |
| `GENAI_API_BASE_URL` | Google GenAI 基础 URL | Gemini 使用时需要 | - |
| `SERPAPI_KEY` | SerpAPI 搜索密钥 | 搜索验证时需要 | - |

### 安全注意事项

- **切勿提交 API 密钥** - 使用提供的 `.gitignore` 文件
- **使用环境变量** 而不是硬编码的值
- **本地创建 `.env` 文件** - 它会自动被 git 忽略
- **定期轮换密钥** 以确保安全最佳实践

## 功能特性

- **多语言支持**: 英文和中文实体提取
- **网页搜索集成**: 使用 MCP 工具进行实时搜索验证
- **答案不变性**: 问题重写时保持原始答案不变
- **基于证据的生成**: 所有生成内容都基于网页证据
- **质量控制**: 多步验证确保数据质量
