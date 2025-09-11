# Synthetic-Data-Browsecomp
Official code repo of Synthetic-Data-Browsecomp [2077 AI]

## Project Structure

```
Synthetic-Data/
├── client/                    # Client-side code and prompts
│   ├── prompt.py             # Consolidated prompt templates
│   ├── synthetic.py          # Main synthetic data generation
│   ├── synthetic_entity.py   # Entity extraction and processing
│   ├── entity_questions_generator.py  # Entity question generation
│   ├── inject_fuzz_gemini.py # Gemini-based question rewriting
│   └── inject_fuzz_gpt5.py   # GPT-5-based question rewriting
├── mcp/                      # Model Context Protocol servers
│   ├── serp_search.py        # Search functionality
│   ├── crawl_extract.py      # Web crawling and extraction
│   └── craw_page.py          # Page crawling utilities
├── result/                   # Generated output files
└── README.md                # This file
```

## Prompt Architecture

All prompt templates are consolidated in `client/prompt.py`:

- **QA Generation**: Templates for generating difficult questions from web pages
- **Entity Extraction**: System prompts for extracting obscure entities in English and Chinese
- **Question Rewriting**: Comprehensive system for injecting and fuzzing questions while maintaining answer invariance
- **Entity Categories**: Predefined categories for entity generation

## Usage

### Basic Synthetic Data Generation
```bash
python client/synthetic.py
```

### Entity Extraction
```bash
python client/synthetic_entity.py --topic "Chinese Cuisine" --locale en
```

### Question Rewriting
```bash
python client/inject_fuzz_gemini.py
python client/inject_fuzz_gpt5.py
```

## Dependencies

- OpenAI Python client
- Google GenAI client
- MCP (Model Context Protocol) libraries
- tqdm for progress bars
- pandas for data handling

## Environment Setup

### Environment Variables

Create a `.env` file in the project root with the following variables:

```bash
# OpenAI API Configuration
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_BASE_URL=https://api.openai.com/v1  # Optional: for proxy usage

# Google GenAI Configuration (for Gemini)
GENAI_API_KEY=your_genai_api_key_here
GENAI_API_BASE_URL=http://your-gemini-endpoint:3000

# SerpAPI Configuration (for search validation)
SERPAPI_KEY=your_serpapi_key_here
```

### Installation

1. **Install dependencies:**
   ```bash
   pip install python-dotenv openai google-genai serpapi crawl4ai tenacity colorlog
   ```

2. **Set up environment variables:**
   ```bash
   # Copy example environment file
   cp .env.example .env
   
   # Edit .env with your actual API keys
   nano .env
   ```

3. **Verify setup:**
   ```bash
   python -c "import os; from dotenv import load_dotenv; load_dotenv(); print('Environment loaded:', bool(os.getenv('OPENAI_API_KEY')))"
   ```

### Environment Variables Reference

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `OPENAI_API_KEY` | OpenAI API key | Yes | - |
| `OPENAI_BASE_URL` | OpenAI API base URL | No | `https://api.openai.com/v1` |
| `GENAI_API_KEY` | Google GenAI API key | For Gemini usage | - |
| `GENAI_API_BASE_URL` | Google GenAI base URL | For Gemini usage | - |
| `SERPAPI_KEY` | SerpAPI key for search | For search validation | - |

### Security Notes

- **Never commit API keys** - use the provided `.gitignore` file
- **Use environment variables** instead of hardcoded values
- **Create `.env` file locally** - it's automatically ignored by git
- **Rotate keys regularly** for security best practices

## Features

- **Multi-language support**: English and Chinese entity extraction
- **Web search integration**: Real-time search validation using MCP tools
- **Answer invariance**: Question rewriting that preserves original answers
- **Evidence-based generation**: All generated content is grounded in web evidence
- **Quality control**: Multiple validation steps to ensure data quality
