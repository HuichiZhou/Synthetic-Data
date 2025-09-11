"""
Prompt templates and system prompts for synthetic data generation.

This module consolidates all prompt templates used across the synthetic data generation pipeline.
"""

# ======================= QA Generation Prompts =======================

QA_SYSTEM = """
You write one difficult, specific QA from a given page.
Rules:
- The question MUST mention the ENTITY and be answerable ONLY from the PAGE TEXT.
- Prefer obscure, buried details (dates, catalog nos., minor participants, captions, footnotes).
- The answer must be short (<= 10 words) and MUST be an exact span copied verbatim from the page.
- Provide an evidence quote (short excerpt that contains the answer). 
- The answer must be unique and unambiguous within the page.
- In ALL cases, there must be only ONE correct answer, never multiple.
- Output strict JSON only.
""".strip()

QA_USER_TMPL = """
ENTITY: {entity}
PAGE TITLE: {title}
PAGE URL: {url}

PAGE TEXT (truncated to {max_chars} chars):
```
{page_text}
```

Return JSON:
{{
  "question": "... must include '{entity}' ...",
  "answer": "short exact span",
  "evidence_quote": "contains the exact answer"
}}
""".strip()

VET_SYSTEM = """
You answer the user's question without tools. If unsure, say "I don't know". Output strict JSON only.
""".strip()

VET_USER_TMPL = """
Q: {question}

Respond with JSON:
{{
  "answer": "<your best answer or 'I don't know'>",
  "confidence": 0.0-1.0
}}
""".strip()

# ======================= Entity Extraction Prompts =======================

EN_ENTITY_EXTRACTION_SYSTEM = """
You are an information extraction expert. Given the fulltext of a web page, extract entities that are relevant to the topic and are *lesser-known / obscure* (projects, prototypes, labs, researchers, datasets, libraries, protocols, benchmarks, workshops, etc.). 
Avoid generic terms and big tech company names. Do not invent facts; rely only on the provided text. 
Return a strict JSON array: [{"entity":str, "why_uncommon":str, "source_url":str, "source_title":str}].
"""

ZH_ENTITY_EXTRACTION_SYSTEM = """
你是信息抽取专家。基于提供的网页正文，只从正文中找出与主题相关且相对小众/不常见的实体——例如项目、研究原型、机构、个人、数据集、库、协议、基准、会议工作坊等。
排除通用术语和大公司的名字。不要引入外部常识，仅依据正文。输出严格 JSON 数组：
[{"entity":str, "why_uncommon":str, "source_url":str, "source_title":str}]。
"""

# ======================= Question Rewriting Prompts =======================

QUESTION_REWRITER_SYSTEM = """
You are a QUESTION REWRITER that can call tools:
- serp_search(query: str, k?: int) → web results (title, url, snippet,...)
- craw_page(url: str) → full HTML/text content of the page

GOAL
Rewrite the input question through two operations—(A) INJECT (REPLACE) and (B) FUZZ—so that:
1) It remains uniquely answerable by web search;
2) The answer/target does NOT change (answer invariance);
3) Each rewrite step is verifiably grounded in external evidence.

LANGUAGE
- Write in the SAME language as the input question.
- Keep the question type (what/when/who/which/how many...) unchanged.

PARAMETERS (replace braces with concrete integers if provided)
- INJECT_ROUNDS = {inject_rounds:=1}
- FUZZ_ROUNDS   = {fuzz_rounds:=1}
- START_WITH    = {start_with:="inject"}   # "inject" or "fuzz"
- MIN_DOMAINS   = {min_domains:=2}         # uniqueness requires ≥ MIN_DOMAINS distinct domains
- MAX_REFINE    = {max_refine:=2}          # extra inject refinements if uniqueness fails
- MAX_ADDED_TOK = {max_added_tokens:=40}   # max extra tokens per inject rewrite
- MAX_FACTS     = {max_facts:=3}           # facts to build the descriptor in inject

DEFINITIONS
A) INJECT (REPLACE):
   - Identify the SINGLE main named entity the original question is about.
   - Use tools to gather 1–MAX_FACTS short, verifiable facts (e.g., official role + jurisdiction + term years,
     unique affiliation + time, signature event + date).
   - **REPLACE** the original named entity mention with a concise, unambiguous descriptive noun phrase composed
     from those facts. **Do NOT include the original name anywhere** in the rewritten text.
   - Keep added tokens ≤ MAX_ADDED_TOK.
   - The rewrite MUST still refer to the SAME entity as the original.

B) FUZZ:
   - Hide 1–2 key slots using hypernyms or ranges while preserving answer & solvability:
       * proper name → type/descriptor (without introducing ambiguity)
       * exact date/year → bounded range (e.g., "early 1930s")
       * exact number → small numeric interval ("about", "mid-", "low-", or x–y)
       * lexical synonym/paraphrase
   - Apply minimally; over-fuzzing is not allowed. The question must stay uniquely answerable.

EVIDENCE & VERIFICATION (required after EVERY rewrite):
1) Use serp_search on the rewritten question (and, if needed, on key entities/phrases).
2) craw_page the top credible results to verify grounding.
3) Map the rewritten description back to the **same canonical entity** as the original.
4) Uniqueness passes only if ≥ MIN_DOMAINS distinct domains support the SAME entity mapping AND contain consistent evidence.
5) Prefer sources: .gov/.edu/official > reputable news/reference > blogs/forums.
6) No hallucinations: Every injected statement must be supported by a quoted span or tight paraphrase from a crawled page.

WORKFLOW
0) Detect main entity of the original question (and capture canonical name).
1) Interleave operations for a total of INJECT_ROUNDS + FUZZ_ROUNDS steps, starting with START_WITH and alternating.
2) For EACH INJECT step:
   a) Search/crawl to collect candidate facts (1–MAX_FACTS).
   b) Compose a descriptor NP from those facts; REPLACE the entity mention with the descriptor.
   c) Verify uniqueness & same-entity. If it FAILS, run up to MAX_REFINE refinement cycles:
      - add/adjust one more precise fact (e.g., years, jurisdiction, affiliation) and re-verify.
      - stop early once uniqueness passes.
3) For EACH FUZZ step:
   a) Choose at most two slots to fuzz (hypernym, time_range, number_range, synonym).
   b) Re-verify uniqueness & same-entity. If FAILS, undo the last fuzz and pick a different, milder fuzz op.
4) Keep the question concise and natural. Do not change what the question asks for.

OUTPUT (STRICT)
Return ONE line only, starting with:
FINAL ANSWER: {{
  "language": "<lang>",
  "original_question": "<q0>",
  "steps": [
    {{
      "type": "inject" | "fuzz",
      "before": "<text_before>",
      "after": "<text_after>",
      "descriptor": "<only for inject: the NP you used>",
      "used_facts": [
        {{"fact": "<short factual clause>", "url": "<source_url>", "quote": "<short span if available>"}}
      ],
      "fuzz_ops": ["hypernym"|"time_range"|"number_range"|"synonym"],  // for fuzz; omit for inject
      "verify": {{
        "unique": true|false,
        "same_entity": true|false,
        "canonical_name": "<name from evidence>",
        "support_domains": ["example.gov","example.edu", "..."],   // distinct domains only
        "support_urls": ["https://...", "..."],
        "note": "<brief reasoning>"
      }}
    }}
  ],
  "final_question": "<the last successful rewrite>",
  "warnings": ["<optional>"]
}}

HARD RULES
- INJECT must **remove the original name** and replace it with a descriptor built ONLY from tool-verified facts.
- Never introduce facts you did not see in evidence. Include at least one URL per fact; prefer 2+ domains overall.
- Maintain answer invariance and question type. Reject a rewrite that changes the target or introduces multiple candidates.
- If a step cannot be made unique after MAX_REFINE refinements, revert that step and explain in `warnings`.
- Keep total verbosity low; avoid long appositives. Aim for precision-first descriptors.
"""

# ======================= Entity Category Definitions =======================

ENTITY_CATEGORIES = [
    # 人物类
    "historical figures (specific individuals)",
    "contemporary artists (specific people)",
    "scientists with specific inventions",
    "political leaders (specific individuals)",
    # 艺术品/创作类
    "Renaissance paintings (specific works)",
    "novels by obscure authors",
    "silent films (specific titles)",
    "indie video games (specific titles)",
    # 地理/建筑类
    "ancient ruins (specific locations)",
    "unique museums (specific institutions)",
    "historical castles (specific structures)",
    "unusual natural landmarks",
    # 组织/团体类
    "local sports teams (specific teams)",
    "specialized universities (specific institutions)",
    "independent music bands",
    "historical secret societies",
    
    # 科技/发明类
    "obsolete technological devices (specific models)",
    "early computer programs (specific titles)",
    "forgotten scientific instruments",
    "patented but unused inventions",
    
    # 文化/传统类
    "regional folk dances (specific styles)",
    "ancient rituals (specific practices)",
    "endangered languages (specific ones)",
    "traditional crafts (specific techniques)",
    
    # 生物/自然类
    "rare animal species (specific types)",
    "endemic plant varieties (specific ones)",
    "extinct prehistoric creatures",
    "unique ecological communities",
    
    # 事件/现象类
    "local historical events (specific occurrences)",
    "seasonal natural phenomena (specific types)",
    "obscure cultural festivals",
    "forgotten scientific discoveries"
]

# ======================= Helper Functions =======================

def get_entity_extraction_system(locale: str) -> str:
    """Get the appropriate entity extraction system prompt based on locale."""
    if locale == "zh":
        return ZH_ENTITY_EXTRACTION_SYSTEM
    return EN_ENTITY_EXTRACTION_SYSTEM

def get_qa_user_prompt(entity: str, title: str, url: str, page_text: str, max_chars: int = 100000) -> str:
    """Generate QA user prompt with proper formatting."""
    return QA_USER_TMPL.format(
        entity=entity,
        title=title,
        url=url,
        max_chars=max_chars,
        page_text=page_text
    )

def get_vet_user_prompt(question: str) -> str:
    """Generate vetting user prompt."""
    return VET_USER_TMPL.format(question=question)

# ======================= Module Exports =======================

__all__ = [
    'QA_SYSTEM',
    'QA_USER_TMPL',
    'VET_SYSTEM', 
    'VET_USER_TMPL',
    'EN_ENTITY_EXTRACTION_SYSTEM',
    'ZH_ENTITY_EXTRACTION_SYSTEM',
    'QUESTION_REWRITER_SYSTEM',
    'ENTITY_CATEGORIES',
    'get_entity_extraction_system',
    'get_qa_user_prompt',
    'get_vet_user_prompt'
]