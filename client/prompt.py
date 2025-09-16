"""
Prompt templates and system prompts for synthetic data generation.

This module consolidates all prompt templates used across the synthetic data generation pipeline.
"""

# ======================= QA Generation Prompts =======================

QA_SYSTEM = """
You are a high-precision QA author. Produce ONE difficult but fair QA about the given ENTITY from the supplied PAGE TEXT.

# INPUTS
- ENTITY: <string>
- PAGE TEXT: <full plain text content of a single page>

# HARD REQUIREMENTS (MUST)

1) Objectivity & Verifiability
   - The question MUST be objectively answerable from THIS PAGE TEXT only (not from UI chrome, URL path, headers, cookie banners, access-denied pages, Cloudflare pages, or search results).
   - Do NOT ask about transient info (deadlines, live schedules, prices, rosters, “today/now/recent/news”), unless the page explicitly states a historical, fixed fact (e.g., “held on 12 May 2014”).
   - Prefer facts that are stable for ≥12 months.

2) Relevance & Uniqueness
   - The question MUST explicitly mention the ENTITY.
   - The answer MUST be a short, unique, unambiguous, verbatim span (≤ 10 words) appearing in the PAGE TEXT body (not title-only, not URL, not alt text, not boilerplate footer).
   - Provide a short evidence quote (20–200 chars) that contains the answer.

3) Difficulty (but not trickery)
   - Target obscure but on-page details (e.g., subclauses, footnotes, minor participants, catalog entries IN TEXT, table cells).
   - Do NOT ask about generic “error messages”, “Cloudflare Ray ID”, “access denied”, “CAPTCHA”, or viewer/UI artifacts.
   - Do NOT ask for strings visible only in the address bar, file name, query params, or PDF viewer UI.
   - The question should require *on-page searching/scrolling* (i.e., non-obvious location in the PAGE TEXT), but MUST NOT require any external web search.

4) Page Quality Gate (FAIL → reject)
   - If the page is gated (login, paywall, human-verification), a soft 404, a blocked “x.com” thread, or has insufficient text to support a unique answer → you MUST reject.

# NEW CONSTRAINTS TO AVOID PRIOR ISSUES

5) Completeness & Non-subjectivity
   - The question MUST be fully specified (include all necessary qualifiers like section, item, version, unit, timeframe if mentioned in PAGE TEXT).
   - NO subjective or evaluative language (e.g., “best”, “notable”, “significant”, “important”) unless the PAGE TEXT explicitly defines these terms and the answer is a direct quote.
   - NO underspecified prompts (avoid vague pronouns like “it/they/this” without clear referents).

6) No Wordplay / Riddles
   - NO puns, trick questions, lateral thinking, acrostics, homophones, riddles, or double meanings.
   - The question must be literal and straightforward.

7) No Meta-Phrasing or URLs in the Question
   - The question text MUST NOT contain phrases like “on this page”, “based on the page”, “according to the webpage/article/site”, etc.
   - The question MUST NOT include any URL, link text, or instructions to visit a site.

8) Source Scope
   - The question MUST be derived solely from the PAGE TEXT content and require searching *within that text* to answer.
   - Do NOT reference or imply external sources, search engines, or navigation.

# AUTHORING CHECKLIST (ENFORCE BEFORE OUTPUT)
- Scan PAGE TEXT to ensure the chosen answer string is unique (or uniquely disambiguated by your question).
- Ensure the question includes ENTITY and all qualifiers to avoid ambiguity.
- Verify the answer is ≤10 words and appears verbatim in the PAGE TEXT body.
- Verify no meta-phrasing and no URLs appear in the question.
- Verify the question is objective, complete, and non-subjective.
- Verify no wordplay or riddle elements are present.
- If any item fails, output a rejection JSON (see below).

# OUTPUT CONTRACT (STRICT JSON ONLY)
- If ALL constraints can be satisfied, output exactly:
{
  "question": "<must include the ENTITY; precise and fully specified; no meta-phrasing; no URLs>",
  "answer": "<<=10-word exact span from PAGE TEXT>",
  "evidence_quote": "<20–200 chars containing the exact answer>",
  "evidence_locator": {
    "section_or_heading": "<best guess or 'unknown'>",
    "why_this_proves": "<1–2 sentences linking quote to question>"
  },
  "checks": {
    "objective_from_page_only": true,
    "link_is_relevant_and_proves": true,
    "answer_unique_on_page": true,
    "answer_not_from_url_or_ui": true,
    "not_error_or_cloudflare_or_captcha": true,
    "no_meta_phrasing_or_url_in_question": true,
    "no_wordplay_or_riddle": true,
    "question_fully_specified_and_non_subjective": true,
    "time_sensitivity": "low|medium|high",
    "time_risk_note": "<why stability is acceptable; 'none' if stable>"
  }
}

- If ANY constraint fails, output exactly:
{
  "reject": true,
  "reason": "<1–2 sentences: which constraint failed and why>"
}
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


INSTRUCTION_DIMENSIONS = {
    "disciplines": {
        "STEM": ["Mathematics", "Physics", "Chemistry", "Biology", "Computer Science", "Engineering", "Materials Science", "Earth Sciences", "Astronomy", "Statistics"],
        "Social Sciences": ["Economics", "Political Science", "Sociology", "Psychology", "Anthropology", "Geography", "Law", "Education", "Communication Studies", "International Relations"],
        "Humanities & Arts": ["History", "Philosophy", "Literature", "Linguistics", "Art History", "Music Studies", "Theater Studies", "Religious Studies", "Archaeology", "Aesthetics"],
        "Applied Sciences": ["Medicine", "Public Health", "Environmental Science", "Urban Planning", "Business", "Management", "Agriculture", "Architecture", "Design Studies", "Information Science"],
        "Interdisciplinary": ["Cognitive Science", "Bioinformatics", "Computational Social Science", "Digital Humanities", "Science & Technology Studies", "Sustainability Studies", "Complex Systems", "Network Science"]
    },
    
    "topics_by_discipline": {
        "Mathematics": ["Topological Data Analysis", "Machine Learning Theory", "Graph Theory Applications", "Optimization Algorithms", "Probability and Statistics", "Numerical Computing", "Cryptography", "Game Theory"],
        "Physics": ["Quantum Computing", "Condensed Matter Physics", "Complex Systems", "Soft Matter", "Photonics", "Plasma Physics", "Astrophysics", "Biophysics"],
        "Chemistry": ["Catalytic Chemistry", "Materials Chemistry", "Green Chemistry", "Pharmaceutical Chemistry", "Computational Chemistry", "Electrochemistry", "Polymer Chemistry", "Analytical Chemistry"],
        "Biology": ["Synthetic Biology", "Systems Biology", "Evolutionary Biology", "Molecular Biology", "Ecology", "Neurobiology", "Developmental Biology", "Conservation Biology"],
        "Computer Science": ["AI Ethics", "Quantum Algorithms", "Privacy-Preserving Computing", "Human-Computer Interaction", "Distributed Systems", "Computer Vision", "Natural Language Processing", "Cybersecurity"],
        "Economics": ["Behavioral Economics", "Experimental Economics", "Digital Economy", "Development Economics", "Environmental Economics", "Labor Economics", "Financial Economics", "Industrial Organization"],
        "Sociology": ["Digital Sociology", "Social Networks", "Social Mobility", "Social Psychology", "Organizational Sociology", "Urban Sociology", "Cultural Sociology", "Social Policy"],
        "Psychology": ["Cognitive Psychology", "Developmental Psychology", "Social Psychology", "Neuropsychology", "Positive Psychology", "Cultural Psychology", "Health Psychology", "Engineering Psychology"],
        "History": ["Digital History", "Environmental History", "History of Science and Technology", "Social History", "Cultural History", "Global History", "Oral History", "Public History"],
        "Philosophy": ["Philosophy of Technology", "Philosophy of Science", "Political Philosophy", "Ethics", "Epistemology", "Philosophy of Mind", "Philosophy of Language", "Aesthetics"],
        "Art": ["Digital Art", "Interactive Art", "Socially Engaged Art", "Eco-Art", "Performance Art", "Visual Culture", "Art Therapy", "Creative Industries"],
        "Environmental Science": ["Climate Change", "Biodiversity", "Pollution Control", "Ecological Restoration", "Sustainable Development", "Environmental Justice", "Ecosystem Services", "Circular Economy"],
        "Medicine": ["Precision Medicine", "Digital Health", "Global Health", "Aging", "Mental Health", "Infectious Disease Control", "Health Inequalities", "Medical AI"],
        "Education": ["Educational Technology", "Online Learning", "Educational Assessment", "Educational Equity", "Lifelong Learning", "STEM Education", "Multicultural Education", "Education Policy"]
    },
    
    "task_types": [
        "Systematic Literature Review", "Prospective Roadmap", "Comparative Analysis", "Mechanism Modeling", "Causal Inference Study", 
        "Evaluation Framework Design", "Policy Analysis", "Technology Assessment", "Risk Assessment", "Ethical Impact Assessment",
        "Experimental Design Protocol", "Survey Research Design", "Monitoring System", "Intervention Strategy", "Best Practices Summary",
        "Interdisciplinary Integration", "Methodological Innovation", "Tool Development", "Standard Setting", "Capacity Building Framework"
    ],
    
    "methodologies": {
        "Quantitative Methods": ["Randomized Controlled Trials", "Quasi-experimental Design", "Difference-in-Differences", "Regression Discontinuity", "Instrumental Variables", "Structural Equation Modeling", 
                               "Bayesian Statistics", "Machine Learning", "Time Series Analysis", "Panel Data Analysis"],
        "Qualitative Methods": ["In-depth Interviews", "Focus Groups", "Participant Observation", "Case Studies", "Grounded Theory", "Discourse Analysis", 
                              "Content Analysis", "Ethnography", "Phenomenological Research", "Action Research"],
        "Mixed Methods": ["Triangulation", "Explanatory Sequential", "Exploratory Sequential", "Concurrent Convergent", "Embedded Design"],
        "Computational Methods": ["Agent-Based Modeling", "Network Analysis", "Text Mining", "Image Analysis", "Simulation Modeling", 
                                "Data Mining", "Complex Systems Analysis", "Artificial Intelligence", "Blockchain Technology", "Quantum Computing"],
        "Participatory Methods": ["Community-Based Participatory Research", "Citizen Science", "Design Thinking", "Co-design", "Stakeholder Engagement"]
    },
    
    "geographical_scope": [
        "Global", "Developed Countries", "Developing Countries", "Least Developed Countries", 
        "North America", "Latin America", "Europe", "East Asia", "Southeast Asia", "South Asia", "Middle East", "Africa", "Oceania",
        "China", "United States", "European Union", "OECD Countries", "G20", "BRICS",
        "Urban Areas", "Rural Areas", "Remote Areas", "Border Regions", "Island Nations"
    ],
    
    "target_populations": [
        "Children and Adolescents", "Adults", "Elderly", "Women", "Men", "LGBTQ+ Communities",
        "Low-income Groups", "Middle Class", "High-income Groups", "Ethnic Minorities", "Indigenous Peoples",
        "Students", "Teachers", "Researchers", "Policymakers", "Entrepreneurs", "Workers",
        "Patients", "People with Disabilities", "Immigrants", "Refugees", "Homeless Individuals", "Farmers", "Fishermen"
    ],
    
    "time_horizons": [
        "Historical Review (10+ years ago)", "Recent History (5-10 years)", "Current Status", 
        "Short-term Forecast (1-2 years)", "Medium-term Planning (3-5 years)", "Long-term Outlook (5-10 years)",
        "Extended Long-term (10-20 years)", "Generational Changes (20+ years)", "Scenarios to 2030", 
        "Scenarios to 2050", "Scenarios to 2100"
    ],
    
    "data_sources": [
        "Academic Literature Databases", "Government Statistics", "International Organization Data", "Commercial Data", 
        "Social Media Data", "Sensor Data", "Remote Sensing Data", "Survey Data", "Experimental Data",
        "Administrative Records", "Electronic Health Records", "Educational Administrative Data", "Financial Transaction Data",
        "Patent Data", "News Media Data", "Historical Archives", "Oral History", "Archaeological Artifacts"
    ],
    
    "ethical_constraints": [
        "Privacy Protection", "Informed Consent", "Data Security", "Algorithmic Fairness", "Transparency", 
        "Accountability", "Human Dignity", "Child Protection", "Vulnerable Population Protection", 
        "Cultural Sensitivity", "Environmental Responsibility", "Social Responsibility", "Research Ethics", "AI Ethics"
    ],
    
    "regulatory_frameworks": [
        "GDPR", "CCPA", "HIPAA", "FERPA", "IRB Review", "Animal Research Ethics",
        "Clinical Trial Standards", "Data Localization", "International Sanctions", "Intellectual Property", 
        "Open Source Licenses", "Academic Integrity", "Conflict of Interest Disclosure"
    ],
    
    "resource_constraints": {
        "Budget": ["Low Budget (<$100K)", "Medium Budget ($100K-$500K)", "High Budget ($500K+)"],
        "Timeline": ["Urgent (<3 months)", "Short-term (3-12 months)", "Medium-term (1-3 years)", "Long-term (3+ years)"],
        "Data Access": ["Open Data", "Restricted Data", "Data Collection Required", "Scarce Data", "Sensitive Data"],
        "Computing Resources": ["Local Computing", "Cloud Computing", "High-Performance Computing", "Quantum Computing", "Edge Computing"],
        "Team Size": ["Individual", "Small Team (2-5 people)", "Medium Team (6-15 people)", "Large Team (15+ people)"]
    },
    
    "output_formats": {
        "Report Types": ["Research Proposal", "White Paper", "Policy Brief", "Technical Report", "Evaluation Report", 
                        "Best Practices Guide", "Standards Document", "Roadmap", "Feasibility Study"],
        "Target Audiences": ["Academic Peers", "Policymakers", "Corporate Executives", "Technical Experts", "General Public", 
                           "Students", "Media", "Investors", "International Organizations", "NGOs"],
        "Report Lengths": ["Brief (500-1000 words)", "Standard (1000-3000 words)", "Detailed (3000-8000 words)", 
                          "Monograph-level (8000+ words)"],
        "Writing Styles": ["Academic Rigorous", "Policy-oriented", "Technical Documentation", "Accessible", "Persuasive", "Neutral Objective"],
        "Visualization Types": ["Flowcharts", "Causal Diagrams", "Network Graphs", "Timelines", "Dashboards", "Maps", 
                              "Statistical Charts", "Concept Maps", "Decision Trees", "Roadmaps"]
    },
    
    "openness_parameters": {
        "Alternative Count": [3, 4, 5, 6, 7],      # N alternative solutions
        "Perspective Count": [2, 3, 4],           # P disciplinary perspectives
        "Uncertainty Count": [2, 3, 4, 5],        # K key uncertainties
        "Tradeoff Dimensions": [2, 3, 4],         # M tradeoff dimensions
        "Scenario Count": [2, 3, 4, 5]            # Number of scenarios
    },
    
    "analysis_requirements": [
        "Critical Analysis", "Comparative Assessment", "Stakeholder Analysis", "Cost-Benefit Analysis",
        "Risk-Benefit Analysis", "SWOT Analysis", "Gap Analysis", "Trend Analysis", 
        "Impact Assessment", "Feasibility Analysis", "Sensitivity Analysis", "Scenario Planning"
    ],
    
    "research_approaches": [
        "Evidence-based", "Theory-driven", "Data-driven", "Problem-oriented", "Solution-focused",
        "Participatory", "Transdisciplinary", "Systems Thinking", "Design Thinking", "Innovation-focused"
    ]
}

ENTITY_GENERATE_PROMPT_TEMPLATE = """Generate one niche, obscure but academically or practically valuable entity based on:

- Primary Domain: {topic}
- Analytical Focus 1 ({random_key1}): {random_value1}
- Analytical Focus 2 ({random_key2}): {random_value2}

Requirements:
- Entity can be any type: institution, discipline, species, organization, concept, technology, cultural phenomenon, etc.
- Must be niche and uncommon, not mainstream
- Should have unique value or research significance
- Must reflect the intersection of both analytical focuses

Output only the entity name, nothing else.
Just Like: "Entity Name"
"""
