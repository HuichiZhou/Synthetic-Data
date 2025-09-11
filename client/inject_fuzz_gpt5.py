from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
base_url = os.getenv("OPENAI_BASE_URL")
api_key = os.getenv("OPENAI_API_KEY")

if not api_key or not base_url:
    raise RuntimeError("Please set OPENAI_API_KEY and OPENAI_BASE_URL environment variables")

client = OpenAI(base_url=base_url, api_key=api_key)


systme_prompt = """
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
       * exact date/year → bounded range (e.g., “early 1930s”)
       * exact number → small numeric interval (“about”, “mid-”, “low-”, or x–y)
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
FINAL ANSWER: {
  "language": "<lang>",
  "original_question": "<q0>",
  "steps": [
    {
      "type": "inject" | "fuzz",
      "before": "<text_before>",
      "after": "<text_after>",
      "descriptor": "<only for inject: the NP you used>",
      "used_facts": [
        {"fact": "<short factual clause>", "url": "<source_url>", "quote": "<short span if available>"}
      ],
      "fuzz_ops": ["hypernym"|"time_range"|"number_range"|"synonym"],  // for fuzz; omit for inject
      "verify": {
        "unique": true|false,
        "same_entity": true|false,
        "canonical_name": "<name from evidence>",
        "support_domains": ["example.gov","example.edu", "..."],   // distinct domains only
        "support_urls": ["https://...", "..."],
        "note": "<brief reasoning>"
      }
    }
  ],
  "final_question": "<the last successful rewrite>",
  "warnings": ["<optional>"]
}

HARD RULES
- INJECT must **remove the original name** and replace it with a descriptor built ONLY from tool-verified facts.
- Never introduce facts you did not see in evidence. Include at least one URL per fact; prefer 2+ domains overall.
- Maintain answer invariance and question type. Reject a rewrite that changes the target or introduces multiple candidates.
- If a step cannot be made unique after MAX_REFINE refinements, revert that step and explain in `warnings`.
- Keep total verbosity low; avoid long appositives. Aim for precision-first descriptors.
"""


data_line = []
# if i only want to extract the 3 keys in the json line, how to do it?

with open("/home/abaka/zhc/curation/client/final/Economics/Behavioral_Economics.jsonl", "r") as f:
    for line in f:
        data_line.append(line)


import json 

def extract_keys(json_line):
    json_data = json.loads(json_line)
    return "Question:" + json_data["question"], "Ground Truth:" + json_data["ground_truth"], "Data Source:" + json_data["data_source"]

data_line = [extract_keys(line) for line in data_line]

from tqdm import tqdm


for item in tqdm(data_line):

    response = client.responses.create(
        model="gpt-5-search",
        tools=[{"type": "web_search_preview"}],
        input=systme_prompt + str(item),
        stream=False
    )
    
    print(response.output_text)

    # save in a file 
    with open("result/Economics/Behavioral_Economics.jsonl", "a") as f:
        f.write(response.output_text + "\n")
