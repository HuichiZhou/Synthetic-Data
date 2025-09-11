```python
python synthetic_entity.py \
  --topic "Chinese Cuisine" \
  --server ../server/serp_search.py --server ../server/craw_page.py \
  --k 25 --per-page 8 --model gpt-4o \
  --locale en --only-nonwiki --out out/hc
