# anatomy-of-microgpt

A document-first companion repository for studying the algorithmic core of
Andrej Karpathy's `microgpt` (a minimal GPT-2-like decoder transformer in pure
Python).

## What is in this repo

- `microgpt_medium_post_en_v2.md`: Main English write-up
- `microgpt_appendix_line_by_line_en_v2.md`: English line-by-line appendix
- `microgpt_medium_post_ko_v2.md`: Main Korean write-up
- `microgpt_appendix_line_by_line_ko_v2.md`: Korean line-by-line appendix
- `fig1_microgpt_overview.png`: End-to-end pipeline diagram
- `fig2_transformer_block_mapping.png`: Transformer block mapping diagram
- `fig3_causal_mask_vs_kv_cache.png`: Causal mask vs KV cache diagram

## Recommended reading order

1. Read `microgpt_medium_post_en_v2.md` for the high-level walkthrough.
2. Open `microgpt_appendix_line_by_line_en_v2.md` for line-indexed code analysis.
3. Refer to `fig1_microgpt_overview.png`, `fig2_transformer_block_mapping.png`,
   and `fig3_causal_mask_vs_kv_cache.png` while reading.

## Scope

- This repository focuses on explanation and analysis.
- It does not include a standalone runnable `microgpt.py` script.

## Original source references

- https://karpathy.github.io/2026/02/12/microgpt/
- https://karpathy.ai/microgpt.html
- https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95

## Korean README

- `README.ko.md`
