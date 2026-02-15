# anatomy-of-microgpt

Andrej Karpathy의 `microgpt`(순수 파이썬으로 구현한 최소 GPT-2 스타일
디코더 트랜스포머)의 알고리즘 핵심을 정리한 문서 중심 저장소입니다.

## 저장소 구성

- `microgpt_medium_post_en_v2.md`: 영문 메인 해설 문서
- `microgpt_appendix_line_by_line_en_v2.md`: 영문 라인별 부록
- `microgpt_medium_post_ko_v2.md`: 국문 메인 해설 문서
- `microgpt_appendix_line_by_line_ko_v2.md`: 국문 라인별 부록
- `fig1_microgpt_overview.png`: 전체 파이프라인 다이어그램
- `fig2_transformer_block_mapping.png`: 트랜스포머 블록 매핑 다이어그램
- `fig3_causal_mask_vs_kv_cache.png`: Causal mask와 KV cache 비교 다이어그램

## 권장 읽기 순서

1. `microgpt_medium_post_ko_v2.md`로 전체 흐름을 먼저 읽습니다.
2. `microgpt_appendix_line_by_line_ko_v2.md`에서 라인 번호 기준으로 코드를 추적합니다.
3. 읽는 동안 `fig1_microgpt_overview.png`, `fig2_transformer_block_mapping.png`,
   `fig3_causal_mask_vs_kv_cache.png`를 함께 참고합니다.

## 범위

- 이 저장소는 설명과 분석에 초점을 둡니다.
- 독립 실행용 `microgpt.py` 스크립트는 포함하지 않습니다.

## 원문 참고 링크

- https://karpathy.github.io/2026/02/12/microgpt/
- https://karpathy.ai/microgpt.html
- https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95

## English README

- `README.md`
