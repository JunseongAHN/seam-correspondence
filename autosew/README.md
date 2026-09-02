# AutoSew 재현 (arXiv 2602.22052, WACV 2026)

2D 세잉 패턴 → 봉제 대응 예측. GraphSAGE + 대칭 self-matching Sinkhorn(dustbin) + NLL.
공개 코드 없음(mslab 페이지에 논문 PDF뿐, 2026-09-02 확인) → 논문 명세 기반 자체 구현.
순 PyTorch (PyG 불필요). 의존성: `torch`, `numpy`.

## 실행

```bash
# 0) 데이터 검증 (학습 전 필수 — 포맷 가정 확인)
python scripts/validate_data.py --data_dir <GCD_PART_DIR> --limit 500

# 1) CPU 정합성 테스트 (~50s)
python tests/test_all.py

# 2) 합성 데이터 스모크
python -m autosew.train --synthetic 24 --epochs 60 --batch 8 --out runs/syn

# 3) 실데이터 학습 (GPU)
python -m autosew.train --data_dir <GCD_PART_DIR> --epochs 18 --batch 16 \
    --out runs/full --cache runs/full/cache.pt
```

`*_specification.json`을 재귀 검색하므로 `--data_dir`는 part 루트면 된다.
논문 하이퍼파라미터가 기본값: L=5/512/128/mean, T=100, τ=0.4, lr 1e-3, 18ep, 80-10-10 split.
논문 미기재 결정은 전부 `autosew/config.py`의 `[GAP]` 주석 + `--set KEY=VAL`로 변경 가능.

## 구조

| 파일 | 내용 |
|---|---|
| `autosew/gcd_parser.py` | specification.json → Pattern (ACW 정규화, 곡률 4종, 실물 GCD.v2 2벌로 검증) |
| `autosew/features.py` | 엣지당 24-dim (supp Table 1) + 패널 내 사이클 그래프 + GT |
| `autosew/model.py` | GraphSAGE 5층 (eq.1), dustbin 스칼라 z (eq.4) |
| `autosew/sinkhorn.py` | log-space Sinkhorn, 자기매칭 대각 마스킹, 패딩 배치, NLL (eq.5) |
| `autosew/metrics.py` | hard assignment (eq.6 + τ_multi) / TP·TR·TF1·MEP·MER·MEF1·GSP |
| `autosew/train.py` | 학습 CLI (Adam, 캐시, history.jsonl, best.pt) |
| `scripts/validate_data.py` | 실데이터 포맷/통계 검증 리포트 |
| `tests/test_all.py` | 파서 불변량, marginal, 패딩 동치, gradcheck, overfit |

자세한 실행 계획·결정 근거·알려진 거동: `HANDOFF.md`.
