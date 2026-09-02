# HANDOFF — AutoSew 재현 학습 (GPU 세션용) · 2026-09-02 오전

**작성**: 클라우드 세션 (CPU, 구현·검증 담당). **수신**: GPU 머신 에이전트.
**목표(오늘 오후)**: 다운로드된 GCD.v2 part(~7GB, one-to-one)로 18 epochs 학습을 돌리고
val TF1/GSP 궤적과 test 지표를 보고한다. 논문 수치 재현이 아니라 **학습이 돌고 지표가
논문 방향으로 가는지**의 확인이 오늘의 정의다.

## 0. 참조점 (논문 Table, GCD.v2 one-to-one 행)

| TP | TR | TF1 | GSP |
|---|---|---|---|
| 97.19 | 96.93 | 97.06 | 80.6 |

논문은 **풀 128K(train 80%)** 학습. 우리 part는 부분집합이므로 **몇 점 낮으면 정상**.
TF1 90+면 구현 건전성으로는 합격. 50대에서 정체하면 구현 문제로 취급하고 §5를 볼 것.
(M-E 멀티엣지 벤치마크는 오늘 범위 밖 — 데이터 비공개, 재구축은 별도 작업.)

## 1. 실행 순서 (이 순서 그대로)

```bash
cd C:\repos\seam-correspondence\autosew   # (경로 표기는 환경에 맞게)

# [1] 데이터 포맷 검증 — 학습 전 필수
python scripts/validate_data.py --data_dir C:\Users\POMCHECKER\Downloads\garments_5000_0\garments_5000_0\default_body\data --limit 500
#   합격 기준: parse_fail=0, loop_violations=0, kt=5(UNKNOWN)=0
#   edges_per_panel [min,max] 을 보고 config.edge_count_minmax 갱신 (기본 (2,40))
#   units_in_meter 가 {100} 외의 값이면 중단하고 보고 (circle radius 스케일 재검토 필요)

# [2] CPU/GPU 정합성 테스트 (~1분)
python tests/test_all.py

# [3] 500벌 스모크 (epochs 2) — 처리량 측정 겸 캐시 생성 확인
python -m autosew.train --data_dir C:\Users\POMCHECKER\Downloads\garments_5000_0\garments_5000_0\default_body\data --limit 500 --epochs 2 --batch 16 --out runs/smoke

# [4] 본 학습
python -m autosew.train --data_dir C:\Users\POMCHECKER\Downloads\garments_5000_0\garments_5000_0\default_body\data --epochs 18 --batch 16 \
    --out runs/full --cache runs/full/cache.pt
#   JSON 파싱이 느리면 첫 실행만 느림 (cache.pt 재사용). --limit 로 규모 조절 가능.

# [5] 보고: runs/full/history.jsonl 전체 + test_metrics.json + [3]의 벌당 처리시간
```

`C:\Users\POMCHECKER\Downloads\garments_5000_0\garments_5000_0\default_body\data` = `C:\Users\POMCHECKER\Downloads\garments_5000_0\garments_5000_0\default_body\data` — GCD.v2 garments_5000_0 / default_body, ~5,000벌 (train ~4,000).
`*_specification.json`을 재귀 검색하므로 루트만 주면 된다. GPU 메모리는 문제될 수 없음
(M~116이면 batch 16에 (16,117,117) 행렬 — 모델 전체 ~1.3M 파라미터).

## 2. 논문에 명시된 것 (기본값으로 이미 설정됨 — 바꾸지 말 것)

feature 24-dim 구성(supp Table 1 + §3.1 전처리), GraphSAGE L=5 / hidden 512 / mean /
ReLU / D=128, Sinkhorn T=100, τ_multi=0.4, lr 1e-3, 18 epochs, 80-10-10 split,
NLL(eq.5), dustbin 학습 스칼라 z 1개(eq.4), P′=½(P̄+P̄ᵀ) 후 행별 argmax+τ 유지(§5.2).

## 3. 논문 미기재 → 우리 결정 (config.py [GAP] 주석과 1:1 대응)

| 항목 | 우리 기본값 | 대안 (`--set KEY=VAL`) |
|---|---|---|
| panel ID u 인코딩 | 파일 내 패널 순서 idx/37 | `panel_id_mode=index_raw / random_norm` |
| 그래프 연결 | 패널 내 사이클만 (패널 간 무연결) | 코드 수정 필요 — 오늘은 고정 |
| 마지막 층 활성 | ReLU (eq.1 축자 독해) | `final_activation=none` |
| 128 출력 위치 | 5층째가 512→128 | `layer_scheme=proj` |
| Sinkhorn ε | 1 (온도 없음, SuperGlue) | `score_scale=rsqrt_d` |
| marginal | 실엣지 1, dustbin M (SuperGlue) | 고정 |
| optimizer | Adam, wd 0 | — |
| loss 방향 | (i,j)+(j,i) 양방향 + 미봉제↔bin 양방향, mean | `loss_both_directions=false` |
| GSP 정의 | strict (pred==GT 집합 일치) | recall-only도 항상 같이 출력됨 |
| curvature frame | 패널 로컬 절대좌표 /100 (circle radius만 /100, 플래그 원값) | `curvature_frame=rel` |

CPU 대조 실험 결과: relu-last vs linear-last vs rsqrt_d — 합성 overfit에서 유의차 없음.
u 인코딩(idx/37 vs raw)도 무차이. 실데이터에서 다를 수 있으니 정체 시에만 스윕.

## 4. 검증된 것 (이 코드에서 참)

- 파서: **실물 GCD.v2 spec 2벌**(리포 `data/rand_00YONAPXZE`, `rand_GMSLZXQMDK`)로 검증.
  18패널/M=116/스티치 54, 곡률 {straight 88, circle 8, quad 8, cubic 12}, UNKNOWN 0,
  루프 폐합·ACW 정규화(역방향 패널 8개 반전 처리)·같은 패널 내 다트 스티치 처리 확인.
  `properties.units_in_meter=100`, `curvature_coords=relative`, `panel_order` 사용 확인.
- Sinkhorn: 실행/열 marginal 수렴(실엣지 행합≈1, bin행합≈M), 패딩 배치 == 단건 계산
  (atol 1e-4), P̄ 근사대칭. **gradcheck 통과** (float64, 임베딩+z).
- 학습: 합성 24벌 overfit — bodice/skirt GSP 1.0, TR 1.0. **실물 2벌 overfit —
  150ep/15s(CPU)에 TF1 1.0, GSP 1.0.**

## 5. 알려진 거동 (버그로 오인 금지)

1. **multi FP 모드**: 동일한 반쪽 패널 두 장이 같은 상대와 봉제될 때 서로를 P′≈0.49로
   끌어당겨 FP 1개 생성. 원인은 self-matching 내적 스코어의 구조: 확신 매칭 엣지는
   ‖f‖가 커지고, 그 근사복제 엣지와의 내적이 공짜로 커진다. 논문 §8.2 "identical
   panels … interchangeable" + MEP 80.4(멀티 FP ~20%)의 재현. one-to-one part에서도
   대칭 의류(좌우 소매 등)에서 τ=0.4 초과 FP가 나올 수 있다 — TP가 TR보다 낮으면 이것.
2. GSP는 strict 기준이라 논문 정의(문면상 recall-only)보다 엄격. 둘 다 출력되니
   비교 시 명시할 것.
3. part에 멀티엣지 GT가 없으므로 MEP/MER/MEF1=0, `has_multi_edge_gt=false` — 정상.
4. 실데이터에 길이 0.1cm 수준의 미소 엣지 존재 (방향벡터 (0,0) 처리됨).

## 6. 문제 시 처방

| 증상 | 처방 |
|---|---|
| loss NaN/발산 | `--set score_scale=rsqrt_d`, 그래도면 `grad_clip=1.0` |
| TF1 50대 정체 | `--set final_activation=none` → `l2_normalize=true` → `panel_id_mode=index_raw` 순서로 1개씩 |
| 파싱 실패 다수 | validate_data 리포트의 fail_examples 5건을 보고 gcd_parser 수정 (curvature 새 타입일 가능성 최대) |
| epoch 너무 느림 | 병목은 Sinkhorn 아닌 파싱 — cache.pt 확인. 그래도면 `--limit 20000` |

## 7. 보고 양식 (이대로 돌려줄 것)

- validate_data 리포트 JSON 전문
- [3] 스모크의 벌당 처리시간 (초/벌, 학습 기준)
- history.jsonl (epoch별 val TF1/TR/TP/GSP/loss/z)
- test_metrics.json + 실패 패턴 수
- 바꾼 config 항목 전부 (기본값이면 "없음")

## 8. 환경

완료 — RTX 4050 Laptop(6GB), torch CUDA 확인됨(True), env `agentic-ai`(Python 3.14). 참고용 원명령: nvidia-smi로 CUDA 확인 후
`pip install torch --index-url https://download.pytorch.org/whl/cu126` (드라이버 13.x면 cu130).
검증: `python -c "import torch; print(torch.cuda.is_available())"`. numpy 필요 (있을 것).
