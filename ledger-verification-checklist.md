# 원장 축자 감사 체크리스트 — 2607.21213

**용도**: 8/30 정독 2회차에서 논문을 펴놓고 한 줄씩 대조한다.
**8/29 저녁 갱신**: G20(consistency loss) 해소 → 남은 ⚠ **12건**. 폐기 목록에 "균일 비례" 추가.
**배경**: 원장 머리에 *"모든 수치가 원문과 일치함을 확인"*이라고 적혀 있다. **수치만이다.**
인용부호 없는 요약문이 축자 옆에 나란히 있어서 같은 등급으로 읽혔고, 그 위에 카드가 세 번 세워졌다
(Chen et al. = Panelformer / §4.3.5 비례 분할 / occupancy·consistency loss).

**규칙**: 원장에서 `>` 또는 `"` 로 감싸이지 않은 문장은 슬라이드에 못 올린다.
발화도 *"제가 읽기로는"* 형태로만.

**등급**
```
✅ 축자    인용부호로 원문이 확보됨 → 그대로 사용 가능
⚠ 요약     원장 작성자의 서술. 인용부호 없음 → 확인 전 단정 금지
❌ 추정    repro-gaps 가 [추정]으로 명시한 것 → 카드 근거로 사용 금지
📐 내 계산  논문 사실이 아님 → "제 산수입니다" 명시 필수
```

---

## §3 문제 설정

| | 항목 | 등급 | 논문에서 확인할 것 |
|---|---|---|---|
| ☐ | stitch specification 4요소 (pairing / orientation / type / multiplicity) | ✅ | — |
| ☐ | "we restrict the problem to predicting boundary edge pairings… assume stitch orientation can be deterministically derived" | ✅ | — |
| ☐ | "To ensure tractability, we exclude such auxiliary information" | ✅ | — |
| ☐ | Fig 3 — 8종 중 5종 지원, "We only consider the first five types…" | ✅ | — |
| ☐ | §3.2 curve-centric 기각 "it scales poorly: the number of nodes and candidate matches grows rapidly…" | ✅ | — |

## §4.1 패널 표현

| | 항목 | 등급 | 논문에서 확인할 것 |
|---|---|---|---|
| ☐ | 이미지 4장 = 5채널 (mask / boundary / tangent 2ch / distance), 256², 1mm/px | ⚠ | **채널 구성과 해상도가 본문에 있는가, 그림에만 있는가** |
| ☐ | **정규화 "최대 치수를 이미지 폭의 0.85배로"** | ⚠ | **0.85 라는 수가 본문에 있는가.** 슬라이드 S-08 의 전제 |
| ☐ | **기하 서술자 5개 목록** — 둘레·면적·bbox 치수 2·래스터화 스케일 팩터 | ⚠ | **이 다섯 개가 나열된 문장이 있는가.** 개수만 있고 목록이 없을 수 있다 |
| ☐ | "panels are typically represented as non-self-intersecting boundary curves… less expressive for global shape characterization" | ✅ | — |
| ☐ | §4.1.1 방향 일관성 점수 `0.5(1−cos θ)` | ⚠ | 식이 본문인가 Fig 5 캡션인가 |
| ☐ | τ_l = 10, w₁ = 0.7, w₂ = w₃ = 0.15 | ✅ | (원장 ✅확인. Fig 5 캡션 출처로 알려짐 — **[그림]인지 확인**) |

## §4.2 Stage 1

| | 항목 | 등급 | 논문에서 확인할 것 |
|---|---|---|---|
| ☐ | **Stage 1 도식** — EfficientNet → MLP → **256차원** + 기하 5개 concat → **GAT 3층** → 헤드 2개 | ⚠ | **256 과 "3층"이 본문에 있는가.** 덱 표 ①층이 이 도식에 걸림 |
| ☐ | 12분류 목록 (collar / waist / L·R sleeves / F·B bodice / F·B skirt / L·R F·B pants) | ✅ | — |
| ☐ | Edge Decoder 출력 = **쌍당 스칼라 1개** | ⚠(본문+그림 종합) | 덱 표 ②층의 근거. **본문 문장이 있는가** |
| ☑ | **손실 두 항의 이름** — supervised topology loss + consistency loss | ✅ | **8/29 해소.** 축자 확보: *"In addition to a supervised topology loss, we introduce a consistency loss enforcing agreement between predicted connectivity and the anatomical semantic graph, promoting structurally and semantically coherent sewing-pattern reconstruction."* → 기여 슬라이드에 **존재만** 올린다 |
| ☐ | 두 손실의 **형태·가중치·조합** / anatomical semantic graph 의 **엣지 집합** | ❌ | **G20 잔여.** 어디에도 정의 없음. **"12분류를 연결성 사전으로 묶는다" 는 내 추측이었다 — 발화 금지** |

## §4.3 Stage 2

| | 항목 | 등급 | 논문에서 확인할 것 |
|---|---|---|---|
| ☐ | "…the distance field image is replaced by the original binary mask rendered using a **uniform scaling** determined by the largest panel dimension" | ✅ | — |
| ☐ | **Stage 2 도식** — `f⁰_sAB ∈ R^128`, `f⁰_snb ∈ R^128`, `f⁰_eAB ∈ R^256`, **GAT 2층**, `f_p ∈ R^332` | ⚠ | **G34 가 이미 "그림 1층 vs 본문 2층 불일치"를 지적.** 도식 자체가 흔들림 |
| ☐ | **occupancy 기제 (f⁰_snb, Fig 9)** | ❌ | **repro-gaps F3 이 "[추정→그림으로 강화]" 로 명시. "AGG 연산자·⊕·Conv 구조는 여전히 미기재".** → **기여 슬라이드의 전거를 [본문]→[그림·추정] 으로 내리거나 삭제** |
| ☐ | §4.3.4 "these fields are intersected with the panel boundary masks… yielding panel-aligned seam maps I_A and I_B" | ✅ | — |
| ☐ | §4.3.5 "seam segments… clustered based on spatial proximity and geometric similarity. **Multiple seam segments on a panel are clustered only when they are connected by darts.**" | ✅ | — |
| ☐ | **★§4.3.5 "학습된 매칭 네트워크 → 세그먼트 수가 적은 쪽에 시작/끝점 재정의 → 비례 길이 분할로 정렬"** | ⚠ | **최우선 확인.** 원장에 인용부호가 없다. 제안 A 의 전체 근거이자 덱 표 ③층 가운데 칸. **"균일" 이라는 단어는 확실히 원장에 없다** |

## §4.4 학습

| | 항목 | 등급 | 논문에서 확인할 것 |
|---|---|---|---|
| ☐ | "fully connected edges (20%), noisy edges (50%), and top-k edges selected using cosine similarity of CNN-extracted shape features (30%)" | ✅ | — |
| ☐ | "During inference, a fully connected graph is used to enable complete message passing." | ✅ | — |
| ☐ | AdamW / lr 1e-4 / cosine / wd 0.01 / mixed precision / **batch size 2** | ⚠ | 인용부호 없음. batch 2 는 카드 4 의 유일한 정량 신호라 확인 필요 |
| ☐ | **2단계 학습 "U-Net 만 pairwise 사전학습(GNN 제외) → 전체 end-to-end 미세조정"** | ⚠ | **기여 슬라이드 "끊기 2종" 의 근거.** 다만 `w/o PT` 행이 표에 있으니 **사전학습의 존재 자체는 확실** — 근거를 표로 갈아탈 것 |
| ☐ | "we curate a sewing-pattern dataset… 2,596 dresses, 393 pairs of pants, and 1,910 tops" | ✅ | — |
| ☐ | 랜덤 분할 80/10/10 | ⚠ | 축자 밖에 있음 |
| ☐ | **OOD "코트·재킷 포함 미지 스타일 90벌"** | ⚠ | **슬라이드 5(최강 카드)가 이 90 에 걸려 있다.** 숫자와 구성 확인 필수 |

## §5 결과

| | 항목 | 등급 | 논문에서 확인할 것 |
|---|---|---|---|
| ☐ | Table 1 전 셀 (LA 90.9 / E-F1 0.9073 / THR 0.74 / E-P 0.8995 / E-R 0.9162 / MAP 0.9719 + ablation 4행) | ✅ | — |
| ☐ | Table 2 전 셀 (Dice 0.8493 / RLE 0.1829 / Overlap 0.0072 + ablation 6행) | ✅ | — |
| ☐ | 지표 정의 — LA = label accuracy / E-F1 = **패널 쌍** 연결성 / Pixel Dice = seam indicator **이미지** 픽셀 일치 | ✅ | §5.1 원문으로 재확인 (사용자 붙여넣기로 확보됨) |
| ☐ | §5.1 이 "cross-style generalization" 을 지표로 열거 | ⚠ | **슬라이드 5 의 핵심.** 그 문장을 축자로 확보할 것 |
| ☐ | §5.3 이 정성 그림 + 문장 하나뿐 | ⚠ | OOD 에 대한 수치가 정말 0 건인지 전수 확인 |
| ☐ | Table 1·2 가 전부 자기 ablation (외부 비교 0건) | ⚠ | **슬라이드 5 의 2위 카드.** 표에 외부 방법 행이 정말 없는지 확인 |
| ☐ | Fig 12/13/14 캡션 "the reconstructed 3D garment with panel semantics represented by panel-wise color coding" / "Red ellipses indicate minor artifacts" | ✅ | — |

## §6 한계

| | 항목 | 등급 |
|---|---|---|
| ☐ | 한계 전문 (semantic classification 의존 / 5종만 / 데이터 다양성 / uncertainty modeling / user-in-the-loop) | ✅ |

---

## 📐 내 계산 — 논문 사실이 아님. 말할 때 "제 산수입니다"

| | 항목 | 전제 |
|---|---|---|
| ☐ | "칼라·커프가 20~30px 로 쪼그라든다" | **0.85배가 Stage 2 에도 적용된다는 가정.** 축자로 확인된 건 "가장 큰 패널 치수로 정한 균일 스케일" 뿐 |
| ☐ | "60cm 패널 ≈ 2.8mm/px, 1m ≈ 4.6mm/px" | 같은 가정 |
| ☐ | 슬라이드 S-08 "최소 패널 <30px 인 벌 7.19%" | 위 가정 + 내 데이터. **`[측정]`이지만 전제가 `[추정]`** |
| ☐ | 관계 제거 −6.0/−4.5 vs 패널 내 −2.9/−1.3 | Table 1 에서 내가 뺀 값. 표 자체는 ✅ |

---

## 이미 확정된 폐기 항목 (재확인 불필요, 쓰지 말 것)

```
Chen et al. 2024 카드        = Panelformer (WACV 2024), 입력이 의류 사진 → 다른 과제
"noisy edges = GT 교란"      §4.4 원문은 두 단어가 전부
"top-k 의 역설"              scaffold 는 attention 마스크지 손실 예제 집합이 아님
"학습·추론 위상이 분포 밖"     완전그래프 추론은 학습 20% 슬라이스 안
"드레이프 결과" / 시뮬레이터    논문에 없음
"11쪽 / ACM TOG"             arXiv v1 [cs.CV] 프리프린트
"성공 예시 30개 이상"          세어본 적 없음
접선장 단독 −2.9              TD 는 두 채널 동시 제거
"같은 생성 파이프라인 OOD"      데이터 생성 방식이 논문에 없음
"§4.3.5 는 균일 비례 분할"     축자는 다트 클러스터링 문장에서 끝난다. "비례"도 "균일"도 인용부호 밖
"12분류를 연결성 사전으로 묶는다"  consistency loss 의 형태에 대한 내 추측
```

---

## 대조 후 조치 — 결과에 따라

**§4.3.5 정렬 절차가 "비례"가 맞으면**
→ 표 ③층 가운데 칸 복구. 단 "균일" 대신 논문의 실제 표현을 쓸 것. 제안 A 근거 복구
**아니면**
→ ③층 가운데 칸은 "정렬 절차 — 논문 표현대로" 로. 제안 A 는 Q&A 카드로 영구 강등

**★ 8/29 결착 — 기여 슬라이드는 5줄로 확정**
```
범위 절단              축자 ✅
curve-centric 기각      축자 ✅
2단 분해                존재 ✅ (w/o PT 행) / 절차 서술은 모호
consistency loss        존재 ✅ (축자) / 정의는 ❌ — 등급을 문장 안에 넣어서 말한다
데이터 큐레이션 4,899벌  축자 ✅
```
occupancy 기제는 **올리지 않는다** (F3 이 [추정] 명시, AGG·⊕·Conv 미기재).
4번째 줄의 발화 형태: *"아이디어는 읽히는데 그게 참조하는 anatomical semantic graph 의
엣지 집합이 논문 어디에도 정의돼 있지 않아서 재현은 안 됩니다."*
→ 등급을 감추면 추정이 되고, **등급을 말하면 정독의 증거가 된다.**
