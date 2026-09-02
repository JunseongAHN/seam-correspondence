# noise-inject

경계에 오차를 주입했을 때 봉제 대응이 얼마나 살아남는지 보는 도구.

GT 드레이프의 봉제 정점은 인접 패널 수만큼 복제 저장되고 **3D 좌표가 정확히 동일**하다
(UV만 다르다). 예측기는 패널마다 따로 예측하므로 오차는 복사본마다 독립이다.
여기에 두 가지 오차 모양을 넣고, 정면 depth map을 통과시킨 뒤 최근접으로 대응을 푼다.

| 모양 | 정의 | 뜻 |
|---|---|---|
| `white` | 점별 독립 `N(0, σ²I)` | 최악의 형태 |
| `ema` | **같은 난수**를 윤곽 따라 zero-phase 순환 EMA로 거른 뒤 크기를 white와 같게 되돌림 | 틀리되 GT 윤곽을 따라감 — AI 출력에 가까운 형태 |

두 모양은 **오차 크기가 같고 모양만 다르다.** 표의 `sep mm` 열이 두 행에서 같은 것이 그 확인이다.

## 쓰는 법 — VTK 뷰어 (권장)

패널 **표면 메시**를 원본 해상도 그대로(57,220면) 띄운다. 간략화 없음.

```bat
pip install vtk
python view_vtk.py
python view_vtk.py --garment rand_023FMIGQK0 --sigma 5 --smooth 20 --back
```

창이 검게만 나오면 그래픽 드라이버 문제다 — 아래의 브라우저 vtk.js 뷰어를 쓴다.

세 뷰포트가 하나의 카메라를 공유한다: **GT · 백색 · 평활(같은 RMS)**.
슬라이더로 σ와 평활 반복 수를 바꾸고, 키로 토글한다.

```
b  뒤쪽 패널 (기본: 앞쪽만)      g  GT 윤곽 겹쳐 보기
w  와이어프레임                  r  시점 초기화      q  종료
```

평활은 그 패널의 **메시 그래프 위 이웃 평균 반복**이다(윤곽 EMA의 곡면판).
좌표가 아니라 **변위에만** 걸고 크기를 백색과 같게 되돌리므로,
두 오른쪽 칸은 **크기가 같고 모양만 다르다**. σ=0이면 아무리 평활해도 GT와 같다.

## 쓰는 법 — 브라우저 vtk.js 뷰어 (표면)

`view_vtk.py` 와 **같은 패널·같은 해상도(57,220면)** 를 브라우저에서 띄운다.
데스크톱 VTK 창이 검게 나오는 기계(OpenGL 드라이버 문제)에서도 된다 — WebGL 은 다른 경로다.

```bat
python export_vtkjs.py                          :: 기본 rand_00YONAPXZE
python export_vtkjs.py --garment rand_023FMIGQK0

python serve.py                                 :: http://localhost:8000/vtkjs.html
```

세 칸이 카메라 하나를 공유한다: **GT · 백색 · 평활(같은 RMS)**.
σ 와 평활 반복 수는 슬라이더, 나머지는 체크박스이고 키도 `view_vtk.py` 와 같다(`b g w r`).

`vendor/vtk.js` 는 vtk.js 36.10.0 UMD 번들을 그대로 받아둔 것이다 — 네트워크 없이 뜬다.
난수는 브라우저의 seeded PRNG 라 `view_vtk.py` 의 numpy draw 와 수치가 일치하지는 않는다.
벌어짐 수치는 같다.

## 쓰는 법 — 브라우저 (윤곽만)

```bat
cd C:\repos\seam-correspondence\noise-inject

python export.py                          :: 기본 rand_00YONAPXZE
python export.py --garment rand_023FMIGQK0
python export.py --root D:\other\data --draws 12 --res 2048

python serve.py                           :: http://localhost:8000 자동으로 열림
```

`index.html` 을 파일로 직접 열면 안 된다 — 브라우저가 `fetch`를 막아서 데이터가 안 뜬다.
반드시 `serve.py` 로 연다. 데이터를 다시 뽑았으면 **새로고침만** 하면 된다(서버가 캐시를 끈다).

의존성은 `numpy` 뿐이다.

## 화면

**3-D 세 칸** — GT / 백색 오차 / EMA 평활. 같은 시점으로 함께 돈다.
드래그 = 회전 · 휠 = 확대 · 우클릭 드래그 = 이동.
`σ` 와 `EMA α` 는 브라우저에서 즉시 다시 뽑는다(시드 고정). 패널 하나만 띄울 수도 있다.

**표** — `export.py` 가 잰 σ × 오차모양 스윕.

| 열 | 뜻 |
|---|---|
| 벌어짐 | 짝인 두 복사본 사이 평균 거리 (mm) |
| 정점 | 최근접이 같은 `stitch_k` 를 공유한 비율 |
| 상호 NN | 서로가 서로의 최근접인 비율 — 학습 없이 나오는 신뢰도 신호 |
| 봉제선 전량 | 그 봉제선의 정점이 **전부** 맞음 |
| 봉제선 투표 | 그 봉제선 정점의 **과반**이 맞음 |
| 벌 전량 | 모든 봉제선이 투표로 성공한 추첨의 비율 |

## 파이프라인

```
패널 윤곽 (면 연결성에서 경계 엣지를 골라 닫힌 루프로 이음)
  → 복사본마다 독립 N(0, σ²I)          [+ 선택: 윤곽 따라 EMA, 크기 복원]
  → 정면 정사영 depth map (res², z-buffer 로 앞면만)
  → 리프트 (픽셀 중심 xy + 그 픽셀의 앞면 depth)
  → 1-최근접 대응
```

EMA는 **좌표가 아니라 변위에만** 건다. 필터가 곡선 자체를 안쪽으로 당기는 것과
노이즈를 깎는 것을 분리하기 위해서다. σ=0 이면 아무리 평활해도 GT와 정확히 같다.

## 한계 (수치를 읽기 전에)

- **오라클**이다. GT 드레이프에 노이즈만 준다. 실제 예측기는 모드 오류(위상 오판)가 더해지므로 이보다 나쁘다.
- **정면에서 보이는** 봉제 정점만 센다. 뒷면 전용 봉제선은 애초에 제외된다.
- depth map을 삼각형 래스터화가 아니라 **정점 splat**으로 만든다 → 가림이 과소평가된다.
- 디코더가 **1-최근접**이다. 상호 최근접·전역 할당(`ot_match`)·위상 게이트(`topo_check`)를 쓰지 않는다. 즉 이 수치는 **하한**이다.
- 한 벌 기준이고 추첨 수가 적다. 벌 단위 값은 `1/draws` 단위로 뛴다.

## 파일

```
view_vtk.py        VTK 뷰어 — 패널 표면 메시, 원본 해상도 (권장)
export.py          데이터 생성 CLI  → data/{contours,sweep,meta}.json
serve.py           로컬 정적 서버 (표준 라이브러리만)
index.html         뷰어 (data/*.json 을 fetch)
ninject/io_gcd.py  ply·segmentation 읽기, 용접, 패널 귀속, 경계 루프 추출
ninject/noise.py   white / zero-phase 순환 EMA / RMS 복원
ninject/pipeline.py 투영·리프트·디코드·채점
ninject/mesh.py    브라우저용 간략화 (경계 정점 보존, 내부만 병합)
```
