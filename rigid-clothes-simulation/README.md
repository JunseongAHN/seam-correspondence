# rigid-clothes-simulation — 사용법

GarmentCodeData 한 벌에서 **평면 패널과 봉제 대응만** 읽어, 외력 0에서 등거리 조립
(Isometric Shell)을 푼다. 중력·충돌 없음, **드레이프(`_sim.ply`의 3D 좌표)는 솔버 입력이
아니다** — 사후 비교용으로만 로드된다.

무엇이 왜 그렇게 돼 있는지는 [HANDOFF.md](HANDOFF.md), 검증 수치는
[README_garment.md](README_garment.md). 이 문서는 돌리는 법만 적는다.

## 환경

```
conda activate flexipanels
```

numpy, scipy, matplotlib, pyyaml 외에는 쓰지 않는다.

## 데이터

`C:\Users\PC\Downloads\data\<garment_id>\` 아래 3450벌. 한 벌에서 실제로 읽는 파일은 넷:

| 파일 | 쓰임 |
|---|---|
| `<id>_sim.ply` | 면(위상), UV(→ 평면 rest 좌표), 좌표 일치 용접(→ 봉제 대응) |
| `<id>_specification.json` | placement — 초기 배치 |
| `<id>_orig_lens.pickle` | UV를 cm로 되돌리는 이방 스케일 `Kx, Ky`의 적합 기준 |
| `<id>_body_measurements.yaml` | `--body` 프록시 (드레이프가 아니라 패턴 생성의 입력) |

데이터셋 경로는 하드코딩돼 있다: [run_garment.py:27](run_garment.py#L27)의 `GARMENT`,
[run_batch.py:36](run_batch.py#L36)의 `DATA`. 데이터를 옮겼으면 이 둘을 고친다. 한 번만
다른 곳을 볼 거면 `--garment`에 전체 경로를 주면 된다.

## 먼저 입력부터 확인

```
python gcd_io.py <garment_dir>          # 전체 경로. 생략하면 rand_00YONAPXZE
```

정점/면/봉제쌍 수, UV 스케일 `Kx, Ky`와 `orig_lens` 대비 엣지 길이 오차, 패널 목록,
placement bbox를 찍는다. 오차가 1%대를 넘으면 그 아래는 볼 필요가 없다.

## 한 벌 풀기

지금의 표준 설정 — `run_batch.py`가 쓰는 것과 같다:

```
python run_garment.py --garment <id> --outdir <dir> --amp 0 --body --sym --mu 0.02 --tag shell
python run_garment.py --fast            # 사다리 4단 × 120 iter, 구조만 빨리 보고 싶을 때
```

| 플래그 | 뜻 |
|---|---|
| `--garment` | garment id 또는 디렉터리 경로 |
| `--outdir` | 출력 폴더 (기본 `result/`) |
| `--amp A` | 초기 섭동 크기 (기본 0.01). `0`이면 spec placement 그대로 시작 |
| `--seed N` | 섭동 난수 seed (기본 1) |
| `--sym` | 좌우 대칭 섭동 — 문제 전체가 대칭으로 유지된다 |
| `--body` | measurements로 세운 해석적 몸(원통 9 + 구 1) 바깥에 유지 |
| `--mu W` | 반공간·몸 페널티 가중치, ARAP 대각 평균 대비 (기본 1.0, 표준은 0.02) |
| `--half-lr` / `--half-fb` | `left_*`는 x≥0·`right_*`는 x≤0 / 앞판은 z≥0·뒤판은 z≤0 |
| `--anchor W` | spec placement 쪽으로 약하게 당김 (0=끔) |
| `--inflate S` | 시작 형상을 S배 부풀린 넓은-basin 프로브 |
| `--lam-start` / `--lam-stop` | λ_b 사다리의 시작(1e-1 위로)·종료 지점 |
| `--per-lambda` / `--max-iter` | 사다리 한 단의 반복 수 / 총 상한 |
| `--rest FILE` | 평면 패턴 좌표를 npy로 교체 (`perturb_pattern.py`용) |
| `--tag NAME` | 출력 파일 이름. 생략하면 플래그 조합으로 자동 생성 |

λ_b 사다리는 기본 1e-1 → 1e-8 여덟 단, 단당 400 iter다.

### 나오는 것

`<outdir>/assembly_<tag>.{ply,npy,json}` — 형상, 좌표, 메타.

### 끝났다고 성공이 아니다

이 프로젝트에서 찢어진 실행은 **전부 exit 0이었다.** json의 세 수치를 직접 본다
([run_batch.py](run_batch.py)의 `gate()`와 같은 기준):

```
mono_violations == 0          단조성 위반
seam_gap_max    <  1e-3       봉제선이 벌어진 정도 (cm)
max_sigma_dev   <  1.0        max |sigma - 1|, 늘어난 정도
E_arap / E_bend / max_sigma_dev 가 유한
```

## 데이터셋 전체 돌리기

```
python run_batch.py [workers] [minutes]        # 기본 8 workers, 60분
```

- `proxy/<id>/assembly_shell.json`이 이미 있으면 건너뛴다 — 언제 끊어도 이어서 돈다.
- garment당 `proxy/<id>/`에 assembly, 패널별 색 ply, `log.txt`, `panel_colours.txt`.
- garment 하나에 json 한 줄씩 `proxy/_batch_log.jsonl`. 상태는 `ok` /
  `SUSPECT(어느 게이트인지)` / `FAILED(...)` 셋 중 하나이고, 실패한 것도 지우지 않는다.
- BLAS 스레드를 2로 고정한다. 안 하면 16코어에서 8프로세스가 6배 느려진다.
- **출력은 gitignore 대상이다** (garment당 약 7 MB). 결과는 커밋하지 말고 다시 돌린다.

## 렌더와 측정

| 명령 | 하는 일 |
|---|---|
| `python render_patches.py <id> <outdir>` | 패널마다 다른 색, 패널 외곽선은 검정. `<outdir>`의 `assembly_*.npy` 전부와 rest_flat / placement / drape_reference를 ply로 |
| `python render_all.py` | `result/`의 모든 run을 matplotlib으로 렌더 |
| `python measure_all.py [tag_prefix]` | 스펙 4절 (a)~(f) 측정 전부 → `result/measurements.json`, `result/angle_deficit.json`. `result/`만 본다 |
| `python gi_complexity.py [id]` | geometry image가 필요로 하는 용량 — 특성 파장, 스펙트럼 |
| `python perturb_pattern.py list \| run <direction> <eps> <outdir> \| report <outdir>` | 패턴을 조금 흔들어 shell이 얼마나 움직이는지 (Lipschitz 비) |
| `python pose_shape.py` | run 간 산포가 pose인지 shape인지 분해 → `result/pose_shape.json` |
| `python make_proxy_ply.py` | 몸 프록시를 ply로 떠서 눈으로 확인 → `result/proxy/*.ply` |
| `python fill_readme.py` | `result/measurements.json`에서 README_garment.md 4절을 다시 씀 |

## 알려진 걸림돌

- `pose_shape.py`, `gi_complexity.py`, 그리고 `render_patches.py`의 폴백 경로는
  `assembly_sym_body_ease5_*` 태그를 읽는다. seam ease가 삭제되면서 새 실행의 태그는
  `shell`이므로, 이 셋은 새 결과에 대고 돌리려면 태그를 고쳐야 한다.
- `perturb_pattern.py run`이 안내로 찍어주는 명령에는 지금은 없는 `--ease 5.0`이 들어 있다.
  그 부분을 빼고 실행한다.
