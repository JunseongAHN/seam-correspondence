"""Write section 4 of README_garment.md straight from result/measurements.json,
so the prose cannot drift from the numbers in the files.

  python measure_all.py > result/measurements.txt
  python fill_readme.py
"""

import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RESULT = os.path.join(HERE, "result")
README = os.path.join(HERE, "README_garment.md")

DRAPE_DXY = {"skirt_front": 15.82, "ftorso": 9.11, "wb_front": 11.13,
             "sleeve_f": 6.26, "cuff_f": 3.09, "hood": 12.88}


def main():
    m = json.load(open(os.path.join(RESULT, "measurements.json")))
    L = []
    A = L.append

    A("## 4. 결과")
    A("")
    A("실행 %d회: `%s`" % (len(m["runs"]), "`, `".join(m["runs"])))
    A("초기값은 전부 **배치 + 난수 섭동**이고, `inflated`만 배치를 1.5배 부풀린 것이다.")
    A("모든 비교는 Procrustes 정합 후이며 **반사는 허용하지 않았다**(거울상 옷은 다른 옷이다).")
    A("")

    # (c)
    A("### (c) 내적 일관성")
    A("")
    A("| 실행 | `max\\|σ−1\\|` 전체 | 개더 제외 | 봉제 갭 max (cm) | 단조성 위반 | 반복 |")
    A("|---|---|---|---|---|---|")
    for t, c in m["consistency"].items():
        A("| `%s` | %.3e | %.3e | %.3e | %d | %d |"
          % (t, c.get("max_sigma_dev", float("nan")),
             c.get("max_sigma_dev_nogather", float("nan")),
             c.get("seam_gap_max", float("nan")),
             c.get("mono_violations", -1), c.get("iterations", -1)))
    A("")

    # half-space constraint compliance
    rows = []
    for f in sorted(glob.glob(os.path.join(RESULT, "assembly_*.json"))):
        j = json.load(open(f))
        if not j.get("half_space"):
            continue
        rows.append((j["tag"], j["half_space"], j.get("mu_rel", 1.0)))
    if rows:
        A("### (c-2) 반공간 제약 준수")
        A("")
        A("좌우 평면 `x=0`(무게중심 기준)과 앞뒤 평면 `z=0`에 대한 단측 2차 벌점이다.")
        A("행렬 대각에 상수 `mu`를 얹으므로 활성집합이 바뀌어도 재분해가 없다"
          " (`mu` = 평균 ARAP 대각 × %.3g)." % rows[0][2])
        A("")
        NAME = {"ax0_sg+1": "left_* : x≥0", "ax0_sg-1": "right_* : x≤0",
                "ax2_sg+1": "front : z≥0", "ax2_sg-1": "back : z≤0"}
        keys = list(rows[0][1].keys())
        A("| 실행 | " + " | ".join(NAME.get(k, k) + " 침범율 / 최대(cm)" for k in keys) + " |")
        A("|---|" + "---|" * len(keys))
        for t, h, _ in rows:
            A("| `%s` | " % t + " | ".join(
                "%.2f%% / %.3f" % (100 * h[k]["frac"], h[k]["max_cm"]) for k in keys) + " |")
        A("")

    # (a)
    A("### (a) 초기값 민감도 — **주 결과**")
    A("")
    A("6개 결과를 Procrustes 정합한 뒤 정점마다 잰 산포다. 작을수록 봉제가 형태를 결정한다.")
    A("")
    A("| 패널 클래스 | 표준편차 p50 (cm) | p90 | 최대편차 p50 | 정점 수 |")
    A("|---|---|---|---|---|")
    for c, v in sorted(m["sensitivity_cm"].items(), key=lambda kv: kv[1]["std_p50"]):
        A("| `%s` | **%.3f** | %.3f | %.3f | %d |"
          % (c, v["std_p50"], v["std_p90"], v["maxdev_p50"], v["n"]))
    A("")

    # (b)
    A("### (b) 각 결손과 (a)와의 상관")
    A("")
    A("| 패널 클래스 | 결손 합 (rad) | p50 | p90 \\|결손\\| | (a)와의 상관 |")
    A("|---|---|---|---|---|")
    corr = m.get("deficit_spread_corr", {}).get("per_class", {})
    for c, v in sorted(m["angle_deficit"].items(), key=lambda kv: -abs(kv[1]["sum"])):
        r = corr.get(c)
        A("| `%s` | %.3f | %+.4f | %.4f | %s |"
          % (c, v["sum"], v["p50"], v["p90_abs"], "%.3f" % r if r is not None else "-"))
    A("")
    A("전체 상관 **%.3f**." % m.get("deficit_spread_corr", {}).get("overall", float("nan")))
    A("")

    # (d)
    A("### (d) 배치로부터의 변위 Δxy")
    A("")
    A("정면 평면 `(x,y)`에서 잰 변위의 p50, cm. 드레이프 기준값은 300벌 p50이다.")
    A("")
    A("| 패널 클래스 | ARAP 조립 | 드레이프 | 비 |")
    A("|---|---|---|---|")
    for c, v in sorted(m["delta_xy_cm"].items()):
        dv = v.get("drape_p50")
        A("| `%s` | %.2f | %s | %s |"
          % (c, v["arap_p50"], "%.2f" % dv if dv else "-",
             "%.2f" % (v["arap_p50"] / dv) if dv else "-"))
    A("")

    # (e)
    A("### (e) 실제 드레이프와의 거리 (맞추려는 것이 아니다)")
    A("")
    A("| 패널 클래스 | p50 (cm) | p90 |")
    A("|---|---|---|")
    for c, v in sorted(m["drape_distance_cm"].items(), key=lambda kv: kv[1]["p50"]):
        A("| `%s` | %.2f | %.2f |" % (c, v["p50"], v["p90"]))
    A("")

    # (f)
    si = m.get("self_intersections", {})
    A("### (f) 자기 관통 (고치지 않고 세기만 함)")
    A("")
    A("교차하는 삼각형 쌍 **%d개**." % si.get("pairs", 0))
    if si.get("by_panel"):
        A("")
        A("| 패널 쌍 | 교차 쌍 수 |")
        A("|---|---|")
        for k, v in sorted(si["by_panel"].items(), key=lambda kv: -kv[1])[:12]:
            A("| %s | %d |" % (k, v))
    A("")

    txt = "\n".join(L)
    src = open(README, encoding="utf-8").read()
    a = src.index("## 4. 결과")
    b = src.index("## 5. 파일")
    open(README, "w", encoding="utf-8").write(src[:a] + txt + "\n---\n\n" + src[b:])
    print("README_garment.md section 4 written from measurements.json")


if __name__ == "__main__":
    main()
