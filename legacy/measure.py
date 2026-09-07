#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
measure.py — GarmentCodeData 스윕 측정기 (슬라이드 수치용)
==========================================================
논문(2607.21213)이 보고하지 않은, 도입 판단에 필요한 데이터 쪽 사실을 GT에서 잰다.

측정 3항목
  1) ΔL   GT 스티치 쌍의 호길이 차이(mm). τ_l = 10mm(=1cm, 논문 §4.1.1)이 타당한 허용치인지
          데이터로 검증한다. 개더/이즈는 길이가 같지 않다 — 꼬리(P>10mm, max)가 논점.
  2) px   Stage 2의 균일 스케일(가장 큰 패널의 최대 치수 → 0.85×256 ≈ 218px)에서
          한 벌 안 가장 작은 패널이 몇 px가 되는가. 칼라·커프의 유효 해상도.
  3) N    벌당 패널 수 / 스티치 수 (AutoSew 보고와 교차 대조용).

호길이는 근사하지 않는다:
  - 직선     : 유클리드
  - circle   : params = [R(절대 cm), large_arc, right] → 2R·asin(c/2R), large_arc면 2πR − 그것
               ★규약 확정 근거: params[0]/chord의 최솟값이 정확히 0.5000(반원 한계) → R은 절대 단위.
  - quadratic/cubic : 제어점은 상대 엣지 프레임(ctrl = p1 + cx·e + cy·perp(e)). 적응 세분 적분.
--check 는 규약·수치의 자기검증만 돌린다(데이터 접근 O, 출력 없음).

사용:
  python3 measure.py --check
  python3 measure.py --list specs.txt --json results/sweep.json
stdlib only. 읽기 전용.
"""
import json, math, argparse, sys, os


# ---------------------------------------------------------------- arc length
def _bez_pt(P, t):
    """de Casteljau on 2D control polygon P."""
    Q = list(P)
    while len(Q) > 1:
        Q = [((1-t)*a[0]+t*b[0], (1-t)*a[1]+t*b[1]) for a, b in zip(Q, Q[1:])]
    return Q[0]


def _gauss_legendre(n):
    """nodes/weights on [-1,1] by Newton iteration on P_n."""
    import math as _m
    xs, ws = [], []
    for i in range(1, n+1):
        x = _m.cos(_m.pi*(i-0.25)/(n+0.5))
        for _ in range(100):
            p0, p1 = 1.0, 0.0
            for j in range(1, n+1):
                p0, p1 = ((2*j-1)*x*p0 - (j-1)*p1)/j, p0
            dp = n*(x*p0 - p1)/(x*x - 1.0)
            dx = -p0/dp
            x += dx
            if abs(dx) < 1e-15:
                break
        xs.append(x); ws.append(2.0/((1-x*x)*dp*dp))
    return xs, ws


_GLX, _GLW = _gauss_legendre(24)


def _bez_len(P):
    """arc length of a 2D Bezier by 24-node Gauss-Legendre on |B'(t)|.
       Exact to ~1e-12 rel for quadratic/cubic (smooth, no cusps here)."""
    n = len(P) - 1
    D = [(n*(P[k+1][0]-P[k][0]), n*(P[k+1][1]-P[k][1])) for k in range(n)]  # derivative ctrl pts
    total = 0.0
    for x, w in zip(_GLX, _GLW):
        t = 0.5*(x + 1.0)
        Q = D
        while len(Q) > 1:
            Q = [((1-t)*a[0]+t*b[0], (1-t)*a[1]+t*b[1]) for a, b in zip(Q, Q[1:])]
        total += w * math.hypot(Q[0][0], Q[0][1])
    return 0.5 * total


def edge_length(v1, v2, curv):
    """exact-as-possible arc length in panel units (cm)."""
    p1 = (float(v1[0]), float(v1[1])); p2 = (float(v2[0]), float(v2[1]))
    chord = math.dist(p1, p2)
    if not curv:
        return chord, 'straight'
    typ = curv.get('type')
    ex, ey = p2[0]-p1[0], p2[1]-p1[1]
    nx, ny = -ey, ex                      # perp(e), same magnitude as e
    if typ == 'circle':
        R = float(curv['params'][0])
        large = bool(curv['params'][1])
        if chord < 1e-12 or R <= 0:
            return chord, 'circle'
        s = min(1.0, chord / (2.0*R))     # clamp float noise at the semicircle limit
        minor = 2.0 * R * math.asin(s)
        return (2.0*math.pi*R - minor if large else minor), 'circle'
    if typ == 'quadratic':
        (cx, cy), = curv['params']
        c = (p1[0]+cx*ex+cy*nx, p1[1]+cx*ey+cy*ny)
        return _bez_len([p1, c, p2]), 'quadratic'
    if typ == 'cubic':
        (ax, ay), (bx, by) = curv['params']
        c1 = (p1[0]+ax*ex+ay*nx, p1[1]+ax*ey+ay*ny)
        c2 = (p1[0]+bx*ex+by*nx, p1[1]+bx*ey+by*ny)
        return _bez_len([p1, c1, c2, p2]), 'cubic'
    return chord, 'unknown:' + str(typ)


def edge_points(v1, v2, curv, K=9):
    """sampled polyline, for panel extent only."""
    p1 = (float(v1[0]), float(v1[1])); p2 = (float(v2[0]), float(v2[1]))
    ex, ey = p2[0]-p1[0], p2[1]-p1[1]
    nx, ny = -ey, ex
    ts = [k/(K-1) for k in range(K)]
    if not curv:
        return [p1, p2]
    typ = curv.get('type')
    if typ == 'quadratic':
        (cx, cy), = curv['params']
        c = (p1[0]+cx*ex+cy*nx, p1[1]+cx*ey+cy*ny)
        return [_bez_pt([p1, c, p2], t) for t in ts]
    if typ == 'cubic':
        (ax, ay), (bx, by) = curv['params']
        c1 = (p1[0]+ax*ex+ay*nx, p1[1]+ax*ey+ay*ny)
        c2 = (p1[0]+bx*ex+by*nx, p1[1]+bx*ey+by*ny)
        return [_bez_pt([p1, c1, c2, p2], t) for t in ts]
    if typ == 'circle':
        R = float(curv['params'][0]); large = bool(curv['params'][1]); right = bool(curv['params'][2])
        chord = math.dist(p1, p2)
        if chord < 1e-12 or R < chord/2:
            return [p1, p2]
        mx, my = 0.5*(p1[0]+p2[0]), 0.5*(p1[1]+p2[1])
        h = math.sqrt(max(R*R - (chord/2)**2, 0.0))
        ux, uy = (p2[0]-p1[0])/chord, (p2[1]-p1[1])/chord
        px, py = -uy, ux
        sgn = 1.0 if right else -1.0
        if large:
            sgn = -sgn
        cxx, cyy = mx + sgn*h*px, my + sgn*h*py
        a1 = math.atan2(p1[1]-cyy, p1[0]-cxx)
        a2 = math.atan2(p2[1]-cyy, p2[0]-cxx)
        d = a2 - a1
        while d <= -math.pi: d += 2*math.pi
        while d > math.pi:   d -= 2*math.pi
        if large:
            d = d - 2*math.pi if d > 0 else d + 2*math.pi
        return [(cxx + R*math.cos(a1 + d*t), cyy + R*math.sin(a1 + d*t)) for t in ts]
    return [p1, p2]


# ---------------------------------------------------------------- per garment
IMG, FRAC = 256, 0.85          # Stage 2 캔버스 · 논문의 최대치수 스케일 계수


def measure_one(path):
    d = json.load(open(path))
    pat = d['pattern']
    units = d.get('properties', {}).get('units_in_meter', 100)
    cm2mm = 1000.0 / units      # units_in_meter=100 → 좌표 1 = 1cm = 10mm

    lens, kinds, extents = {}, {}, {}
    for name, p in pat['panels'].items():
        V = p['vertices']
        pts = []
        for k, e in enumerate(p['edges']):
            i, j = e['endpoints']
            L, kind = edge_length(V[i], V[j], e.get('curvature'))
            lens[(name, k)] = L * cm2mm
            kinds[(name, k)] = kind
            pts.extend(edge_points(V[i], V[j], e.get('curvature')))
        xs = [q[0] for q in pts]; ys = [q[1] for q in pts]
        extents[name] = max(max(xs)-min(xs), max(ys)-min(ys))

    dls = []
    for s in pat['stitches']:
        sides = [x for x in s if isinstance(x, dict)]
        if len(sides) != 2:
            continue
        a = (sides[0]['panel'], sides[0]['edge'])
        b = (sides[1]['panel'], sides[1]['edge'])
        if a not in lens or b not in lens:
            continue
        La, Lb = lens[a], lens[b]
        dls.append(dict(dl=abs(La-Lb), lo=min(La, Lb), hi=max(La, Lb),
                        ratio=(max(La, Lb)/min(La, Lb) if min(La, Lb) > 1e-9 else None),
                        ka=kinds[a], kb=kinds[b],
                        pa=a[0], pb=b[0], same_panel=(a[0] == b[0])))

    emax = max(extents.values()) if extents else 0.0
    emin = min(extents.values()) if extents else 0.0
    scale = (FRAC*IMG/emax) if emax > 1e-9 else 0.0     # Stage 2 균일 스케일
    return dict(id=os.path.basename(path).replace('.json', ''),
                n_panels=len(pat['panels']), n_stitches=len(pat['stitches']),
                n_pairs=len(dls),
                min_panel_px=emin*scale, max_panel_px=emax*scale,
                panel_px_ratio=(emax/emin if emin > 1e-9 else None),
                deltas=dls)


# ---------------------------------------------------------------- stats
def q(v, p):
    if not v: return None
    v = sorted(v); i = min(len(v)-1, max(0, int(round(p*(len(v)-1)))))
    return v[i]


def summarize(rows):
    all_dl = [x['dl'] for r in rows for x in r['deltas']]
    ratios = [x['ratio'] for r in rows for x in r['deltas'] if x['ratio']]
    minpx = [r['min_panel_px'] for r in rows]
    npan = [r['n_panels'] for r in rows]
    nst = [r['n_stitches'] for r in rows]
    n = len(all_dl)
    out = dict(
        n_garments=len(rows), n_pairs=n,
        dl_mm={k: (round(q(all_dl, p), 3) if all_dl else None) for k, p in
               (('p50', .5), ('p75', .75), ('p90', .9), ('p95', .95), ('p99', .99), ('max', 1.0))},
        dl_mean=round(sum(all_dl)/n, 3) if n else None,
        frac_gt_1mm=round(sum(1 for x in all_dl if x > 1)/n, 4) if n else None,
        frac_gt_10mm=round(sum(1 for x in all_dl if x > 10)/n, 4) if n else None,
        frac_gt_20mm=round(sum(1 for x in all_dl if x > 20)/n, 4) if n else None,
        count_gt_10mm=sum(1 for x in all_dl if x > 10),
        garments_with_gt_10mm=sum(1 for r in rows if any(x['dl'] > 10 for x in r['deltas'])),
        ratio={k: (round(q(ratios, p), 4) if ratios else None) for k, p in
               (('p50', .5), ('p95', .95), ('p99', .99), ('max', 1.0))},
        min_panel_px={k: (round(q(minpx, p), 2) if minpx else None) for k, p in
                      (('min', 0.0), ('p1', .01), ('p10', .10), ('p50', .5), ('p90', .9), ('max', 1.0))},
        frac_minpanel_lt_30px=round(sum(1 for x in minpx if x < 30)/len(minpx), 4) if minpx else None,
        frac_minpanel_lt_20px=round(sum(1 for x in minpx if x < 20)/len(minpx), 4) if minpx else None,
        panels={k: q(npan, p) for k, p in (('min', 0.0), ('p50', .5), ('p99', .99), ('max', 1.0))},
        panels_mean=round(sum(npan)/len(npan), 2) if npan else None,
        stitches={k: q(nst, p) for k, p in (('min', 0.0), ('p50', .5), ('max', 1.0))},
        stitches_mean=round(sum(nst)/len(nst), 2) if nst else None,
    )
    return out


# ---------------------------------------------------------------- self-check
def check():
    ok = True
    def t(name, got, want, tol=1e-7):
        nonlocal ok
        good = abs(got-want) <= tol*max(1.0, abs(want))
        print(('PASS ' if good else 'FAIL '), f'{name}: got {got!r} want {want!r}')
        ok &= good
    # straight
    L, _ = edge_length([0, 0], [3, 4], None); t('straight 3-4-5', L, 5.0)
    # semicircle: chord 10, R 5, minor arc = pi*R
    L, _ = edge_length([0, 0], [10, 0], dict(type='circle', params=[5, 0, 0]))
    t('semicircle minor', L, math.pi*5)
    # same, large_arc -> 2piR - piR = piR (semicircle is self-complementary)
    L, _ = edge_length([0, 0], [10, 0], dict(type='circle', params=[5, 1, 0]))
    t('semicircle large', L, math.pi*5)
    # 60deg arc: chord 10 -> R = 10/(2 sin30) = 10; arc = R*theta = 10*pi/3
    L, _ = edge_length([0, 0], [10, 0], dict(type='circle', params=[10, 0, 0]))
    t('60deg arc', L, 10*math.pi/3)
    # large arc complement: 2*pi*10 - 10*pi/3
    L, _ = edge_length([0, 0], [10, 0], dict(type='circle', params=[10, 1, 0]))
    t('300deg arc', L, 2*math.pi*10 - 10*math.pi/3)
    # degenerate bezier with collinear controls == straight
    L, _ = edge_length([0, 0], [10, 0], dict(type='cubic', params=[[0.3, 0.0], [0.7, 0.0]]))
    t('flat cubic', L, 10.0, 1e-9)
    L, _ = edge_length([0, 0], [10, 0], dict(type='quadratic', params=[[0.5, 0.0]]))
    t('flat quadratic', L, 10.0, 1e-9)
    # quadratic vs dense sampling
    cv = dict(type='quadratic', params=[[0.5, 0.4]])
    L, _ = edge_length([0, 0], [10, 0], cv)
    pts = edge_points([0, 0], [10, 0], cv, K=4001)
    t('quadratic vs 4001-sample polyline', L,
      sum(math.dist(a, b) for a, b in zip(pts, pts[1:])), 1e-6)
    cv = dict(type='cubic', params=[[0.25, 0.5], [0.75, -0.3]])
    L, _ = edge_length([0, 0], [10, 0], cv)
    pts = edge_points([0, 0], [10, 0], cv, K=4001)
    t('cubic vs 4001-sample polyline', L,
      sum(math.dist(a, b) for a, b in zip(pts, pts[1:])), 1e-6)
    # circle sampler consistency with analytic arc
    cv = dict(type='circle', params=[7.0, 0, 1])
    L, _ = edge_length([0, 0], [10, 0], cv)
    pts = edge_points([0, 0], [10, 0], cv, K=8001)
    t('circle sampler vs analytic', sum(math.dist(a, b) for a, b in zip(pts, pts[1:])), L, 1e-6)
    print('CHECK', 'OK' if ok else 'FAILED')
    return 0 if ok else 1


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--list')
    ap.add_argument('--spec', nargs='*', default=[])
    ap.add_argument('--json')
    ap.add_argument('--check', action='store_true')
    ap.add_argument('--progress', type=int, default=500)
    a = ap.parse_args()
    if a.check:
        sys.exit(check())
    paths = list(a.spec)
    if a.list:
        paths += [ln.strip() for ln in open(a.list) if ln.strip()]
    if not paths:
        sys.exit('no specs')
    rows, skipped = [], []
    for i, p in enumerate(paths):
        try:
            rows.append(measure_one(p))
        except Exception as ex:
            skipped.append((p, repr(ex)))
        if a.progress and (i+1) % a.progress == 0:
            print(f'  {i+1}/{len(paths)}', file=sys.stderr, flush=True)
    s = summarize(rows)
    s['skipped'] = len(skipped)
    print(json.dumps(s, indent=1, ensure_ascii=False))
    if skipped:
        print('first skips:', skipped[:3], file=sys.stderr)
    if a.json:
        os.makedirs(os.path.dirname(a.json) or '.', exist_ok=True)
        json.dump(dict(summary=s,
                       per_garment=[{k: v for k, v in r.items() if k != 'deltas'} for r in rows],
                       tail_pairs=sorted([dict(x, id=r['id']) for r in rows for x in r['deltas']
                                          if x['dl'] > 10], key=lambda z: -z['dl'])[:400]),
                  open(a.json, 'w'), indent=1, ensure_ascii=False)
        print('json ->', a.json)


if __name__ == '__main__':
    main()
