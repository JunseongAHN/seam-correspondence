#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
topo_check.py — 위상 게이트 검사기 + 오류 주입 catch-rate (제안 C 실증)
=====================================================================
GarmentCode(Data) specification JSON을 읽어, 예측/GT 스티치 집합이 만드는
접합 복합체의 위상 불변량을 검사한다. 순수 조합론 — 시뮬레이터·물성 불필요.
불변량 4종 + 다양체 검사:
  (1) 연결 성분 수          (panels via stitches)
  (2) 경계 루프 수 b        (자유 엣지 그래프의 사이클)
  (3) 방향성(orientability)  (parity union-find; 홀수 사이클 → 뫼비우스)
  (4) genus g               (χ = V - E + F,  g = (2 - χ - b)/2, 성분별)
  (+) 다양체: 엣지 재사용(한 엣지가 2개 스티치에), 경계 정점 차수 ≠ 2
한 방향으로만 성립: 불변량 위반 → 오답 증명. 통과 → 정답 보장 아님.
맹점(설계상): 대칭 스왑(좌우 소매 맞교환)은 위상 동형이라 원리적으로 못 잡는다.
오류 주입 (기준 불변량 = 주입 전 GT 자신 → 의미 분류·외부 사전지식 불필요):
  delete     스티치 1개 삭제 (전수)
  swap       스티치 2개의 파트너 교환 (전수 또는 --max-swaps 샘플)
  flip       스티치 1개의 방향(끝점 짝) 반전 (전수)
  add        자유 엣지 2개 사이 가짜 seam 추가 (전수 또는 샘플)
  reuse      이미 봉제된 엣지에 두 번째 스티치 (다양체 검사로 즉시 검출 — 전수)
끝점 짝 결정: 패널 translation/rotation(3D 배치)으로 리프트한 뒤 두 짝 중
거리합이 작은 쪽. Euler 규약은 intrinsic XYZ / extrinsic XYZ 둘 다 시도해
전 스티치 거리합이 작은 쪽을 자동 선택.
사용:
  python3 topo_check.py --selftest                       # 합성 위상 단위테스트
  python3 topo_check.py --spec path/to/specification.json
  python3 topo_check.py --spec ... --inject [--max-swaps 2000] [--json out.json]
stdlib only. 저장소·데이터 읽기 전용.
"""
import json, math, argparse, sys, itertools, random


# ---------------------------------------------------------------- union-find
class UF:
    def __init__(self):
        self.p = {}

    def find(self, x):
        p = self.p
        if x not in p:
            p[x] = x
            return x
        r = x
        while p[r] != r:
            r = p[r]
        while p[x] != r:
            p[x], x = r, p[x]
        return r

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb
        return rb


class ParityUF:
    """union-find with parity: rel=0 same orientation, rel=1 flipped.
       returns False on contradiction (odd cycle -> non-orientable)."""
    def __init__(self):
        self.p = {}
        self.r = {}   # parity to parent

    def find(self, x):
        if x not in self.p:
            self.p[x] = x; self.r[x] = 0
            return x, 0
        path = []
        while self.p[x] != x:
            path.append(x); x = self.p[x]
        root = x
        # recompute parity along path
        for n in reversed(path):
            self.r[n] ^= self.r[self.p[n]] if self.p[n] != root else 0
        # simpler: full recompute
        def par(n):
            q = 0
            while self.p[n] != n:
                q ^= self.r[n]; n = self.p[n]
            return q
        return root, None  # parity via par() below

    def parity(self, x):
        if x not in self.p:
            self.p[x] = x; self.r[x] = 0
        q = 0
        while self.p[x] != x:
            q ^= self.r[x]; x = self.p[x]
        return x, q

    def union(self, a, b, rel):
        ra, pa = self.parity(a)
        rb, pb = self.parity(b)
        if ra == rb:
            return (pa ^ pb) == rel      # consistent?
        self.p[ra] = rb
        self.r[ra] = pa ^ pb ^ rel
        return True


# ---------------------------------------------------------------- geometry
def euler_mat(rx, ry, rz, order):
    rx, ry, rz = (math.radians(v) for v in (rx, ry, rz))
    cx, sx, cy, sy, cz, sz = math.cos(rx), math.sin(rx), math.cos(ry), math.sin(ry), math.cos(rz), math.sin(rz)
    Rx = [[1,0,0],[0,cx,-sx],[0,sx,cx]]
    Ry = [[cy,0,sy],[0,1,0],[-sy,0,cy]]
    Rz = [[cz,-sz,0],[sz,cz,0],[0,0,1]]
    def mm(A,B):
        return [[sum(A[i][k]*B[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
    if order == 'intrinsic_xyz':   # R = Rx @ Ry @ Rz  (applied Rz first in body frame)
        return mm(mm(Rx,Ry),Rz)
    else:                          # extrinsic xyz: R = Rz @ Ry @ Rx
        return mm(mm(Rz,Ry),Rx)


def lift(v2, T, R):
    x, y = float(v2[0]), float(v2[1])
    p = [x, y, 0.0]
    q = [sum(R[i][k]*p[k] for k in range(3)) + T[i] for i in range(3)]
    return q


def curve2d(p1, p2, curv, K):
    """sample K points of the edge curve in 2D panel coords.
       GarmentCode curvature control points are in relative edge frame:
       ctrl = p1 + cx*(p2-p1) + cy*perp(p2-p1)."""
    x1, y1 = float(p1[0]), float(p1[1]); x2, y2 = float(p2[0]), float(p2[1])
    ex, ey = x2 - x1, y2 - y1
    nx, ny = -ey, ex
    ts = [k / (K - 1) for k in range(K)]
    if not curv:
        return [(x1 + t*ex, y1 + t*ey) for t in ts]
    typ = curv.get('type')
    if typ == 'quadratic':
        (cx, cy), = curv['params']
        qx, qy = x1 + cx*ex + cy*nx, y1 + cx*ey + cy*ny
        return [((1-t)**2*x1 + 2*(1-t)*t*qx + t*t*x2,
                 (1-t)**2*y1 + 2*(1-t)*t*qy + t*t*y2) for t in ts]
    if typ == 'cubic':
        (ax, ay), (bx, by) = curv['params']
        q1x, q1y = x1 + ax*ex + ay*nx, y1 + ax*ey + ay*ny
        q2x, q2y = x1 + bx*ex + by*nx, y1 + bx*ey + by*ny
        return [((1-t)**3*x1 + 3*(1-t)**2*t*q1x + 3*(1-t)*t*t*q2x + t**3*x2,
                 (1-t)**3*y1 + 3*(1-t)**2*t*q1y + 3*(1-t)*t*t*q2y + t**3*y2) for t in ts]
    if typ == 'circle':
        # ★params[0] 은 절대 반지름(cm). 근거: 데이터 전수에서 params[0]/chord 의 최솟값이
        # 정확히 0.5000(반원 한계) — 상대 해석이면 params[0] 자체가 0.5에서 하한을 쳐야 한다.
        R = float(curv['params'][0]); large = bool(curv['params'][1]); right = bool(curv['params'][2])
        chord = math.hypot(ex, ey)
        if chord < 1e-12 or R < chord/2 - 1e-9:
            return [(x1 + t*ex, y1 + t*ey) for t in ts]
        mx, my = 0.5*(x1+x2), 0.5*(y1+y2)
        h = math.sqrt(max(R*R - (chord/2)**2, 0.0))
        ux, uy = ex/chord, ey/chord
        px, py = -uy, ux
        sgn = 1.0 if right else -1.0
        if large:
            sgn = -sgn
        cxx, cyy = mx + sgn*h*px, my + sgn*h*py
        a1 = math.atan2(y1-cyy, x1-cxx); a2 = math.atan2(y2-cyy, x2-cxx)
        dth = a2 - a1
        while dth <= -math.pi: dth += 2*math.pi
        while dth > math.pi:   dth -= 2*math.pi
        if large:
            dth = dth - 2*math.pi if dth > 0 else dth + 2*math.pi
        return [(cxx + R*math.cos(a1 + dth*t), cyy + R*math.sin(a1 + dth*t)) for t in ts]
    # unknown type: chord fallback
    return [(x1 + t*ex, y1 + t*ey) for t in ts]


def d3(a, b):
    return math.dist(a, b)


# ---------------------------------------------------------------- pattern
class Pattern:
    def __init__(self, spec_path=None, data=None):
        if data is None:
            data = json.load(open(spec_path))
        pat = data['pattern'] if 'pattern' in data else data
        self.panels = {}          # name -> dict(edges=[(i,j),...], verts=[...], T, R)
        for name, p in pat['panels'].items():
            edges = [tuple(e['endpoints']) for e in p['edges']]
            curvs = [e.get('curvature') for e in p['edges']]
            self.panels[name] = dict(
                edges=edges, curvs=curvs, verts=p['vertices'],
                T=p.get('translation', [0,0,0]), R=p.get('rotation', [0,0,0]))
            # verify boundary is one closed ordered loop
            for k in range(len(edges)):
                if edges[k][1] != edges[(k+1) % len(edges)][0]:
                    self.loop_ok = False
                    break
        self.stitches = [ (s[0]['panel'], s[0]['edge'], s[1]['panel'], s[1]['edge'])
                          for s in pat['stitches'] ]
        self._choose_euler()

    def _choose_euler(self):
        best = None
        for order in ('intrinsic_xyz', 'extrinsic_xyz'):
            pos = {}
            for name, p in self.panels.items():
                R = euler_mat(*p['R'], order=order)
                pos[name] = [lift(v, p['T'], R) for v in p['verts']]
            tot = 0.0
            for (pa, ea, pb, eb) in self.stitches:
                a1, a2 = self.panels[pa]['edges'][ea]
                b1, b2 = self.panels[pb]['edges'][eb]
                A1, A2 = pos[pa][a1], pos[pa][a2]
                B1, B2 = pos[pb][b1], pos[pb][b2]
                tot += min(d3(A1,B1)+d3(A2,B2), d3(A1,B2)+d3(A2,B1))
            if best is None or tot < best[0]:
                best = (tot, order, pos)
        self.euler_order = best[1]
        self.pos3 = best[2]

    def edge_curve3(self, panel, eidx, K=9):
        key = (panel, eidx, K)
        if not hasattr(self, '_ccache'):
            self._ccache = {}
        if key in self._ccache:
            return self._ccache[key]
        p = self.panels[panel]
        i, j = p['edges'][eidx]
        pts2 = curve2d(p['verts'][i], p['verts'][j], p['curvs'][eidx], K)
        R = euler_mat(*p['R'], order=self.euler_order)
        out = [lift(q, p['T'], R) for q in pts2]
        self._ccache[key] = out
        return out

    def pairing(self, stitch, with_margin=False):
        """'anti' (a1<->b2, a2<->b1) or 'para' (a1<->b1, a2<->b2), by whole-curve
           3D distance (curvature-aware, K samples)."""
        pa, ea, pb, eb = stitch
        A = self.edge_curve3(pa, ea)
        B = self.edge_curve3(pb, eb)
        K = len(A)
        dp = sum(d3(A[k], B[k]) for k in range(K))
        da = sum(d3(A[k], B[K-1-k]) for k in range(K))
        res = 'para' if dp <= da else 'anti'
        if with_margin:
            return res, abs(dp - da) / K
        return res


# ---------------------------------------------------------------- invariants
def invariants(pat, stitches, pairings=None, flip_set=frozenset()):
    """stitches: list of (pa,ea,pb,eb). pairings: parallel/anti per stitch (default: geometric).
       flip_set: indices whose pairing is inverted (for flip injection).
       Returns dict of invariants, or manifold violation report."""
    report = dict(manifold=True, violations=[])
    # --- edge reuse
    use = {}
    for (pa, ea, pb, eb) in stitches:
        for key in ((pa, ea), (pb, eb)):
            use[key] = use.get(key, 0) + 1
    reused = [k for k, c in use.items() if c > 1]
    if reused:
        report['manifold'] = False
        report['violations'].append(('edge_reuse', len(reused)))
    # --- vertex classes & panel components & parity
    vuf, cuf, ouf = UF(), UF(), ParityUF()
    for name in pat.panels:
        cuf.find(name); ouf.parity(name)
        for i in range(len(pat.panels[name]['verts'])):
            vuf.find((name, i))
    orientable = True
    for idx, st in enumerate(stitches):
        pa, ea, pb, eb = st
        a1, a2 = pat.panels[pa]['edges'][ea]
        b1, b2 = pat.panels[pb]['edges'][eb]
        pr = pairings[idx] if pairings else pat.pairing(st)
        if idx in flip_set:
            pr = 'anti' if pr == 'para' else 'para'
        if pr == 'anti':
            vuf.union((pa, a1), (pb, b2)); vuf.union((pa, a2), (pb, b1))
            ok = ouf.union(pa, pb, 0)      # anti-parallel gluing keeps orientation
        else:
            vuf.union((pa, a1), (pb, b1)); vuf.union((pa, a2), (pb, b2))
            ok = ouf.union(pa, pb, 1)      # parallel gluing needs a flip
        if not ok:
            orientable = False
        cuf.union(pa, pb)
    # --- counts (global and per component)
    comp_of = {name: cuf.find(name) for name in pat.panels}
    comps = sorted(set(comp_of.values()))
    F = {c: 0 for c in comps}; E = {c: 0 for c in comps}; Vset = {c: set() for c in comps}
    for name, p in pat.panels.items():
        c = comp_of[name]
        F[c] += 1
        E[c] += len(p['edges'])
        for i in range(len(p['verts'])):
            Vset[c].add(vuf.find((name, i)))
    for (pa, ea, pb, eb) in stitches:
        E[comp_of[pa]] -= 1                       # two edges merged into one
    # --- boundary loops: free-edge graph on vertex classes
    freedeg = {}; fedges = {c: [] for c in comps}
    for name, p in pat.panels.items():
        for k, (i, j) in enumerate(p['edges']):
            if use.get((name, k), 0) == 0:
                ci, cj = vuf.find((name, i)), vuf.find((name, j))
                fedges[comp_of[name]].append((ci, cj))
                freedeg[ci] = freedeg.get(ci, 0) + 1
                freedeg[cj] = freedeg.get(cj, 0) + 1
    bad_deg = [v for v, dgr in freedeg.items() if dgr != 2]
    if bad_deg:
        report['manifold'] = False
        report['violations'].append(('boundary_degree', len(bad_deg)))
    b = {}
    for c in comps:
        buf = UF()
        nodes = set()
        for (ci, cj) in fedges[c]:
            buf.union(ci, cj); nodes.add(ci); nodes.add(cj)
        b[c] = len({buf.find(n) for n in nodes})
    # --- Euler characteristic & genus per component
    genus, chi = {}, {}
    for c in comps:
        chi[c] = len(Vset[c]) - E[c] + F[c]
        g2 = 2 - chi[c] - b[c]
        genus[c] = g2 / 2 if orientable else None   # crosscaps g2 if non-orientable
        if orientable and g2 % 2 != 0:
            report['violations'].append(('odd_chi', c))
            report['manifold'] = False
    report.update(dict(
        components=len(comps), orientable=orientable,
        boundary_loops=sorted(b.values(), reverse=True),
        chi=sorted(chi.values()), genus=sorted([genus[c] for c in comps],
                                               key=lambda x: (x is None, x)),
        total_loops=sum(b.values())))
    return report


def signature(rep):
    return (rep['manifold'], rep['components'], tuple(rep['boundary_loops']),
            rep['orientable'], tuple(rep['chi']))


# ---------------------------------------------------------------- calibration
def calibrate_pairings(pat):
    """GT 전용: spec에 per-stitch 방향이 없으므로, 'GT는 orientable'이라는 물리 제약으로
       끝점 짝을 보정한다. 패널 2-채색(성분별 전수 탐색)으로 기하 증거(margin 가중)와의
       불일치 비용을 최소화. 예측 검사에는 쓰지 말 것 — 예측은 자기 방향을 갖고 온다.
       returns (pairings, repaired:[(stitch, margin)])."""
    geo = [pat.pairing(s, with_margin=True) for s in pat.stitches]
    names = list(pat.panels)
    cuf = UF()
    for n in names:
        cuf.find(n)
    for (pa, _, pb, _) in pat.stitches:
        cuf.union(pa, pb)
    comps = {}
    for n in names:
        comps.setdefault(cuf.find(n), []).append(n)
    # 2-coloring. 기하 증거를 margin 큰 순으로 parity union-find 에 넣고, 모순되는 제약만
    # 버린다(=repaired). 전 제약이 무모순이면 비용 0 = 전역 최적이므로 전수 탐색과 동일한 답.
    # (원래 코드는 성분당 2^(패널수-1) 전수 탐색이라 34패널 벌에서 정지하지 않는다.)
    order = sorted(range(len(pat.stitches)), key=lambda i: -geo[i][1])
    puf = ParityUF()
    for n in names:
        puf.parity(n)
    dropped = set()
    for i in order:
        pa, _, pb, _ = pat.stitches[i]
        rel = 1 if geo[i][0] == 'para' else 0
        if not puf.union(pa, pb, rel):
            dropped.add(i)
    color = {}
    for n in names:
        root, par = puf.parity(n)
        color[n] = par
    pairings, repaired = [], []
    for i, s in enumerate(pat.stitches):
        pa, _, pb, _ = s
        rel = color[pa] ^ color[pb]
        forced = 'para' if rel else 'anti'
        pairings.append(forced)
        if forced != geo[i][0]:
            repaired.append((s, round(geo[i][1], 4)))
    return pairings, repaired


# ---------------------------------------------------------------- injections
LR = (('left', 'right'), ('right', 'left'), ('_l', '_r'), ('_r', '_l'),
      ('lf', 'rf'), ('rf', 'lf'), ('lb', 'rb'), ('rb', 'lb'))


def mirror_name(n):
    for a, bb in LR:
        if a in n:
            return n.replace(a, bb)
    return None


def is_symmetric_swap(s1, s2):
    """heuristic: the two stitches connect mirror-named panel sets."""
    n1 = {s1[0], s1[2]}; n2 = {s2[0], s2[2]}
    m1 = {mirror_name(x) or x for x in n1}
    return m1 == n2


def run_injections(pat, max_swaps=3000, max_adds=1500, seed=0):
    """catch = 오류 조합의 '어떤' 끝점 짝 선택으로도 GT 불변량을 재현할 수 없을 때만.
       (best-case 판정 → catch rate는 하한. 과대평가 방지.)"""
    rng = random.Random(seed)
    base_pair, repaired = calibrate_pairings(pat)
    base = invariants(pat, pat.stitches, base_pair)
    base_sig = signature(base)
    out = dict(baseline=base, calibration_repaired=len(repaired))
    S = pat.stitches
    # delete — exhaustive
    caught = 0
    for i in range(len(S)):
        rep = invariants(pat, S[:i] + S[i+1:], base_pair[:i] + base_pair[i+1:])
        caught += signature(rep) != base_sig
    out['delete'] = dict(n=len(S), caught=caught)
    # swap — partners exchanged; try all 4 pairing combos for the two new stitches
    pairs = list(itertools.combinations(range(len(S)), 2))
    if len(pairs) > max_swaps:
        pairs = rng.sample(pairs, max_swaps)
    caught = 0; sym_n = sym_caught = 0
    for (i, j) in pairs:
        pa, ea, pb, eb = S[i]; pc, ec, pd, ed = S[j]
        ns = list(S)
        ns[i] = (pa, ea, pd, ed); ns[j] = (pc, ec, pb, eb)
        hit = True
        for pi in ('anti', 'para'):
            for pj in ('anti', 'para'):
                np_ = list(base_pair); np_[i] = pi; np_[j] = pj
                if signature(invariants(pat, ns, np_)) == base_sig:
                    hit = False; break
            if not hit:
                break
        caught += hit
        if is_symmetric_swap(S[i], S[j]):
            sym_n += 1; sym_caught += hit
    out['swap'] = dict(n=len(pairs), caught=caught, sym_n=sym_n, sym_caught=sym_caught)
    # flip — invert the calibrated pairing of one stitch (exhaustive)
    caught = 0; absorbed = []
    for i in range(len(S)):
        rep = invariants(pat, S, base_pair, flip_set=frozenset([i]))
        if signature(rep) != base_sig:
            caught += 1
        else:
            absorbed.append(S[i][0] + '#' + str(S[i][1]) + '<->' + S[i][2] + '#' + str(S[i][3]))
    out['flip'] = dict(n=len(S), caught=caught, absorbed=absorbed[:12])
    # add — spurious seam between two free edges; try both pairings
    free = []
    used = {(pa, ea) for (pa, ea, _, _) in S} | {(pb, eb) for (_, _, pb, eb) in S}
    for name, p in pat.panels.items():
        for k in range(len(p['edges'])):
            if (name, k) not in used:
                free.append((name, k))
    pairs = list(itertools.combinations(range(len(free)), 2))
    if len(pairs) > max_adds:
        pairs = rng.sample(pairs, max_adds)
    caught = 0
    for (i, j) in pairs:
        ns = S + [(free[i][0], free[i][1], free[j][0], free[j][1])]
        hit = True
        for pn in ('anti', 'para'):
            if signature(invariants(pat, ns, base_pair + [pn])) == base_sig:
                hit = False; break
        caught += hit
    out['add'] = dict(n=len(pairs), caught=caught, free_edges=len(free))
    # reuse — second stitch on an already-stitched edge (manifold check catches by construction)
    caught = n = 0
    for (pa, ea) in sorted(used)[:20]:
        if not free:
            break
        f = free[n % len(free)]
        ns = S + [(pa, ea, f[0], f[1])]
        rep = invariants(pat, ns, base_pair + ['anti'])
        caught += not rep['manifold']
        n += 1
    out['reuse'] = dict(n=n, caught=caught)
    return out


# ---------------------------------------------------------------- self-tests
def _mk(panels, stitches):
    """panels: dict name->list of 2D verts (closed loop, edges consecutive).
       stitches: list of ((p,e),(p,e)) + optional forced pairing via 5th elem."""
    pat = Pattern(data=dict(pattern=dict(
        panels={n: dict(vertices=v, edges=[dict(endpoints=[i, (i+1) % len(v)])
                                           for i in range(len(v))],
                        translation=t, rotation=[0, 0, 0])
                for n, (v, t) in panels.items()},
        stitches=[[dict(panel=a[0], edge=a[1]), dict(panel=b[0], edge=b[1])]
                  for a, b in stitches])))
    return pat


def selftest():
    sq = [[0,0],[10,0],[10,10],[0,10]]
    ok = True

    def chk(name, rep, comp, loops, orient, genus_list):
        nonlocal ok
        good = (rep['components'] == comp and rep['total_loops'] == loops and
                rep['orientable'] == orient and
                (genus_list is None or rep['genus'] == genus_list))
        print(('PASS ' if good else 'FAIL '), name, '->', {k: rep[k] for k in
              ('components', 'total_loops', 'orientable', 'genus', 'manifold')})
        ok &= good

    # 1) two squares side by side glued along facing edges -> disk
    pat = _mk({'A': (sq, [0,0,0]), 'B': (sq, [10.5,0,0])}, [(('A',1),('B',3))])
    chk('disk (2 panels, 1 seam)', invariants(pat, pat.stitches), 1, 1, True, [0.0])
    # 2) cylinder: front/back glued along both side seams (place B behind A)
    pat = _mk({'A': (sq, [0,0,0]), 'B': (sq, [0,0,5])},
              [(('A',1),('B',3)), (('A',3),('B',1))])
    chk('cylinder', invariants(pat, pat.stitches), 1, 2, True, [0.0])
    # 3) torus: cylinder + glue top/bottom loops (A-top<->B-top? no: A-top with A's own?…)
    #    simplest torus: one square, glue e0<->e2 and e1<->e3, both anti (forced)
    pat = _mk({'A': (sq, [0,0,0])}, [(('A',0),('A',2)), (('A',1),('A',3))])
    rep = invariants(pat, pat.stitches, ['anti', 'anti'])
    chk('torus (forced anti,anti)', rep, 1, 0, True, [1.0])
    # 4) Möbius: one square, glue e0<->e2 parallel
    pat = _mk({'A': (sq, [0,0,0])}, [(('A',0),('A',2))])
    rep = invariants(pat, pat.stitches, ['para'])
    chk('Möbius (forced para)', rep, 1, None if False else 1, False, None)
    # 5) dart: pentagon with a notch, glue the two notch edges -> disk
    dartp = [[0,0],[10,0],[10,10],[5,4],[0,10]]   # edges: ...(2,3),(3,4) are dart sides
    pat = _mk({'A': (dartp, [0,0,0])}, [(('A',2),('A',3))])
    chk('dart self-stitch -> disk', invariants(pat, pat.stitches), 1, 1, True, [0.0])
    # 6a) pillow: two squares glued along ALL four edge pairs -> sphere (g0, b0)
    pat = _mk({'A': (sq, [0,0,0]), 'B': (sq, [0,0,5])},
              [(('A',1),('B',3)), (('A',3),('B',1)),
               (('A',0),('B',0)), (('A',2),('B',2))])
    rep = invariants(pat, pat.stitches, ['anti']*4)
    chk('pillow -> sphere', rep, 1, 0, True, [0.0])
    # 6b) cuff-to-hem: cylinder(A,B), then glue TOP loop to BOTTOM loop -> torus (g1)
    pat = _mk({'A': (sq, [0,0,0]), 'B': (sq, [0,0,5])},
              [(('A',1),('B',3)), (('A',3),('B',1)),
               (('A',0),('A',2)), (('B',0),('B',2))])
    rep = invariants(pat, pat.stitches, ['anti', 'anti', 'anti', 'anti'])
    chk('cuff-to-hem (top loop glued to bottom loop) -> genus 1', rep, 1, 0, True, [1.0])
    print('SELFTEST', 'OK' if ok else 'FAILED')
    return 0 if ok else 1


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--spec')
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--inject', action='store_true')
    ap.add_argument('--max-swaps', type=int, default=3000)
    ap.add_argument('--json')
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())
    pat = Pattern(a.spec)
    base_pair, repaired = calibrate_pairings(pat)
    base = invariants(pat, pat.stitches, base_pair)
    print(f"{a.spec}")
    print(f"  euler order  : {pat.euler_order}   calibration repaired {len(repaired)} pairings"
          + (f" (max margin {max(m for _, m in repaired):.3f})" if repaired else ""))
    print(f"  panels {len(pat.panels)}  stitches {len(pat.stitches)}")
    print(f"  GT invariants: comp {base['components']}  loops {base['boundary_loops']}"
          f"  orientable {base['orientable']}  genus {base['genus']}  manifold {base['manifold']}")
    out = dict(spec=a.spec, baseline=base)
    if a.inject:
        res = run_injections(pat, max_swaps=a.max_swaps)
        out.update(res)
        for k in ('delete', 'swap', 'flip', 'add', 'reuse'):
            r = res[k]
            pct = 100.0 * r['caught'] / max(r['n'], 1)
            extra = ''
            if k == 'swap' and r['sym_n']:
                extra = (f"   (거울이름 스티치쌍 스왑 {r['sym_caught']}/{r['sym_n']}"
                         f" = {100.0*r['sym_caught']/r['sym_n']:.0f}% — 단일 스왑은 배선이 꼬여 잡힘."
                         f" 진짜 맹점은 좌우 서브어셈블리 전체의 일관 교환 = 동형)")
            if k == 'flip' and r['caught'] < r['n']:
                extra = f"   흡수됨(bridge): {r['n']-r['caught']}건"
            print(f"  {k:<7} caught {r['caught']:>5}/{r['n']:<5} = {pct:5.1f}%{extra}")
    if a.json:
        json.dump(out, open(a.json, 'w'), indent=1, ensure_ascii=False, default=str)
        print('  json ->', a.json)


if __name__ == '__main__':
    main()
