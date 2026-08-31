#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
baseline_match.py — 휴리스틱 스티치 매칭 베이스라인 (실험 A1)
==============================================================
"복잡한 걸 도입하기 전에 단순한 게 우리 데이터에서 어디까지 되는지 잰다."
논문(2607.21213)이 하지 않은 유일한 비교 — 휴리스틱 대비 — 를 GT 위에서 측정한다.
두 휴리스틱 (사다리 이원 구성):
  H-weak    레이아웃 근접성: 3D 배치 좌표에서 엣지 곡선 간 평균 거리(방향 min).
            ⚠ GarmentCodeData의 배치는 시뮬 배치라 informative — 실무 CAD 레이아웃(패킹)에서는
            이 신호가 사라진다. 따라서 H-weak 성능은 실무 상한으로 읽을 것.
  H-strong  위치 무시, 형상만: (호길이, 총 회전각, 현/호 비율) 유사도.
            대칭 패널·동일 길이 엣지에서 원리적으로 모호 → 실패가 어디 몰리는지가 측정 목적.
매칭: 전 후보쌍(같은 패널 포함 → 다트 커버) 점수 오름차순 greedy, 1:1 제약, 임계값 τ.
τ는 전역 그리드에서 best-F1로 선택(튜닝된 휴리스틱 = 휴리스틱의 상한임을 보고에 명시).
지표: 쌍 P/R/F1 + ★벌 단위 전량 정답률(예측 집합 == GT 집합; AutoSew GSP 대응 지표)
     + 실패 분포(다트 / 길이 불일치>10% / 거울 이름 관여).
사용:
  python3 baseline_match.py --spec a.json [b.json ...]        # 파일들 통합 평가
  python3 baseline_match.py --list specs.txt --json out.json  # 스윕 집합(줄당 경로)
stdlib only. 읽기 전용. Euler 규약은 intrinsic XYZ 고정(데이터셋 상수, topo_check로 검증됨).
"""
import json, math, argparse, sys, itertools

K = 9  # curve samples per edge


def euler_mat(rx, ry, rz):
    rx, ry, rz = (math.radians(v) for v in (rx, ry, rz))
    cx, sx, cy, sy, cz, sz = math.cos(rx), math.sin(rx), math.cos(ry), math.sin(ry), math.cos(rz), math.sin(rz)
    Rx = [[1,0,0],[0,cx,-sx],[0,sx,cx]]
    Ry = [[cy,0,sy],[0,1,0],[-sy,0,cy]]
    Rz = [[cz,-sz,0],[sz,cz,0],[0,0,1]]
    def mm(A,B):
        return [[sum(A[i][k]*B[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
    return mm(mm(Rx,Ry),Rz)  # intrinsic XYZ


def curve2d(p1, p2, curv):
    x1, y1 = float(p1[0]), float(p1[1]); x2, y2 = float(p2[0]), float(p2[1])
    ex, ey = x2-x1, y2-y1; nx, ny = -ey, ex
    ts = [k/(K-1) for k in range(K)]
    if not curv:
        return [(x1+t*ex, y1+t*ey) for t in ts]
    typ = curv.get('type')
    if typ == 'quadratic':
        (cx, cy), = curv['params']
        qx, qy = x1+cx*ex+cy*nx, y1+cx*ey+cy*ny
        return [((1-t)**2*x1+2*(1-t)*t*qx+t*t*x2, (1-t)**2*y1+2*(1-t)*t*qy+t*t*y2) for t in ts]
    if typ == 'cubic':
        (ax, ay), (bx, by) = curv['params']
        q1x, q1y = x1+ax*ex+ay*nx, y1+ax*ey+ay*ny
        q2x, q2y = x1+bx*ex+by*nx, y1+bx*ey+by*ny
        return [((1-t)**3*x1+3*(1-t)**2*t*q1x+3*(1-t)*t*t*q2x+t**3*x2,
                 (1-t)**3*y1+3*(1-t)**2*t*q1y+3*(1-t)*t*t*q2y+t**3*y2) for t in ts]
    return [(x1+t*ex, y1+t*ey) for t in ts]  # circle arc 등: chord 근사


def d3(a, b):
    return math.dist(a, b)


LRS = (('left','right'),('right','left'),('_l','_r'),('_r','_l'),('lf','rf'),('rf','lf'))


def mirror_name(n):
    for a,b in LRS:
        if a in n:
            return n.replace(a,b)
    return None


class Garment:
    def __init__(self, path):
        data = json.load(open(path))
        pat = data['pattern'] if 'pattern' in data else data
        self.path = path
        self.edges = []            # list of dict(panel, idx, pts3, len, turn, chordarc)
        self.eid = {}              # (panel, idx) -> index into self.edges
        for name, p in pat['panels'].items():
            R = euler_mat(*p.get('rotation',[0,0,0]))
            T = p.get('translation',[0,0,0])
            for k, e in enumerate(p['edges']):
                i, j = e['endpoints']
                pts2 = curve2d(p['vertices'][i], p['vertices'][j], e.get('curvature'))
                pts3 = [[sum(R[r][c]*q[c] for c in range(2)) + T[r] for r in range(3)] for q in
                        [(x, y, 0) for (x, y) in pts2]]
                # arc length & turning (2D, panel-local)
                L = sum(math.dist(pts2[t], pts2[t+1]) for t in range(K-1))
                turn = 0.0
                for t in range(1, K-1):
                    a = (pts2[t][0]-pts2[t-1][0], pts2[t][1]-pts2[t-1][1])
                    b = (pts2[t+1][0]-pts2[t][0], pts2[t+1][1]-pts2[t][1])
                    na, nb = math.hypot(*a), math.hypot(*b)
                    if na > 1e-12 and nb > 1e-12:
                        cross = a[0]*b[1]-a[1]*b[0]; dot = a[0]*b[0]+a[1]*b[1]
                        turn += abs(math.atan2(cross, dot))
                chord = math.dist(pts2[0], pts2[-1])
                self.eid[(name, k)] = len(self.edges)
                self.edges.append(dict(panel=name, idx=k, pts3=pts3, len=L, turn=turn,
                                       chordarc=(chord/L if L > 1e-9 else 1.0)))
        self.gt = set()
        for s in pat['stitches']:
            a = self.eid[(s[0]['panel'], s[0]['edge'])]
            b = self.eid[(s[1]['panel'], s[1]['edge'])]
            self.gt.add(frozenset((a, b)))

    # ---- scores ----
    def score_weak(self, a, b):
        A, B = self.edges[a]['pts3'], self.edges[b]['pts3']
        d_para = sum(d3(A[k], B[k]) for k in range(K)) / K
        d_anti = sum(d3(A[k], B[K-1-k]) for k in range(K)) / K
        return min(d_para, d_anti)

    def score_strong(self, a, b):
        ea, eb = self.edges[a], self.edges[b]
        ml = max(ea['len'], eb['len'], 1e-9)
        return (0.6*abs(ea['len']-eb['len'])/ml
                + 0.25*abs(ea['turn']-eb['turn'])/math.pi
                + 0.15*abs(ea['chordarc']-eb['chordarc']))

    def _cands(self, kind):
        """쌍 점수는 벌당·종류당 1회만 계산해 캐시 (τ 스윕이 공짜가 되도록)."""
        if not hasattr(self, '_cc'):
            self._cc = {}
        if kind in self._cc:
            return self._cc[kind]
        score = self.score_weak if kind == 'weak' else self.score_strong
        n = len(self.edges)
        cands = sorted((score(a, b), a, b) for a, b in itertools.combinations(range(n), 2))
        self._cc[kind] = cands
        return cands

    def predict(self, kind, tau):
        used, pred = set(), set()
        for s, a, b in self._cands(kind):
            if s > tau:
                break
            if a in used or b in used:
                continue
            used.add(a); used.add(b)
            pred.add(frozenset((a, b)))
        return pred

    def pair_class(self, pair):
        """GT 쌍의 접합 유형: dart/self · near(<20cm 배치) · far(>=20cm 배치 — 옆선·어깨선류)"""
        a, b = tuple(pair)
        if self.edges[a]['panel'] == self.edges[b]['panel']:
            return 'dart/self'
        return 'near_placed' if self.score_weak(a, b) < 20.0 else 'far_placed'

    def miss_detail(self, pair, pred):
        """놓친 GT 쌍에 대해: 예측이 거울 상대와 바꿔치기했는지"""
        a, b = tuple(pair)
        partner = {}
        for pp in pred:
            x, y = tuple(pp)
            partner[x] = y; partner[y] = x
        for (u, v) in ((a, b), (b, a)):
            got = partner.get(u)
            if got is not None:
                mn = mirror_name(self.edges[v]['panel'])
                if mn and self.edges[got]['panel'] == mn:
                    return 'mirror_swap'
        return None


def evaluate(garments, kind, tau):
    TP = FP = FN = 0; perfect = 0
    cls_tot, cls_hit = {}, {}
    mirror_swaps = 0
    per_g = []
    for g in garments:
        pred = g.predict(kind, tau)
        tp = pred & g.gt
        TP += len(tp); FP += len(pred - g.gt); FN += len(g.gt - pred)
        ok = pred == g.gt
        perfect += ok
        per_g.append(dict(path=g.path, gt=len(g.gt), tp=len(tp),
                          fp=len(pred-g.gt), fn=len(g.gt-pred), perfect=bool(ok)))
        for pair in g.gt:
            c = g.pair_class(pair)
            cls_tot[c] = cls_tot.get(c, 0) + 1
            if pair in tp:
                cls_hit[c] = cls_hit.get(c, 0) + 1
        for pair in (g.gt - pred):
            if g.miss_detail(pair, pred) == 'mirror_swap':
                mirror_swaps += 1
    P = TP/max(TP+FP, 1); R = TP/max(TP+FN, 1)
    F1 = 2*P*R/max(P+R, 1e-12)
    cls_recall = {c: round(cls_hit.get(c, 0)/cls_tot[c], 3) for c in sorted(cls_tot)}
    return dict(tau=tau, precision=round(P, 4), recall=round(R, 4), f1=round(F1, 4),
                perfect=perfect, n_garments=len(garments),
                class_recall=cls_recall, class_total=cls_tot,
                mirror_swaps=mirror_swaps, per_garment=per_g)


GRIDS = dict(weak=[0.5, 1, 2, 3, 5, 8, 12, 20, 30, 45, 60],
             strong=[0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.2, 0.3])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--spec', nargs='*', default=[])
    ap.add_argument('--list', help='text file, one spec path per line')
    ap.add_argument('--json')
    a = ap.parse_args()
    paths = list(a.spec)
    if a.list:
        paths += [ln.strip() for ln in open(a.list) if ln.strip()]
    if not paths:
        sys.exit('no specs')
    garments, skipped = [], 0
    for p in paths:
        try:
            garments.append(Garment(p))
        except Exception as ex:
            skipped += 1
            print(f'  [skip] {p}: {ex}', file=sys.stderr)
    ne = sum(len(g.edges) for g in garments)
    ns = sum(len(g.gt) for g in garments)
    print(f'garments {len(garments)} (skipped {skipped})   edges {ne}   GT stitches {ns}')
    out = dict(n_garments=len(garments), edges=ne, gt_stitches=ns, results={})
    for kind in ('weak', 'strong'):
        best = None
        for tau in GRIDS[kind]:
            r = evaluate(garments, kind, tau)
            if best is None or r['f1'] > best['f1']:
                best = r
        pg = 100.0*best['perfect']/max(best['n_garments'], 1)
        print(f"H-{kind:<6} best τ={best['tau']:<5} P {best['precision']:.3f}  R {best['recall']:.3f}"
              f"  F1 {best['f1']:.3f}   ★전량 정답 {best['perfect']}/{best['n_garments']} = {pg:.1f}%")
        print(f"          유형별 recall: {best['class_recall']}  (모수 {best['class_total']})"
              f"   거울 바꿔치기 {best['mirror_swaps']}건")
        best.pop('per_garment', None) if len(garments) > 50 else None
        out['results'][kind] = best
    if a.json:
        json.dump(out, open(a.json, 'w'), indent=1, ensure_ascii=False)
        print('json ->', a.json)


if __name__ == '__main__':
    main()
