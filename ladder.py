#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ladder.py — 정보원별 휴리스틱 사다리 (실험 A1 재설계본 v2)
==========================================================
왜 다시 짜는가 (2026-08-29 진단):
  구 하네스(baseline_match.py)는 "내 휴리스틱이 얼마나 나쁜가"를 쟀다.
  ① 전역 임계값 τ=8cm 가 GT 쌍의 30.3%만 덮어 recall 이 0.30에 갇힌다.
  ② 배치 거리 자체가 약하다 — GarmentCodeData 의 translation/rotation 은 시뮬 '전' 초기 배치라
     붙을 엣지가 맞닿아 있지 않다(GT 쌍 거리 중앙값 14.2cm, 참 파트너 1등 41.5%).
  → 최적 할당으로 바꿔도 천장이 낮다. 피처를 바꿔야 한다.

이 파일이 재는 것: **각 정보원이 얼마를 주는가.** 논문 §3 정보 계층과 같은 축.
  R0 길이만 / R1 +형상 / R2 +배치 / R3 +패널 이름
  R0~R2 = 논문의 "geometry alone" 과 같은 조건.  R3 = 그 자기부과 제약을 풀면 얼마가 들어오나(별도 축).

v2 에서 고친 것
  (a) 가중치를 손으로 정하지 않는다. rung 마다 새 항의 가중치를 그리드로 훑어 그 rung 의 best-F1 을
      쓴다 = "튜닝된 휴리스틱 = 그 rung 의 상한". 앞 rung 의 가중치는 고정(전방 선택).
  (b) 이름 항이 같은 패널 쌍을 0점으로 만들어 가짜 다트가 순위를 덮던 문제 → 같은 패널은 이름 항 중립(0.5).
  (c) 모든 항 [0,1] 정규화(배치는 벌 bbox 대각선으로 나눔) — 전역 cm 임계값 폐기.

두 축을 따로 보고한다:
  신호  참 파트너가 그 점수에서 1등인 비율 (매칭과 무관한 피처 품질 — 오늘의 교훈)
  결과  쌍 P/R/F1 + ★벌 단위 전량 정답률(AutoSew GSP 대응)

호길이는 measure.py 의 정확 호길이(원호 해석해 + Gauss-Legendre). stdlib only. 읽기 전용.
"""
import json, math, argparse, os, sys, itertools, importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))


def _load(n, p):
    sp = importlib.util.spec_from_file_location(n, p); m = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(m); return m


MS = _load('measure', os.path.join(HERE, 'measure.py'))
K = 9
TAUS = [0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.14, 0.2, 0.27, 0.35, 0.45, 0.6, 0.8, 1.0]
W_POS = [0.0, 0.2, 0.35, 0.5, 0.65, 0.8, 1.0]
W_NAME = [0.0, 0.1, 0.2, 0.3, 0.45, 0.6]


def euler_mat(rx, ry, rz):
    rx, ry, rz = (math.radians(v) for v in (rx, ry, rz))
    cx, sx, cy, sy, cz, sz = math.cos(rx), math.sin(rx), math.cos(ry), math.sin(ry), math.cos(rz), math.sin(rz)
    Rx = [[1,0,0],[0,cx,-sx],[0,sx,cx]]; Ry = [[cy,0,sy],[0,1,0],[-sy,0,cy]]; Rz = [[cz,-sz,0],[sz,cz,0],[0,0,1]]
    def mm(A, B): return [[sum(A[i][k]*B[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
    return mm(mm(Rx, Ry), Rz)


SIDE_TOK = {'left', 'right', 'l', 'r'}


def side_of(n):
    if 'left' in n: return 'L'
    if 'right' in n: return 'R'
    return '-'


class Garment:
    def __init__(self, path):
        d = json.load(open(path)); pat = d['pattern']
        self.path = path; self.E = []; eid = {}; allpts = []
        for name, p in pat['panels'].items():
            R = euler_mat(*p.get('rotation', [0,0,0])); T = p.get('translation', [0,0,0])
            toks = set(name.split('_')) - SIDE_TOK          # 부위 토큰만 (좌우는 d_side 가 따로 본다)
            sd = side_of(name); V = p['vertices']
            for k, e in enumerate(p['edges']):
                i, j = e['endpoints']; curv = e.get('curvature')
                L, _ = MS.edge_length(V[i], V[j], curv)
                pts2 = MS.edge_points(V[i], V[j], curv, K)
                if len(pts2) != K:
                    q0, q1 = pts2[0], pts2[-1]
                    pts2 = [(q0[0]+(q1[0]-q0[0])*t/(K-1), q0[1]+(q1[1]-q0[1])*t/(K-1)) for t in range(K)]
                pts3 = [[sum(R[r][c]*q[c] for c in range(2)) + T[r] for r in range(3)] for q in pts2]
                allpts.extend(pts3)
                turn = 0.0
                for t in range(1, K-1):
                    a = (pts2[t][0]-pts2[t-1][0], pts2[t][1]-pts2[t-1][1])
                    b = (pts2[t+1][0]-pts2[t][0], pts2[t+1][1]-pts2[t][1])
                    if math.hypot(*a) > 1e-12 and math.hypot(*b) > 1e-12:
                        turn += abs(math.atan2(a[0]*b[1]-a[1]*b[0], a[0]*b[0]+a[1]*b[1]))
                chord = math.dist(pts2[0], pts2[-1])
                eid[(name, k)] = len(self.E)
                self.E.append(dict(panel=name, pts3=pts3, L=L, turn=turn,
                                   ca=(chord/L if L > 1e-9 else 1.0), toks=toks, side=sd))
        xs = [q[0] for q in allpts]; ys = [q[1] for q in allpts]; zs = [q[2] for q in allpts]
        self.diag = max(math.dist((min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs))), 1e-6)
        self.gt = set()
        for s in pat['stitches']:
            sd = [x for x in s if isinstance(x, dict)]
            if len(sd) != 2: continue
            a, b = (sd[0]['panel'], sd[0]['edge']), (sd[1]['panel'], sd[1]['edge'])
            if a in eid and b in eid: self.gt.add(frozenset((eid[a], eid[b])))
        self.n = len(self.E)
        self.T = {}
        for a, b in itertools.combinations(range(self.n), 2):
            A, B = self.E[a], self.E[b]
            ml = max(A['L'], B['L'], 1e-9)
            shape = (0.6*abs(A['L']-B['L'])/ml + 0.25*abs(A['turn']-B['turn'])/math.pi
                     + 0.15*abs(A['ca']-B['ca']))
            P, Q = A['pts3'], B['pts3']
            dpos = min(sum(math.dist(P[k], Q[k]) for k in range(K)),
                       sum(math.dist(P[k], Q[K-1-k]) for k in range(K)))/K/self.diag
            if A['panel'] == B['panel']:
                dname = 0.5                                  # 같은 패널: 이름 항 중립
            else:
                u, v = A['toks'], B['toks']
                jac = len(u & v)/max(len(u | v), 1)
                opp = 1.0 if (A['side'] != '-' and B['side'] != '-' and A['side'] != B['side']) else 0.0
                dname = 0.5*(1.0-jac) + 0.5*opp
            self.T[(a, b)] = (abs(A['L']-B['L'])/ml, shape, min(dpos, 1.0), dname)

    def cands(self, w_pos, w_name, len_only=False):
        if len_only:
            it = ((t[0], ab[0], ab[1]) for ab, t in self.T.items())
        else:
            wb = 1.0 - w_pos - w_name
            it = ((wb*t[1] + w_pos*t[2] + w_name*t[3], ab[0], ab[1]) for ab, t in self.T.items())
        return sorted(it)

    def rank1(self, c):
        best = {}
        for s, a, b in c:
            if a not in best: best[a] = b
            if b not in best: best[b] = a
        return sum(1 for p in self.gt
                   if best.get(tuple(p)[0]) == tuple(p)[1] or best.get(tuple(p)[1]) == tuple(p)[0])


def predict(c, tau):
    used, pred = set(), set()
    for s, a, b in c:
        if s > tau: break
        if a in used or b in used: continue
        used.add(a); used.add(b); pred.add(frozenset((a, b)))
    return pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--list', required=True)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--shard', type=int, default=250)
    ap.add_argument('--json')
    ap.add_argument('--fix-wpos', type=float, default=None,
                    help='R2/R3 의 배치 가중치를 고정 (튜닝 집합에서 구한 값으로 전수 평가할 때)')
    ap.add_argument('--fix-wname', type=float, default=None)
    a = ap.parse_args()
    paths = [ln.strip() for ln in open(a.list) if ln.strip()]
    if a.limit: paths = paths[:a.limit]

    # 설정: (라벨, len_only, w_pos 후보, w_name 후보)
    wp2 = [a.fix_wpos] if a.fix_wpos is not None else W_POS
    wp3 = [a.fix_wpos] if a.fix_wpos is not None else W_POS
    wn3 = [a.fix_wname] if a.fix_wname is not None else W_NAME
    CFG = [('R0_len',   True,  [0.0],  [0.0]),
           ('R1_shape', False, [0.0],  [0.0]),
           ('R2_place', False, wp2,    [0.0]),
           ('R3_name',  False, wp3,    wn3)]
    acc = {}     # (label, w_pos, w_name, tau) -> counters ; rank1[(label,w_pos,w_name)]
    r1 = {}
    NG = ngt = 0
    for s0 in range(0, len(paths), a.shard):
        for p in paths[s0:s0+a.shard]:
            try: g = Garment(p)
            except Exception as ex:
                print(f'  [skip] {p}: {ex}', file=sys.stderr); continue
            NG += 1; ngt += len(g.gt)
            for lab, lo, WP, WN in CFG:
                for wp in WP:
                    for wn in WN:
                        if wp + wn > 1.0: continue
                        c = g.cands(wp, wn, len_only=lo)
                        r1[(lab, wp, wn)] = r1.get((lab, wp, wn), 0) + g.rank1(c)
                        for tau in TAUS:
                            A = acc.setdefault((lab, wp, wn, tau), [0, 0, 0, 0])
                            pr = predict(c, tau); tp = pr & g.gt
                            A[0] += len(tp); A[1] += len(pr-g.gt); A[2] += len(g.gt-pr); A[3] += (pr == g.gt)
            del g
        print(f'  {min(s0+a.shard, len(paths))}/{len(paths)}', file=sys.stderr, flush=True)

    print(f'garments {NG}   GT stitches {ngt}\n')
    print(f'{"rung":<10} {"w_pos":>6} {"w_name":>7} {"τ":>6} | {"참파트너1등":>11} {"P":>6} {"R":>6} '
          f'{"F1":>6} {"★전량정답":>14}')
    out = dict(n_garments=NG, gt=ngt, rungs={})
    for lab, lo, WP, WN in CFG:
        best = None
        for (l, wp, wn, tau), A in acc.items():
            if l != lab: continue
            P = A[0]/max(A[0]+A[1], 1); R = A[0]/max(A[0]+A[2], 1)
            F1 = 2*P*R/max(P+R, 1e-12)
            if best is None or F1 > best[0]:
                best = (F1, wp, wn, tau, P, R, A[3])
        F1, wp, wn, tau, P, R, pf = best
        sig = r1[(lab, wp, wn)]/max(ngt, 1)
        print(f'{lab:<10} {wp:6.2f} {wn:7.2f} {tau:6.3f} | {100*sig:10.1f}% {P:6.3f} {R:6.3f} {F1:6.3f} '
              f'{pf:5d}/{NG} = {100.0*pf/max(NG,1):5.2f}%')
        out['rungs'][lab] = dict(w_pos=wp, w_name=wn, tau=tau, rank1_rate=round(sig, 4),
                                 precision=round(P, 4), recall=round(R, 4), f1=round(F1, 4),
                                 perfect=pf, n_garments=NG)
    if a.json:
        os.makedirs(os.path.dirname(a.json) or '.', exist_ok=True)
        json.dump(out, open(a.json, 'w'), indent=1, ensure_ascii=False)
        print('json ->', a.json)


if __name__ == '__main__':
    main()
