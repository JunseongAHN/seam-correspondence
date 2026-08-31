#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ot_match.py — 스티치 매칭을 최적수송(OT)으로 푼다. solver 를 분리해 재는 실험.
================================================================================
왜: ladder.py 의 R3 은 "점수 오름차순 greedy + 전역 임계값" 이다. 이게 병목인지
    피처가 병목인지 갈라야 한다. **비용 함수를 R3 과 똑같이 고정하고 solver 만 바꾼다.**

문제의 구조 (여기서부터 설계가 나온다)
  · 스티치는 1:1 이다 (GarmentCodeData 전수 확인 — 한 엣지가 두 스티치에 쓰인 사례 0건)
  · 짝짓기는 **대칭**이다. i↔j 이면 j↔i. 즉 이분 할당이 아니라 한 집합 위의 대합(involution).
  · 모든 엣지가 짝을 갖지는 않는다 — 네크라인·밑단은 열린 경계로 남는다 → **dustbin 필요.**
  · 좌우 거울 패널이 있으면 스티치도 거울로 짝지어진다 (아래 MIRROR).

rung 3종
  greedy    R3 재현 — 점수 오름차순 1:1 greedy + 임계값
  sinkhorn  대칭 비용행렬 + dustbin 을 augment 해 log-domain Sinkhorn.
            비용이 대칭이고 주변부가 같으므로 수송계획 P 도 대칭이 된다 = 대합 구조를 solver 가 안다.
  mirror    Sinkhorn 후 P 를 거울 맵으로 대칭화하고 다시 정규화(교대 사영).
            ★MIRROR 맵은 (거울 패널, **역순 엣지 인덱스**) — 데이터에서 94.6% 성립을 확인하고 썼다.
            같은 인덱스로 매핑하면 3.7% 밖에 안 맞는다 (거울은 경계 순회 방향을 뒤집는다).

라운딩: P 내림차순 greedy 1:1, 단 (i,j) 는 양쪽 dustbin 질량을 모두 넘겨야 채택.
        → "붙일 짝이 없으면 비워둔다" 가 구조적으로 표현된다 (greedy 에는 이 개념이 없다).

사용:
  python3 ot_match.py --list specs.txt --limit 400 --tune          # eps·dustbin·τ 그리드
  python3 ot_match.py --list specs.txt --eps .05 --z -.6 --tau .3 --json results/ot.json
numpy 필요. 읽기 전용.
"""
import json, math, os, sys, argparse, importlib.util
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def _load(n, p):
    sp = importlib.util.spec_from_file_location(n, p); m = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(m); return m


LD = _load('ladder', os.path.join(HERE, 'ladder.py'))
W_SHAPE, W_POS, W_NAME = 0.30, 0.50, 0.20        # ladder.py 가 400벌에서 튜닝한 R3 가중치


def mirror_panel(n):
    if 'left' in n:  return n.replace('left', 'right')
    if 'right' in n: return n.replace('right', 'left')
    return None


class Prob:
    """한 벌의 비용행렬 + 거울 맵."""
    def __init__(self, path):
        g = LD.Garment(path)
        self.g = g
        n = g.n
        C = np.full((n, n), np.inf)
        for (a, b), t in g.T.items():
            v = W_SHAPE*t[1] + W_POS*t[2] + W_NAME*t[3]
            C[a, b] = v; C[b, a] = v
        np.fill_diagonal(C, np.inf)
        self.C = C
        self.n = n
        # 거울 맵: (거울 패널, 역순 엣지 인덱스)
        per = {}
        for i, e in enumerate(g.E):
            per.setdefault(e['panel'], []).append(i)
        idx = {}
        for name, ids in per.items():
            for k, i in enumerate(ids):
                idx[(name, k)] = i
        cnt = {name: len(ids) for name, ids in per.items()}
        self.mir = np.full(n, -1, dtype=int)
        for name, ids in per.items():
            mn = mirror_panel(name)
            if mn is None or mn not in per: continue
            if cnt[mn] != len(ids): continue
            for k, i in enumerate(ids):
                self.mir[i] = idx[(mn, cnt[mn]-1-k)]
        self.gt = set(map(lambda p: tuple(sorted(tuple(p))), g.gt))


def sinkhorn(C, eps, z, iters=25):
    """SuperGlue 식 augment: dustbin 행/열 1개, 실제 노드 질량 1, dustbin 질량 n."""
    n = C.shape[0]
    S = np.full((n+1, n+1), z, dtype=np.float64)
    S[:n, :n] = -C                                  # 점수 = -비용
    S[:n, :n][~np.isfinite(S[:n, :n])] = -1e9
    K = S/eps
    la = np.concatenate([np.zeros(n), [math.log(n)]])
    lb = la.copy()
    u = np.zeros(n+1); v = np.zeros(n+1)
    for _ in range(iters):
        u = la - _lse(K + v[None, :], axis=1)
        v = lb - _lse(K + u[:, None], axis=0)
    P = np.exp(K + u[:, None] + v[None, :])
    return P


def _lse(M, axis):
    m = M.max(axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    return (m + np.log(np.exp(M - m).sum(axis=axis, keepdims=True))).squeeze(axis)


def symmetrize_mirror(P, mir, n):
    """P[i,j] 와 P[m(i),m(j)] 를 평균 — 좌우 대칭 규약을 수송계획에 주입."""
    Q = P.copy()
    ok = mir >= 0
    ii = np.where(ok)[0]
    if len(ii) == 0: return Q
    mi = mir[ii]
    blk = P[:n, :n]
    sub = 0.5*(blk[np.ix_(ii, ii)] + blk[np.ix_(mi, mi)])
    Q[np.ix_(ii, ii)] = sub
    Q[np.ix_(mi, mi)] = sub
    d = P[:n, n].copy()
    d[ii] = 0.5*(P[ii, n] + P[mi, n])                 # 거울이 있는 엣지만 dustbin 질량 대칭화
    Q[:n, n] = d
    Q[n, :n] = d
    return Q


def round_plan(P, n, tau):
    """P 내림차순 greedy 1:1. dustbin 질량을 못 넘으면 비워둔다."""
    blk = P[:n, :n].copy()
    dust = P[:n, n]
    iu = np.triu_indices(n, 1)
    order = np.argsort(-blk[iu])
    used = np.zeros(n, dtype=bool)
    out = set()
    for o in order:
        i, j = iu[0][o], iu[1][o]
        p = blk[i, j]
        if p < tau: break
        if used[i] or used[j]: continue
        if p < dust[i] or p < dust[j]: continue
        used[i] = used[j] = True
        out.add((int(i), int(j)))
    return out


def round_plan_mirror(P, n, tau, mir):
    """대칭을 계획이 아니라 **결정**에 강제한다.
       (i,j) 를 채택하면 거울 짝 (m(i),m(j)) 도 같이 채택 — 좌우 규약이 결정 단위로 전파된다.
       비용행렬이 이미 거의 거울 대칭이라 P 를 평균 내는 것만으로는 아무것도 안 바뀐다."""
    blk = P[:n, :n]
    dust = P[:n, n]
    iu = np.triu_indices(n, 1)
    order = np.argsort(-blk[iu])
    used = np.zeros(n, dtype=bool)
    out = set()

    def take(i, j):
        if used[i] or used[j] or i == j: return False
        if blk[i, j] < dust[i] or blk[i, j] < dust[j]: return False
        used[i] = used[j] = True
        out.add((min(i, j), max(i, j)))
        return True

    for o in order:
        i, j = int(iu[0][o]), int(iu[1][o])
        if blk[i, j] < tau: break
        if not take(i, j): continue
        mi, mj = int(mir[i]), int(mir[j])
        if mi >= 0 and mj >= 0:
            take(mi, mj)                              # 거울 짝은 임계값을 다시 묻지 않는다
    return out


def greedy_ref(C, n, tau):
    iu = np.triu_indices(n, 1)
    vals = C[iu]
    order = np.argsort(vals)
    used = np.zeros(n, dtype=bool); out = set()
    for o in order:
        v = vals[o]
        if not np.isfinite(v) or v > tau: break
        i, j = iu[0][o], iu[1][o]
        if used[i] or used[j]: continue
        used[i] = used[j] = True
        out.add((int(i), int(j)))
    return out


def score(pred, gt):
    tp = len(pred & gt)
    return tp, len(pred)-tp, len(gt)-tp, pred == gt


def run(paths, eps, z, taus, tau_g, want=('greedy', 'sinkhorn', 'mirror'), quiet=True):
    acc = {r: {t: [0, 0, 0, 0] for t in (taus if r != 'greedy' else [tau_g])} for r in want}
    ng = 0; ngt = 0
    for p in paths:
        try: pr = Prob(p)
        except Exception as ex:
            print('  [skip]', p, ex, file=sys.stderr); continue
        n, gt = pr.n, pr.gt
        ng += 1; ngt += len(gt)
        if 'greedy' in want:
            a = acc['greedy'][tau_g]
            tp, fp, fn, ok = score(greedy_ref(pr.C, n, tau_g), gt)
            a[0] += tp; a[1] += fp; a[2] += fn; a[3] += ok
        if 'sinkhorn' in want or 'mirror' in want:
            P = sinkhorn(pr.C, eps, z)
            if 'sinkhorn' in want:
                for t in taus:
                    a = acc['sinkhorn'][t]
                    tp, fp, fn, ok = score(round_plan(P, n, t), gt)
                    a[0] += tp; a[1] += fp; a[2] += fn; a[3] += ok
            if 'mirror' in want:
                Q = symmetrize_mirror(P, pr.mir, n)
                for t in taus:
                    a = acc['mirror'][t]
                    tp, fp, fn, ok = score(round_plan_mirror(Q, n, t, pr.mir), gt)
                    a[0] += tp; a[1] += fp; a[2] += fn; a[3] += ok
    return acc, ng, ngt


def best(acc, rung, ng):
    b = None
    for t, a in acc[rung].items():
        P = a[0]/max(a[0]+a[1], 1); R = a[0]/max(a[0]+a[2], 1)
        F = 2*P*R/max(P+R, 1e-12)
        if b is None or F > b[0]: b = (F, t, P, R, a[3])
    return b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--list', required=True)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--tune', action='store_true')
    ap.add_argument('--eps', type=float, default=0.05)
    ap.add_argument('--z', type=float, default=-0.6)
    ap.add_argument('--tau-g', type=float, default=0.20)
    ap.add_argument('--json')
    ap.add_argument('--from', dest='lo', type=int, default=0)
    ap.add_argument('--to', dest='hi', type=int, default=0)
    ap.add_argument('--acc', help='샤드 누적 파일 (여러 호출에 나눠 돌릴 때)')
    a = ap.parse_args()
    paths = [l.strip() for l in open(a.list) if l.strip()]
    if a.limit: paths = paths[:a.limit]
    if a.hi: paths = paths[a.lo:a.hi]
    TAUS = [1e-4, 3e-4, 1e-3, 3e-3, 0.01, 0.03, 0.1, 0.2, 0.35, 0.5]

    if a.tune:
        print(f'{"eps":>6} {"z":>6} | {"sinkhorn F1":>12} {"τ":>7} {"GSP":>7} | {"mirror F1":>10} {"τ":>7} {"GSP":>7}')
        for eps in (0.005, 0.01, 0.02):
            for z in (-0.15, -0.3, -0.5):
                acc, ng, ngt = run(paths, eps, z, TAUS, a.tau_g, want=('sinkhorn', 'mirror'))
                s = best(acc, 'sinkhorn', ng); m = best(acc, 'mirror', ng)
                print(f'{eps:6.3f} {z:6.2f} | {s[0]:12.4f} {s[1]:7.4g} {100*s[4]/ng:6.2f}% |'
                      f' {m[0]:10.4f} {m[1]:7.4g} {100*m[4]/ng:6.2f}%')
        return

    TAUS = [0.01, 0.03, 0.1, 0.2, 0.35]
    acc, ng, ngt = run(paths, a.eps, a.z, TAUS, a.tau_g)
    if a.acc:
        prev = json.load(open(a.acc)) if os.path.exists(a.acc) else None
        if prev:
            ng += prev['ng']; ngt += prev['ngt']
            for r in acc:
                for t in acc[r]:
                    for i in range(4): acc[r][t][i] += prev['acc'][r][str(t)][i]
        json.dump(dict(ng=ng, ngt=ngt,
                       acc={r:{str(t):list(map(int,v)) for t,v in d.items()} for r,d in acc.items()}),
                  open(a.acc,'w'))
    print(f'garments {ng}   GT stitches {ngt}   (eps={a.eps}, dustbin z={a.z})\n')
    print(f'{"rung":<10} {"τ":>8} {"P":>7} {"R":>7} {"F1":>7} {"★벌 전량 정답":>16}')
    out = dict(n_garments=ng, gt=ngt, eps=a.eps, z=a.z, rungs={})
    for r in ('greedy', 'sinkhorn', 'mirror'):
        F, t, P, R, ok = best(acc, r, ng)
        print(f'{r:<10} {t:8.4g} {P:7.3f} {R:7.3f} {F:7.3f} {ok:8d}/{ng} = {100.0*ok/ng:5.2f}%')
        out['rungs'][r] = dict(tau=t, precision=round(P, 4), recall=round(R, 4), f1=round(F, 4),
                               perfect=ok, n_garments=ng)
    if a.json:
        os.makedirs(os.path.dirname(a.json) or '.', exist_ok=True)
        json.dump(out, open(a.json, 'w'), indent=1, ensure_ascii=False)
        print('json ->', a.json)


if __name__ == '__main__':
    main()
