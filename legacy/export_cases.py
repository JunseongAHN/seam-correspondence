#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
export_cases.py — 뷰어가 먹을 케이스 데이터를 굽는다 (results/cases.json)
=========================================================================
ladder.py 의 R3(배치+이름) 예측을 지정한 벌들에 돌려서, 3D 렌더에 필요한 것만 뽑는다.
  패널: 2D 외곽선(곡선 샘플링됨) + 회전행렬 R + 이동 T   → 브라우저가 ShapeGeometry 로 채운다
  접합: 대응하는 9쌍의 3D 점 + ΔL(mm) + 끝점 짝(anti/para), TP/FP/FN 으로 분류

사용:
  python3 export_cases.py                                   # 기본 두 벌(성공/실패)
  python3 export_cases.py rand_XXXX rand_YYYY               # 벌 직접 지정
  python3 export_cases.py --specs <dir>                     # spec 폴더 지정
"""
import json, math, os, sys, argparse, importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))


def _load(n, p):
    sp = importlib.util.spec_from_file_location(n, p); m = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(m); return m


LD = _load('ladder', os.path.join(HERE, 'ladder.py'))
MS = _load('measure', os.path.join(HERE, 'measure.py'))

W_POS, W_NAME, TAU, K = 0.5, 0.2, 0.2, 9      # ladder.py 가 400벌에서 튜닝한 값
DEFAULT = ['rand_ATMPLQP6KY', 'rand_1A8YV15ISI']
KIND = {'rand_ATMPLQP6KY': 'perfect', 'rand_1A8YV15ISI': 'fail'}


def build(spec_path, kind):
    raw = json.load(open(spec_path)); pat = raw['pattern']
    g = LD.Garment(spec_path)
    pred = LD.predict(g.cands(W_POS, W_NAME), TAU)
    gt = g.gt
    panels, gid = [], 0
    for name, pp in pat['panels'].items():
        R = LD.euler_mat(*pp.get('rotation', [0, 0, 0])); T = pp.get('translation', [0, 0, 0])
        V = pp['vertices']; outline, emeta = [], []
        for k, e in enumerate(pp['edges']):
            i, j = e['endpoints']; curv = e.get('curvature')
            pts = MS.edge_points(V[i], V[j], curv, K)
            if len(pts) != K:                                  # 직선은 두 점만 온다
                q0, q1 = pts[0], pts[-1]
                pts = [(q0[0]+(q1[0]-q0[0])*t/(K-1), q0[1]+(q1[1]-q0[1])*t/(K-1)) for t in range(K)]
            start = len(outline)
            outline.extend([[round(q[0], 2), round(q[1], 2)] for q in pts[:-1]])
            emeta.append(dict(k=k, g=gid, s=start, n=K-1,
                              L=round(MS.edge_length(V[i], V[j], curv)[0]*10, 1)))
            gid += 1
        panels.append(dict(name=name, label=pp.get('label', ''), outline=outline, edges=emeta,
                           R=[round(v, 6) for row in R for v in row],
                           T=[round(v, 3) for v in T]))

    def links(pairs):
        out = []
        for pr in pairs:
            a, b = sorted(tuple(pr))
            A, B = g.E[a]['pts3'], g.E[b]['pts3']
            dp = sum(math.dist(A[k], B[k]) for k in range(K))
            da = sum(math.dist(A[k], B[K-1-k]) for k in range(K))
            anti = da < dp
            out.append(dict(a=a, b=b, anti=bool(anti),
                            dL=round(abs(g.E[a]['L']-g.E[b]['L'])*10, 1),
                            pa=[[round(v, 2) for v in q] for q in A],
                            pb=[[round(v, 2) for v in q] for q in (B[::-1] if anti else B)]))
        return out

    tp, fp, fn = gt & pred, pred - gt, gt - pred
    names = {}
    for p in panels:
        for e in p['edges']:
            names[e['g']] = '%s#%d' % (p['name'], e['k'])
    return dict(id=os.path.basename(spec_path).replace('.json', ''), kind=kind,
                panels=panels, names=names,
                tp=links(tp), fp=links(fp), fn=links(fn),
                stats=dict(panels=len(panels), edges=gid, gt=len(gt),
                           tp=len(tp), fp=len(fp), fn=len(fn),
                           precision=round(len(tp)/max(len(pred), 1), 3),
                           recall=round(len(tp)/max(len(gt), 1), 3)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('ids', nargs='*', default=[])
    ap.add_argument('--specs', default=os.path.expanduser('~/work/specs'),
                    help='<id>.json 이 들어 있는 폴더')
    a = ap.parse_args()
    ids = a.ids or DEFAULT
    out = []
    for gid in ids:
        p = os.path.join(a.specs, gid + '.json')
        if not os.path.exists(p):
            sys.exit('not found: ' + p)
        c = build(p, KIND.get(gid, 'fail' if len(out) else 'perfect'))
        print(gid, c['stats'])
        out.append(c)
    dst = os.path.join(HERE, 'results', 'cases.json')
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    json.dump(out, open(dst, 'w'), ensure_ascii=False, separators=(',', ':'))
    print('->', dst, os.path.getsize(dst), 'bytes')


if __name__ == '__main__':
    main()
