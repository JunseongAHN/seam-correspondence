#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_baseline.py — baseline_match.py 를 3450벌에 돌리기 위한 드라이버
====================================================================
baseline_match.py 는 손대지 않는다. 이 파일이 하는 일은 둘뿐:

(1) 샤딩.  baseline_match.main() 은 전 벌을 메모리에 올린 뒤 τ 그리드를 돈다.
    60벌에 70MB → 3450벌이면 ~4GB로 이 VM(3.9GB)에서 터진다.
    샤드별로 τ별 카운터(TP/FP/FN/perfect)만 누적한 뒤 마지막에 전역 best-F1 τ를 고른다.
    → 지표 정의는 원본과 동일. 스트리밍으로 바꿨을 뿐.

(2) 실패 분류를 예측기와 독립인 축으로 교체.
    ★원본의 pair_class 는 near/far 를 score_weak < 20 으로 가른다. 그런데 H-weak 의
    best τ 가 8이므로 far_placed(=score≥20)의 recall 은 정의상 항상 0이다 — 측정이 아니라
    항등식. 슬라이드에 올리면 안 된다.
    대신 GT만으로 결정되는 축을 쓴다:
      dart/self  같은 패널 안의 접합 (다트·슬릿)
      equal      ΔL ≤ 1mm      길이가 같은 접합
      ease       1 < ΔL ≤ 10mm  논문 τ_l=10mm 안쪽
      gather     ΔL > 10mm      τ_l 밖 — 개더·이즈
    ΔL 은 measure.py 의 정확 호길이(원호 해석해 + Gauss-Legendre)로 계산한다.
    baseline_match 자신의 len 은 circle 엣지를 현으로 근사하므로 분류에 쓰지 않는다.

사용: python3 run_baseline.py --list specs.txt --shard 300 --json results/baseline_sweep.json
"""
import json, argparse, sys, os, importlib.util


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


HERE = os.path.dirname(os.path.abspath(__file__))
BM = _load('baseline_match', os.path.join(HERE, 'baseline_match.py'))
MS = _load('measure', os.path.join(HERE, 'measure.py'))


def gt_classes(path):
    """(panel,edge) 쌍 -> class, using exact arc lengths. predictor-independent."""
    d = json.load(open(path))
    pat = d['pattern']
    units = d.get('properties', {}).get('units_in_meter', 100)
    cm2mm = 1000.0 / units
    L = {}
    for name, p in pat['panels'].items():
        V = p['vertices']
        for k, e in enumerate(p['edges']):
            i, j = e['endpoints']
            ln, _ = MS.edge_length(V[i], V[j], e.get('curvature'))
            L[(name, k)] = ln * cm2mm
    out = {}
    for s in pat['stitches']:
        sides = [x for x in s if isinstance(x, dict)]
        if len(sides) != 2:
            continue
        a = (sides[0]['panel'], sides[0]['edge'])
        b = (sides[1]['panel'], sides[1]['edge'])
        if a not in L or b not in L:
            continue
        if a[0] == b[0]:
            c = 'dart/self'
        else:
            dl = abs(L[a] - L[b])
            c = 'equal' if dl <= 1.0 else ('ease' if dl <= 10.0 else 'gather')
        out[frozenset((a, b))] = c
    return out


def key_of(g, pair):
    a, b = tuple(pair)
    return frozenset(((g.edges[a]['panel'], g.edges[a]['idx']),
                      (g.edges[b]['panel'], g.edges[b]['idx'])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--list', required=True)
    ap.add_argument('--shard', type=int, default=300)
    ap.add_argument('--json')
    a = ap.parse_args()
    paths = [ln.strip() for ln in open(a.list) if ln.strip()]

    acc = {k: {t: dict(TP=0, FP=0, FN=0, perfect=0, n=0, mirror=0,
                       hit={}, tot={}) for t in BM.GRIDS[k]} for k in ('weak', 'strong')}
    skipped = 0
    for s0 in range(0, len(paths), a.shard):
        chunk = paths[s0:s0+a.shard]
        gs, cls = [], []
        for p in chunk:
            try:
                gs.append(BM.Garment(p)); cls.append(gt_classes(p))
            except Exception as ex:
                skipped += 1
                print(f'  [skip] {p}: {ex}', file=sys.stderr)
        for kind in ('weak', 'strong'):
            for tau in BM.GRIDS[kind]:
                A = acc[kind][tau]
                for g, cm in zip(gs, cls):
                    pred = g.predict(kind, tau)
                    tp = pred & g.gt
                    A['TP'] += len(tp); A['FP'] += len(pred - g.gt); A['FN'] += len(g.gt - pred)
                    A['perfect'] += (pred == g.gt); A['n'] += 1
                    for pair in g.gt:
                        c = cm.get(key_of(g, pair), 'unknown')
                        A['tot'][c] = A['tot'].get(c, 0) + 1
                        if pair in tp:
                            A['hit'][c] = A['hit'].get(c, 0) + 1
                    for pair in (g.gt - pred):
                        if g.miss_detail(pair, pred) == 'mirror_swap':
                            A['mirror'] += 1
            for g in gs:
                g._cc = {}          # drop cached candidate lists between kinds
        del gs, cls
        print(f'  shard {s0}-{s0+len(chunk)} done', file=sys.stderr, flush=True)

    out = dict(n_garments=len(paths)-skipped, skipped=skipped, results={})
    for kind in ('weak', 'strong'):
        best = None
        for tau, A in acc[kind].items():
            P = A['TP']/max(A['TP']+A['FP'], 1); R = A['TP']/max(A['TP']+A['FN'], 1)
            F1 = 2*P*R/max(P+R, 1e-12)
            r = dict(tau=tau, precision=round(P, 4), recall=round(R, 4), f1=round(F1, 4),
                     perfect=A['perfect'], n=A['n'], mirror_swaps=A['mirror'],
                     class_recall={c: round(A['hit'].get(c, 0)/A['tot'][c], 4) for c in sorted(A['tot'])},
                     class_total=A['tot'])
            if best is None or r['f1'] > best['f1']:
                best = r
        pg = 100.0*best['perfect']/max(best['n'], 1)
        print(f"H-{kind:<6} best τ={best['tau']:<5} P {best['precision']:.3f}  R {best['recall']:.3f}"
              f"  F1 {best['f1']:.3f}   ★전량 정답 {best['perfect']}/{best['n']} = {pg:.2f}%")
        print(f"          유형별 recall: {best['class_recall']}")
        print(f"          모수: {best['class_total']}   거울 바꿔치기 {best['mirror_swaps']}건")
        out['results'][kind] = best
        out['results'][kind]['tau_grid'] = {
            str(t): dict(f1=round(2*(acc[kind][t]['TP']/max(acc[kind][t]['TP']+acc[kind][t]['FP'], 1))
                                  * (acc[kind][t]['TP']/max(acc[kind][t]['TP']+acc[kind][t]['FN'], 1))
                                  / max((acc[kind][t]['TP']/max(acc[kind][t]['TP']+acc[kind][t]['FP'], 1))
                                        + (acc[kind][t]['TP']/max(acc[kind][t]['TP']+acc[kind][t]['FN'], 1)), 1e-12), 4),
                         perfect=acc[kind][t]['perfect'])
            for t in BM.GRIDS[kind]}
    if a.json:
        os.makedirs(os.path.dirname(a.json) or '.', exist_ok=True)
        json.dump(out, open(a.json, 'w'), indent=1, ensure_ascii=False)
        print('json ->', a.json)


if __name__ == '__main__':
    main()
