#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_topo.py — topo_check.py 를 무작위 표본에 돌려 catch-rate 를 집계한다.
topo_check.py 는 손대지 않는다. 전수(3450벌 × ~0.9s ≈ 53분)는 오늘 예산 밖이라
고정 시드 무작위 표본으로 비율만 추정한다. 벌 단위가 아니라 주입 사례 단위로 합산하고,
벌별 비율의 분포(min/med/max)도 같이 낸다 — 큰 벌이 평균을 지배하는 것을 보이기 위해.
사용: python3 run_topo.py --list specs.txt --n 200 --seed 0 --budget 150 --json results/topo_sweep.json
"""
import json, argparse, os, random, time, importlib.util, sys


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


HERE = os.path.dirname(os.path.abspath(__file__))
TC = _load('topo_check', os.path.join(HERE, 'topo_check.py'))
KINDS = ('delete', 'swap', 'flip', 'add', 'reuse')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--list', required=True)
    ap.add_argument('--n', type=int, default=200)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--max-swaps', type=int, default=1500)
    ap.add_argument('--budget', type=float, default=150.0, help='seconds')
    ap.add_argument('--json')
    a = ap.parse_args()
    paths = [ln.strip() for ln in open(a.list) if ln.strip()]
    rng = random.Random(a.seed)
    sample = rng.sample(paths, min(a.n, len(paths)))

    tot = {k: [0, 0] for k in KINDS}          # caught, n
    per_g_rate = {k: [] for k in KINDS}
    base_sigs, rows, errs = {}, [], 0
    sym = [0, 0]
    t0 = time.time()
    done = 0
    for p in sample:
        if time.time() - t0 > a.budget:
            break
        try:
            pat = TC.Pattern(p)
            res = TC.run_injections(pat, max_swaps=a.max_swaps)
        except Exception as ex:
            errs += 1
            print(f'  [err] {os.path.basename(p)}: {ex}', file=sys.stderr)
            continue
        b = res['baseline']
        sig = (b['components'], tuple(b['boundary_loops']), b['orientable'],
               tuple(str(x) for x in b['genus']), b['manifold'])
        base_sigs[sig] = base_sigs.get(sig, 0) + 1
        row = dict(id=os.path.basename(p).replace('.json', ''),
                   panels=len(pat.panels), stitches=len(pat.stitches),
                   euler=pat.euler_order, repaired=res['calibration_repaired'],
                   comp=b['components'], loops=b['boundary_loops'],
                   orientable=b['orientable'], genus=[str(x) for x in b['genus']],
                   manifold=b['manifold'])
        for k in KINDS:
            r = res[k]
            tot[k][0] += r['caught']; tot[k][1] += r['n']
            if r['n']:
                per_g_rate[k].append(r['caught']/r['n'])
            row[k] = [r['caught'], r['n']]
        sym[0] += res['swap'].get('sym_caught', 0); sym[1] += res['swap'].get('sym_n', 0)
        rows.append(row); done += 1

    def q(v, p):
        if not v: return None
        v = sorted(v); return round(v[min(len(v)-1, int(round(p*(len(v)-1))))], 4)

    print(f'garments {done} (errors {errs})   elapsed {time.time()-t0:.1f}s')
    print(f'GT baseline invariant signatures: {len(base_sigs)} distinct')
    for sig, c in sorted(base_sigs.items(), key=lambda z: -z[1])[:6]:
        print(f'   x{c:<4} comp={sig[0]} loops={sig[1]} orient={sig[2]} genus={sig[3]} manifold={sig[4]}')
    print(f'non-manifold GT: {sum(1 for r in rows if not r["manifold"])}/{done}   '
          f'non-orientable GT: {sum(1 for r in rows if not r["orientable"])}/{done}   '
          f'calibration repaired>0: {sum(1 for r in rows if r["repaired"])}/{done}')
    out = dict(n_garments=done, errors=errs, seed=a.seed, max_swaps=a.max_swaps, kinds={})
    for k in KINDS:
        c, n = tot[k]
        pr = per_g_rate[k]
        print(f'  {k:<7} caught {c:>7}/{n:<7} = {100.0*c/max(n,1):5.1f}%   '
              f'벌별 비율 min {q(pr,0)} p25 {q(pr,.25)} med {q(pr,.5)} max {q(pr,1)}')
        out['kinds'][k] = dict(caught=c, n=n, rate=round(c/max(n, 1), 4),
                               per_garment=dict(min=q(pr, 0), p25=q(pr, .25), med=q(pr, .5),
                                                p75=q(pr, .75), max=q(pr, 1)))
    if sym[1]:
        print(f'  거울이름 스티치쌍 단일 스왑: {sym[0]}/{sym[1]} = {100.0*sym[0]/sym[1]:.1f}%')
        out['mirror_named_single_swap'] = dict(caught=sym[0], n=sym[1])
    out['per_garment'] = rows
    if a.json:
        os.makedirs(os.path.dirname(a.json) or '.', exist_ok=True)
        json.dump(out, open(a.json, 'w'), indent=1, ensure_ascii=False, default=str)
        print('json ->', a.json)


if __name__ == '__main__':
    main()
