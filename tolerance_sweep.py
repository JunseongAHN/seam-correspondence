"""Fine sigma sweep of the depth-error -> stitching-failure curve. Emits JSON."""
import numpy as np, os, glob, json, sys, depth_oracle as D
from depth_tolerance import front_visible
AX_Z = 2
SIG = [0.05,0.1,0.15,0.2,0.25,0.3,0.4,0.5,0.7,1.0,1.5,2.0,3.0,5.0]   # cm

def run(root, n, seed=0):
    rng = np.random.default_rng(seed)
    res = {m:{ "pair":[[] for _ in SIG], "seam":[[] for _ in SIG], "garment":[[] for _ in SIG]}
           for m in ("iid","panel")}
    margins_all, seam_counts, seam_minmargin = [], [], []
    used = 0
    for gd in sorted(glob.glob(os.path.join(root,"rand_*")))[:n]:
        o = D.load(gd)
        if o is None: continue
        W, lab = o; si, sets = D.seam_sets(lab)
        if len(si) < 20: continue
        vis = front_visible(W, si)
        P = W[si][vis]; S=[sets[i] for i in np.where(vis)[0]]
        if len(P) < 20: continue
        used += 1
        m = D.margins(P, S); margins_all.append(m)
        ids = sorted({x for fs in S for x in fs})
        seam_counts.append(len(ids))
        for sid in ids:
            mm = np.array([sid in fs for fs in S])
            if mm.sum(): seam_minmargin.append(float(m[mm].min()))
        base = np.repeat(P,2,axis=0); owner=np.repeat(np.arange(len(P)),2)
        Sx = [S[i] for i in owner]; m2=len(base)
        ok = np.zeros((m2,m2),bool)
        for i in range(m2):
            a=Sx[i]; ok[i]=[not a.isdisjoint(Sx[j]) for j in range(m2)]
        np.fill_diagonal(ok,False); eye=np.eye(m2,dtype=bool)
        seam_mem=[np.array([sid in Sx[i] for i in range(m2)]) for sid in ids]
        for k,s in enumerate(SIG):
            for model in ("iid","panel"):
                if model=="iid": dz = rng.normal(0,s,m2)
                else:
                    off={}; dz=np.empty(m2)
                    for i in range(m2):
                        key=(owner[i]%1, i%2)
                        dz[i]=off.setdefault(i%2, rng.normal(0,s))
                Q=base.copy(); Q[:,AX_Z]+=dz
                Dm=np.linalg.norm(Q[:,None,:]-Q[None,:,:],axis=2); Dm[eye]=np.inf
                good=ok[np.arange(m2),Dm.argmin(1)]
                res[model]["pair"][k].append(float(good.mean()))
                sok=[bool(good[mm].all()) for mm in seam_mem if mm.sum()>1]
                res[model]["seam"][k].append(float(np.mean(sok)))
                res[model]["garment"][k].append(float(all(sok)))
    M=np.concatenate(margins_all)
    out=dict(sigma_mm=[s*10 for s in SIG], n_garments=used,
             n_seam_vertices=int(len(M)),
             seam_count=dict(mean=float(np.mean(seam_counts)),min=int(min(seam_counts)),
                             max=int(max(seam_counts))),
             margin_mm=dict(p1=float(np.percentile(M,1)*10),p10=float(np.percentile(M,10)*10),
                            p25=float(np.percentile(M,25)*10),p50=float(np.percentile(M,50)*10),
                            p75=float(np.percentile(M,75)*10),min=float(M.min()*10)),
             margin_hist=[float(x) for x in np.histogram(np.clip(M*10,0,150),bins=30,range=(0,150))[0]],
             seam_minmargin_mm=sorted(float(x*10) for x in seam_minmargin))
    for model in ("iid","panel"):
        out[model]={k:[float(np.mean(v)) for v in res[model][k]] for k in ("pair","seam","garment")}
        out[model]["seam_spread"]=[[float(np.percentile(v,10)),float(np.percentile(v,90))]
                                   for v in res[model]["seam"]]
    return out

if __name__=="__main__":
    o=run(os.path.expanduser("~/mnt/data"), int(sys.argv[1]) if len(sys.argv)>1 else 15)
    json.dump(o, open(os.path.expanduser("~/mnt/seam-correspondence/tolerance.json"),"w"), indent=1)
    print("garments",o["n_garments"],"seam verts",o["n_seam_vertices"],
          "seams/garment",round(o["seam_count"]["mean"],1))
    print("margin mm", {k:round(v,2) for k,v in o["margin_mm"].items()})
    print(f"{'sig_mm':>7} {'pair':>7} {'seam':>7} {'garment':>8} | {'p_seam':>7} {'p_gar':>7}")
    for i,s in enumerate(o["sigma_mm"]):
        print(f"{s:>7.1f} {o['iid']['pair'][i]:>7.4f} {o['iid']['seam'][i]:>7.4f} "
              f"{o['iid']['garment'][i]:>8.4f} | {o['panel']['seam'][i]:>7.4f} {o['panel']['garment'][i]:>7.4f}")
