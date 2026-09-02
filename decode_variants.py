"""Same noise, four decoders. How much of the 'seam death' was the decoder, not the noise?
  1nn-all     1-NN; seam correct only if EVERY vertex correct   <- what was reported
  1nn-vote    1-NN; seam correct if >50% of vertices correct
  mutual-*    same, but a non-mutual nearest neighbour counts as wrong (abstain)
"""
import numpy as np, os, glob, sys, depth_oracle as D
from depth_tolerance import front_visible
AX_Z=2; SIG=[1.0,2.0,3.0,5.0,10.0]; R=12
KEYS=("1nn-all","1nn-vote","mutual-all","mutual-vote","_vert","_mut")

rng=np.random.default_rng(3)
n=int(sys.argv[1]) if len(sys.argv)>1 else 10
acc={s:{k:[] for k in KEYS} for s in SIG}
used=0
for gd in sorted(glob.glob(os.path.expanduser("~/mnt/data/rand_*")))[:n]:
    o=D.load(gd)
    if o is None: continue
    W,lab=o; si,sets=D.seam_sets(lab)
    if len(si)<20: continue
    vis=front_visible(W,si); P=W[si][vis]; S=[sets[i] for i in np.where(vis)[0]]
    if len(P)<20: continue
    ids=sorted({x for fs in S for x in fs}); used+=1
    # ---- built ONCE per garment ----
    base=np.repeat(P,2,axis=0); owner=np.repeat(np.arange(len(P)),2); m2=len(base)
    Sx=[S[i] for i in owner]
    M=np.zeros((m2,len(ids)),bool)                      # membership matrix
    for k,sd in enumerate(ids): M[:,k]=[sd in Sx[i] for i in range(m2)]
    ok=(M.astype(np.uint8)@M.astype(np.uint8).T)>0      # share a stitch id
    np.fill_diagonal(ok,False); eye=np.eye(m2,dtype=bool)
    big=[k for k in range(len(ids)) if M[:,k].sum()>1]
    for s in SIG:
        for _ in range(R):
            Q=base.copy(); Q[:,AX_Z]+=rng.normal(0,s/10.0,m2)
            Dm=np.linalg.norm(Q[:,None,:]-Q[None,:,:],axis=2); Dm[eye]=np.inf
            nn=Dm.argmin(1); good=ok[np.arange(m2),nn]
            mut=(nn[nn]==np.arange(m2)); gm=good&mut
            a=acc[s]
            a["1nn-all"].append(np.mean([good[M[:,k]].all()      for k in big]))
            a["1nn-vote"].append(np.mean([good[M[:,k]].mean()>0.5 for k in big]))
            a["mutual-all"].append(np.mean([gm[M[:,k]].all()      for k in big]))
            a["mutual-vote"].append(np.mean([gm[M[:,k]].mean()>0.5 for k in big]))
            a["_vert"].append(good.mean()); a["_mut"].append(mut.mean())
print(f"garments={used}  draws/sigma={R}\n")
print(f"{'sig':>6} {'vert':>7} {'mutual':>7} | {'1nn-all':>8} {'1nn-vote':>9} | {'mut-all':>8} {'mut-vote':>9}")
print("-"*66)
for s in SIG:
    a=acc[s]
    print(f"{s:>4.0f}mm {np.mean(a['_vert']):>7.4f} {np.mean(a['_mut']):>7.3f} |"
          f" {np.mean(a['1nn-all']):>8.4f} {np.mean(a['1nn-vote']):>9.4f} |"
          f" {np.mean(a['mutual-all']):>8.4f} {np.mean(a['mutual-vote']):>9.4f}")
