"""AI-like error: inaccurate but SMOOTH — the predicted contour follows the GT
contour's shape while being displaced.  Then: how easy is the seam decode?

noise model per panel loop
  white N(0, s^2 I) per vertex
  -> zero-phase circular EMA along the loop (forward+backward, 2 laps each)
  -> rescale so the per-point RMS is back to s        (same magnitude, smooth shape)
interior (non-boundary) vertices get white noise at the same s — they only act as
z-buffer occluders.

pipeline: noised points -> front depth map (1024^2, z-buffer) -> lift -> 1-NN decode
"""
import numpy as np, os, sys, json
from collections import defaultdict
import depth_oracle as D
from depth_tolerance import front_visible
from pipeline_tolerance import project_lift
from panel_depth import load_full, panel_sets
AX_Z=2; RES=1024

def loops_of(faces):
    cnt=defaultdict(int)
    for f in faces:
        for a,b in ((f[0],f[1]),(f[1],f[2]),(f[2],f[0])): cnt[(min(a,b),max(a,b))]+=1
    be=[e for e,c in cnt.items() if c==1]
    adj=defaultdict(list)
    for a,b in be: adj[a].append(b); adj[b].append(a)
    seen=set(); out=[]
    for st in list(adj):
        if st in seen: continue
        L=[st]; seen.add(st); prev=None; cur=st
        while True:
            nx=[v for v in adj[cur] if v!=prev and v not in seen]
            if not nx: break
            L.append(nx[0]); seen.add(nx[0]); prev,cur=cur,nx[0]
        if len(L)>=4: out.append(L)
    return out

def ema_loop(X,a):
    """zero-phase circular EMA on an (n,3) array"""
    n=len(X); f=np.empty_like(X); s=X[0].copy()
    for lap in range(2):
        for i in range(n):
            s=a*X[i]+(1-a)*s
            if lap: f[i]=s
    out=np.empty_like(X); t=f[-1].copy()
    for lap in range(2):
        for i in range(n-1,-1,-1):
            t=a*f[i]+(1-a)*t
            if lap: out[i]=t
    return out

g="rand_00YONAPXZE"; gd=os.path.expanduser("~/mnt/data/"+g)
Wv,Wf,lab=load_full(gd,g)
panels,memb,is_seam=panel_sets(Wf,lab)
loops=[]                                  # (panel_idx, [welded vertex ids])
for k in range(len(panels)):
    own=np.array([k in memb[v] for v in range(len(lab))]); fs=own[Wf].all(1)
    if fs.sum()<4: continue
    for L in loops_of(Wf[fs]): loops.append((k,L))
print(f"panels {len(panels)}  loops {len(loops)}  loop pts {sum(len(L) for _,L in loops)}")

# copies of each seam vertex = its occurrences across panel loops
occ=defaultdict(list)                     # welded vid -> [(loop_i, pos)]
for li,(pk,L) in enumerate(loops):
    for pos,v in enumerate(L): occ[v].append((li,pos))
sets_all={i:frozenset(lab[i].split(",")) for i in range(len(lab)) if lab[i].startswith("stitch")}
seam_v=[v for v in sets_all if len(occ[v])>=2]
vis=front_visible(Wv,np.array(seam_v))
seam_v=[v for v,ok in zip(seam_v,vis) if ok]
copies=[(v,li,pos) for v in seam_v for (li,pos) in occ[v][:2]]
vid=np.array([c[0] for c in copies]); m2=len(copies)
Sx=[sets_all[v] for v in vid]
ids=sorted({s for fs in Sx for s in fs},key=lambda s:int(s.split("_")[1]))
M=np.zeros((m2,len(ids)),bool)
for k,sd in enumerate(ids): M[:,k]=[sd in Sx[i] for i in range(m2)]
ok=(M.astype(np.uint8)@M.astype(np.uint8).T)>0
np.fill_diagonal(ok,False); eye=np.eye(m2,dtype=bool)
big=[k for k in range(len(ids)) if M[:,k].sum()>1]
print(f"seam verts (front-visible, >=2 copies) {len(seam_v)}  copies {m2}  seams {len(big)}")

loopv=[np.array(L) for _,L in loops]
inloop=np.zeros(len(lab),bool)
for L in loopv: inloop[L]=True
interior=np.where(~inloop)[0]
frame=(Wv[:,0].min(),Wv[:,0].max(),Wv[:,1].min(),Wv[:,1].max())

def run(sig_mm, alpha, rng):
    s=sig_mm/10.0
    disp={}                                        # (loop_i) -> (n,3) displacement
    tot=0.0; cntp=0
    for li,L in enumerate(loopv):
        n=len(L); w=rng.normal(0,s,(n,3))
        if alpha is not None and n>3:
            w=ema_loop(w,alpha)
            r=np.sqrt((w**2).sum(1)).mean()
            r0=s*np.sqrt(2/np.pi)*np.sqrt(3)*0.9213                # E|N3| ~ s*1.5958
            r0=s*1.59577
            if r>1e-12: w*= (r0/r)                                 # restore magnitude
        disp[li]=w
        tot+=np.linalg.norm(w,axis=1).sum(); cntp+=n
    pts=np.empty((m2,3))
    for i,(v,li,pos) in enumerate(copies): pts[i]=Wv[v]+disp[li][pos]
    # occluders: every loop vertex at its (panel-specific) noised spot + interior white
    occl=[]
    for li,L in enumerate(loopv): occl.append(Wv[L]+disp[li])
    occl.append(Wv[interior]+rng.normal(0,s,(len(interior),3)))
    allp=np.vstack([pts]+occl)
    L_,_,_,step=project_lift(allp,RES,frame); Ls=L_[:m2]
    Dm=np.linalg.norm(Ls[:,None,:]-Ls[None,:,:],axis=2); Dm[eye]=np.inf
    nn=Dm.argmin(1); good=ok[np.arange(m2),nn]
    sep=(tot/cntp)*np.sqrt(2)*10
    return (good.mean(), (nn[nn]==np.arange(m2)).mean(),
            np.mean([good[M[:,k]].all() for k in big]),
            np.mean([good[M[:,k]].mean()>0.5 for k in big]),
            float(all(good[M[:,k]].mean()>0.5 for k in big)), sep)

SIG=[1,2,3,5,10,20]; ALPHAS=[(None,'백색'),(0.6,'α=0.6'),(0.3,'α=0.3'),(0.1,'α=0.1')]
R=6
print(f"\n{'σ':>5} {'평활':>7} {'벌어짐mm':>9} {'정점':>7} {'상호NN':>7} {'seam전량':>9} {'seam투표':>9} {'벌':>6}")
print("-"*70)
res={}
for s in SIG:
    for a,nm in ALPHAS:
        rng=np.random.default_rng(101)
        v=np.mean([run(s,a,rng) for _ in range(R)],axis=0)
        res[(s,nm)]=v
        print(f"{s:>4}mm {nm:>7} {v[5]:>9.1f} {v[0]:>7.4f} {v[1]:>7.3f} {v[2]:>9.4f} {v[3]:>9.4f} {v[4]:>6.2f}")
    print()
json.dump({f"{k[0]}|{k[1]}":[float(x) for x in v] for k,v in res.items()},
          open(os.path.expanduser("~/mnt/seam-correspondence/ema_sweep.json"),"w"))
