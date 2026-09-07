"""Per-panel boundary CONTOURS in 3D, for the viewer.

For each panel: take its faces, find edges used by exactly one of those faces
(= the panel's boundary), chain them into closed loops.  Each panel owns its own
copy of a shared seam vertex, so applying independent N(0, s^2 I) per panel makes
the two loops of a seam separate — which is the thing to look at.
"""
import numpy as np, os, json, sys
from collections import defaultdict
from panel_depth import load_full, panel_sets
AX_Z=2

g = sys.argv[1] if len(sys.argv)>1 else "rand_00YONAPXZE"
gd = os.path.expanduser("~/mnt/data/"+g)
Wv, Wf, lab = load_full(gd, g)
panels, memb, is_seam = panel_sets(Wf, lab)
print(f"{g}: verts {len(lab)}  faces {len(Wf)}  panels {len(panels)}")

def loops_of(faces):
    cnt = defaultdict(int); 
    for f in faces:
        for a,b in ((f[0],f[1]),(f[1],f[2]),(f[2],f[0])):
            cnt[(min(a,b),max(a,b))] += 1
    bedges = [e for e,c in cnt.items() if c==1]
    if not bedges: return []
    adj = defaultdict(list)
    for a,b in bedges: adj[a].append(b); adj[b].append(a)
    seen=set(); out=[]
    for start in list(adj):
        if start in seen: continue
        loop=[start]; seen.add(start); prev=None; cur=start
        while True:
            nxts=[v for v in adj[cur] if v!=prev and v not in seen]
            if not nxts:
                if start in adj[cur] and len(loop)>2: pass
                break
            nxt=nxts[0]; loop.append(nxt); seen.add(nxt); prev,cur=cur,nxt
        if len(loop)>=4: out.append(loop)
    return out

out_panels=[]
for k,p in enumerate(panels):
    own=np.array([k in memb[v] for v in range(len(lab))])
    fsel=own[Wf].all(1)
    if fsel.sum()<4: continue
    ls=loops_of(Wf[fsel])
    if not ls: continue
    ls.sort(key=len, reverse=True)
    pts=[[[round(float(c),2) for c in Wv[v]] for v in L] for L in ls[:3]]
    allz=[Wv[v][AX_Z] for L in ls for v in L]
    out_panels.append(dict(name=p, loops=pts, zmed=round(float(np.median(allz)),2),
                           n=int(sum(len(L) for L in ls[:3]))))
    print(f"  {p:<18} loops={len(ls)}  pts={sum(len(L) for L in ls[:3]):>4}  zmed={np.median(allz):>7.2f}")

# faint context cloud
rs=np.random.default_rng(2); pk=rs.choice(len(Wv), 2500, replace=False)
lo=Wv.min(0); hi=Wv.max(0)
out=dict(garment=g, panels=out_panels,
         cloud=[[round(float(c),1) for c in Wv[i]] for i in pk],
         bbox=dict(lo=[round(float(x),2) for x in lo], hi=[round(float(x),2) for x in hi]),
         center=[round(float(x),2) for x in (lo+hi)/2])
p=os.path.expanduser("~/mnt/seam-correspondence/contours_3d.json")
json.dump(out, open(p,"w"), separators=(",",":"))
print("panels out", len(out_panels), " total pts",
      sum(x["n"] for x in out_panels), " KB", round(os.path.getsize(p)/1024))
