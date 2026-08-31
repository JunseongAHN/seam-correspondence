"""Per-panel depth maps + panel stacking, for the front view.

The single front depth map keeps only the front-most surface, so panel identity
and the depth GAP between stacked panels are both destroyed.  This renders each
panel on its own (no z-buffer competition) and measures how much the single map
throws away.

Panel attribution: _sim_segmentation.txt already labels non-seam vertices with a
panel name.  Seam vertices carry stitch ids instead, so they inherit the panel
labels of their mesh neighbours.
"""
import numpy as np, os, json, io, base64
from PIL import Image

AX_Z=2
BARY=np.array([[1,0,0],[0,1,0],[0,0,1],[.5,.5,0],[0,.5,.5],[.5,0,.5],[1/3,1/3,1/3]])

def load_full(gdir,g):
    raw=open(os.path.join(gdir,g+"_sim.ply"),"rb").read()
    end=raw.find(b"end_header\n")+11; hdr=raw[:end].decode()
    nv=int([l for l in hdr.split("\n") if l.startswith("element vertex")][0].split()[-1])
    nf=int([l for l in hdr.split("\n") if l.startswith("element face")][0].split()[-1])
    dt=np.dtype([("x","<f4"),("y","<f4"),("z","<f4"),("s","<f8"),("t","<f8")])
    V=np.frombuffer(raw,dtype=dt,count=nv,offset=end)
    xyz=np.stack([V["x"],V["y"],V["z"]],1).astype(np.float64)
    F=np.frombuffer(raw,dtype=np.dtype([("n","u1"),("v","<i4",3)]),count=nf,
                    offset=end+nv*dt.itemsize)["v"]
    lab=np.array([l.strip() for l in open(os.path.join(gdir,g+"_sim_segmentation.txt"))],dtype=object)
    key=np.ascontiguousarray(xyz).view([("a","<f8"),("b","<f8"),("c","<f8")]).ravel()
    _,first,inv,cnt=np.unique(key,return_index=True,return_inverse=True,return_counts=True)
    order=np.argsort(first); pos=np.empty(len(cnt),np.int64); pos[order]=np.arange(len(order))
    wid=pos[inv]
    Wv=np.zeros((len(lab),3)); Wv[wid]=xyz
    return Wv, wid[F], lab              # welded verts, welded faces, labels

def panel_sets(Wf, lab):
    """panel membership per welded vertex (seam verts inherit from neighbours)"""
    n=len(lab)
    is_seam=np.array([l.startswith("stitch") for l in lab])
    panels=sorted({l for l in lab[~is_seam]})
    pidx={p:k for k,p in enumerate(panels)}
    memb=[set() for _ in range(n)]
    for i in np.where(~is_seam)[0]: memb[i].add(pidx[lab[i]])
    # propagate to seam vertices through faces (2 sweeps covers junction chains)
    for _ in range(2):
        for f in Wf:
            ps=set().union(*[memb[v] for v in f])
            for v in f:
                if is_seam[v]: memb[v] |= ps
    return panels, memb, is_seam

def render(pts_tris, frame, res):
    x0,x1,y0,y1=frame; step=max(x1-x0,y1-y0)/res
    S=np.einsum("kb,fbc->fkc",BARY,pts_tris).reshape(-1,3)
    ix=np.floor((S[:,0]-x0)/step).astype(np.int64); iy=np.floor((S[:,1]-y0)/step).astype(np.int64)
    m=(ix>=0)&(ix<res)&(iy>=0)&(iy<res)
    buf=np.full(res*res,-np.inf)
    np.maximum.at(buf,iy[m]*res+ix[m],S[m,AX_Z])
    return buf.reshape(res,res)

def png(depth,z0,z1,tint=None):
    m=np.isfinite(depth)
    v=np.zeros(depth.shape,np.uint8)
    if m.any(): v[m]=(35+220*np.clip((depth[m]-z0)/(z1-z0),0,1)).astype(np.uint8)
    a=(m*255).astype(np.uint8)
    rgb=np.dstack([v,v,v]) if tint is None else np.dstack([(v*tint[i]/255).astype(np.uint8) for i in range(3)])
    im=Image.fromarray(np.dstack([rgb,a])[::-1],"RGBA")
    b=io.BytesIO(); im.save(b,"PNG",optimize=True)
    return "data:image/png;base64,"+base64.b64encode(b.getvalue()).decode()

if __name__=="__main__":
    RES=256; g="rand_00YONAPXZE"; gd=os.path.expanduser("~/mnt/data/"+g)
    Wv,Wf,lab=load_full(gd,g)
    panels,memb,is_seam=panel_sets(Wf,lab)
    frame=(Wv[:,0].min(),Wv[:,0].max(),Wv[:,1].min(),Wv[:,1].max())
    z0,z1=Wv[:,AX_Z].min(),Wv[:,AX_Z].max()
    print(f"panels={len(panels)}  z range {z0:.1f}..{z1:.1f} cm")
    maps=[]; stack=np.zeros((RES,RES)); zmin=np.full((RES,RES),np.inf); zmax=np.full((RES,RES),-np.inf)
    for k,p in enumerate(panels):
        own=np.array([k in memb[v] for v in range(len(lab))])
        fsel=own[Wf].all(1)
        if fsel.sum()<4: 
            maps.append(None); continue
        d=render(Wv[Wf[fsel]],frame,RES)
        m=np.isfinite(d)
        stack+=m; zmin=np.minimum(zmin,np.where(m,d,np.inf)); zmax=np.maximum(zmax,np.where(m,d,-np.inf))
        zz=d[m]
        maps.append(dict(name=p, png=png(d,z0,z1), cover=float(m.mean()),
                         zmin=round(float(zz.min()),2), zmed=round(float(np.median(zz)),2),
                         zmax=round(float(zz.max()),2), faces=int(fsel.sum())))
        print(f"  {p:<18} faces={fsel.sum():>6}  cover={m.mean():.3f}  z med={np.median(zz):>7.2f}  [{zz.min():.1f},{zz.max():.1f}]")
    spread=np.where(stack>0,zmax-zmin,np.nan)
    ov=stack.copy(); ov[stack==0]=np.nan
    print(f"\npixels with >=2 panels: {(stack>=2).sum()/(stack>0).sum():.3f} of covered")
    sp=spread[np.isfinite(spread)&(stack>=2)]
    print(f"depth spread where stacked (cm): p50={np.median(sp):.2f} p90={np.percentile(sp,90):.2f} max={sp.max():.2f}")
    out=dict(res=RES, z0=round(float(z0),2), z1=round(float(z1),2),
             frame=[round(float(f),2) for f in frame],
             panels=[m for m in maps if m],
             stack_png=png(np.where(stack>0,stack,-np.inf),0,max(1,stack.max())),
             spread_png=png(np.where(stack>=2,spread,-np.inf),0,float(np.nanmax(spread))),
             stack_max=int(stack.max()),
             spread_stats=dict(p50=round(float(np.median(sp)),2),p90=round(float(np.percentile(sp,90)),2),
                               mx=round(float(sp.max()),2),
                               frac_stacked=round(float((stack>=2).sum()/(stack>0).sum()),4)))
    p=os.path.expanduser("~/mnt/seam-correspondence/panel_depth.json")
    json.dump(out,open(p,"w"),separators=(",",":"))
    print("KB",round(os.path.getsize(p)/1024))
