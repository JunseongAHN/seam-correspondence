"""Export ONE garment's corrected pipeline for visual audit.
Per sigma, ONE fixed draw: GT seam polylines, the lifted (noise->depth->lift)
polylines for both panel copies, which seams the vote got wrong, and the actual
depth map that the decode ran on."""
import numpy as np, os, json, base64, io
from PIL import Image
import depth_oracle as D
from depth_tolerance import front_visible
from pipeline_tolerance import project_lift

RES=1024; DISP=256; SIG=[0.0,1.0,2.0,3.0,5.0,10.0]
g="rand_00YONAPXZE"; gd=os.path.expanduser("~/mnt/data/"+g)
W,lab=D.load(gd); si,sets=D.seam_sets(lab)
vis=front_visible(W,si); P=W[si][vis]; S=[sets[i] for i in np.where(vis)[0]]
ids=sorted({x for fs in S for x in fs},key=lambda s:int(s.split("_")[1]))
marg=D.margins(P,S)
other=np.delete(W,si,axis=0)
base=np.repeat(P,2,axis=0); m2=len(base)
Sx=[S[i] for i in np.repeat(np.arange(len(P)),2)]
M=np.zeros((m2,len(ids)),bool)
for k,sd in enumerate(ids): M[:,k]=[sd in Sx[i] for i in range(m2)]
ok=(M.astype(np.uint8)@M.astype(np.uint8).T)>0
np.fill_diagonal(ok,False); eye=np.eye(m2,dtype=bool)
frame=(W[:,0].min(),W[:,0].max(),W[:,1].min(),W[:,1].max())

# order each seam's vertices into a path (greedy NN chain from an extreme point)
def chain(idx):
    pts=P[idx]; c=pts.mean(0); start=int(np.argmax(np.linalg.norm(pts-c,axis=1)))
    order=[start]; left=set(range(len(idx)))-{start}
    while left:
        d=np.linalg.norm(pts[list(left)]-pts[order[-1]],axis=1)
        nxt=list(left)[int(np.argmin(d))]; order.append(nxt); left.discard(nxt)
    return [idx[o] for o in order]
seam_vidx=[]   # vertex indices per seam, ordered
for k,sd in enumerate(ids):
    idx=[i for i in range(len(P)) if sd in S[i]]
    seam_vidx.append(chain(idx) if len(idx)>2 else idx)

def depth_png(pts):
    (x0,x1,y0,y1)=frame; step=max(x1-x0,y1-y0)/RES
    ix=np.clip(np.floor((pts[:,0]-x0)/step).astype(int),0,RES)
    iy=np.clip(np.floor((pts[:,1]-y0)/step).astype(int),0,RES)
    buf=np.full((RES+1,RES+1),-np.inf)
    np.maximum.at(buf,(iy,ix),pts[:,2])
    f=buf[:DISP*(RES//DISP),:DISP*(RES//DISP)].reshape(DISP,RES//DISP,DISP,RES//DISP).max(axis=(1,3))
    m=np.isfinite(f)
    z0,z1=pts[:,2].min(),pts[:,2].max()
    v=np.zeros((DISP,DISP),np.uint8); v[m]=(30+225*(f[m]-z0)/(z1-z0)).astype(np.uint8)
    a=(m*255).astype(np.uint8)
    im=Image.fromarray(np.dstack([v,v,v,a])[::-1],"RGBA")   # flip: image y down
    b=io.BytesIO(); im.save(b,"PNG",optimize=True)
    return "data:image/png;base64,"+base64.b64encode(b.getvalue()).decode()

rng=np.random.default_rng(7)
frames=[]
for s_mm in SIG:
    s=s_mm/10.0
    Qs=base+(rng.normal(0,s,(m2,3)) if s>0 else 0)
    Qo=other+(rng.normal(0,s,other.shape) if s>0 else 0)
    allp=np.vstack([Qs,Qo])
    L,_,_,step=project_lift(allp,RES,frame); Ls=L[:m2]
    Dm=np.linalg.norm(Ls[:,None,:]-Ls[None,:,:],axis=2); Dm[eye]=np.inf
    nn=Dm.argmin(1); good=ok[np.arange(m2),nn]
    seam_ok=[bool(good[M[:,k]].mean()>0.5) for k in range(len(ids))]
    seam_all=[bool(good[M[:,k]].all()) for k in range(len(ids))]
    paths=[]; vok=[]
    for k in range(len(ids)):
        vi=seam_vidx[k]
        paths.append([[[round(float(Ls[2*i+c][0]),2),round(float(Ls[2*i+c][1]),2)] for i in vi]
                      for c in (0,1)])
        vok.append([bool(good[2*i] and good[2*i+1]) for i in vi])
    frames.append(dict(sigma_mm=s_mm, seam_ok=seam_ok, seam_all=seam_all,
                       paths=paths, vert_ok=vok,
                       vert_acc=round(float(good.mean()),4),
                       mutual=round(float((nn[nn]==np.arange(m2)).mean()),4),
                       seam_vote=round(float(np.mean(seam_ok)),4),
                       seam_allrate=round(float(np.mean(seam_all)),4),
                       garment=bool(all(seam_ok)),
                       depth=depth_png(allp)))
    print(f"sigma {s_mm:>5.1f}mm  vert {good.mean():.4f}  vote {np.mean(seam_ok):.4f}  "
          f"all {np.mean(seam_all):.4f}  garment {all(seam_ok)}  broken={[k for k in range(len(ids)) if not seam_ok[k]]}")

spec=json.load(open(os.path.join(gd,g+"_specification.json")))["pattern"]["stitches"]
names=[]
for sd in ids:
    k=int(sd.split("_")[1])
    names.append(f"{spec[k][0]['panel']}·e{spec[k][0]['edge']} ↔ {spec[k][1]['panel']}·e{spec[k][1]['edge']}"
                 if k<len(spec) else sd)
rs=np.random.default_rng(1); pick=rs.choice(len(W),3000,replace=False)
out=dict(garment=g, res=RES, px_mm=round(float(step*10),2), n_seams=len(ids),
         n_pts=int(len(P)), frame=[round(float(f),2) for f in frame],
         seam_name=names, seam_size=[len(v) for v in seam_vidx],
         seam_minmargin_mm=[round(float(marg[v].min()*10),2) for v in seam_vidx],
         gt_paths=[[[round(float(P[i][0]),2),round(float(P[i][1]),2)] for i in v] for v in seam_vidx],
         silhouette=[[round(float(W[i][0]),1),round(float(W[i][1]),1)] for i in pick],
         frames=frames)
p=os.path.expanduser("~/mnt/seam-correspondence/visual_one.json")
json.dump(out,open(p,"w"),separators=(",",":"))
print("\nKB",round(os.path.getsize(p)/1024))
