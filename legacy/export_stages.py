"""One garment, four pipeline stages per sigma, for the renderer.
  0 GT 경계   1 노이즈 먹인 경계(투영 전)   2 depth map   3 리프트 후 연결 결과
"""
import numpy as np, os, json, io, base64
from PIL import Image
import depth_oracle as D
from depth_tolerance import front_visible
from pipeline_tolerance import project_lift
AX_Z=2; RES=1024; DISP=320; SIG=[0.0,1.0,2.0,3.0,5.0,10.0,20.0]
g="rand_00YONAPXZE"; gd=os.path.expanduser("~/mnt/data/"+g)
W,lab=D.load(gd); si,sets=D.seam_sets(lab)
vis=front_visible(W,si); P=W[si][vis]; S=[sets[i] for i in np.where(vis)[0]]
ids=sorted({x for fs in S for x in fs},key=lambda s:int(s.split("_")[1]))
marg=D.margins(P,S); other=np.delete(W,si,axis=0)
base=np.repeat(P,2,axis=0); m2=len(base)
Sx=[S[i] for i in np.repeat(np.arange(len(P)),2)]
M=np.zeros((m2,len(ids)),bool)
for k,sd in enumerate(ids): M[:,k]=[sd in Sx[i] for i in range(m2)]
ok=(M.astype(np.uint8)@M.astype(np.uint8).T)>0
np.fill_diagonal(ok,False); eye=np.eye(m2,dtype=bool)
frame=(W[:,0].min(),W[:,0].max(),W[:,1].min(),W[:,1].max())
z0,z1=W[:,AX_Z].min(),W[:,AX_Z].max()
def chain(idx):
    pts=P[idx]; c=pts.mean(0); s=int(np.argmax(np.linalg.norm(pts-c,axis=1)))
    o=[s]; left=set(range(len(idx)))-{s}
    while left:
        d=np.linalg.norm(pts[list(left)]-pts[o[-1]],axis=1)
        n=list(left)[int(np.argmin(d))]; o.append(n); left.discard(n)
    return [idx[i] for i in o]
seam_v=[chain([i for i in range(len(P)) if sd in S[i]]) if sum(sd in S[i] for i in range(len(P)))>2
        else [i for i in range(len(P)) if sd in S[i]] for sd in ids]
def dpng(pts):
    x0,x1,y0,y1=frame; step=max(x1-x0,y1-y0)/RES
    ix=np.clip(np.floor((pts[:,0]-x0)/step).astype(int),0,RES)
    iy=np.clip(np.floor((pts[:,1]-y0)/step).astype(int),0,RES)
    buf=np.full((RES+1,RES+1),-np.inf); np.maximum.at(buf,(iy,ix),pts[:,2])
    k=RES//DISP
    f=buf[:DISP*k,:DISP*k].reshape(DISP,k,DISP,k).max(axis=(1,3))
    m=np.isfinite(f); v=np.zeros((DISP,DISP),np.uint8)
    v[m]=(35+220*np.clip((f[m]-z0)/(z1-z0),0,1)).astype(np.uint8)
    im=Image.fromarray(np.dstack([v,v,v,(m*255).astype(np.uint8)])[::-1],"RGBA")
    b=io.BytesIO(); im.save(b,"PNG",optimize=True)
    return "data:image/png;base64,"+base64.b64encode(b.getvalue()).decode()
rng=np.random.default_rng(7); frames=[]
for s_mm in SIG:
    s=s_mm/10.0
    Qs=base+(rng.normal(0,s,(m2,3)) if s>0 else 0)
    Qo=other+(rng.normal(0,s,other.shape) if s>0 else 0)
    allp=np.vstack([Qs,Qo])
    L,_,_,step=project_lift(allp,RES,frame); Ls=L[:m2]
    Dm=np.linalg.norm(Ls[:,None,:]-Ls[None,:,:],axis=2); Dm[eye]=np.inf
    nn=Dm.argmin(1); good=ok[np.arange(m2),nn]
    frames.append(dict(sigma_mm=s_mm,
      noisy=[[round(float(a),3) for a in p] for p in Qs],
      lift=[[round(float(a),3) for a in p] for p in Ls],
      nn=[int(x) for x in nn], good=[bool(x) for x in good],
      depth=dpng(allp),
      vert_acc=round(float(good.mean()),4),
      mutual=round(float((nn[nn]==np.arange(m2)).mean()),4),
      seam_allrate=round(float(np.mean([good[M[:,k]].all() for k in range(len(ids))])),4),
      seam_vote=round(float(np.mean([good[M[:,k]].mean()>0.5 for k in range(len(ids))])),4),
      seam_ok=[bool(good[M[:,k]].mean()>0.5) for k in range(len(ids))],
      garment=bool(all(good[M[:,k]].mean()>0.5 for k in range(len(ids))))))
    print(f"{s_mm:>5.1f}mm vert={good.mean():.4f} vote={frames[-1]['seam_vote']:.4f} "
          f"all={frames[-1]['seam_allrate']:.4f} wrong_links={int((~good).sum())}")
spec=json.load(open(os.path.join(gd,g+"_specification.json")))["pattern"]["stitches"]
names=[f"{spec[int(sd.split('_')[1])][0]['panel']}·e{spec[int(sd.split('_')[1])][0]['edge']} ↔ "
       f"{spec[int(sd.split('_')[1])][1]['panel']}·e{spec[int(sd.split('_')[1])][1]['edge']}"
       if int(sd.split('_')[1])<len(spec) else sd for sd in ids]
rs=np.random.default_rng(1); pk=rs.choice(len(W),3000,replace=False)
out=dict(garment=g,res=RES,px_mm=round(float(step*10),2),n_pts=int(len(P)),n_seams=len(ids),
  seam_name=names,seam_v=[[int(i) for i in v] for v in seam_v],
  seam_minmargin_mm=[round(float(marg[v].min()*10),2) for v in seam_v],
  gt=[[round(float(a),3) for a in P[i]] for i in range(len(P))],
  margin_mm=[round(float(x*10),2) for x in marg],
  silhouette=[[round(float(W[i][0]),1),round(float(W[i][1]),1)] for i in pk],
  frames=frames)
p=os.path.expanduser("~/mnt/seam-correspondence/stages_one.json")
json.dump(out,open(p,"w"),separators=(",",":"))
print("KB",round(os.path.getsize(p)/1024))
