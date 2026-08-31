"""Export ONE garment's seam-decode situation for visual inspection."""
import numpy as np, os, json, sys, depth_oracle as D
from depth_tolerance import front_visible
AX_Z = 2
SIG = [0.5,1.0,1.5,2.0,2.5,3.0,4.0,5.0,7.0,10.0,15.0,20.0,30.0,50.0]  # mm
R = 24                                                                 # draws per sigma

g = sys.argv[1] if len(sys.argv) > 1 else "rand_00YONAPXZE"
gd = os.path.expanduser("~/mnt/data/" + g)
W, lab = D.load(gd)
si, sets = D.seam_sets(lab)
vis = front_visible(W, si)
P = W[si][vis]; S = [sets[i] for i in np.where(vis)[0]]
m = D.margins(P, S)
ids = sorted({x for fs in S for x in fs}, key=lambda s: int(s.split("_")[1]))
sid_of = {s: k for k, s in enumerate(ids)}

base = np.repeat(P, 2, axis=0); owner = np.repeat(np.arange(len(P)), 2)
Sx = [S[i] for i in owner]; m2 = len(base)
ok = np.zeros((m2, m2), bool)
for i in range(m2):
    a = Sx[i]; ok[i] = [not a.isdisjoint(Sx[j]) for j in range(m2)]
np.fill_diagonal(ok, False); eye = np.eye(m2, dtype=bool)
mem = {s: np.array([s in Sx[i] for i in range(m2)]) for s in ids}

rng = np.random.default_rng(7)
vert_fail = []      # [sigma][vertex] failure rate
seam_fail = []      # [sigma][seam]   failure rate (seam not fully correct)
for s_mm in SIG:
    s = s_mm / 10.0                                   # cm
    vf = np.zeros(len(P)); sf = np.zeros(len(ids))
    for _ in range(R):
        Q = base.copy(); Q[:, AX_Z] += rng.normal(0, s, m2)
        Dm = np.linalg.norm(Q[:, None, :] - Q[None, :, :], axis=2); Dm[eye] = np.inf
        good = ok[np.arange(m2), Dm.argmin(1)]
        vf += (~good).reshape(-1, 2).any(1)
        for k, sd in enumerate(ids):
            sf[k] += not good[mem[sd]].all()
    vert_fail.append((vf / R).round(3).tolist())
    seam_fail.append((sf / R).round(3).tolist())

# context silhouette: subsample the full welded mesh
rs = np.random.default_rng(1)
pick = rs.choice(len(W), size=min(4000, len(W)), replace=False)
sil = W[pick]

out = dict(
    garment=g, n_seams=len(ids), n_seam_vertices=int(len(P)),
    sigma_mm=SIG, draws=R,
    bbox=dict(x=[float(W[:,0].min()), float(W[:,0].max())],
              y=[float(W[:,1].min()), float(W[:,1].max())],
              z=[float(W[:,2].min()), float(W[:,2].max())]),
    silhouette=[[round(float(a),1), round(float(b),1), round(float(c),1)] for a,b,c in sil],
    seam_pts=[[round(float(p[0]),2), round(float(p[1]),2), round(float(p[2]),2)] for p in P],
    seam_margin_mm=[round(float(x*10),2) for x in m],
    seam_of_pt=[[sid_of[s] for s in sorted(S[i], key=lambda z:int(z.split('_')[1]))] for i in range(len(P))],
    seam_ids=[int(s.split("_")[1]) for s in ids],
    seam_size=[int(sum(1 for i in range(len(P)) if ids[k] in S[i])) for k in range(len(ids))],
    seam_minmargin_mm=[round(float(m[[ids[k] in S[i] for i in range(len(P))]].min()*10),2)
                       for k in range(len(ids))],
    vert_fail=vert_fail, seam_fail=seam_fail,
)
json.dump(out, open(os.path.expanduser(f"~/mnt/seam-correspondence/one_{g}.json"), "w"),
          separators=(",", ":"))
print(g, "seams", len(ids), "front-visible seam verts", len(P))
print("seam-level survival by sigma:")
for k, s in enumerate(SIG):
    print(f"  {s:>5.1f}mm  seams intact {1-np.mean(seam_fail[k]):.3f}   verts ok {1-np.mean(vert_fail[k]):.3f}")
