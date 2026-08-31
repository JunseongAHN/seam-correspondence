import sys, numpy as np
sys.path.insert(0, r'C:\repos\seam-correspondence\rigid-clothes-simulation')
import gcd_io, assembly as A
from collections import defaultdict
d = gcd_io.load(r'C:\Users\PC\Downloads\data\rand_00YONAPXZE')
faces, wid, rest = d['faces'], d['wid'], d['rest']
pn = np.array(d['panel_names'], dtype=object)

# welded edges owned by 2+ panels, with the per-panel rest length
seen = defaultdict(dict)
for t,f in enumerate(faces):
    p = d['panel_of_face'][t]
    for a,b in ((f[0],f[1]),(f[1],f[2]),(f[2],f[0])):
        seen[(min(wid[a],wid[b]), max(wid[a],wid[b]))][p] = float(np.linalg.norm(rest[a]-rest[b]))
ratio, keys = [], []
for k,v in seen.items():
    if len(v)>=2:
        L=sorted(v.values()); ratio.append(L[-1]/L[0]); keys.append(k)
ratio=np.array(ratio)
RUFFLE_T = 1.05                                    # >5% = designed gathering, not a match error
print('seam edges: %d | designed-ruffle (ratio > %.2f): %d (%.2f%% of seam edges)'
      % (len(ratio), RUFFLE_T, (ratio>RUFFLE_T).sum(), 100*(ratio>RUFFLE_T).mean()))
print('  non-ruffle seam edges: length-ratio p50 %.5f  p99 %.5f  max %.5f'
      % tuple(np.quantile(ratio[ratio<=RUFFLE_T],[.5,.99,1.0])))
print('  ruffled seam edges   : length-ratio p50 %.5f  max %.5f  (design top_ruffle = 1.84426)'
      % (np.median(ratio[ratio>RUFFLE_T]), ratio[ratio>RUFFLE_T].max()))

# mark faces touching a ruffled seam edge
bad_w = set()
for r,k in zip(ratio, keys):
    if r > RUFFLE_T: bad_w.add(k[0]); bad_w.add(k[1])
touch = np.array([any(w in bad_w for w in wid[f]) for f in faces])
print('  faces touching a ruffled seam: %d (%.3f%% of mesh)' % (touch.sum(), 100*touch.mean()))

P = np.load(r'C:\repos\seam-correspondence\rigid-clothes-simulation\result\assembly_fast_seed1.npy')
G, area = A.shape_gradients(rest[faces])
F = A.deformation_gradients(P, faces, G); _, s = A.best_rotations(F)
dev = np.abs(s-1).max(1)
print()
print('max|sigma-1|  ALL faces                    : %.3e' % dev.max())
print('max|sigma-1|  EXCLUDING ruffled-seam faces : %.3e' % dev[~touch].max())
print('   p50 %.3e  p99 %.3e  p99.9 %.3e' % tuple(np.quantile(dev[~touch],[.5,.99,.999])))
print('   ruffled-seam faces only: p50 %.3e  max %.3e' % (np.median(dev[touch]), dev[touch].max()))
