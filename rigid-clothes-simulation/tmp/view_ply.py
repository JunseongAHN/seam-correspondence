import sys, os, numpy as np
sys.path.insert(0, r'C:\repos\seam-correspondence\rigid-clothes-simulation')
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plyio, gcd_io
TMP = r'C:\repos\seam-correspondence\rigid-clothes-simulation\tmp'
path = sys.argv[1]
P, F = plyio.read_ply(path)                     # read back the PLY itself
d = gcd_io.load(r'C:\Users\PC\Downloads\data\rand_00YONAPXZE')
pf = d['panel_of_face']
print('read %s : %d verts, %d faces' % (os.path.basename(path), len(P), len(F)))
UP = lambda X: np.stack([X[:,0], X[:,2], X[:,1]],1)
Q = UP(P)
views = [(8,-90,'front'), (8,0,'left side'), (8,90,'back'), (8,180,'right side'),
         (60,-90,'top-down'), (-25,-90,'from below')]
fig = plt.figure(figsize=(18,11))
for k,(ev,az,ttl) in enumerate(views):
    ax = fig.add_subplot(2,3,k+1, projection='3d')
    plyio.draw(ax, Q, F, pf, ttl, elev=ev, azim=az)
fig.suptitle(os.path.basename(path), fontsize=12)
fig.tight_layout()
out = os.path.join(TMP, os.path.basename(path).replace('.ply','_views.png'))
fig.savefig(out, dpi=115); plt.close(fig)
print('wrote', out)
