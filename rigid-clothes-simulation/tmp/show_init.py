import sys, os, numpy as np
sys.path.insert(0, r'C:\repos\seam-correspondence\rigid-clothes-simulation')
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import gcd_io, plyio

TMP = r'C:\repos\seam-correspondence\rigid-clothes-simulation\tmp'
d = gcd_io.load(r'C:\Users\PC\Downloads\data\rand_00YONAPXZE')
F, pf, pr = d['faces'], d['panel_of_face'], d['panel_of_raw']
UP = lambda P: np.stack([P[:,0], P[:,2], P[:,1]], 1)   # GarmentCode y-up -> matplotlib z-up

flat = np.hstack([d['rest'], np.zeros((len(d['rest']),1))])
plyio.write_ply(os.path.join(TMP,'rest_flat.ply'), flat, F, pr)
plyio.write_ply(os.path.join(TMP,'placed_init.ply'), d['placed'], F, pr)
rng = np.random.default_rng(1)
scale = float(np.linalg.norm(np.ptp(d['placed'],axis=0)))
P0 = d['placed'] + (0.01*scale/np.sqrt(3))*rng.standard_normal(d['placed'].shape)
plyio.write_ply(os.path.join(TMP,'placed_init_seed1.ply'), P0, F, pr)

fig = plt.figure(figsize=(17,5.2))
ax = fig.add_subplot(1,4,1, projection='3d')
plyio.draw(ax, np.stack([d['rest'][:,0], d['rest'][:,1], np.zeros(len(d['rest']))],1), F, pf,
           'INPUT rest: flat pattern (cm)', elev=90, azim=-90)
for k,(az,ttl) in enumerate([(-90,'INITIAL VALUE - front'), (0,'INITIAL VALUE - side'), (-55,'INITIAL VALUE - 3/4')]):
    ax = fig.add_subplot(1,4,k+2, projection='3d')
    plyio.draw(ax, UP(d['placed']), F, pf, ttl, elev=8, azim=az)
fig.tight_layout(); fig.savefig(os.path.join(TMP,'initial.png'), dpi=125); plt.close(fig)

fig = plt.figure(figsize=(11,5.2))
for k,(P,ttl) in enumerate([(d['placed'],'INITIAL VALUE (placement)'), (d['drape'],'the drape - NOT an input, comparison only')]):
    ax = fig.add_subplot(1,2,k+1, projection='3d')
    plyio.draw(ax, UP(P), F, pf, ttl, elev=8, azim=-70)
fig.tight_layout(); fig.savefig(os.path.join(TMP,'initial_vs_drape.png'), dpi=125); plt.close(fig)
print('wrote tmp/initial.png, tmp/initial_vs_drape.png and the three PLYs')
