import sys, os, glob, numpy as np
sys.path.insert(0, r'C:\repos\seam-correspondence\rigid-clothes-simulation')
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import gcd_io, plyio, analyze
R = r'C:\repos\seam-correspondence\rigid-clothes-simulation\result'
TMP = r'C:\repos\seam-correspondence\rigid-clothes-simulation\tmp'
tag = sys.argv[1] if len(sys.argv)>1 else 'fast_seed1'
d = gcd_io.load(r'C:\Users\PC\Downloads\data\rand_00YONAPXZE')
P = np.load(os.path.join(R,'assembly_%s.npy'%tag))
F, pf = d['faces'], d['panel_of_face']
UP = lambda X: np.stack([X[:,0], X[:,2], X[:,1]],1)
# align the assembly to the drape so the three panels are comparable at a glance
Pa = analyze.procrustes(P, d['drape'])
fig = plt.figure(figsize=(17,5.4))
for k,(X,ttl) in enumerate([(d['placed'],'initial: placement'),
                            (Pa,'ARAP assembly (%s)'%tag),
                            (d['drape'],'drape (comparison only)')]):
    ax = fig.add_subplot(1,3,k+1, projection='3d')
    plyio.draw(ax, UP(X), F, pf, ttl, elev=8, azim=-70)
fig.tight_layout(); fig.savefig(os.path.join(TMP,'result_%s.png'%tag), dpi=125); plt.close(fig)
print('wrote tmp/result_%s.png' % tag)
dist = np.linalg.norm(Pa - d['drape'], axis=1)
cls = analyze.class_of_raw(d)
print('distance to drape after Procrustes (cm): overall p50 %.2f  p90 %.2f' % (np.median(dist), np.quantile(dist,.9)))
for c,v in sorted(analyze.by_class(dist, cls).items()): print('   %-12s %6.2f' % (c,v))
