import sys, time, numpy as np
sys.path.insert(0, r'C:\repos\seam-correspondence\rigid-clothes-simulation')
import gcd_io, assembly as A
import scipy.sparse.linalg as spl

d = gcd_io.load(r'C:\Users\PC\Downloads\data\rand_00YONAPXZE')
rest_tri = d['rest'][d['faces']]
G, area = A.shape_gradients(rest_tri)
print('rest areas now all positive: %s  (min %.3e cm^2)' % (bool((area>0).all()), area.min()))
H, R4 = A.build_hinges(d['faces'], d['wid'], rest_tri)
Kb, wb = A.hinge_stencils(R4)
gar = dict(faces=d['faces'], n=len(d['rest']), G=G, area=area, hinges=H, Kb=Kb, wb=wb, pairs=d['pairs'])

import assembly
t0=time.time(); asm = assembly.Assembly(gar, lam_b=1e-4, woodbury=False); tF=time.time()-t0
print('factorization only          %6.2fs   nnz %d' % (tF, asm.L0.nnz))
t0=time.time(); Y = asm.lu.solve(np.asarray(asm.g and __import__('scipy.sparse',fromlist=['x']).coo_matrix(
    (np.concatenate([np.ones(len(d['pairs'])),-np.ones(len(d['pairs']))]),
     (np.concatenate([np.arange(len(d['pairs']))]*2), np.concatenate([d['pairs'][:,0],d['pairs'][:,1]]))),
    shape=(len(d['pairs']), gar['n'])).T.todense())); tY=time.time()-t0
print('Woodbury Y (967 solves)     %6.2fs   %.0f MB' % (tY, Y.nbytes/1e6))
# eigen sanity: is L0 positive definite now?
ev = spl.eigsh(asm.L0, k=2, which='SA', return_eigenvectors=False, tol=1e-6)
print('L0 smallest eigenvalues     %s  -> positive definite: %s' % (np.sort(ev), bool((ev>0).all())))

asm = assembly.Assembly(gar, lam_b=1e-4)
P = d['placed'].copy()
sched = [(1.0, 12)]
t0=time.time(); P,h,v = assembly.solve(asm, P, sched, max_iter=12); per=(time.time()-t0)/12
print('per iteration (energy incl) %6.3fs' % per)
LAD, PER, ST = 8, 400, 21
iters = PER + 10*ST + (LAD-1)*(PER+10)
one = iters*per + LAD*(tF+tY)
print()
print('one run: %d iters -> %.1f min   |  3 parallel: %.1f min for 6 runs (2 waves)'
      % (iters, one/60, 2*one/60))
