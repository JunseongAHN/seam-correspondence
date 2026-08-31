import sys, time, numpy as np
sys.path.insert(0, r'C:\repos\seam-correspondence\rigid-clothes-simulation')
import gcd_io, assembly as A

t0=time.time(); d = gcd_io.load(r'C:\Users\PC\Downloads\data\rand_00YONAPXZE'); print('load            %6.2fs' % (time.time()-t0))
rest_tri = d['rest'][d['faces']]
t0=time.time(); G, area = A.shape_gradients(rest_tri); print('shape_gradients %6.2fs  min area %.3e cm^2' % (time.time()-t0, area.min()))
t0=time.time(); H, R4 = A.build_hinges(d['faces'], d['wid'], rest_tri); print('build_hinges    %6.2fs  -> %d hinges' % (time.time()-t0, len(H)))
t0=time.time(); Kb, wb = A.hinge_stencils(R4); print('stencils        %6.2fs' % (time.time()-t0))
print('  stencil sanity: max|sum K| %.2e   max|sum K x| %.2e'
      % (np.abs(Kb.sum(1)).max(), np.abs(np.einsum('ha,had->hd',Kb,R4)).max()))
gar = dict(faces=d['faces'], n=len(d['rest']), G=G, area=area, hinges=H, Kb=Kb, wb=wb, pairs=d['pairs'])

t0=time.time(); asm = A.Assembly(gar, lam_b=1e-4); tA=time.time()-t0
print('Assembly (factorize + Woodbury Y)  %6.2fs   nnz(L0)=%d  Y is %.0f MB'
      % (tA, asm.L0.nnz, asm.Y.nbytes/1e6))
P = d['placed'].copy()
t0=time.time()
for _ in range(5):
    F = A.deformation_gradients(P, gar['faces'], G); R,_ = A.best_rotations(F)
    b = A.arap_rhs(R, gar['faces'], G, area, gar['n']); P = asm.solve_global(b, 1.0); P -= P.mean(0)
per = (time.time()-t0)/5
print('per local+global iteration         %6.3fs' % per)
t0=time.time(); asm.energies(P); print('per energy evaluation              %6.3fs' % (time.time()-t0))
print()
LADDER=8; PER=400; STAGES=21
iters = PER + 10*STAGES + (LADDER-1)*(PER+10)
print('one run: ~%d iterations x %.3fs + %d factorizations x %.2fs = %.1f min'
      % (iters, per+0.35, LADDER, tA, (iters*(per+0.35) + LADDER*tA)/60))
print('six runs: %.1f min' % (6*(iters*(per+0.35) + LADDER*tA)/60))
