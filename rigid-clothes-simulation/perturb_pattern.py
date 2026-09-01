"""Is the map from pattern to shell continuous?

  python perturb_pattern.py list
  python perturb_pattern.py run  <direction> <epsilon> <outdir>
  python perturb_pattern.py report <outdir>

The seed spread says the shell is a function of the pattern.  It says nothing
about whether it is a CONTINUOUS function of the pattern, and that is what a
network needs: nearby patterns must give nearby shells, or training fails
wherever they do not.

The test is a Lipschitz ratio.  Perturb the FLAT PANEL COORDINATES by eps along a
direction a design parameter would actually move, solve again, and measure
||d shell|| / ||d pattern||.  Bounded and roughly constant as eps -> 0 means
smooth.  A jump at some eps means a bifurcation, and no amount of capacity will
learn across it.

Two things make this cheap and exact.  The mesh topology never changes, so the
vertex correspondence between the base and the perturbed run is the identity --
no registration, no Chamfer distance.  And `--amp 0` turns off the random
perturbation, which removes the 1.375 cm seed noise that would otherwise be the
floor on what can be measured.

The skirt is the direction to watch.  Its panel is a flat annulus and its
assembled section is an envelope rather than a tube; raising the lambda_b start
moved that section ratio 4.92 -> 3.90, so the envelope has more than one settled
state and a small change in pattern might swap between them.
"""

import os
import sys

import numpy as np

import gcd_io
import run_garment as RG

# each direction stretches the named panels along one axis of the FLAT pattern,
# which is what a design parameter does to the pattern it generates
DIRECTIONS = {
    "skirt_length":  (("skirt",), 1),        # panel y = down the skirt
    "waist_girth":   (("wb_",), 0),          # panel x = around the body
    "sleeve_length": (("sleeve",), 1),
    "torso_girth":   (("ftorso", "btorso"), 0),
}


def perturbed_rest(d, direction, eps):
    """d["rest"] with one family of panels scaled by (1+eps) along one axis.

    Scaling is about each panel's own centroid, so the panel keeps its place in
    the layout and only its shape changes.
    """
    keys, axis = DIRECTIONS[direction]
    pn = np.array(d["panel_names"], dtype=object)[np.maximum(d["panel_of_raw"], 0)]
    R = d["rest"].copy()
    hit = np.array([any(k in str(x) for k in keys) for x in pn])
    if not hit.any():
        raise SystemExit("garment has no panel matching %s" % (keys,))
    for nm in sorted(set(str(x) for x in pn[hit])):
        k = pn == nm
        c = R[k, axis].mean()
        R[k, axis] = c + (1.0 + eps) * (R[k, axis] - c)
    return R, hit


def main():
    if len(sys.argv) < 2 or sys.argv[1] == "list":
        print("directions:", ", ".join(sorted(DIRECTIONS)))
        return

    if sys.argv[1] == "run":
        direction, eps, outdir = sys.argv[2], float(sys.argv[3]), sys.argv[4]
        os.makedirs(outdir, exist_ok=True)
        d = gcd_io.load(RG.GARMENT)
        if eps != 0.0:
            R, hit = perturbed_rest(d, direction, eps)
            np.save(os.path.join(outdir, "rest_%s_%g.npy" % (direction, eps)), R)
            print("perturbed %d vertices, mean |d pattern| = %.4f cm"
                  % (hit.sum(), np.linalg.norm(R - d["rest"], axis=1).mean()))
        print("now run:  python run_garment.py --amp 0 --body --sym --ease 5.0 "
              "--mu 0.02 --outdir %s --tag %s_%g" % (outdir, direction, eps))
        return

    if sys.argv[1] == "report":
        outdir = sys.argv[2]
        d = gcd_io.load(RG.GARMENT)
        base = np.load(os.path.join(outdir, "assembly_base_0.npy"))
        scale = float(np.linalg.norm(np.ptp(base, axis=0)))
        print("garment scale %.1f cm;  all distances are per-vertex RMS\n" % scale)
        print("  %-16s %8s %11s %11s %9s %9s"
              % ("direction", "eps", "|dpattern|", "|dshell|", "ratio", "max |dshell|"))
        import glob
        rows = []
        for f in sorted(glob.glob(os.path.join(outdir, "assembly_*.npy"))):
            tag = os.path.basename(f)[len("assembly_"):-4]
            if tag == "base_0":
                continue
            direction, e = tag.rsplit("_", 1)
            eps = float(e)
            R, _ = perturbed_rest(d, direction, eps)
            dp = float(np.sqrt((np.linalg.norm(R - d["rest"], axis=1) ** 2).mean()))
            P = np.load(f)
            dv = np.linalg.norm(P - base, axis=1)
            ds = float(np.sqrt((dv ** 2).mean()))
            rows.append((direction, eps, dp, ds, ds / max(dp, 1e-12), dv.max()))
        for r in sorted(rows):
            print("  %-16s %8.4f %11.4f %11.4f %9.2f %9.3f" % r)
        print()
        print("  A ratio that stays flat as eps shrinks is a Lipschitz constant and the")
        print("  map is smooth there.  A ratio that grows as eps shrinks means the shell")
        print("  is jumping, and training cannot cross that point.")
        return

    raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
