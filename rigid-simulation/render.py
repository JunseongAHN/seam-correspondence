"""Render the input/output PLY pairs in result/ to PNG.

  python render.py
"""

import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import mesh as meshmod

HERE = os.path.dirname(os.path.abspath(__file__))
RESULT = os.path.join(HERE, "result")


def draw(ax, P, F, title):
    ax.plot_trisurf(P[:, 0], P[:, 1], P[:, 2], triangles=F,
                    linewidth=0.08, edgecolor=(0, 0, 0, 0.25),
                    color=(0.42, 0.62, 0.85), shade=True, antialiased=True)
    c = P.mean(0)
    r = float(np.abs(P - c).max()) * 1.05
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_zlim(c[2] - r, c[2] + r)
    ax.set_box_aspect((1, 1, 1))
    ax.set_title(title, fontsize=9)
    ax.tick_params(labelsize=6)


def main():
    pairs = sorted(glob.glob(os.path.join(RESULT, "*_input.ply")))
    if not pairs:
        print("nothing in %s -- run run_case1.py / run_case2.py first" % RESULT)
        return
    for ip in pairs:
        tag = os.path.basename(ip)[:-len("_input.ply")]
        op = os.path.join(RESULT, tag + "_output.ply")
        if not os.path.exists(op):
            continue
        Pi, Fi = meshmod.read_ply(ip)
        Po, Fo = meshmod.read_ply(op)
        fig = plt.figure(figsize=(11, 5))
        a1 = fig.add_subplot(1, 2, 1, projection="3d")
        a2 = fig.add_subplot(1, 2, 2, projection="3d")
        draw(a1, Pi, Fi, "input: flat panel + random z perturbation")
        draw(a2, Po, Fo, "output: %s" % tag)
        a2.view_init(elev=22, azim=35)
        fig.suptitle(tag, fontsize=11)
        fig.tight_layout()
        out = os.path.join(RESULT, tag + ".png")
        fig.savefig(out, dpi=130)
        plt.close(fig)
        print("wrote %s  (%d verts, %d faces)" % (out, len(Po), len(Fo)))


if __name__ == "__main__":
    main()
