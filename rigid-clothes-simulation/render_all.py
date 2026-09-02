"""Renders for result/ (spec 7): every run, plus the drape comparison.

  python render_all.py
"""

import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import analyze
import gcd_io
import plyio

HERE = os.path.dirname(os.path.abspath(__file__))
RESULT = os.path.join(HERE, "result")
GARMENT = r"C:\Users\PC\Downloads\data\rand_00YONAPXZE"

UP = lambda X: np.stack([X[:, 0], X[:, 2], X[:, 1]], 1)   # GarmentCode y-up -> matplotlib z-up


def panel_figure(P, F, pf, title, path, views=((8, -90, "front"), (8, 0, "side"),
                                               (8, 90, "back"), (55, -90, "top"))):
    fig = plt.figure(figsize=(4.3 * len(views), 5.0))
    for k, (ev, az, ttl) in enumerate(views):
        ax = fig.add_subplot(1, len(views), k + 1, projection="3d")
        plyio.draw(ax, UP(P), F, pf, ttl, elev=ev, azim=az)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=115)
    plt.close(fig)
    print("wrote", os.path.basename(path))


def main():
    d = gcd_io.load(GARMENT)
    F, pf, pr = d["faces"], d["panel_of_face"], d["panel_of_raw"]

    plyio.write_ply(os.path.join(RESULT, "rest_flat.ply"),
                    np.hstack([d["rest"], np.zeros((len(d["rest"]), 1))]), F, pr)
    plyio.write_ply(os.path.join(RESULT, "placed_init.ply"), d["placed"], F, pr)
    plyio.write_ply(os.path.join(RESULT, "drape_reference.ply"), d["drape"], F, pr)
    panel_figure(d["placed"], F, pf, "initial value: specification placement",
                 os.path.join(RESULT, "render_placed_init.png"))

    runs = sorted(glob.glob(os.path.join(RESULT, "assembly_*.npy")))
    Ps = {}
    for f in runs:
        tag = os.path.basename(f)[len("assembly_"):-4]
        P = np.load(f)
        Ps[tag] = P
        panel_figure(P, F, pf, "ARAP assembly - %s" % tag,
                     os.path.join(RESULT, "render_%s.png" % tag))
        plyio.write_ply(os.path.join(RESULT, "assembly_%s_aligned.ply" % tag),
                        analyze.procrustes(P, d["drape"]), F, pr)

    if Ps:
        ref = list(Ps.values())[0]
        cols = [("placement (initial)", d["placed"]),
                ("ARAP assembly", analyze.procrustes(ref, d["drape"])),
                ("drape (comparison only)", d["drape"])]
        fig = plt.figure(figsize=(4.6 * len(cols), 5.2))
        for k, (ttl, X) in enumerate(cols):
            ax = fig.add_subplot(1, len(cols), k + 1, projection="3d")
            plyio.draw(ax, UP(X), F, pf, ttl, elev=8, azim=-70)
        fig.tight_layout()
        fig.savefig(os.path.join(RESULT, "render_vs_drape.png"), dpi=125)
        plt.close(fig)
        print("wrote render_vs_drape.png")

        # spread across runs, painted on the mean shape
        if len(Ps) > 1:
            S = np.stack([analyze.procrustes(P, ref) for P in Ps.values()])
            spread = np.linalg.norm(S.std(0), axis=1)
            M = S.mean(0)
            fv = spread[F].mean(1)
            fig = plt.figure(figsize=(13, 5.2))
            for k, (ev, az, ttl) in enumerate([(8, -90, "front"), (8, 0, "side"), (8, 90, "back")]):
                ax = fig.add_subplot(1, 3, k + 1, projection="3d")
                from mpl_toolkits.mplot3d.art3d import Poly3DCollection
                T = UP(M)[F]
                cmap = plt.get_cmap("inferno")
                cn = fv / max(np.quantile(fv, 0.98), 1e-9)
                pc = Poly3DCollection(T, facecolors=cmap(np.clip(cn, 0, 1)), edgecolors="none")
                ax.add_collection3d(pc)
                c = UP(M).mean(0)
                r = float(np.abs(UP(M) - c).max()) * 1.02
                ax.set_xlim(c[0] - r, c[0] + r); ax.set_ylim(c[1] - r, c[1] + r)
                ax.set_zlim(c[2] - r, c[2] + r); ax.set_box_aspect((1, 1, 1))
                ax.view_init(elev=ev, azim=az); ax.set_title(ttl, fontsize=9)
                ax.tick_params(labelsize=6)
            fig.suptitle("initial-value spread across %d runs (bright = less determined), cm"
                         % len(Ps), fontsize=11)
            fig.tight_layout()
            fig.savefig(os.path.join(RESULT, "render_spread.png"), dpi=125)
            plt.close(fig)
            print("wrote render_spread.png")


if __name__ == "__main__":
    main()
