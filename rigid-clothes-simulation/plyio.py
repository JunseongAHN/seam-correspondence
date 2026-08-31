"""ASCII PLY with a per-vertex panel id, plus a matplotlib renderer."""

import numpy as np

PALETTE = np.array([
    [0.85, 0.33, 0.31], [0.30, 0.52, 0.74], [0.45, 0.68, 0.36], [0.90, 0.62, 0.24],
    [0.60, 0.40, 0.72], [0.35, 0.71, 0.68], [0.85, 0.55, 0.70], [0.55, 0.55, 0.55],
    [0.72, 0.75, 0.28], [0.25, 0.45, 0.55], [0.95, 0.78, 0.35], [0.50, 0.30, 0.25],
    [0.40, 0.62, 0.85], [0.78, 0.42, 0.55], [0.30, 0.65, 0.45], [0.65, 0.65, 0.85],
    [0.88, 0.48, 0.20], [0.20, 0.35, 0.30],
])


def write_ply(path, P, faces, panel=None):
    P = np.asarray(P, float)
    faces = np.asarray(faces, np.int64)
    has = panel is not None
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write("element vertex %d\n" % len(P))
        f.write("property float x\nproperty float y\nproperty float z\n")
        if has:
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
            f.write("property int panel\n")
        f.write("element face %d\n" % len(faces))
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        if has:
            c = (PALETTE[np.asarray(panel) % len(PALETTE)] * 255).astype(int)
            for p, col, pid in zip(P, c, panel):
                f.write("%.6f %.6f %.6f %d %d %d %d\n" % (p[0], p[1], p[2], col[0], col[1], col[2], pid))
        else:
            for p in P:
                f.write("%.6f %.6f %.6f\n" % (p[0], p[1], p[2]))
        for t in faces:
            f.write("3 %d %d %d\n" % (t[0], t[1], t[2]))


def read_ply(path):
    lines = open(path).read().split("\n")
    nv = nf = 0
    props = 0
    h = 0
    for i, ln in enumerate(lines):
        if ln.startswith("element vertex"):
            nv = int(ln.split()[-1])
        elif ln.startswith("element face"):
            nf = int(ln.split()[-1])
        elif ln.startswith("property") and nf == 0:
            props += 1
        elif ln.strip() == "end_header":
            h = i + 1
            break
    P = np.array([[float(x) for x in lines[h + i].split()[:3]] for i in range(nv)])
    F = np.array([[int(x) for x in lines[h + nv + i].split()[1:4]] for i in range(nf)], np.int64)
    return P, F


def draw(ax, P, faces, panel_of_face, title, elev=18, azim=-70, lw=0.0):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    tris = P[faces]
    col = PALETTE[np.asarray(panel_of_face) % len(PALETTE)]
    # cheap lambert shading so the form reads
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1)[:, None], 1e-12)
    sh = 0.55 + 0.45 * np.abs(n @ np.array([0.3, 0.4, 0.87]))
    pc = Poly3DCollection(tris, facecolors=np.clip(col * sh[:, None], 0, 1),
                          edgecolors="none", linewidths=lw)
    ax.add_collection3d(pc)
    c = P.mean(0)
    r = float(np.abs(P - c).max()) * 1.02
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_zlim(c[2] - r, c[2] + r)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(title, fontsize=9)
    ax.tick_params(labelsize=6)
