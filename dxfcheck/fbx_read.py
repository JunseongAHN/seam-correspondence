"""Minimal reader for binary FBX 7.x — enough to pull meshes and their placement.

FBX binary is a tree of nodes; each node has a name, a list of typed properties,
and children.  Arrays (f/d/i/l) may be zlib-deflated.  We only need Geometry
(Vertices, PolygonVertexIndex) and Model (Lcl Translation / Rotation / Scaling),
plus Connections to tie a Geometry to its Model.
"""
import struct, zlib, sys
from collections import defaultdict

class Node:
    __slots__ = ("name", "props", "children")
    def __init__(self, name, props, children):
        self.name, self.props, self.children = name, props, children
    def find(self, name):
        return [c for c in self.children if c.name == name]


def _array(f, kind):
    n, enc, clen = struct.unpack("<III", f.read(12))
    raw = f.read(clen)
    if enc: raw = zlib.decompress(raw)
    fmt = {"f": "f", "d": "d", "i": "i", "l": "q", "b": "b"}[kind]
    return list(struct.unpack("<%d%s" % (n, fmt), raw))


def _prop(f):
    t = f.read(1).decode()
    if t == "Y": return struct.unpack("<h", f.read(2))[0]
    if t == "C": return bool(f.read(1)[0])
    if t == "I": return struct.unpack("<i", f.read(4))[0]
    if t == "F": return struct.unpack("<f", f.read(4))[0]
    if t == "D": return struct.unpack("<d", f.read(8))[0]
    if t == "L": return struct.unpack("<q", f.read(8))[0]
    if t in "fdilb": return _array(f, t)
    if t in ("S", "R"):
        n = struct.unpack("<I", f.read(4))[0]
        b = f.read(n)
        return b.decode("utf-8", "replace") if t == "S" else b
    raise ValueError("unknown property type %r" % t)


def _node(f, ver):
    W = 8 if ver >= 7500 else 4
    fmt = "<QQQ" if W == 8 else "<III"
    hdr = f.read(3 * W)
    if len(hdr) < 3 * W: return None
    end, nprops, plen = struct.unpack(fmt, hdr)
    nlen = f.read(1)[0]
    name = f.read(nlen).decode("utf-8", "replace")
    if end == 0: return None
    props = [_prop(f) for _ in range(nprops)]
    children = []
    while f.tell() < end:
        c = _node(f, ver)
        if c is None: break
        children.append(c)
    f.seek(end)
    return Node(name, props, children)


def read(path):
    f = open(path, "rb")
    hdr = f.read(27)
    ver = struct.unpack("<I", hdr[23:27])[0]
    root = []
    while True:
        n = _node(f, ver)
        if n is None: break
        root.append(n)
    return Node("root", [], root), ver


def meshes(path):
    """-> list of dicts: name, verts (N,3), polys (list of index lists), transform."""
    root, ver = read(path)
    objs = root.find("Objects")
    geo, mdl = {}, {}
    for o in objs:
        for g in o.find("Geometry"):
            gid = g.props[0]
            v = next((c.props[0] for c in g.find("Vertices")), None)
            pi = next((c.props[0] for c in g.find("PolygonVertexIndex")), None)
            if v is None: continue
            geo[gid] = dict(verts=v, idx=pi, name=g.props[1].split("\x00")[0] if len(g.props) > 1 else "")
        for m in o.find("Model"):
            mid = m.props[0]
            name = m.props[1].split("\x00")[0] if len(m.props) > 1 else ""
            t = [0.0, 0.0, 0.0]; r = [0.0, 0.0, 0.0]; s = [1.0, 1.0, 1.0]
            for p70 in m.find("Properties70"):
                for p in p70.find("P"):
                    if not p.props: continue
                    k = p.props[0]
                    if k == "Lcl Translation": t = [float(x) for x in p.props[4:7]]
                    elif k == "Lcl Rotation": r = [float(x) for x in p.props[4:7]]
                    elif k == "Lcl Scaling": s = [float(x) for x in p.props[4:7]]
            mdl[mid] = dict(name=name, t=t, r=r, s=s)
    link = {}
    for c in root.find("Connections"):
        for cc in c.find("C"):
            if len(cc.props) >= 3 and cc.props[0] == "OO":
                link[cc.props[1]] = cc.props[2]
    out = []
    for gid, g in geo.items():
        m = mdl.get(link.get(gid))
        verts = [g["verts"][i:i+3] for i in range(0, len(g["verts"]), 3)]
        polys, cur = [], []
        for i in (g["idx"] or []):
            if i < 0: cur.append(-i - 1); polys.append(cur); cur = []
            else: cur.append(i)
        out.append(dict(name=(m["name"] if m else g["name"]), verts=verts, polys=polys,
                        t=(m["t"] if m else [0,0,0]), r=(m["r"] if m else [0,0,0]),
                        s=(m["s"] if m else [1,1,1])))
    return out, ver


if __name__ == "__main__":
    ms, ver = meshes(sys.argv[1])
    print(f"FBX {ver}: {len(ms)} meshes")
    for m in ms:
        import numpy as np
        v = np.array(m["verts"])
        print(f"  {m['name'][:34]:36} verts {len(v):6} polys {len(m['polys']):6} "
              f"t=({m['t'][0]:8.1f},{m['t'][1]:8.1f},{m['t'][2]:8.1f}) "
              f"bbox {np.ptp(v,0).round(1).tolist()}")
