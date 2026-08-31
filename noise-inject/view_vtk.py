#!/usr/bin/env python
"""VTK viewer — panel surfaces + boundary contours under injected error.

Three viewports on one shared camera:
    GT   |   white N(0, s^2 I)   |   smoothed, same RMS

The smoothed field is the SAME draw low-passed over the panel's own mesh graph
(the surface analogue of the contour EMA), then rescaled so the mean per-point
displacement matches the white draw.  So the two right panes differ in SHAPE
only, not in size.  Smoothing is applied to the DISPLACEMENT, never to the
coordinates, so sigma = 0 is exactly the GT however hard you smooth.

    pip install vtk
    python view_vtk.py
    python view_vtk.py --garment rand_023FMIGQK0 --back

keys   b back panels   g GT overlay   w wireframe   e edges   r reset   q quit
"""
import argparse, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ninject.io_gcd import load, panel_loops, panel_membership, AX_Z

DEF_ROOT = r"C:\Users\PC\Downloads\data"
COL_FRONT = (0.22, 0.53, 0.90)
COL_BACK  = (0.85, 0.35, 0.15)
COL_GT    = (0.55, 0.54, 0.51)
BG        = (0.086, 0.086, 0.102)


# ---------------------------------------------------------------- geometry
class Panel:
    __slots__ = ("name", "front", "V", "F", "loops", "nbr_idx", "nbr_ptr", "disp")

    def __init__(self, name, V, F, loops):
        self.name = name
        self.V = V
        self.F = F
        self.loops = loops
        self.front = float(np.median(V[:, AX_Z])) >= 0.0
        # neighbour lists (CSR) for smoothing the displacement over the panel
        pairs = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
        pairs = np.vstack([pairs, pairs[:, ::-1]])
        order = np.argsort(pairs[:, 0], kind="stable")
        p = pairs[order]
        self.nbr_idx = p[:, 1].astype(np.int64)
        counts = np.bincount(p[:, 0], minlength=len(V))
        self.nbr_ptr = np.r_[0, np.cumsum(counts)].astype(np.int64)
        self.disp = np.zeros_like(V)


def build_panels(root, garment):
    gd = os.path.join(root, garment)
    W, F, labels = load(gd, garment)
    panels, loops = panel_loops(W, F, labels)
    _, memb, _ = panel_membership(F, labels)
    by = {}
    for pk, L in loops:
        by.setdefault(pk, []).append(L)
    out = []
    for pk in sorted(by):
        own = np.array([pk in memb[v] for v in range(len(labels))])
        sel = own[F].all(1)
        if sel.sum() < 4:
            continue
        vids = np.unique(F[sel])
        idx = {v: i for i, v in enumerate(vids)}
        V = W[vids]
        Fl = np.array([[idx[a], idx[b], idx[c]] for a, b, c in F[sel]], np.int64)
        ls = [[idx[v] for v in L if v in idx] for L in sorted(by[pk], key=len, reverse=True)[:3]]
        out.append(Panel(panels[pk], V, Fl, [l for l in ls if len(l) >= 4]))
    lo, hi = W.min(0), W.max(0)
    return out, (lo + hi) / 2, W


def smooth_disp(P, d, w, iters):
    """d <- (1-w) d + w * mean(neighbours), repeated; then restore the magnitude."""
    if iters <= 0 or w <= 0:
        return d
    m0 = np.linalg.norm(d, axis=1).mean()
    x = d.copy()
    for _ in range(iters):
        acc = np.add.reduceat(x[P.nbr_idx], P.nbr_ptr[:-1], axis=0)
        cnt = np.maximum(np.diff(P.nbr_ptr), 1)[:, None]
        x = (1 - w) * x + w * (acc / cnt)
    m1 = np.linalg.norm(x, axis=1).mean()
    return x * (m0 / m1) if m1 > 1e-12 else x


# ---------------------------------------------------------------- vtk scene
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=DEF_ROOT)
    ap.add_argument("--garment", default="rand_00YONAPXZE")
    ap.add_argument("--sigma", type=float, default=10.0, help="mm")
    ap.add_argument("--smooth", type=int, default=12, help="smoothing iterations")
    ap.add_argument("--seed", type=int, default=20260831)
    ap.add_argument("--back", action="store_true", help="show back panels too")
    ap.add_argument("--screenshot", default=None, help="render offscreen to this png and exit")
    ap.add_argument("--size", default="1680x820")
    a = ap.parse_args()

    try:
        import vtk
    except ImportError:
        sys.exit("vtk is not installed.   pip install vtk")

    panels, center, W = build_panels(a.root, a.garment)
    nf = sum(len(p.F) for p in panels)
    print(f"{a.garment}: panels {len(panels)}  verts {sum(len(p.V) for p in panels)}  faces {nf}")

    state = dict(sigma=a.sigma, iters=a.smooth, back=a.back, gt=True, wire=False)
    rng = np.random.default_rng(a.seed)

    def regen():
        rng2 = np.random.default_rng(a.seed)
        s = state["sigma"] / 10.0
        for P in panels:
            P.disp = rng2.normal(0.0, s, P.V.shape) if s > 0 else np.zeros_like(P.V)

    regen()

    from vtkmodules.util import numpy_support

    # ---- vtk polydata builders
    def make_poly(V, F):
        pts = vtk.vtkPoints()
        pts.SetData(vtk_np(V))
        cells = vtk.vtkCellArray()
        arr = np.empty((len(F), 4), np.int64)
        arr[:, 0] = 3; arr[:, 1:] = F
        if hasattr(cells, "SetData"):
            off = numpy_support.numpy_to_vtkIdTypeArray(
                np.arange(0, 3 * len(F) + 1, 3, dtype=np.int64), deep=1)
            con = numpy_support.numpy_to_vtkIdTypeArray(
                np.ascontiguousarray(F.astype(np.int64).ravel()), deep=1)
            cells.SetData(off, con)
        else:
            ida = numpy_support.numpy_to_vtkIdTypeArray(
                np.ascontiguousarray(arr.ravel()), deep=1)
            cells.SetCells(len(F), ida)
        pd = vtk.vtkPolyData(); pd.SetPoints(pts); pd.SetPolys(cells)
        return pd

    def make_lines(V, loops):
        pts = vtk.vtkPoints(); pts.SetData(vtk_np(V))
        cells = vtk.vtkCellArray()
        for L in loops:
            cells.InsertNextCell(len(L) + 1)
            for i in L:
                cells.InsertCellPoint(int(i))
            cells.InsertCellPoint(int(L[0]))
        pd = vtk.vtkPolyData(); pd.SetPoints(pts); pd.SetLines(cells)
        return pd

    def vtk_np(V):
        return numpy_support.numpy_to_vtk(np.ascontiguousarray(V - center), deep=1)

    # ---- three renderers, one camera
    W_, H_ = (int(x) for x in a.size.lower().split("x"))
    renwin = vtk.vtkRenderWindow()
    renwin.SetSize(W_, H_)
    renwin.SetWindowName(f"noise-inject — {a.garment}")
    if a.screenshot:
        renwin.SetOffScreenRendering(1)

    titles = ["GT", f"white  N(0, s^2 I)", "smoothed  ·  same RMS"]
    rends, actors = [], []
    for k in range(3):
        r = vtk.vtkRenderer()
        r.SetViewport(k / 3.0, 0.0, (k + 1) / 3.0, 1.0)
        r.SetBackground(*BG)
        r.SetLayer(0)
        r.SetUseDepthPeeling(0)
        renwin.AddRenderer(r)
        rends.append(r)
        group = []
        for P in panels:
            col = COL_FRONT if P.front else COL_BACK
            surf = make_poly(P.V, P.F)
            sm = vtk.vtkPolyDataMapper(); sm.SetInputData(surf)
            sa = vtk.vtkActor(); sa.SetMapper(sm)
            sa.GetProperty().SetColor(*col)
            sa.GetProperty().SetOpacity(0.92)
            sa.GetProperty().SetInterpolationToGouraud()
            sa.GetProperty().SetAmbient(0.30); sa.GetProperty().SetDiffuse(0.75)
            sa.GetProperty().SetSpecular(0.10)
            sa.GetProperty().BackfaceCullingOff()
            r.AddActor(sa)

            ln = make_lines(P.V, P.loops)
            lm = vtk.vtkPolyDataMapper(); lm.SetInputData(ln)
            la = vtk.vtkActor(); la.SetMapper(lm)
            la.GetProperty().SetColor(1.0, 1.0, 1.0)
            la.GetProperty().SetLineWidth(2.0)
            r.AddActor(la)

            gt = make_lines(P.V, P.loops)
            gm = vtk.vtkPolyDataMapper(); gm.SetInputData(gt)
            ga = vtk.vtkActor(); ga.SetMapper(gm)
            ga.GetProperty().SetColor(*COL_GT)
            ga.GetProperty().SetLineWidth(1.6)
            r.AddActor(ga)
            group.append(dict(P=P, surf=surf, ln=ln, gt=gt, sa=sa, la=la, ga=ga))
        actors.append(group)

        t = vtk.vtkTextActor()
        t.SetInput(titles[k])
        t.GetTextProperty().SetFontSize(19)
        t.GetTextProperty().SetColor(0.88, 0.88, 0.85)
        t.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
        t.GetPositionCoordinate().SetValue(0.05, 0.945)
        r.AddViewProp(t)

    renwin.SetNumberOfLayers(2)
    overlay = vtk.vtkRenderer()
    overlay.SetLayer(1)
    overlay.InteractiveOff()
    overlay.SetViewport(0.0, 0.0, 1.0, 1.0)
    renwin.AddRenderer(overlay)
    info = vtk.vtkTextActor()
    info.GetTextProperty().SetFontSize(15)
    info.GetTextProperty().SetColor(0.80, 0.80, 0.77)
    info.SetDisplayPosition(int(W_ * 0.30), 22)
    overlay.AddViewProp(info)

    keys = vtk.vtkTextActor()
    keys.SetInput("b back panels   g GT overlay   w wireframe   r reset   q quit")
    keys.GetTextProperty().SetFontSize(13)
    keys.GetTextProperty().SetColor(0.55, 0.55, 0.53)
    keys.SetDisplayPosition(int(W_ * 0.30), 46)
    overlay.AddViewProp(keys)

    cam = rends[0].GetActiveCamera()
    for r in rends[1:]:
        r.SetActiveCamera(cam)

    # ---- update
    def apply():
        s = state["sigma"] / 10.0
        w, it = 0.5, state["iters"]
        rawtot = smtot = cnt = 0.0
        for gi, group in enumerate(actors):
            for g in group:
                P = g["P"]
                show = state["back"] or P.front
                g["sa"].SetVisibility(show)
                g["la"].SetVisibility(show)
                g["ga"].SetVisibility(show and state["gt"] and gi > 0)
                g["sa"].GetProperty().SetRepresentationToWireframe() if state["wire"] \
                    else g["sa"].GetProperty().SetRepresentationToSurface()
                if not show:
                    continue
                if gi == 0:
                    V = P.V
                elif gi == 1:
                    V = P.V + P.disp
                else:
                    V = P.V + smooth_disp(P, P.disp, w, it)
                g["surf"].GetPoints().SetData(vtk_np(V))
                g["ln"].GetPoints().SetData(vtk_np(V))
                g["surf"].Modified(); g["ln"].Modified()
                if gi == 1:
                    rawtot += np.linalg.norm(P.disp, axis=1).sum(); cnt += len(P.V)
                elif gi == 2:
                    smtot += np.linalg.norm(smooth_disp(P, P.disp, w, it), axis=1).sum()
        sep_w = (rawtot / cnt) * np.sqrt(2) * 10 if cnt else 0
        sep_s = (smtot / cnt) * np.sqrt(2) * 10 if cnt else 0
        info.SetInput(f"sigma {state['sigma']:.1f} mm   smoothing {state['iters']} it   "
                      f"separation  white {sep_w:.1f} mm | smooth {sep_s:.1f} mm   "
                      f"[{'back+front' if state['back'] else 'front only'}]")
        renwin.Render()

    # ---- interactor + sliders
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renwin)
    style = vtk.vtkInteractorStyleTrackballCamera()
    iren.SetInteractorStyle(style)

    def slider(title, lo, hi, val, y, cb):
        rep = vtk.vtkSliderRepresentation2D()
        rep.SetMinimumValue(lo); rep.SetMaximumValue(hi); rep.SetValue(val)
        rep.SetTitleText(title)
        rep.GetPoint1Coordinate().SetCoordinateSystemToNormalizedDisplay()
        rep.GetPoint1Coordinate().SetValue(0.035, y)
        rep.GetPoint2Coordinate().SetCoordinateSystemToNormalizedDisplay()
        rep.GetPoint2Coordinate().SetValue(0.235, y)
        rep.SetSliderLength(0.02); rep.SetSliderWidth(0.02)
        rep.SetEndCapLength(0.008); rep.SetTubeWidth(0.006)
        rep.SetLabelFormat("%.1f")
        for p in (rep.GetSliderProperty(), rep.GetSelectedProperty()):
            p.SetColor(0.30, 0.62, 0.92)
        rep.GetTubeProperty().SetColor(0.32, 0.32, 0.32)
        rep.GetCapProperty().SetColor(0.5, 0.5, 0.5)
        rep.GetTitleProperty().SetColor(0.85, 0.85, 0.82)
        rep.GetTitleProperty().SetFontSize(11)
        rep.GetTitleProperty().ShadowOff()
        rep.GetLabelProperty().SetColor(0.85, 0.85, 0.82)
        rep.GetLabelProperty().SetFontSize(11)
        rep.GetLabelProperty().ShadowOff()
        w_ = vtk.vtkSliderWidget()
        w_.SetInteractor(iren); w_.SetRepresentation(rep)
        w_.SetAnimationModeToAnimate(); w_.EnabledOn()
        w_.AddObserver("InteractionEvent", cb)
        return w_

    def on_sigma(obj, ev):
        state["sigma"] = obj.GetRepresentation().GetValue(); regen(); apply()

    def on_smooth(obj, ev):
        state["iters"] = int(round(obj.GetRepresentation().GetValue())); apply()

    s1 = slider("sigma (mm)", 0.0, 30.0, state["sigma"], 0.175, on_sigma)
    s2 = slider("smoothing (iterations)", 0.0, 60.0, state["iters"], 0.085, on_smooth)

    def on_key(obj, ev):
        k = obj.GetKeySym().lower()
        if k == "b": state["back"] = not state["back"]
        elif k == "g": state["gt"] = not state["gt"]
        elif k in ("w", "e"): state["wire"] = not state["wire"]
        elif k == "r":
            for r in rends[:1]:
                r.ResetCamera()
        else:
            return
        apply()
    iren.AddObserver("KeyPressEvent", on_key)

    rends[0].ResetCamera()
    cam.Azimuth(28); cam.Elevation(8); cam.Zoom(1.25)
    apply()

    if a.screenshot:
        renwin.Render()
        w2i = vtk.vtkWindowToImageFilter(); w2i.SetInput(renwin); w2i.Update()
        wr = vtk.vtkPNGWriter(); wr.SetFileName(a.screenshot)
        wr.SetInputConnection(w2i.GetOutputPort()); wr.Write()
        print("wrote", a.screenshot)
        return

    print("keys:  b back panels   g GT overlay   w wireframe   r reset   q quit")
    iren.Initialize()
    iren.Start()


if __name__ == "__main__":
    main()
