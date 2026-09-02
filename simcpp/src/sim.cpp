#include "sim.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <unordered_map>

namespace sim {

// ---------------------------------------------------------------------------
// ARAP with per-triangle rest
// ---------------------------------------------------------------------------

void shape_gradients(const std::vector<int>& faces, const std::vector<double>& rest,
                     std::vector<double>& G, std::vector<double>& area) {
  const int M = (int)faces.size() / 3;
  G.assign((size_t)M * 6, 0.0);
  area.assign(M, 0.0);
  for (int t = 0; t < M; ++t) {
    const double* x0 = &rest[(size_t)faces[t * 3 + 0] * 2];
    const double* x1 = &rest[(size_t)faces[t * 3 + 1] * 2];
    const double* x2 = &rest[(size_t)faces[t * 3 + 2] * 2];
    const double d1x = x1[0] - x0[0], d1y = x1[1] - x0[1];
    const double d2x = x2[0] - x0[0], d2y = x2[1] - x0[1];
    // E = [[d1x, d2x], [d1y, d2y]]
    const double det = d1x * d2y - d2x * d1y;
    if (std::abs(det) < 1e-14) {
      std::fprintf(stderr, "degenerate rest triangle %d: |det| = %.3e\n", t, std::abs(det));
      std::abort();
    }
    const double gjx = d2y / det, gjy = -d2x / det;   // Einv row 0
    const double gkx = -d1y / det, gky = d1x / det;   // Einv row 1
    double* g = &G[(size_t)t * 6];
    g[0] = -(gjx + gkx); g[1] = -(gjy + gky);
    g[2] = gjx;          g[3] = gjy;
    g[4] = gkx;          g[5] = gky;
    // |signed area|: back panels are parametrised mirrored, and a reflection
    // leaves the singular values of F -- hence ||F-R||^2 -- unchanged, so the
    // absolute value is exact, not a patch.  Signed areas would put negative
    // weights on half the mesh and make the stiffness matrix indefinite.
    area[t] = 0.5 * std::abs(det);
  }
}

void deformation_gradients(const Mat3X& P, const std::vector<int>& faces,
                           const std::vector<double>& G, std::vector<double>& F) {
  const int M = (int)faces.size() / 3;
  F.assign((size_t)M * 6, 0.0);
  for (int t = 0; t < M; ++t) {
    double* f = &F[(size_t)t * 6];
    const double* g = &G[(size_t)t * 6];
    for (int v = 0; v < 3; ++v) {
      const int i = faces[t * 3 + v];
      const double g0 = g[v * 2 + 0], g1 = g[v * 2 + 1];
      for (int d = 0; d < 3; ++d) {
        const double p = P(i, d);
        f[d * 2 + 0] += p * g0;
        f[d * 2 + 1] += p * g1;
      }
    }
  }
}

void best_rotations(const std::vector<double>& F, std::vector<double>& R,
                    std::vector<double>& sig) {
  const int M = (int)F.size() / 6;
  R.assign((size_t)M * 6, 0.0);
  sig.assign((size_t)M * 2, 0.0);
  for (int t = 0; t < M; ++t) {
    const double* f = &F[(size_t)t * 6];
    // C = F^T F, 2x2 SPD
    double c00 = 0.0, c01 = 0.0, c11 = 0.0;
    for (int d = 0; d < 3; ++d) {
      const double a = f[d * 2 + 0], b = f[d * 2 + 1];
      c00 += a * a; c01 += a * b; c11 += b * b;
    }
    const double tr = c00 + c11;
    const double det = c00 * c11 - c01 * c01;
    const double disc = std::sqrt(std::max(tr * tr - 4.0 * det, 0.0));
    sig[t * 2 + 0] = std::sqrt(std::max((tr + disc) * 0.5, 0.0));
    sig[t * 2 + 1] = std::sqrt(std::max((tr - disc) * 0.5, 0.0));
    // U V^T = F (F^T F)^{-1/2}; for a 2x2 SPD C,
    //   sqrt(C) = (C + sqrt(det C) I) / sqrt(tr C + 2 sqrt(det C))
    // which inverts in closed form -- no SVD needed, and no determinant fix:
    // the Stiefel manifold V_2(R^3) is connected, unlike SO(3) inside O(3).
    const double sd = std::sqrt(std::max(det, 0.0));
    const double den = std::sqrt(std::max(tr + 2.0 * sd, 1e-300));
    const double s00 = (c00 + sd) / den, s01 = c01 / den;
    const double s10 = c01 / den, s11 = (c11 + sd) / den;
    const double dS = std::max(s00 * s11 - s01 * s10, 1e-300);
    const double i00 = s11 / dS, i11 = s00 / dS, i01 = -s01 / dS, i10 = -s10 / dS;
    double* r = &R[(size_t)t * 6];
    for (int d = 0; d < 3; ++d) {
      const double a = f[d * 2 + 0], b = f[d * 2 + 1];
      r[d * 2 + 0] = a * i00 + b * i10;
      r[d * 2 + 1] = a * i01 + b * i11;
    }
  }
}

double arap_energy(const std::vector<double>& F, const std::vector<double>& R,
                   const std::vector<double>& area) {
  double e = 0.0;
  for (size_t t = 0; t < area.size(); ++t) {
    double s = 0.0;
    for (int k = 0; k < 6; ++k) {
      const double d = F[t * 6 + k] - R[t * 6 + k];
      s += d * d;
    }
    e += area[t] * s;
  }
  return e;
}

// ---------------------------------------------------------------------------
// hinges: unfold every interior edge of the WELDED topology
// ---------------------------------------------------------------------------

namespace {
struct Inc { int t[2]; int c[2]; int cnt; };
inline double dist2d(const double* a, const double* b) {
  const double dx = a[0] - b[0], dy = a[1] - b[1];
  return std::sqrt(dx * dx + dy * dy);
}
}  // namespace

void build_hinges(const std::vector<int>& faces, const std::vector<int>& wid,
                  const std::vector<double>& rest, const std::vector<int>& panel_of_face,
                  std::vector<int>& hinges, std::vector<double>& rest4) {
  const int M = (int)faces.size() / 3;
  // Topology comes from the WELDED mesh, so seams -- where the raw mesh is torn
  // into separate panels -- still produce hinges.  ALL interior edges are
  // penalised, no exceptions.  Insertion order is preserved so the hinge list
  // matches the Python dict iteration order exactly.
  std::vector<Inc> incs;
  incs.reserve((size_t)M * 3 / 2 + 16);
  std::unordered_map<uint64_t, int> idx;
  idx.reserve((size_t)M * 3);
  for (int t = 0; t < M; ++t) {
    for (int c = 0; c < 3; ++c) {
      const int a = wid[faces[t * 3 + c]], b = wid[faces[t * 3 + (c + 1) % 3]];
      const uint64_t lo = (uint64_t)(a < b ? a : b), hi = (uint64_t)(a < b ? b : a);
      const uint64_t key = (lo << 32) | hi;
      auto it = idx.find(key);
      if (it == idx.end()) {
        idx.emplace(key, (int)incs.size());
        incs.push_back(Inc{{t, -1}, {c, -1}, 1});
      } else {
        Inc& e = incs[it->second];
        if (e.cnt < 2) { e.t[1] = t; e.c[1] = c; }
        e.cnt++;
      }
    }
  }

  hinges.clear();
  rest4.clear();
  for (const Inc& e : incs) {
    if (e.cnt != 2) continue;                      // boundary or non-manifold
    const int tA = e.t[0], cA = e.c[0], tB = e.t[1], cB = e.c[1];
    const int a0 = faces[tA * 3 + cA];
    const int a1 = faces[tA * 3 + (cA + 1) % 3];
    const int a2 = faces[tA * 3 + (cA + 2) % 3];
    const int b3 = faces[tB * 3 + (cB + 2) % 3];
    // orient B's shared edge the same way as A's
    int ib0 = cB, ib1 = (cB + 1) % 3;
    if (wid[faces[tB * 3 + ib0]] != wid[a0]) std::swap(ib0, ib1);

    const double* A0 = &rest[(size_t)a0 * 2];
    const double* A1 = &rest[(size_t)a1 * 2];
    const double* A2 = &rest[(size_t)a2 * 2];
    const double* B0 = &rest[(size_t)faces[tB * 3 + ib0] * 2];
    const double* B1 = &rest[(size_t)faces[tB * 3 + ib1] * 2];
    const double* B3 = &rest[(size_t)b3 * 2];

    hinges.push_back(a0); hinges.push_back(a1);
    hinges.push_back(a2); hinges.push_back(b3);

    if (!panel_of_face.empty() && panel_of_face[tA] == panel_of_face[tB]) {
      // Same panel: the two rest frames already agree, so no unfold is needed.
      // This matters because a panel can weld two of its own distinct vertices
      // onto one point (a dart or notch), and there the welded-id matching that
      // orients the unfold is ambiguous.
      rest4.push_back(A0[0]); rest4.push_back(A0[1]);
      rest4.push_back(A1[0]); rest4.push_back(A1[1]);
      rest4.push_back(A2[0]); rest4.push_back(A2[1]);
      rest4.push_back(B3[0]); rest4.push_back(B3[1]);
    } else {
      const double LA = dist2d(A1, A0), LB = dist2d(B1, B0);
      const double L = 0.5 * (LA + LB);            // seams differ slightly; average
      const double p = dist2d(A2, A0), q = dist2d(A2, A1);
      const double pp = dist2d(B3, B0), qq = dist2d(B3, B1);
      const double X2 = (L * L + p * p - q * q) / (2 * L);
      const double X3 = (L * L + pp * pp - qq * qq) / (2 * L);
      const double Y2 = std::sqrt(std::max(p * p - X2 * X2, 1e-24));
      const double Y3 = std::sqrt(std::max(pp * pp - X3 * X3, 1e-24));
      rest4.push_back(0.0); rest4.push_back(0.0);
      rest4.push_back(L);   rest4.push_back(0.0);
      rest4.push_back(X2);  rest4.push_back(Y2);
      rest4.push_back(X3);  rest4.push_back(-Y3);
    }
  }
}

namespace {
inline double cot2(double ux, double uy, double vx, double vy) {
  const double dot = ux * vx + uy * vy;
  const double crs = std::abs(ux * vy - uy * vx);
  return dot / std::max(crs, 1e-300);
}
inline double tri_area(const double* a, const double* b, const double* c) {
  return 0.5 * std::abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]));
}
}  // namespace

void hinge_stencils(const std::vector<double>& rest4, std::vector<double>& Kb,
                    std::vector<double>& wb) {
  const int H = (int)rest4.size() / 8;
  Kb.assign((size_t)H * 4, 0.0);
  wb.assign(H, 0.0);
  for (int h = 0; h < H; ++h) {
    const double* x0 = &rest4[(size_t)h * 8 + 0];
    const double* x1 = &rest4[(size_t)h * 8 + 2];
    const double* x2 = &rest4[(size_t)h * 8 + 4];
    const double* x3 = &rest4[(size_t)h * 8 + 6];
    const double ex = x1[0] - x0[0], ey = x1[1] - x0[1];
    const double c01 = cot2(ex, ey, x2[0] - x0[0], x2[1] - x0[1]);
    const double c02 = cot2(ex, ey, x3[0] - x0[0], x3[1] - x0[1]);
    const double c03 = cot2(-ex, -ey, x2[0] - x1[0], x2[1] - x1[1]);
    const double c04 = cot2(-ex, -ey, x3[0] - x1[0], x3[1] - x1[1]);
    // Bergou quadratic bending: K is the affine dependency of the four unfolded
    // rest points (sum K = 0, sum K x = 0), scaled by the cotangent formula.
    Kb[h * 4 + 0] = c03 + c04;
    Kb[h * 4 + 1] = c01 + c02;
    Kb[h * 4 + 2] = -c01 - c03;
    Kb[h * 4 + 3] = -c02 - c04;
    const double A = tri_area(x0, x1, x2) + tri_area(x0, x1, x3);
    wb[h] = 3.0 / std::max(A, 1e-300);
  }
}

double bending_energy(const Mat3X& P, const std::vector<int>& hinges,
                      const std::vector<double>& Kb, const std::vector<double>& wb) {
  double e = 0.0;
  for (size_t h = 0; h < wb.size(); ++h) {
    double kp[3] = {0.0, 0.0, 0.0};
    for (int a = 0; a < 4; ++a) {
      const double k = Kb[h * 4 + a];
      const int i = hinges[h * 4 + a];
      for (int d = 0; d < 3; ++d) kp[d] += k * P(i, d);
    }
    e += wb[h] * (kp[0] * kp[0] + kp[1] * kp[1] + kp[2] * kp[2]);
  }
  return e;
}

Garment build_garment(const Input& in) {
  Garment g;
  g.n = in.n;
  g.faces = in.faces;
  g.pairs = in.pairs;
  shape_gradients(in.faces, in.rest, g.G, g.area);
  std::vector<double> rest4;
  build_hinges(in.faces, in.wid, in.rest, in.panel_of_face, g.hinges, rest4);
  hinge_stencils(rest4, g.Kb, g.wb);
  return g;
}

double diag_scale(const Garment& g) {
  std::vector<double> dg(g.n, 0.0);
  const int M = g.nFaces();
  for (int t = 0; t < M; ++t)
    for (int v = 0; v < 3; ++v) {
      const double g0 = g.G[(size_t)t * 6 + v * 2 + 0], g1 = g.G[(size_t)t * 6 + v * 2 + 1];
      dg[g.faces[t * 3 + v]] += g.area[t] * (g0 * g0 + g1 * g1);
    }
  double s = 0.0;
  for (double v : dg) s += v;
  return s / (double)g.n;
}

// ---------------------------------------------------------------------------
// Assembly:  L(w_s) = L0 + w_s D^T D,  L0 = K_cot + lam_b H_bend + eps I + diag
// ---------------------------------------------------------------------------

Assembly::Assembly(const Garment& gar, double lam_b_, const Vec& mu_, const Vec& nu_,
                   double eps_, bool woodbury, WoodMode mode)
    : g(gar), lam_b(lam_b_), eps(eps_), mu(mu_), nu(nu_), mode_(mode) {
  has_mu = mu.size() > 0;
  has_nu = nu.size() > 0;
  const int n = g.n, M = g.nFaces(), H = g.nHinges();

  std::vector<Eigen::Triplet<double>> T;
  T.reserve((size_t)M * 9 + (size_t)H * 16 + n);
  double abs_sum = 0.0;
  for (int t = 0; t < M; ++t) {
    const double* gg = &g.G[(size_t)t * 6];
    for (int u = 0; u < 3; ++u)
      for (int v = 0; v < 3; ++v) {
        const double w = g.area[t] * (gg[u * 2] * gg[v * 2] + gg[u * 2 + 1] * gg[v * 2 + 1]);
        abs_sum += std::abs(w);
        T.emplace_back(g.faces[t * 3 + u], g.faces[t * 3 + v], w);
      }
  }
  eps = eps_ * (abs_sum / (double)n);
  for (int h = 0; h < H; ++h)
    for (int u = 0; u < 4; ++u)
      for (int v = 0; v < 4; ++v)
        T.emplace_back(g.hinges[h * 4 + u], g.hinges[h * 4 + v],
                       lam_b * g.wb[h] * g.Kb[h * 4 + u] * g.Kb[h * 4 + v]);
  for (int i = 0; i < n; ++i) {
    // The obstacle penalty mu is CONSTANT (every constrained vertex carries it,
    // violating or not), so the matrix does not change with the active set and
    // the single factorisation stands; the active set enters only via the rhs.
    double d = eps;
    if (has_mu) d += mu[i];
    if (has_nu) d += nu[i];
    T.emplace_back(i, i, d);
  }
  SpMat L0(n, n);
  L0.setFromTriplets(T.begin(), T.end());
  L0.makeCompressed();
  ldlt_.compute(L0);
  if (ldlt_.info() != Eigen::Success) {
    std::fprintf(stderr, "SimplicialLDLT failed on L0 (lam_b = %g)\n", lam_b);
    std::abort();
  }

  const int K = g.nPairs();
  woodbury_ = woodbury && K > 0;
  if (woodbury_) {
    // D^T D has rank <= n_pairs, so one Woodbury update carries the whole w_s
    // continuation on a single factorisation of L0.  lambda_b sits inside L0,
    // so each rung of the lambda_b ladder costs one factorisation.
    std::vector<Eigen::Triplet<double>> TD;
    TD.reserve((size_t)K * 2);
    for (int k = 0; k < K; ++k) {
      TD.emplace_back(k, g.pairs[k * 2 + 0], 1.0);
      TD.emplace_back(k, g.pairs[k * 2 + 1], -1.0);
    }
    D_ = SpMat(K, n);
    D_.setFromTriplets(TD.begin(), TD.end());
    D_.makeCompressed();
    Y_ = ldlt_.solve(MatX(D_.transpose()));            // (n, K)
    MatX C = D_ * Y_;                                  // (K, K) symmetric PSD
    if (mode_ == WoodMode::Eigh) {
      // The w_s continuation only shifts the spectrum of C = D L0^-1 D^T:
      //   (C + I/w_s)^-1 = V (Lam + I/w_s)^-1 V^T
      // so ONE eigendecomposition per factorisation carries every rung of the
      // w_s ladder, instead of one dense inverse per w_s value.
      const auto t0 = std::chrono::steady_clock::now();
      Eigen::SelfAdjointEigenSolver<MatX> es(C);
      lam_ = es.eigenvalues();
      V_ = es.eigenvectors();
      rebuild_s_ += std::chrono::duration<double>(
          std::chrono::steady_clock::now() - t0).count();
    } else {
      C_ = C;                                          // rebuild the inverse per w_s
    }
  }
}

Mat3X Assembly::solve_global(const Mat3X& b, double w_s) const {
  Mat3X z = ldlt_.solve(b);
  if (!woodbury_ || w_s == 0.0) return z;
  if (mode_ == WoodMode::Eigh) {
    MatX t = V_.transpose() * (D_ * z);                // (K, 3)
    const double inv_w = 1.0 / w_s;
    for (int i = 0; i < t.rows(); ++i) t.row(i) /= (lam_[i] + inv_w);
    return z - Y_ * (V_ * t);
  }
  if (Minv_.size() == 0 || Minv_w_ != w_s) {
    const auto t0 = std::chrono::steady_clock::now();
    MatX M = C_;
    M.diagonal().array() += 1.0 / w_s;
    Minv_ = M.inverse();
    Minv_w_ = w_s;
    rebuild_s_ += std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();
  }
  return z - Y_ * (Minv_ * (D_ * z));
}

// ---------------------------------------------------------------------------
// solve / solve_annealed
// ---------------------------------------------------------------------------

void arap_rhs(const std::vector<double>& R, const std::vector<int>& faces,
              const std::vector<double>& G, const std::vector<double>& area, int n,
              Mat3X& b) {
  b.setZero(n, 3);
  const int M = (int)faces.size() / 3;
  for (int t = 0; t < M; ++t) {
    const double* r = &R[(size_t)t * 6];
    const double* g = &G[(size_t)t * 6];
    const double A = area[t];
    for (int v = 0; v < 3; ++v) {
      const int i = faces[t * 3 + v];
      const double g0 = g[v * 2 + 0], g1 = g[v * 2 + 1];
      for (int d = 0; d < 3; ++d) b(i, d) += A * (r[d * 2 + 0] * g0 + r[d * 2 + 1] * g1);
    }
  }
}

namespace {

std::vector<std::pair<double, int>> geometric_schedule(double w0, double w1, double factor,
                                                       int iters, int tail) {
  std::vector<std::pair<double, int>> s;
  double w = w0;
  while (w < w1) { s.emplace_back(w, iters); w *= factor; }
  s.emplace_back(w1, tail);
  return s;
}

struct StageOut { int iters_used; int mono; };

StageOut solve_stages(const Assembly& asmb, Mat3X& P,
                      const std::vector<std::pair<double, int>>& schedule, int max_iter,
                      double tol, const ClampFn& clamp, const Mat3X& anchor, bool recenter,
                      bool verbose, int stage_off, int it_off, double lam_b,
                      std::vector<HistRec>& hist) {
  const Garment& g = asmb.g;
  std::vector<double> F, R, sig;
  Mat3X b, Z;
  int it = 0, mono = 0;
  for (size_t stage = 0; stage < schedule.size(); ++stage) {
    const double w_s = schedule[stage].first;
    const int n_it = schedule[stage].second;
    bool have_prev = false;
    double prev = 0.0;
    for (int k = 0; k < n_it; ++k) {
      if (it >= max_iter) break;
      deformation_gradients(P, g.faces, g.G, F);
      best_rotations(F, R, sig);
      // energy at the current P with its own optimal rotations -- the same F and
      // R the global step is about to use, so no second polar decomposition
      const double e_a = arap_energy(F, R, g.area);
      const double e_b = bending_energy(P, g.hinges, g.Kb, g.wb);
      double e_s = 0.0;
      for (int p = 0; p < g.nPairs(); ++p)
        for (int d = 0; d < 3; ++d) {
          const double dd = P(g.pairs[p * 2 + 0], d) - P(g.pairs[p * 2 + 1], d);
          e_s += dd * dd;
        }
      arap_rhs(R, g.faces, g.G, g.area, g.n, b);
      double e_c = 0.0, e_n = 0.0;
      if (asmb.has_nu) {
        // two-sided anchor to the specification placement (an input, not the
        // drape): picks the point of the isometric continuum nearest the pose
        // the pattern was designed around.
        for (int i = 0; i < g.n; ++i)
          for (int d = 0; d < 3; ++d) {
            const double dd = P(i, d) - anchor(i, d);
            e_n += asmb.nu[i] * dd * dd;
            b(i, d) += asmb.nu[i] * anchor(i, d);
          }
      }
      if (asmb.has_mu) {
        // Z = the feasible point nearest P.  Adding mu|x-Z|^2 to the energy makes
        // the local step exact for a fixed Z and the global step a linear solve
        // with the SAME matrix; on free coordinates Z = P so the term is purely
        // proximal and vanishes at the fixed point, on violating ones it is the
        // half-space penalty itself.
        Z = P;
        if (clamp) clamp(Z);
        for (int i = 0; i < g.n; ++i)
          for (int d = 0; d < 3; ++d) {
            const double dd = P(i, d) - Z(i, d);
            e_c += asmb.mu[i] * dd * dd;
            b(i, d) += asmb.mu[i] * Z(i, d);
          }
      }
      const double tot = e_a + asmb.lam_b * e_b + w_s * e_s + e_c + e_n;
      P = asmb.solve_global(b, w_s);
      if (recenter) {
        for (int d = 0; d < 3; ++d) P.col(d).array() -= P.col(d).mean();
      }
      hist.push_back(HistRec{it + it_off, (int)stage + stage_off, lam_b, w_s, tot,
                             e_a, e_b, e_s, e_c, e_n});
      ++it;
      if (have_prev) {
        if (tot > prev * (1 + 1e-9) + 1e-14) ++mono;
        if (std::abs(prev - tot) <= tol * std::max(std::abs(prev), 1e-30) &&
            stage == schedule.size() - 1) {
          prev = tot;
          break;
        }
      }
      prev = tot;
      have_prev = true;
    }
    if (verbose) {
      const HistRec& e = hist.back();
      std::printf("    stage %2d  w_s=%9.3g  it=%5d  E=%.8g  E_arap=%.4g E_bend=%.4g "
                  "E_stitch=%.3g E_obst=%.3g E_anch=%.3g\n",
                  (int)stage, w_s, it, e.E, e.E_arap, e.E_bend, e.E_stitch, e.E_half,
                  e.E_anchor);
      std::fflush(stdout);
    }
    if (it >= max_iter) break;
  }
  return StageOut{it, mono};
}

}  // namespace

Result solve_annealed(const Garment& g, const Mat3X& P0,
                      const std::vector<double>& ladder, const Vec& mu, const Vec& nu,
                      const Mat3X& anchor, const ClampFn& clamp, const SolveOpts& opt) {
  const auto t_start = std::chrono::steady_clock::now();
  Result res;
  res.P = P0;
  int off = 0, used = 0;
  // lambda_b continuation, stiff -> target.  From a flat/placed start a single
  // small lambda_b folds the sheet flat instead of finding the shape; starting
  // stiff picks the long-wavelength mode and softening afterwards removes the
  // lambda_b bias.
  for (size_t i = 0; i < ladder.size(); ++i) {
    if (opt.verbose) {
      std::printf("  lambda_b = %g  (rung %d/%d)\n", ladder[i], (int)i + 1, (int)ladder.size());
      std::fflush(stdout);
    }
    Assembly asmb(g, ladder[i], mu, nu, 1e-8, true, opt.wood);
    res.factorizations++;
    auto sched = geometric_schedule(i == 0 ? opt.w0 : opt.w1, opt.w1, opt.factor,
                                    opt.iters_per_stage, opt.per_lambda);
    const int budget = std::min<int>(opt.per_lambda + opt.iters_per_stage * (int)sched.size(),
                                     opt.max_iter - used);
    if (budget <= 0) break;
    StageOut so = solve_stages(asmb, res.P, sched, budget, opt.tol, clamp, anchor,
                               opt.recenter, opt.verbose, off, used, ladder[i], res.hist);
    off += (int)sched.size();
    used += so.iters_used;
    res.mono_violations += so.mono;
    res.rebuild_seconds += asmb.rebuild_seconds();
  }
  res.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
  return res;
}

// ---------------------------------------------------------------------------
// body.py: analytic obstacle proxy (plain cylinders + one sphere)
// ---------------------------------------------------------------------------

BodyProxy::BodyProxy(const std::vector<double>& cyl, const std::vector<double>& sph) {
  const int K = (int)cyl.size() / 7, S = (int)sph.size() / 4;
  P0_.resize(K, 3); N_.resize(K, 3); L_.resize(K); R_.resize(K);
  P0N_.resize(K); P0P0_.resize(K);
  for (int k = 0; k < K; ++k) {
    Eigen::Vector3d p0(cyl[k * 7 + 0], cyl[k * 7 + 1], cyl[k * 7 + 2]);
    Eigen::Vector3d p1(cyl[k * 7 + 3], cyl[k * 7 + 4], cyl[k * 7 + 5]);
    Eigen::Vector3d d = p1 - p0;
    const double L = d.norm();
    Eigen::Vector3d nrm = d / L;
    P0_.row(k) = p0.transpose(); N_.row(k) = nrm.transpose();
    L_[k] = L; R_[k] = cyl[k * 7 + 6];
    P0N_[k] = p0.dot(nrm); P0P0_[k] = p0.dot(p0);
  }
  Sc_.resize(S, 3); Sr_.resize(S); ScSc_.resize(S);
  for (int s = 0; s < S; ++s) {
    Eigen::Vector3d c(sph[s * 4 + 0], sph[s * 4 + 1], sph[s * 4 + 2]);
    Sc_.row(s) = c.transpose(); Sr_[s] = sph[s * 4 + 3]; ScSc_[s] = c.dot(c);
  }
}

void BodyProxy::penetration(const Mat3X& X, Vec& dep, Mat3X& out) const {
  const int n = (int)X.rows(), K = (int)R_.size(), S = (int)Sr_.size();
  dep.resize(n);
  out = X;
  // The test itself needs only scalars: |X - P0|^2 comes from three matrix
  // products and the radial distance from |W|^2 - s^2, so no (n, k, 3) array is
  // ever formed.  Only vertices actually inside get their exit point built.
  const MatX XN = K ? MatX(X * N_.transpose()) : MatX(n, 0);
  const MatX XP0 = K ? MatX(X * P0_.transpose()) : MatX(n, 0);
  const Vec X2 = X.rowwise().squaredNorm();
  const MatX XS = S ? MatX(X * Sc_.transpose()) : MatX(n, 0);

  for (int i = 0; i < n; ++i) {
    int kbest = 0;
    double best = -1e300, best_wall = 0.0, best_c0 = 0.0, best_rho = 0.0;
    for (int k = 0; k < K; ++k) {
      const double s = XN(i, k) - P0N_[k];
      const double w2 = X2[i] - 2.0 * XP0(i, k) + P0P0_[k];
      const double rho = std::sqrt(std::max(w2 - s * s, 0.0));
      const double d_wall = R_[k] - rho;               // all three > 0 <=> inside
      const double d = std::min(d_wall, std::min(s, L_[k] - s));
      if (d > best) {                                  // argmax, first max wins
        best = d; kbest = k; best_wall = d_wall; best_c0 = s; best_rho = rho;
      }
    }
    dep[i] = best;
    if (K > 0 && best > 0.0) {
      // inside a finite cylinder the nearest boundary point is exactly one of
      // three: radially out to the wall, or straight out through either cap
      const double c0 = best_c0, c1 = L_[kbest] - c0;
      const Eigen::Vector3d Nj = N_.row(kbest).transpose();
      const Eigen::Vector3d base = P0_.row(kbest).transpose() + c0 * Nj;
      const Eigen::Vector3d Xi = X.row(i).transpose();
      const Eigen::Vector3d v = Xi - base;
      const Eigen::Vector3d u = best_rho > 1e-9
                                    ? Eigen::Vector3d(v / std::max(best_rho, 1e-9))
                                    : Eigen::Vector3d(1.0, 0.0, 0.0);  // on the axis: +x
      int pick = 0;                                    // argmin, first min wins
      double pv = best_wall;
      if (c0 < pv) { pick = 1; pv = c0; }
      if (c1 < pv) { pick = 2; }
      Eigen::Vector3d o;
      if (pick == 0) o = base + R_[kbest] * u;
      else if (pick == 1) o = Xi - c0 * Nj;
      else o = Xi + c1 * Nj;
      out.row(i) = o.transpose();
    }
    if (S > 0) {
      int qbest = 0;
      double sbest = -1e300;
      for (int s = 0; s < S; ++s) {
        const double ds = std::sqrt(std::max(X2[i] - 2.0 * XS(i, s) + ScSc_[s], 0.0));
        const double sd = Sr_[s] - ds;
        if (sd > sbest) { sbest = sd; qbest = s; }
      }
      // BOTH conditions: the sphere must be deeper AND actually contain the
      // vertex.  Without "sd > 0" every vertex merely nearer the sphere than to
      // any cylinder -- both distances negative -- gets teleported onto it.
      if (sbest > dep[i] && sbest > 0.0) {
        const Eigen::Vector3d Xi = X.row(i).transpose();
        const Eigen::Vector3d c = Sc_.row(qbest).transpose();
        const Eigen::Vector3d v = Xi - c;
        const double nv = v.norm();
        const Eigen::Vector3d u = nv > 1e-9 ? Eigen::Vector3d(v / std::max(nv, 1e-9))
                                            : Eigen::Vector3d(0.0, 1.0, 0.0);
        out.row(i) = (c + Sr_[qbest] * u).transpose();
        dep[i] = sbest;
      }
    }
  }
}

void BodyProxy::clamp(Mat3X& P, int sweeps) const {
  // Leaving one primitive can land a vertex inside a neighbour, so the union is
  // swept a few times; each sweep is exact for the primitive it acts on.
  Vec dep;
  Mat3X out;
  for (int s = 0; s < sweeps; ++s) {
    penetration(P, dep, out);
    bool any = false;
    for (int i = 0; i < dep.size(); ++i)
      if (dep[i] > 0.0) { any = true; break; }
    if (!any) break;
    P = out;
  }
}

}  // namespace sim
