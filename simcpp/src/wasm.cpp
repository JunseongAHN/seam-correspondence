// Emscripten surface.  Two ways in:
//
//   * a flat C API (extern "C", exported as _sim_*) that takes raw heap pointers,
//     so plain JS can malloc, fill, call and read back;
//   * an embind wrapper over the same functions for convenience.
//
// The JS side is not written here; test_wasm.js shows the C API being driven.
#include "sim.h"

#include <cstring>
#include <memory>
#include <vector>

#ifdef __EMSCRIPTEN__
#include <emscripten/bind.h>
#include <emscripten/emscripten.h>
#define EXPORT EMSCRIPTEN_KEEPALIVE
#else
#define EXPORT
#endif

using namespace sim;

namespace {

// One live problem, so JS can build it once and run several ladders against it.
struct Session {
  Input in;
  Garment g;
  std::unique_ptr<BodyProxy> body;
  Result res;
};

Session* g_session = nullptr;

}  // namespace

extern "C" {

// Build the garment from raw arrays.  All pointers are into the wasm heap.
//   faces          (n_faces*3) int32
//   wid            (n)         int32   raw -> welded vertex id
//   panel_of_face  (n_faces)   int32   (may be null)
//   rest           (n*2)       float64 flat pattern coordinates, cm
//   pairs          (n_pairs*2) int32   seam vertex pairs
//   mu             (n)         float64 penalty diagonal (may be null)
//   cyl            (n_cyl*7)   float64 obstacle cylinders p0 p1 r (may be null)
//   sph            (n_sph*4)   float64 obstacle spheres c r (may be null)
// Returns the hinge count, or -1 on failure.
EXPORT int sim_build(int n, int n_faces, const int* faces, const int* wid,
                     const int* panel_of_face, const double* rest, int n_pairs,
                     const int* pairs, const double* mu, int n_cyl, const double* cyl,
                     int n_sph, const double* sph) {
  delete g_session;
  g_session = new Session();
  Input& in = g_session->in;
  in.n = n;
  in.faces.assign(faces, faces + (size_t)n_faces * 3);
  in.wid.assign(wid, wid + (size_t)n);
  if (panel_of_face) in.panel_of_face.assign(panel_of_face, panel_of_face + (size_t)n_faces);
  in.rest.assign(rest, rest + (size_t)n * 2);
  if (n_pairs > 0) in.pairs.assign(pairs, pairs + (size_t)n_pairs * 2);
  if (mu) in.mu.assign(mu, mu + (size_t)n);
  if (n_cyl > 0) in.cyl.assign(cyl, cyl + (size_t)n_cyl * 7);
  if (n_sph > 0) in.sph.assign(sph, sph + (size_t)n_sph * 4);
  g_session->g = build_garment(in);
  if (n_cyl > 0 || n_sph > 0)
    g_session->body.reset(new BodyProxy(in.cyl, in.sph));
  return g_session->g.nHinges();
}

// Run the lambda_b ladder from P0 (n*3 float64, row-major) and write the result
// into out (n*3 float64).  Returns the iteration count, or -1 if not built.
EXPORT int sim_solve(const double* P0, int n_ladder, const double* ladder, double w0,
                     double w1, double factor, int iters_per_stage, int per_lambda,
                     int max_iter, double tol, int recenter, int verbose, double* out) {
  if (!g_session) return -1;
  Session& S = *g_session;
  const int n = S.g.n;
  Mat3X P(n, 3);
  for (int i = 0; i < n; ++i)
    for (int d = 0; d < 3; ++d) P(i, d) = P0[(size_t)i * 3 + d];

  Vec mu, nu;
  Mat3X anchor;
  if (!S.in.mu.empty()) mu = Eigen::Map<const Vec>(S.in.mu.data(), n);

  ClampFn clamp;
  if (S.body && !S.body->empty()) {
    BodyProxy* bp = S.body.get();
    clamp = [bp](Mat3X& X) { bp->clamp(X); };
  }

  SolveOpts opt;
  opt.w0 = w0; opt.w1 = w1; opt.factor = factor;
  opt.iters_per_stage = iters_per_stage;
  opt.per_lambda = per_lambda;
  opt.max_iter = max_iter;
  opt.tol = tol;
  opt.recenter = recenter != 0;
  opt.verbose = verbose != 0;

  std::vector<double> lad(ladder, ladder + n_ladder);
  S.res = solve_annealed(S.g, P, lad, mu, nu, anchor, clamp, opt);
  for (int i = 0; i < n; ++i)
    for (int d = 0; d < 3; ++d) out[(size_t)i * 3 + d] = S.res.P(i, d);
  return (int)S.res.hist.size();
}

EXPORT int sim_n_hinges() { return g_session ? g_session->g.nHinges() : -1; }
EXPORT int sim_mono_violations() { return g_session ? g_session->res.mono_violations : -1; }
EXPORT double sim_seconds() { return g_session ? g_session->res.seconds : -1.0; }

// last-iteration energies: 0 arap, 1 bend, 2 stitch, 3 obstacle, 4 anchor
EXPORT double sim_energy(int which) {
  if (!g_session || g_session->res.hist.empty()) return -1.0;
  const HistRec& e = g_session->res.hist.back();
  switch (which) {
    case 0: return e.E_arap;
    case 1: return e.E_bend;
    case 2: return e.E_stitch;
    case 3: return e.E_half;
    case 4: return e.E_anchor;
    default: return e.E;
  }
}

EXPORT void sim_free() { delete g_session; g_session = nullptr; }

}  // extern "C"

#ifdef __EMSCRIPTEN__
// embind mirror of the same surface, taking heap offsets as numbers
EMSCRIPTEN_BINDINGS(simcpp) {
  emscripten::function("simBuild", emscripten::optional_override(
      [](int n, int n_faces, uintptr_t faces, uintptr_t wid, uintptr_t pof, uintptr_t rest,
         int n_pairs, uintptr_t pairs, uintptr_t mu, int n_cyl, uintptr_t cyl, int n_sph,
         uintptr_t sph) {
        return sim_build(n, n_faces, (const int*)faces, (const int*)wid,
                         pof ? (const int*)pof : nullptr, (const double*)rest, n_pairs,
                         (const int*)pairs, mu ? (const double*)mu : nullptr, n_cyl,
                         cyl ? (const double*)cyl : nullptr, n_sph,
                         sph ? (const double*)sph : nullptr);
      }));
  emscripten::function("simSolve", emscripten::optional_override(
      [](uintptr_t P0, int n_ladder, uintptr_t ladder, double w0, double w1, double factor,
         int iters_per_stage, int per_lambda, int max_iter, double tol, int recenter,
         int verbose, uintptr_t out) {
        return sim_solve((const double*)P0, n_ladder, (const double*)ladder, w0, w1, factor,
                         iters_per_stage, per_lambda, max_iter, tol, recenter, verbose,
                         (double*)out);
      }));
  emscripten::function("simNHinges", &sim_n_hinges);
  emscripten::function("simMonoViolations", &sim_mono_violations);
  emscripten::function("simSeconds", &sim_seconds);
  emscripten::function("simEnergy", &sim_energy);
  emscripten::function("simFree", &sim_free);
}
#endif
