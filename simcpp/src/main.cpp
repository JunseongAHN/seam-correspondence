// Native driver: read the dump, run solve_annealed, write assembly_<tag>.npy and
// the same four gates run_garment.py reports.
#include "sim.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

using namespace sim;

namespace {

std::vector<double> parse_ladder(const std::string& s) {
  std::vector<double> v;
  size_t i = 0;
  while (i < s.size()) {
    size_t j = s.find(',', i);
    if (j == std::string::npos) j = s.size();
    v.push_back(std::atof(s.substr(i, j - i).c_str()));
    i = j + 1;
  }
  return v;
}

double median(std::vector<double> v) {
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  const size_t m = v.size() / 2;
  return v.size() % 2 ? v[m] : 0.5 * (v[m - 1] + v[m]);
}

}  // namespace

int main(int argc, char** argv) {
  std::string in_path, out_prefix = "assembly";
  // the reference command line: --amp 0 --body --sym --mu 0.02 --per-lambda 30
  //                             --lam-start 1e-3 --lam-stop 1e-5
  std::string ladder_s = "1e-1,1e-2,1e-3,1e-4,1e-5";
  std::string selftest;
  SolveOpts opt;
  opt.per_lambda = 30;
  bool verbose = true;

  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() { return std::string(argv[++i]); };
    if (a == "--in") in_path = next();
    else if (a == "--out") out_prefix = next();
    else if (a == "--ladder") ladder_s = next();
    else if (a == "--per-lambda") opt.per_lambda = std::atoi(next().c_str());
    else if (a == "--iters-per-stage") opt.iters_per_stage = std::atoi(next().c_str());
    else if (a == "--w0") opt.w0 = std::atof(next().c_str());
    else if (a == "--w1") opt.w1 = std::atof(next().c_str());
    else if (a == "--factor") opt.factor = std::atof(next().c_str());
    else if (a == "--max-iter") opt.max_iter = std::atoi(next().c_str());
    else if (a == "--tol") opt.tol = std::atof(next().c_str());
    else if (a == "--quiet") verbose = false;
    else if (a == "--selftest") selftest = next();
    else if (a == "--wood") {
      const std::string m = next();
      opt.wood = (m == "inverse") ? WoodMode::Inverse : WoodMode::Eigh;
    }
    else { std::fprintf(stderr, "unknown argument %s\n", a.c_str()); return 2; }
  }
  if (in_path.empty()) {
    std::fprintf(stderr, "usage: simcpp --in <dump.bin> [--out <prefix>] "
                         "[--ladder 1e-1,...] [--per-lambda 30] [--quiet]\n");
    return 2;
  }
  opt.verbose = verbose;

  Input in;
  std::string err;
  if (!read_input(in_path, in, err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }

  Garment g = build_garment(in);
  std::printf("built: %d verts, %d faces, %d hinges, %d seam pairs\n", g.n, g.nFaces(),
              g.nHinges(), g.nPairs());

  Mat3X P0(in.n, 3);
  for (int i = 0; i < in.n; ++i)
    for (int d = 0; d < 3; ++d) P0(i, d) = in.P0[(size_t)i * 3 + d];

  Vec mu, nu;
  Mat3X anchor;
  if (!in.mu.empty()) mu = Eigen::Map<const Vec>(in.mu.data(), in.n);
  if (!in.nu.empty()) {
    nu = Eigen::Map<const Vec>(in.nu.data(), in.n);
    anchor.resize(in.n, 3);
    for (int i = 0; i < in.n; ++i)
      for (int d = 0; d < 3; ++d) anchor(i, d) = in.anchor[(size_t)i * 3 + d];
  }

  // pluggable clamp: here the analytic body proxy, but any projector works
  BodyProxy bodyp(in.cyl, in.sph);
  ClampFn clamp;
  if (!bodyp.empty()) clamp = [&bodyp](Mat3X& P) { bodyp.clamp(P); };

  opt.recenter = in.recenter != 0;
  std::vector<double> ladder = parse_ladder(ladder_s);

  if (!selftest.empty()) {
    // Dump every intermediate the Python computes, so each piece can be diffed
    // on its own before the full pipeline is trusted (see selftest.py).
    const int M = g.nFaces(), H = g.nHinges();
    write_npy(selftest + "_G.npy", g.G.data(), M, 6);
    write_npy(selftest + "_area.npy", g.area.data(), M, 1);
    write_npy_i32(selftest + "_hinges.npy", g.hinges.data(), H, 4);
    write_npy(selftest + "_Kb.npy", g.Kb.data(), H, 4);
    write_npy(selftest + "_wb.npy", g.wb.data(), H, 1);

    std::vector<double> F0, R0, sig0;
    deformation_gradients(P0, g.faces, g.G, F0);
    best_rotations(F0, R0, sig0);
    write_npy(selftest + "_F0.npy", F0.data(), M, 6);
    write_npy(selftest + "_R0.npy", R0.data(), M, 6);
    write_npy(selftest + "_sig0.npy", sig0.data(), M, 2);

    Mat3X b0;
    arap_rhs(R0, g.faces, g.G, g.area, g.n, b0);
    Mat3X Z0 = P0;
    if (clamp) clamp(Z0);
    write_npy3(selftest + "_Z0.npy", Z0, err);
    if (mu.size() > 0)
      for (int i = 0; i < g.n; ++i)
        for (int d = 0; d < 3; ++d) b0(i, d) += mu[i] * Z0(i, d);
    write_npy3(selftest + "_b0.npy", b0, err);

    // one global step at the first rung of the ladder and the first w_s
    Assembly a0(g, ladder[0], mu, nu, 1e-8, true, opt.wood);
    Mat3X P1 = a0.solve_global(b0, opt.w0);
    write_npy3(selftest + "_P1.npy", P1, err);
    std::printf("selftest written to %s_*.npy  "
                "(E_arap=%.15g E_bend=%.15g at P0)\n", selftest.c_str(),
                arap_energy(F0, R0, g.area),
                bending_energy(P0, g.hinges, g.Kb, g.wb));
    return 0;
  }
  Result res = solve_annealed(g, P0, ladder, mu, nu, anchor, clamp, opt);

  // the four gates run_garment.py writes
  const Mat3X& P = res.P;
  double gap_max = 0.0;
  std::vector<double> gaps;
  gaps.reserve(g.nPairs());
  for (int p = 0; p < g.nPairs(); ++p) {
    double s = 0.0;
    for (int d = 0; d < 3; ++d) {
      const double dd = P(g.pairs[p * 2 + 0], d) - P(g.pairs[p * 2 + 1], d);
      s += dd * dd;
    }
    const double gp = std::sqrt(s);
    gaps.push_back(gp);
    gap_max = std::max(gap_max, gp);
  }
  std::vector<double> F, R, sig;
  deformation_gradients(P, g.faces, g.G, F);
  best_rotations(F, R, sig);
  double max_sig = 0.0;
  std::vector<double> per_tri;
  per_tri.reserve(g.nFaces());
  for (int t = 0; t < g.nFaces(); ++t) {
    const double a = std::abs(sig[t * 2 + 0] - 1.0), b = std::abs(sig[t * 2 + 1] - 1.0);
    max_sig = std::max(max_sig, std::max(a, b));
    per_tri.push_back(std::max(a, b));
  }
  const HistRec& e = res.hist.back();

  std::string npy = out_prefix + ".npy";
  if (!write_npy3(npy, P, err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
  std::ofstream js(out_prefix + ".json");
  js.precision(17);
  js << "{\"n\": " << g.n << ", \"n_faces\": " << g.nFaces()
     << ", \"n_hinges\": " << g.nHinges() << ", \"n_pairs\": " << g.nPairs()
     << ", \"iterations\": " << res.hist.size()
     << ", \"factorizations\": " << res.factorizations
     << ", \"seconds\": " << res.seconds
     << ", \"woodbury_rebuild_seconds\": " << res.rebuild_seconds
     << ", \"mono_violations\": " << res.mono_violations
     << ", \"E_arap\": " << e.E_arap << ", \"E_bend\": " << e.E_bend
     << ", \"E_stitch\": " << e.E_stitch << ", \"E_half\": " << e.E_half
     << ", \"max_sigma_dev\": " << max_sig
     << ", \"p50_sigma_dev\": " << median(per_tri)
     << ", \"seam_gap_max\": " << gap_max
     << ", \"seam_gap_p50\": " << median(gaps) << "}\n";

  std::printf("%s: %d iters, %d fac, %.2f s (%.3f s in the Woodbury dense operator) | "
              "max|s-1| %.3e  gap max %.3e cm  mono viol %d\n",
              out_prefix.c_str(), (int)res.hist.size(), res.factorizations, res.seconds,
              res.rebuild_seconds, max_sig, gap_max, res.mono_violations);
  return 0;
}
