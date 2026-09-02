// Isometric assembly solver -- C++ port of rigid-clothes-simulation/assembly.py
// (plus the obstacle projector from body.py).  See README.md for what is and is
// not ported.
#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace sim {

// positions are (n,3) row-major so a row is one vertex and contiguous
using Mat3X = Eigen::MatrixXd;  // always (n,3), column-major
using MatX = Eigen::MatrixXd;
using Vec = Eigen::VectorXd;
using SpMat = Eigen::SparseMatrix<double>;

// ---------------------------------------------------------------------------
// input: everything gcd_io.load() + run_garment produce, read from a flat file
// ---------------------------------------------------------------------------
struct Input {
  int n = 0;
  int recenter = 1;
  std::vector<int> faces;          // (M,3)
  std::vector<int> wid;            // (n,)   raw -> welded id
  std::vector<int> panel_of_face;  // (M,)
  std::vector<double> rest;        // (n,2)  flat pattern, cm
  std::vector<double> P0;          // (n,3)  initial placement
  std::vector<int> pairs;          // (K,2)  seam vertex pairs
  std::vector<double> mu;          // (n,) or empty -- obstacle penalty diagonal
  std::vector<double> nu;          // (n,) or empty -- anchor penalty diagonal
  std::vector<double> anchor;      // (n,3) or empty
  std::vector<double> cyl;         // (NC,7) p0(3) p1(3) r
  std::vector<double> sph;         // (NS,4) c(3) r

  int nFaces() const { return (int)faces.size() / 3; }
  int nPairs() const { return (int)pairs.size() / 2; }
};

bool read_input(const std::string& path, Input& in, std::string& err);
bool write_npy3(const std::string& path, const Mat3X& P, std::string& err);
// generic .npy writers, so intermediate stages can be diffed against numpy
bool write_npy(const std::string& path, const double* data, long long rows, long long cols);
bool write_npy_i32(const std::string& path, const int* data, long long rows, long long cols);

// ---------------------------------------------------------------------------
// the "gar" dict of run_garment.build()
// ---------------------------------------------------------------------------
struct Garment {
  int n = 0;
  std::vector<int> faces;    // (M,3)
  std::vector<double> G;     // (M,3,2) shape gradients, flat
  std::vector<double> area;  // (M,)    |signed rest area|
  std::vector<int> hinges;   // (H,4)
  std::vector<double> Kb;    // (H,4)
  std::vector<double> wb;    // (H,)
  std::vector<int> pairs;    // (K,2)

  int nFaces() const { return (int)faces.size() / 3; }
  int nHinges() const { return (int)hinges.size() / 4; }
  int nPairs() const { return (int)pairs.size() / 2; }
};

// assembly.shape_gradients: rest_tri (M,3,2) -> G (M,3,2), |area| (M,)
void shape_gradients(const std::vector<int>& faces, const std::vector<double>& rest,
                     std::vector<double>& G, std::vector<double>& area);

// assembly.deformation_gradients: F (M,3,2) with F_t = sum_v p_v g_v^T
void deformation_gradients(const Mat3X& P, const std::vector<int>& faces,
                           const std::vector<double>& G, std::vector<double>& F);

// assembly.best_rotations: closed-form 2x2 polar decomposition, no SVD.
// R (M,3,2) and sig (M,2) descending.
void best_rotations(const std::vector<double>& F, std::vector<double>& R,
                    std::vector<double>& sig);

double arap_energy(const std::vector<double>& F, const std::vector<double>& R,
                   const std::vector<double>& area);

// assembly.build_hinges + hinge_stencils
void build_hinges(const std::vector<int>& faces, const std::vector<int>& wid,
                  const std::vector<double>& rest, const std::vector<int>& panel_of_face,
                  std::vector<int>& hinges, std::vector<double>& rest4);
void hinge_stencils(const std::vector<double>& rest4, std::vector<double>& Kb,
                    std::vector<double>& wb);

double bending_energy(const Mat3X& P, const std::vector<int>& hinges,
                      const std::vector<double>& Kb, const std::vector<double>& wb);

// assembly.arap_rhs: b_v = sum_t A_t R_t g_tv
void arap_rhs(const std::vector<double>& R, const std::vector<int>& faces,
              const std::vector<double>& G, const std::vector<double>& area, int n,
              Mat3X& b);

Garment build_garment(const Input& in);

// run_garment.diag_scale -- mean ARAP stiffness diagonal
double diag_scale(const Garment& g);

// ---------------------------------------------------------------------------
// assembly.Assembly
// ---------------------------------------------------------------------------
// How (C + I/w_s)^-1 in the Woodbury update is carried across the w_s ladder.
//   Eigh    -- one eigendecomposition of C per factorisation (what Python does now)
//   Inverse -- a dense inverse rebuilt at every change of w_s (the older Python way)
// Both are exact; see README.md for the measured cost of each in C++.
enum class WoodMode { Eigh, Inverse };

class Assembly {
 public:
  Assembly(const Garment& g, double lam_b, const Vec& mu, const Vec& nu,
           double eps = 1e-8, bool woodbury = true, WoodMode mode = WoodMode::Eigh);

  // seconds spent (re)building the small dense operator, summed over calls
  double rebuild_seconds() const { return rebuild_s_; }

  // one global step: (L0 + w_s D^T D)^-1 b via the Woodbury identity
  Mat3X solve_global(const Mat3X& b, double w_s) const;

  const Garment& g;
  double lam_b, eps;
  Vec mu, nu;                       // empty => term off
  bool has_mu = false, has_nu = false;

 private:
  Eigen::SimplicialLDLT<SpMat> ldlt_;
  SpMat D_;
  MatX Y_;                          // L0^-1 D^T   (n, K)
  MatX V_;                          // eigenvectors of C = D L0^-1 D^T
  Vec lam_;                         // eigenvalues of C
  MatX C_;                          // kept only for WoodMode::Inverse
  mutable MatX Minv_;               // cached (C + I/w_s)^-1
  mutable double Minv_w_ = 0.0;
  mutable double rebuild_s_ = 0.0;
  WoodMode mode_ = WoodMode::Eigh;
  bool woodbury_ = false;
};

// ---------------------------------------------------------------------------
// assembly.solve / solve_annealed
// ---------------------------------------------------------------------------
struct HistRec {
  int it, stage;
  double lam_b, w_s, E, E_arap, E_bend, E_stitch, E_half, E_anchor;
};

// clamp(P) must project P onto the feasible set in place.  Empty = no obstacle.
using ClampFn = std::function<void(Mat3X&)>;

struct SolveOpts {
  double w0 = 1e-2, w1 = 1e4, factor = 2.0;
  int iters_per_stage = 10;
  int per_lambda = 400;
  int max_iter = 20000;
  double tol = 1e-10;
  bool recenter = true;
  bool verbose = true;
  WoodMode wood = WoodMode::Eigh;
};

struct Result {
  Mat3X P;
  std::vector<HistRec> hist;
  int mono_violations = 0;
  int factorizations = 0;
  double seconds = 0.0;
  double rebuild_seconds = 0.0;   // time inside the Woodbury dense operator
};

Result solve_annealed(const Garment& g, const Mat3X& P0,
                      const std::vector<double>& ladder, const Vec& mu, const Vec& nu,
                      const Mat3X& anchor, const ClampFn& clamp, const SolveOpts& opt);

// ---------------------------------------------------------------------------
// body.py: analytic obstacle proxy (cylinders + spheres) and its projector
// ---------------------------------------------------------------------------
class BodyProxy {
 public:
  BodyProxy(const std::vector<double>& cyl, const std::vector<double>& sph);
  bool empty() const { return R_.size() == 0 && Sr_.size() == 0; }
  // (depth, exit point) for the primitive each vertex is deepest inside
  void penetration(const Mat3X& X, Vec& dep, Mat3X& out) const;
  void clamp(Mat3X& P, int sweeps = 2) const;

 private:
  MatX P0_, N_, Sc_;                // (K,3) / (K,3) / (S,3)
  Vec L_, R_, Sr_, P0N_, P0P0_, ScSc_;
};

}  // namespace sim
