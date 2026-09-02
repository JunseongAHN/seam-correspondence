// Reader for the flat file dump_garment.py writes, and a minimal .npy writer so
// the result can be diffed against the Python reference with numpy.load.
#include "sim.h"

#include <cstdio>
#include <cstring>
#include <fstream>

namespace sim {

namespace {

template <typename T>
bool rd(std::ifstream& f, std::vector<T>& v, size_t count) {
  v.resize(count);
  if (count == 0) return true;
  f.read(reinterpret_cast<char*>(v.data()), (std::streamsize)(count * sizeof(T)));
  return (bool)f;
}

bool rd_i32(std::ifstream& f, std::vector<int>& v, size_t count) {
  static_assert(sizeof(int) == 4, "int must be 32 bit");
  return rd(f, v, count);
}

}  // namespace

bool read_input(const std::string& path, Input& in, std::string& err) {
  std::ifstream f(path, std::ios::binary);
  if (!f) { err = "cannot open " + path; return false; }
  char magic[8];
  f.read(magic, 8);
  if (std::memcmp(magic, "SIMCPP01", 8) != 0) { err = "bad magic in " + path; return false; }
  int hdr[8];
  f.read(reinterpret_cast<char*>(hdr), sizeof(hdr));
  const int n = hdr[0], M = hdr[1], K = hdr[2], NC = hdr[3], NS = hdr[4];
  const int has_mu = hdr[5], has_nu = hdr[6];
  in.n = n;
  in.recenter = hdr[7];
  bool ok = true;
  ok &= rd_i32(f, in.faces, (size_t)M * 3);
  ok &= rd_i32(f, in.wid, (size_t)n);
  ok &= rd_i32(f, in.panel_of_face, (size_t)M);
  ok &= rd(f, in.rest, (size_t)n * 2);
  ok &= rd(f, in.P0, (size_t)n * 3);
  ok &= rd_i32(f, in.pairs, (size_t)K * 2);
  if (has_mu) ok &= rd(f, in.mu, (size_t)n);
  if (has_nu) {
    ok &= rd(f, in.nu, (size_t)n);
    ok &= rd(f, in.anchor, (size_t)n * 3);
  }
  ok &= rd(f, in.cyl, (size_t)NC * 7);
  ok &= rd(f, in.sph, (size_t)NS * 4);
  if (!ok) { err = "short read on " + path; return false; }
  return true;
}

namespace {

// write the .npy header for a C-order 2-D array of `descr`
bool npy_header(std::ofstream& f, const char* descr, long long rows, long long cols) {
  char dict[256];
  if (cols == 1)
    std::snprintf(dict, sizeof(dict),
                  "{'descr': '%s', 'fortran_order': False, 'shape': (%lld,), }", descr, rows);
  else
    std::snprintf(dict, sizeof(dict),
                  "{'descr': '%s', 'fortran_order': False, 'shape': (%lld, %lld), }", descr,
                  rows, cols);
  const size_t total = 10 + std::strlen(dict) + 1;   // magic(6)+ver(2)+len(2)+dict+\n
  std::string d(dict);
  d.append((64 - total % 64) % 64, ' ');
  d.push_back('\n');
  const unsigned short hlen = (unsigned short)d.size();
  f.write("\x93NUMPY\x01\x00", 8);
  f.write(reinterpret_cast<const char*>(&hlen), 2);
  f.write(d.data(), (std::streamsize)d.size());
  return (bool)f;
}

}  // namespace

bool write_npy3(const std::string& path, const Mat3X& P, std::string& err) {
  std::ofstream f(path, std::ios::binary);
  if (!f) { err = "cannot write " + path; return false; }
  npy_header(f, "<f8", P.rows(), 3);
  double row[3];
  for (Eigen::Index i = 0; i < P.rows(); ++i) {
    row[0] = P(i, 0); row[1] = P(i, 1); row[2] = P(i, 2);
    f.write(reinterpret_cast<const char*>(row), 3 * sizeof(double));
  }
  return (bool)f;
}

bool write_npy(const std::string& path, const double* data, long long rows, long long cols) {
  std::ofstream f(path, std::ios::binary);
  if (!f) return false;
  npy_header(f, "<f8", rows, cols);
  f.write(reinterpret_cast<const char*>(data),
          (std::streamsize)(rows * (cols ? cols : 1) * sizeof(double)));
  return (bool)f;
}

bool write_npy_i32(const std::string& path, const int* data, long long rows, long long cols) {
  std::ofstream f(path, std::ios::binary);
  if (!f) return false;
  npy_header(f, "<i4", rows, cols);
  f.write(reinterpret_cast<const char*>(data),
          (std::streamsize)(rows * (cols ? cols : 1) * sizeof(int)));
  return (bool)f;
}

}  // namespace sim
