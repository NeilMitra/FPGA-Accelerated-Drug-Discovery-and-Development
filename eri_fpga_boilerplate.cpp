// SPDX-License-Identifier: MIT
// Boilerplate for FPGA-based ERI compute + compression pipeline.
// Target: Intel oneAPI DPC++ with FPGA Add-on.
// Paper reference: "Computing and Compressing Electron Repulsion Integrals on FPGAs" (Wu et al., 2023).
// Stages: Setup -> Recurrence Relations -> Quadrature -> Compress/Store.

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <cstdint>
#include <cmath>

struct GTOPrimitive {
  float alpha;
  float center[3];
  int   a[3];
  float norm;
};

struct GTOShell {
  GTOPrimitive prim;
  int L;
};

struct Quartet {
  GTOShell a, b, c, d;
};

template<int N_RYS>
struct RysData {
  float roots[N_RYS];
  float weights[N_RYS];
};

constexpr int shell_size(int L) { return (L + 1) * (L + 2) / 2; }

template<int L_A, int L_B, int L_C, int L_D, int N_RYS, int NBIT,
         int MAX_IJKL=84>
struct ERIKernelCfg {
  static constexpr int NGA = shell_size(L_A);
  static constexpr int NGB = shell_size(L_B);
  static constexpr int NGC = shell_size(L_C);
  static constexpr int NGD = shell_size(L_D);
  static constexpr int NERIS = NGA * NGB * NGC * NGD;
  static_assert(NBIT >= 8 && NBIT <= 32, "Supported compressed integer bitwidth range.");
};

template<int NBIT>
struct Pack512 {
  static int pack(const int32_t* in, int count, uint64_t* out512) {
    const int per_word = 512 / NBIT;
    int out_words = 0;
    int idx = 0;
    while (idx < count) {
      uint64_t lanes[8] = {0,0,0,0,0,0,0,0};
      int bitpos = 0;
      for (int i = 0; i < per_word && idx < count; ++i, ++idx) {
        uint64_t u = static_cast<uint64_t>(static_cast<uint32_t>(in[idx])) & ((NBIT==64)?~0ull:((1ull<<NBIT)-1ull));
        int lane   = bitpos / 64;
        int shift  = bitpos % 64;
        lanes[lane] |= (u << shift);
        if (shift + NBIT > 64 && lane < 7) {
          lanes[lane+1] |= (u >> (64 - shift));
        }
        bitpos += NBIT;
      }
      for (int i=0;i<8;i++) out512[i + out_words*8] = lanes[i];
      out_words++;
    }
    return out_words;
  }
};

struct CompressionResultHeader {
  float bmax;
  int   nvals;
};

template<int NBIT>
inline void compress_quartet(const float* eri, int nvals, CompressionResultHeader& hdr,
                             int32_t* out_q) {
  float bmax = 0.0f;
  for (int i=0;i<nvals;i++) bmax = fmaxf(bmax, fabsf(eri[i]));
  hdr.bmax = (bmax == 0.0f) ? 1.0f : bmax;
  hdr.nvals = nvals;
  const float eps = hdr.bmax / (float)((1u<<NBIT) - 1u);
  for (int i=0;i<nvals;i++) {
    float qf = eri[i] / eps;
    int32_t q = (int32_t) llroundf(qf);
    out_q[i] = q;
  }
}

template<typename Cfg>
class ERIKernel;

template<int L_A, int L_B, int L_C, int L_D, int N_RYS, int NBIT, int MAX_IJKL>
void LaunchERIKernel(sycl::queue& q,
                     const Quartet& quartet,
                     const RysData<N_RYS>& rys,
                     uint64_t* out_packed_512,
                     CompressionResultHeader* out_hdr) {

  using C = ERIKernelCfg<L_A,L_B,L_C,L_D,N_RYS,NBIT,MAX_IJKL>;

  q.submit([&](sycl::handler& h) {
    h.single_task<ERIKernel<C>>([=]() [[intel::kernel_args_restrict]] {
      [[intel::fpga_memory]] float Ibuf[C::NERIS];
      [[intel::fpga_memory]] int32_t qvals[C::NERIS];
      CompressionResultHeader hdr;

      for (int idx = 0; idx < C::NERIS; ++idx) {
        Ibuf[idx] = 0.0f; // placeholder
      }

      compress_quartet<NBIT>(Ibuf, C::NERIS, hdr, qvals);

      uint64_t* out64 = out_packed_512;
      out64[0] = *reinterpret_cast<uint64_t const*>(&hdr.bmax);
      out64[1] = static_cast<uint64_t>(hdr.nvals);
      Pack512<NBIT>::pack(qvals, C::NERIS, out64 + 2);
      *out_hdr = hdr;
    });
  }).wait();
}

int main() {
#if defined(FPGA_EMULATOR)
  sycl::ext::intel::fpga_emulator_selector selector;
#elif defined(FPGA_HARDWARE)
  sycl::ext::intel::fpga_selector selector;
#else
  sycl::ext::intel::fpga_emulator_selector selector;
#endif
  sycl::queue q(selector, sycl::property_list{sycl::property::queue::in_order{}});

  constexpr int L_A=0, L_B=1, L_C=1, L_D=0;
  constexpr int N_RYS=4, NBIT=16;

  Quartet quartet{};
  RysData<N_RYS> rys{};

  uint64_t* out_packed_512 = sycl::malloc_device<uint64_t>(1024, q);
  auto* out_hdr = sycl::malloc_device<CompressionResultHeader>(1, q);

  LaunchERIKernel<L_A,L_B,L_C,L_D,N_RYS,NBIT,84>(q, quartet, rys, out_packed_512, out_hdr);

  sycl::free(out_packed_512, q);
  sycl::free(out_hdr, q);
  return 0;
}
