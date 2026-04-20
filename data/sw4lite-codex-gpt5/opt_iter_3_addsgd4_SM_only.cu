#include <cuda_runtime.h>

#ifndef BX
#define BX 32
#endif

#ifndef BY
#define BY 8
#endif

// Kernel must keep the same signature and name.
__global__ void addsgd4_SM (int ifirst, int ilast, int jfirst, int jlast, int kfirst, int klast,
                         int ni, int nj, int nk,
                         float_sw4* __restrict__ a_up,
                         const float_sw4* __restrict__ a_u,
                         const float_sw4* __restrict__ a_um,
                         const float_sw4* __restrict__ a_rho,
                         const float_sw4* __restrict__ a_dcx,
                         const float_sw4* __restrict__ a_dcy,
                         const float_sw4* __restrict__ a_dcz,
                         const float_sw4* __restrict__ a_strx,
                         const float_sw4* __restrict__ a_stry,
                         const float_sw4* __restrict__ a_strz,
                         const float_sw4* __restrict__ a_cox,
                         const float_sw4* __restrict__ a_coy,
                         const float_sw4* __restrict__ a_coz,
                         const float_sw4 beta) {
  // SoA access helpers (use nijk defined below; only used after nijk is set)
  #define U(c, IDX)   a_u[(c)*(nijk) + (IDX)]
  #define UM(c, IDX)  a_um[(c)*(nijk) + (IDX)]
  #define UP(c, IDX)  a_up[(c)*(nijk) + (IDX)]

  int index;
  float_sw4 u_km2[3], u_km1[3];

  // Shared-memory ring buffer: s=0 current, s=1 next, s=2 next-next
  __shared__ float_sw4 shu[3][3][BY+4][BX+4];

  const int i = threadIdx.x + blockIdx.x * BX;
  const int j = jfirst + threadIdx.y + blockIdx.y * BY;

  const int ti = threadIdx.x + 2;
  const int tj = threadIdx.y + 2;

  const int nij  = ni * nj;
  const int nijk = nij * nk;

  int active = 0, loader = 0, loady2 = 0, loaderx2 = 0;

  if (i+2 >= ifirst && i+2 <= ilast && j <= jlast)
    active = 1;

  if (i < ni && j-2 <= jlast+2) {
    loader = 1;
    if (threadIdx.x < 4 && i+BX < ni)
      loaderx2 = 1;
    if (threadIdx.y < 4 && j-2+BY <= jlast+2)
      loady2 = 1;
  }

  // Preload k-2 and k-1 into registers
  if (active) {
    int idxkm2 = (kfirst - 2) * nij + j * ni + i + 2;
    #pragma unroll 3
    for (int c=0; c<3; c++) {
      float_sw4 u0  = __ldg(&U(c,idxkm2));
      float_sw4 um0 = __ldg(&UM(c,idxkm2));
      u_km2[c] = u0 - um0;
    }
    int idxkm1 = idxkm2 + nij;
    #pragma unroll 3
    for (int c=0; c<3; c++) {
      float_sw4 u0  = __ldg(&U(c,idxkm1));
      float_sw4 um0 = __ldg(&UM(c,idxkm1));
      u_km1[c] = u0 - um0;
    }
  }

  index = kfirst * nij + (j - 2) * ni + i;

  // Stage two k-planes into shared memory (s=1,2)
  #pragma unroll 3
  for (int c=0; c<3; c++) {
    if (loader) {
      int idx0 = index;
      float_sw4 u10  = __ldg(&U(c,idx0));
      float_sw4 um10 = __ldg(&UM(c,idx0));
      float_sw4 u11  = __ldg(&U(c,idx0+nij));
      float_sw4 um11 = __ldg(&UM(c,idx0+nij));
      shu[c][1][threadIdx.y][threadIdx.x] = u10 - um10;
      shu[c][2][threadIdx.y][threadIdx.x] = u11 - um11;
    }
    if (loaderx2) {
      int idxx = index + BX;
      float_sw4 u20  = __ldg(&U(c,idxx));
      float_sw4 um20 = __ldg(&UM(c,idxx));
      float_sw4 u21  = __ldg(&U(c,idxx+nij));
      float_sw4 um21 = __ldg(&UM(c,idxx+nij));
      shu[c][1][threadIdx.y][threadIdx.x+BX] = u20 - um20;
      shu[c][2][threadIdx.y][threadIdx.x+BX] = u21 - um21;
    }
    if (loady2) {
      int idxy = index + BY * ni;
      float_sw4 u30  = __ldg(&U(c,idxy));
      float_sw4 um30 = __ldg(&UM(c,idxy));
      float_sw4 u31  = __ldg(&U(c,idxy+nij));
      float_sw4 um31 = __ldg(&UM(c,idxy+nij));
      shu[c][1][threadIdx.y+BY][threadIdx.x] = u30 - um30;
      shu[c][2][threadIdx.y+BY][threadIdx.x] = u31 - um31;
    }
    if (loady2 && loaderx2) {
      int idxyx = index + BY * ni + BX;
      float_sw4 u40  = __ldg(&U(c,idxyx));
      float_sw4 um40 = __ldg(&UM(c,idxyx));
      float_sw4 u41  = __ldg(&U(c,idxyx+nij));
      float_sw4 um41 = __ldg(&UM(c,idxyx+nij));
      shu[c][1][threadIdx.y+BY][threadIdx.x+BX] = u40 - um40;
      shu[c][2][threadIdx.y+BY][threadIdx.x+BX] = u41 - um41;
    }
  }

  index += 2 * nij;

  for (int k = kfirst; k <= klast; k++) {
    __syncthreads();

    // Roll the shared memory ring buffer along k and load the next plane
    #pragma unroll 3
    for (int c=0; c<3; c++) {
      if (loader) {
        #pragma unroll 2
        for (int s=0; s<2; s++)
          shu[c][s][threadIdx.y][threadIdx.x] = shu[c][s+1][threadIdx.y][threadIdx.x];
        int idx0 = index;
        float_sw4 u0  = __ldg(&U(c,idx0));
        float_sw4 um0 = __ldg(&UM(c,idx0));
        shu[c][2][threadIdx.y][threadIdx.x] = u0 - um0;
      }
      if (loaderx2) {
        #pragma unroll 2
        for (int s=0; s<2; s++)
          shu[c][s][threadIdx.y][threadIdx.x+BX] = shu[c][s+1][threadIdx.y][threadIdx.x+BX];
        int idx1 = index + BX;
        float_sw4 u1  = __ldg(&U(c,idx1));
        float_sw4 um1 = __ldg(&UM(c,idx1));
        shu[c][2][threadIdx.y][threadIdx.x+BX] = u1 - um1;
      }
      if (loady2) {
        #pragma unroll 2
        for (int s=0; s<2; s++)
          shu[c][s][threadIdx.y+BY][threadIdx.x] = shu[c][s+1][threadIdx.y+BY][threadIdx.x];
        int idx2 = index + BY * ni;
        float_sw4 u2  = __ldg(&U(c,idx2));
        float_sw4 um2 = __ldg(&UM(c,idx2));
        shu[c][2][threadIdx.y+BY][threadIdx.x] = u2 - um2;
      }
      if (loady2 && loaderx2) {
        #pragma unroll 2
        for (int s=0; s<2; s++)
          shu[c][s][threadIdx.y+BY][threadIdx.x+BX] = shu[c][s+1][threadIdx.y+BY][threadIdx.x+BX];
        int idx3 = index + BY * ni + BX;
        float_sw4 u3  = __ldg(&U(c,idx3));
        float_sw4 um3 = __ldg(&UM(c,idx3));
        shu[c][2][threadIdx.y+BY][threadIdx.x+BX] = u3 - um3;
      }
    }

    index += nij;
    __syncthreads();

    if (active) {
      const int idx = k * nij + j * ni + i + 2;

      // Hoist frequently reused coefficients and neighbor rho values to registers
      const float_sw4 rho_c   = __ldg(&a_rho[idx]);
      const float_sw4 rho_ip1 = __ldg(&a_rho[idx+1]);
      const float_sw4 rho_im1 = __ldg(&a_rho[idx-1]);
      const float_sw4 rho_jp1 = __ldg(&a_rho[idx+ni]);
      const float_sw4 rho_jm1 = __ldg(&a_rho[idx-ni]);
      const float_sw4 rho_kp1 = __ldg(&a_rho[idx+nij]);
      const float_sw4 rho_km1 = __ldg(&a_rho[idx-nij]);

      const float_sw4 birho = beta / rho_c;

      // Metric/derivative coefficients
      const float_sw4 strx_i2 = __ldg(&a_strx[i+2]);
      const float_sw4 stry_j  = __ldg(&a_stry[j]);
      const float_sw4 strz_k  = __ldg(&a_strz[k]);

      const float_sw4 cox_i2  = __ldg(&a_cox[i+2]);
      const float_sw4 coy_j   = __ldg(&a_coy[j]);
      const float_sw4 coz_k   = __ldg(&a_coz[k]);

      const float_sw4 dcx_i1  = __ldg(&a_dcx[i+1]);
      const float_sw4 dcx_i2  = __ldg(&a_dcx[i+2]);
      const float_sw4 dcx_i3  = __ldg(&a_dcx[i+3]);

      const float_sw4 dcy_jm1 = __ldg(&a_dcy[j-1]);
      const float_sw4 dcy_j   = __ldg(&a_dcy[j]);
      const float_sw4 dcy_jp1 = __ldg(&a_dcy[j+1]);

      const float_sw4 dcz_km1 = __ldg(&a_dcz[k-1]);
      const float_sw4 dcz_k   = __ldg(&a_dcz[k]);
      const float_sw4 dcz_kp1 = __ldg(&a_dcz[k+1]);

      // Precompute directional prefactors
      const float_sw4 pref_x = strx_i2 * coy_j * coz_k;
      const float_sw4 pref_y = stry_j  * cox_i2 * coz_k;
      const float_sw4 pref_z = strz_k  * cox_i2 * coy_j;

      #pragma unroll 3
      for (int c=0; c<3; c++) {
        // Fetch shared tile neighborhood for this component (current k plane is s=0)
        const float_sw4 s_ti_m2 = shu[c][0][tj][ti-2];
        const float_sw4 s_ti_m1 = shu[c][0][tj][ti-1];
        const float_sw4 s_ti_0  = shu[c][0][tj][ti  ];
        const float_sw4 s_ti_p1 = shu[c][0][tj][ti+1];
        const float_sw4 s_ti_p2 = shu[c][0][tj][ti+2];

        const float_sw4 s_tj_m2 = shu[c][0][tj-2][ti];
        const float_sw4 s_tj_m1 = shu[c][0][tj-1][ti];
        const float_sw4 s_tj_0  = shu[c][0][tj  ][ti];
        const float_sw4 s_tj_p1 = shu[c][0][tj+1][ti];
        const float_sw4 s_tj_p2 = shu[c][0][tj+2][ti];

        const float_sw4 s_km2   = u_km2[c];
        const float_sw4 s_km1   = u_km1[c];
        const float_sw4 s_k0    = shu[c][0][tj][ti];
        const float_sw4 s_k1    = shu[c][1][tj][ti];
        const float_sw4 s_k2    = shu[c][2][tj][ti];

        // Discrete operators in x, y, z directions
        const float_sw4 lapx =
          rho_ip1 * dcx_i3 * (s_ti_p2 - 2.0 * s_ti_p1 + s_ti_0) -
          2.0    * rho_c   * dcx_i2 * (s_ti_p1 - 2.0 * s_ti_0  + s_ti_m1) +
          rho_im1 * dcx_i1 * (s_ti_0 - 2.0 * s_ti_m1 + s_ti_m2);

        const float_sw4 lapy =
          rho_jp1 * dcy_jp1 * (s_tj_p2 - 2.0 * s_tj_p1 + s_tj_0) -
          2.0    * rho_c   * dcy_j   * (s_tj_p1 - 2.0 * s_tj_0  + s_tj_m1) +
          rho_jm1 * dcy_jm1 * (s_tj_0 - 2.0 * s_tj_m1 + s_tj_m2);

        const float_sw4 lapz =
          rho_kp1 * dcz_kp1 * (s_k2 - 2.0 * s_k1 + s_k0) -
          2.0    * rho_c   * dcz_k   * (s_k1 - 2.0 * s_k0 + s_km1) +
          rho_km1 * dcz_km1 * (s_k0 - 2.0 * s_km1 + s_km2);

        UP(c,idx) -= birho * (pref_x * lapx + pref_y * lapy + pref_z * lapz);

        // Rotate z-history registers
        u_km2[c] = s_km1;
        u_km1[c] = s_k0;
      }
    }
  }

  #undef U
  #undef UM
  #undef UP
}