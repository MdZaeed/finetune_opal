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
  int index;
  float_sw4 u_km2[3], u_km1[3];

  __shared__ float_sw4 shu[3][3][BY+4][BX+4];

  const int i = threadIdx.x + blockIdx.x * BX;
  const int j = jfirst + threadIdx.y + blockIdx.y * BY;

  const int ti = threadIdx.x + 2;
  const int tj = threadIdx.y + 2;

  const int nij = ni * nj;
  const int nijk = nij * nk;

  // Precompute flags as bool to aid compiler optimization
  const bool active  = (i + 2 >= ifirst) && (i + 2 <= ilast) && (j <= jlast);
  const bool loader  = (i < ni) && (j - 2 <= jlast + 2);
  const bool loaderx2 = loader && (threadIdx.x < 4) && (i + BX < ni);
  const bool loady2  = loader && (threadIdx.y < 4) && (j - 2 + BY <= jlast + 2);

  if (active) {
    int idx = (kfirst - 2) * nij + j * ni + i + 2;
    #pragma unroll 3
    for (int c=0; c<3; c++) {
      u_km2[c] = u(c,idx) - um(c,idx);
    }
    idx += nij;
    #pragma unroll 3
    for (int c=0; c<3; c++) {
      u_km1[c] = u(c,idx) - um(c,idx);
    }
  }

  index = kfirst * nij + (j - 2) * ni + i;

  #pragma unroll 3
  for (int c=0; c<3; c++) {
    if (loader) {
      int idx = index;
      shu[c][1][threadIdx.y][threadIdx.x] = u(c,idx) - um(c,idx);
      shu[c][2][threadIdx.y][threadIdx.x] = u(c,idx+nij) - um(c,idx+nij);
    }
    if (loaderx2) {
      int idx = index + BX;
      shu[c][1][threadIdx.y][threadIdx.x+BX] = u(c,idx) - um(c,idx);
      shu[c][2][threadIdx.y][threadIdx.x+BX] = u(c,idx+nij) - um(c,idx+nij);
    }
    if (loady2) {
      int idx = index + BY * ni;
      shu[c][1][threadIdx.y+BY][threadIdx.x] = u(c,idx) - um(c,idx);
      shu[c][2][threadIdx.y+BY][threadIdx.x] = u(c,idx+nij) - um(c,idx+nij);
    }
    if (loady2 && loaderx2) {
      int idx = index + BY * ni + BX;
      shu[c][1][threadIdx.y+BY][threadIdx.x+BX] = u(c,idx) - um(c,idx);
      shu[c][2][threadIdx.y+BY][threadIdx.x+BX] = u(c,idx+nij) - um(c,idx+nij);
    }
  }

  index += 2 * nij;

  for (int k=kfirst; k <= klast; k++) {
    __syncthreads();

    #pragma unroll 3
    for (int c=0; c<3; c++) {
      if (loader) {
        #pragma unroll 2
        for (int s=0; s<2; s++)
          shu[c][s][threadIdx.y][threadIdx.x] = shu[c][s+1][threadIdx.y][threadIdx.x];
        int idx = index;
        shu[c][2][threadIdx.y][threadIdx.x] = u(c,idx) - um(c,idx);
      }
      if (loaderx2) {
        #pragma unroll 2
        for (int s=0; s<2; s++)
          shu[c][s][threadIdx.y][threadIdx.x+BX] = shu[c][s+1][threadIdx.y][threadIdx.x+BX];
        int idx = index + BX;
        shu[c][2][threadIdx.y][threadIdx.x+BX] = u(c,idx) - um(c,idx);
      }
      if (loady2) {
        #pragma unroll 2
        for (int s=0; s<2; s++)
          shu[c][s][threadIdx.y+BY][threadIdx.x] = shu[c][s+1][threadIdx.y+BY][threadIdx.x];
        int idx = index + BY * ni;
        shu[c][2][threadIdx.y+BY][threadIdx.x] = u(c,idx) - um(c,idx);
      }
      if (loady2 && loaderx2) {
        #pragma unroll 2
        for (int s=0; s<2; s++)
          shu[c][s][threadIdx.y+BY][threadIdx.x+BX] = shu[c][s+1][threadIdx.y+BY][threadIdx.x+BX];
        int idx = index + BY * ni + BX;
        shu[c][2][threadIdx.y+BY][threadIdx.x+BX] = u(c,idx) - um(c,idx);
      }
    }

    index += nij;
    __syncthreads();

    if (active) {
      const int idx = k * nij + j * ni + i + 2;

      // Read-only cached coefficients and rho values
      const float_sw4 r_c   = __ldg(&a_rho[idx]);
      const float_sw4 birho = beta / r_c;

      const float_sw4 strx_i2 = __ldg(&a_strx[i+2]);
      const float_sw4 stry_j  = __ldg(&a_stry[j]);
      const float_sw4 strz_k  = __ldg(&a_strz[k]);

      const float_sw4 cox_i2  = __ldg(&a_cox[i+2]);
      const float_sw4 coy_j   = __ldg(&a_coy[j]);
      const float_sw4 coz_k   = __ldg(&a_coz[k]);

      const float_sw4 wx = strx_i2 * coy_j * coz_k;
      const float_sw4 wy = stry_j  * cox_i2 * coz_k;
      const float_sw4 wz = strz_k  * cox_i2 * coy_j;

      const float_sw4 dcx_i1 = __ldg(&a_dcx[i+1]);
      const float_sw4 dcx_i2 = __ldg(&a_dcx[i+2]);
      const float_sw4 dcx_i3 = __ldg(&a_dcx[i+3]);

      const float_sw4 dcy_jm1 = __ldg(&a_dcy[j-1]);
      const float_sw4 dcy_j   = __ldg(&a_dcy[j]);
      const float_sw4 dcy_jp1 = __ldg(&a_dcy[j+1]);

      const float_sw4 dcz_km1 = __ldg(&a_dcz[k-1]);
      const float_sw4 dcz_k   = __ldg(&a_dcz[k]);
      const float_sw4 dcz_kp1 = __ldg(&a_dcz[k+1]);

      const float_sw4 r_ip1 = __ldg(&a_rho[idx+1]);
      const float_sw4 r_im1 = __ldg(&a_rho[idx-1]);
      const float_sw4 r_jp  = __ldg(&a_rho[idx+ni]);
      const float_sw4 r_jm  = __ldg(&a_rho[idx-ni]);
      const float_sw4 r_kp  = __ldg(&a_rho[idx+nij]);
      const float_sw4 r_km  = __ldg(&a_rho[idx-nij]);

      #pragma unroll 3
      for (int c=0; c<3; c++) {
        // Gather shared values into registers to reduce repeated shared mem traffic
        const float_sw4 s00     = shu[c][0][tj][ti];
        const float_sw4 s0_ip1  = shu[c][0][tj][ti+1];
        const float_sw4 s0_ip2  = shu[c][0][tj][ti+2];
        const float_sw4 s0_im1  = shu[c][0][tj][ti-1];
        const float_sw4 s0_im2  = shu[c][0][tj][ti-2];

        const float_sw4 s0_jp1  = shu[c][0][tj+1][ti];
        const float_sw4 s0_jp2  = shu[c][0][tj+2][ti];
        const float_sw4 s0_jm1  = shu[c][0][tj-1][ti];
        const float_sw4 s0_jm2  = shu[c][0][tj-2][ti];

        const float_sw4 s1      = shu[c][1][tj][ti];
        const float_sw4 s2      = shu[c][2][tj][ti];

        // X-direction 2nd derivative stencils
        const float_sw4 dx2_ip = (s0_ip2 - 2.0 * s0_ip1 + s00);
        const float_sw4 dx2_i  = (s0_ip1 - 2.0 * s00     + s0_im1);
        const float_sw4 dx2_im = (s00     - 2.0 * s0_im1 + s0_im2);

        const float_sw4 termx =
          wx * ( r_ip1 * dcx_i3 * dx2_ip - 2.0 * r_c * dcx_i2 * dx2_i + r_im1 * dcx_i1 * dx2_im );

        // Y-direction 2nd derivative stencils
        const float_sw4 dy2_jp = (s0_jp2 - 2.0 * s0_jp1 + s00);
        const float_sw4 dy2_j  = (s0_jp1 - 2.0 * s00     + s0_jm1);
        const float_sw4 dy2_jm = (s00     - 2.0 * s0_jm1 + s0_jm2);

        const float_sw4 termy =
          wy * ( r_jp * dcy_jp1 * dy2_jp - 2.0 * r_c * dcy_j * dy2_j + r_jm * dcy_jm1 * dy2_jm );

        // Z-direction 2nd derivative stencils (uses rolling registers u_km1/u_km2)
        const float_sw4 dz2_kp = (s2 - 2.0 * s1 + s00);
        const float_sw4 dz2_k  = (s1 - 2.0 * s00 + u_km1[c]);
        const float_sw4 dz2_km = (s00 - 2.0 * u_km1[c] + u_km2[c]);

        const float_sw4 termz =
          wz * ( r_kp * dcz_kp1 * dz2_kp - 2.0 * r_c * dcz_k * dz2_k + r_km * dcz_km1 * dz2_km );

        up(c,idx) -= birho * (termx + termy + termz);

        // Roll the z-direction history
        u_km2[c] = u_km1[c];
        u_km1[c] = s00;
      }
    }

  }
}