__global__ void dpdmt_dev( int ifirst, int ilast, int jfirst, int jlast, int kfirst, int klast,
			   float_sw4* up, float_sw4* u, float_sw4* um,
			   float_sw4* u2, float_sw4 dt2i, int ghost_points )
{
   const size_t myi = static_cast<size_t>(threadIdx.x) + static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x);
   const size_t nthreads = static_cast<size_t>(gridDim.x) * static_cast<size_t>(blockDim.x);
   const size_t npts = static_cast<size_t>(ilast - ifirst + 1) *
                       static_cast<size_t>(jlast - jfirst + 1) *
                       static_cast<size_t>(klast - kfirst + 1);
   const size_t N = npts * static_cast<size_t>(3);

   const float_sw4 c = dt2i;

   // Create restricted, read-only views for better caching on Hopper
   const float_sw4* __restrict__ cup = up;
   const float_sw4* __restrict__ cu  = u;
   const float_sw4* __restrict__ cum = um;
   float_sw4* __restrict__ cu2 = u2;

   const size_t stride = nthreads;

   // Manually unrolled grid-stride loop to increase ILP and hide memory latency
   for (size_t base = myi; base < N; base += stride * 8)
   {
      size_t i0 = base + 0 * stride;
      size_t i1 = base + 1 * stride;
      size_t i2 = base + 2 * stride;
      size_t i3 = base + 3 * stride;
      size_t i4 = base + 4 * stride;
      size_t i5 = base + 5 * stride;
      size_t i6 = base + 6 * stride;
      size_t i7 = base + 7 * stride;

      if (i0 < N) {
         float_sw4 up0 = __ldg(cup + i0);
         float_sw4 u0  = __ldg(cu  + i0);
         float_sw4 um0 = __ldg(cum + i0);
         cu2[i0] = (up0 - static_cast<float_sw4>(2) * u0 + um0) * c;
      }
      if (i1 < N) {
         float_sw4 up1 = __ldg(cup + i1);
         float_sw4 u1  = __ldg(cu  + i1);
         float_sw4 um1 = __ldg(cum + i1);
         cu2[i1] = (up1 - static_cast<float_sw4>(2) * u1 + um1) * c;
      }
      if (i2 < N) {
         float_sw4 up2 = __ldg(cup + i2);
         float_sw4 u2v = __ldg(cu  + i2);
         float_sw4 um2 = __ldg(cum + i2);
         cu2[i2] = (up2 - static_cast<float_sw4>(2) * u2v + um2) * c;
      }
      if (i3 < N) {
         float_sw4 up3 = __ldg(cup + i3);
         float_sw4 u3  = __ldg(cu  + i3);
         float_sw4 um3 = __ldg(cum + i3);
         cu2[i3] = (up3 - static_cast<float_sw4>(2) * u3 + um3) * c;
      }
      if (i4 < N) {
         float_sw4 up4 = __ldg(cup + i4);
         float_sw4 u4  = __ldg(cu  + i4);
         float_sw4 um4 = __ldg(cum + i4);
         cu2[i4] = (up4 - static_cast<float_sw4>(2) * u4 + um4) * c;
      }
      if (i5 < N) {
         float_sw4 up5 = __ldg(cup + i5);
         float_sw4 u5  = __ldg(cu  + i5);
         float_sw4 um5 = __ldg(cum + i5);
         cu2[i5] = (up5 - static_cast<float_sw4>(2) * u5 + um5) * c;
      }
      if (i6 < N) {
         float_sw4 up6 = __ldg(cup + i6);
         float_sw4 u6  = __ldg(cu  + i6);
         float_sw4 um6 = __ldg(cum + i6);
         cu2[i6] = (up6 - static_cast<float_sw4>(2) * u6 + um6) * c;
      }
      if (i7 < N) {
         float_sw4 up7 = __ldg(cup + i7);
         float_sw4 u7  = __ldg(cu  + i7);
         float_sw4 um7 = __ldg(cum + i7);
         cu2[i7] = (up7 - static_cast<float_sw4>(2) * u7 + um7) * c;
      }
   }
}