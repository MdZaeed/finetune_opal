__global__ void rhs4_v2 (int ifirst, int ilast, int jfirst, int jlast, int kfirst, int klast,
int ni, int nj, int nk,
float_sw4* __restrict__ a_up,
const float_sw4* __restrict__ a_u,
const float_sw4* __restrict__ a_um,
const float_sw4* __restrict__ a_mu,
const float_sw4* __restrict__ a_lambda,
const float_sw4* __restrict__ a_rho,
const float_sw4* __restrict__ a_fo,
const float_sw4* __restrict__ a_strx,
const float_sw4* __restrict__ a_stry,
const float_sw4* __restrict__ a_strz,
const float_sw4 h,
const float_sw4 dt) {
int index;
const float_sw4 i6 = 1.0/6;
const float_sw4 i144 = 1.0/144;
const float_sw4 tf = 0.75;
float_sw4 strx_im2, strx_im1, strx_i, strx_ip1, strx_ip2;
float_sw4 stry_jm2, stry_jm1, stry_j, stry_jp1, stry_jp2;
float_sw4 strz_km2, strz_km1, strz_k, strz_kp1, strz_kp2;
float_sw4 la_km2, la_km1, la_ijk, la_kp1, la_kp2;
float_sw4 la_im2, la_im1, la_ip1, la_ip2;
float_sw4 la_jm2, la_jm1, la_jp1, la_jp2;
float_sw4 mu_km2, mu_km1, mu_ijk, mu_kp1, mu_kp2;
float_sw4 mu_im2, mu_im1, mu_ip1, mu_ip2;
float_sw4 mu_jm2, mu_jm1, mu_jp1, mu_jp2;
float_sw4 fo0, fo1, fo2, rho;
float_sw4 up0, up1, up2;
float_sw4 um0, um1, um2;
float_sw4 a0, a1, a2, b0, b1, b2, c0, c1, c2, d0, d1, d2;
float_sw4 la_next_k_val_0, mu_next_k_val_0;
float_sw4 la_next_k_val_1, mu_next_k_val_1;
float_sw4 la_next_k_val_2, mu_next_k_val_2;
float_sw4 la_next_k_val_3, mu_next_k_val_3;
float_sw4 r1, r2, r3, cof;
int active=0, loader=0, loady2=0, loadx2=0;
__shared__ float_sw4 shu[3][5][BY+4][BX+4];
__shared__ float_sw4 sh_lambda[5][BY+4][BX+4];
__shared__ float_sw4 sh_mu[5][BY+4][BX+4];
const int i = threadIdx.x + blockIdx.x * BX;
const int j = jfirst + threadIdx.y + blockIdx.y * BY;
const int ti = threadIdx.x + 2;
const int tj = threadIdx.y + 2;
const int tk = 2;
const int nij = ni * nj;
const int nijk = nij * nk;
int kthm0=3, kthm1=2, kthm2=1, kthm3=0, kthm4=4, kthtmp;
if (i >= ifirst && i <= ilast && j <= jlast)
active = 1;
if (j-2 <= jlast + 2) {
if (i-2 >= 0 && i-2 < ni)
loader = 1;
if (threadIdx.x < 4 && i+BX-2 < ni)
loadx2 = 1;
if (threadIdx.y < 4 && j+BY-2 <= jlast+2)
loady2 = 1;
}
index = (kfirst - 2) * nij + (j - 2) * ni + i - 2;
if (loader) {
int idx = index;
shu[0][0][threadIdx.y][threadIdx.x] = u(0,idx);
shu[1][0][threadIdx.y][threadIdx.x] = u(1,idx);
shu[2][0][threadIdx.y][threadIdx.x] = u(2,idx);
sh_lambda[kthm4][threadIdx.y][threadIdx.x] = a_lambda[idx];
sh_mu[kthm4][threadIdx.y][threadIdx.x] = a_mu[idx];
idx += nij;
shu[0][1][threadIdx.y][threadIdx.x] = u(0,idx);
shu[1][1][threadIdx.y][threadIdx.x] = u(1,idx);
shu[2][1][threadIdx.y][threadIdx.x] = u(2,idx);
sh_lambda[kthm3][threadIdx.y][threadIdx.x] = a_lambda[idx];
sh_mu[kthm3][threadIdx.y][threadIdx.x] = a_mu[idx];
idx += nij;
shu[0][2][threadIdx.y][threadIdx.x] = u(0,idx);
shu[1][2][threadIdx.y][threadIdx.x] = u(1,idx);
shu[2][2][threadIdx.y][threadIdx.x] = u(2,idx);
sh_lambda[kthm2][threadIdx.y][threadIdx.x] = a_lambda[idx];
sh_mu[kthm2][threadIdx.y][threadIdx.x] = a_mu[idx];
idx += nij;
shu[0][3][threadIdx.y][threadIdx.x] = u(0,idx);
shu[1][3][threadIdx.y][threadIdx.x] = u(1,idx);
shu[2][3][threadIdx.y][threadIdx.x] = u(2,idx);
sh_lambda[kthm1][threadIdx.y][threadIdx.x] = a_lambda[idx];
sh_mu[kthm1][threadIdx.y][threadIdx.x] = a_mu[idx];
idx += nij;
a0 = u(0,idx);
a1 = u(1,idx);
a2 = u(2,idx);
la_next_k_val_0 = a_lambda[idx];
mu_next_k_val_0 = a_mu[idx];
}
if (loadx2) {
int idx = index + BX;
shu[0][0][threadIdx.y][threadIdx.x+BX] = u(0,idx);
shu[1][0][threadIdx.y][threadIdx.x+BX] = u(1,idx);
shu[2][0][threadIdx.y][threadIdx.x+BX] = u(2,idx);
sh_lambda[kthm4][threadIdx.y][threadIdx.x+BX] = a_lambda[idx];
sh_mu[kthm4][threadIdx.y][threadIdx.x+BX] = a_mu[idx];
idx += nij;
shu[0][1][threadIdx.y][threadIdx.x+BX] = u(0,idx);
shu[1][1][threadIdx.y][threadIdx.x+BX] = u(1,idx);
shu[2][1][threadIdx.y][threadIdx.x+BX] = u(2,idx);
sh_lambda[kthm3][threadIdx.y][threadIdx.x+BX] = a_lambda[idx];
sh_mu[kthm3][threadIdx.y][threadIdx.x+BX] = a_mu[idx];
idx += nij;
shu[0][2][threadIdx.y][threadIdx.x+BX] = u(0,idx);
shu[1][2][threadIdx.y][threadIdx.x+BX] = u(1,idx);
shu[2][2][threadIdx.y][threadIdx.x+BX] = u(2,idx);
sh_lambda[kthm2][threadIdx.y][threadIdx.x+BX] = a_lambda[idx];
sh_mu[kthm2][threadIdx.y][threadIdx.x+BX] = a_mu[idx];
idx += nij;
shu[0][3][threadIdx.y][threadIdx.x+BX] = u(0,idx);
shu[1][3][threadIdx.y][threadIdx.x+BX] = u(1,idx);
shu[2][3][threadIdx.y][threadIdx.x+BX] = u(2,idx);
sh_lambda[kthm1][threadIdx.y][threadIdx.x+BX] = a_lambda[idx];
sh_mu[kthm1][threadIdx.y][threadIdx.x+BX] = a_mu[idx];
idx += nij;
b0 = u(0,idx);
b1 = u(1,idx);
b2 = u(2,idx);
la_next_k_val_1 = a_lambda[idx];
mu_next_k_val_1 = a_mu[idx];
}
if (loader && loady2) {
int idx = index + BY * ni;
shu[0][0][threadIdx.y+BY][threadIdx.x] = u(0,idx);
shu[1][0][threadIdx.y+BY][threadIdx.x] = u(1,idx);
shu[2][0][threadIdx.y+BY][threadIdx.x] = u(2,idx);
sh_lambda[kthm4][threadIdx.y+BY][threadIdx.x] = a_lambda[idx];
sh_mu[kthm4][threadIdx.y+BY][threadIdx.x] = a_mu[idx];
idx += nij;
shu[0][1][threadIdx.y+BY][threadIdx.x] = u(0,idx);
shu[1][1][threadIdx.y+BY][threadIdx.x] = u(1,idx);
shu[2][1][threadIdx.y+BY][threadIdx.x] = u(2,idx);
sh_lambda[kthm3][threadIdx.y+BY][threadIdx.x] = a_lambda[idx];
sh_mu[kthm3][threadIdx.y+BY][threadIdx.x] = a_mu[idx];
idx += nij;
shu[0][2][threadIdx.y+BY][threadIdx.x] = u(0,idx);
shu[1][2][threadIdx.y+BY][threadIdx.x] = u(1,idx);
shu[2][2][threadIdx.y+BY][threadIdx.x] = u(2,idx);
sh_lambda[kthm2][threadIdx.y+BY][threadIdx.x] = a_lambda[idx];
sh_mu[kthm2][threadIdx.y+BY][threadIdx.x] = a_mu[idx];
idx += nij;
shu[0][3][threadIdx.y+BY][threadIdx.x] = u(0,idx);
shu[1][3][threadIdx.y+BY][threadIdx.x] = u(1,idx);
shu[2][3][threadIdx.y+BY][threadIdx.x] = u(2,idx);
sh_lambda[kthm1][threadIdx.y+BY][threadIdx.x] = a_lambda[idx];
sh_mu[kthm1][threadIdx.y+BY][threadIdx.x] = a_mu[idx];
idx += nij;
c0 = u(0,idx);
c1 = u(1,idx);
c2 = u(2,idx);
la_next_k_val_2 = a_lambda[idx];
mu_next_k_val_2 = a_mu[idx];
}
if (loadx2 && loady2) {
int idx = index + BY * ni + BX;
shu[0][0][threadIdx.y+BY][threadIdx.x+BX] = u(0,idx);
shu[1][0][threadIdx.y+BY][threadIdx.x+BX] = u(1,idx);
shu[2][0][threadIdx.y+BY][threadIdx.x+BX] = u(2,idx);
sh_lambda[kthm4][threadIdx.y+BY][threadIdx.x+BX] = a_lambda[idx];
sh_mu[kthm4][threadIdx.y+BY][threadIdx.x+BX] = a_mu[idx];
idx += nij;
shu[0][1][threadIdx.y+BY][threadIdx.x+BX] = u(0,idx);
shu[1][1][threadIdx.y+BY][threadIdx.x+BX] = u(1,idx);
shu[2][1][threadIdx.y+BY][threadIdx.x+BX] = u(2,idx);
sh_lambda[kthm3][threadIdx.y+BY][threadIdx.x+BX] = a_lambda[idx];
sh_mu[kthm3][threadIdx.y+BY][threadIdx.x+BX] = a_mu[idx];
idx += nij;
shu[0][2][threadIdx.y+BY][threadIdx.x+BX] = u(0,idx);
shu[1][2][threadIdx.y+BY][threadIdx.x+BX] = u(1,idx);
shu[2][2][threadIdx.y+BY][threadIdx.x+BX] = u(2,idx);
sh_lambda[kthm2][threadIdx.y+BY][threadIdx.x+BX] = a_lambda[idx];
sh_mu[kthm2][threadIdx.y+BY][threadIdx.x+BX] = a_mu[idx];
idx += nij;
shu[0][3][threadIdx.y+BY][threadIdx.x+BX] = u(0,idx);
shu[1][3][threadIdx.y+BY][threadIdx.x+BX] = u(1,idx);
shu[2][3][threadIdx.y+BY][threadIdx.x+BX] = u(2,idx);
sh_lambda[kthm1][threadIdx.y+BY][threadIdx.x+BX] = a_lambda[idx];
sh_mu[kthm1][threadIdx.y+BY][threadIdx.x+BX] = a_mu[idx];
idx += nij;
d0 = u(0,idx);
d1 = u(1,idx);
d2 = u(2,idx);
la_next_k_val_3 = a_lambda[idx];
mu_next_k_val_3 = a_mu[idx];
}
if (active) {
strx_im2 = a_strx[i-2];
strx_im1 = a_strx[i-1];
strx_i = a_strx[i];
strx_ip1 = a_strx[i+1];
strx_ip2 = a_strx[i+2];
stry_jm2 = a_stry[j-2];
stry_jm1 = a_stry[j-1];
stry_j = a_stry[j];
stry_jp1 = a_stry[j+1];
stry_jp2 = a_stry[j+2];
// These are for the current thread's (i,j) location, loaded into registers
int current_k_idx_base = (kfirst - 2) * nij + j * ni + i;
la_km2 = a_lambda[current_k_idx_base];
mu_km2 = a_mu[current_k_idx_base];
current_k_idx_base += nij;
la_km1 = a_lambda[current_k_idx_base];
mu_km1 = a_mu[current_k_idx_base];
current_k_idx_base += nij;
la_ijk = a_lambda[current_k_idx_base];
mu_ijk = a_mu[current_k_idx_base];
current_k_idx_base += nij;
la_kp1 = a_lambda[current_k_idx_base];
mu_kp1 = a_mu[current_k_idx_base];
}
index += 4 * nij; // index now points to kfirst+2 layer for (j-2, i-2)
strz_km2 = a_strz[kfirst-2];
strz_km1 = a_strz[kfirst-1];
strz_k = a_strz[kfirst];
strz_kp1 = a_strz[kfirst+1];
cof = 1.0/(h*h);
for (int k = kfirst; k <= klast; k++) {
kthtmp = kthm4;
kthm4 = kthm3;
kthm3 = kthm2;
kthm2 = kthm1;
kthm1 = kthm0;
kthm0 = kthtmp;
__syncthreads();
shu[0][kthm0][threadIdx.y][threadIdx.x] = a0;
shu[1][kthm0][threadIdx.y][threadIdx.x] = a1;
shu[2][kthm0][threadIdx.y][threadIdx.x] = a2;
if (loader) { // loader covers the base threadIdx.x, threadIdx.y
    sh_lambda[kthm0][threadIdx.y][threadIdx.x] = la_next_k_val_0;
    sh_mu[kthm0][threadIdx.y][threadIdx.x] = mu_next_k_val_0;
}
if (threadIdx.x < 4) {
    if (loadx2) {
        shu[0][kthm0][threadIdx.y][threadIdx.x+BX] = b0;
        shu[1][kthm0][threadIdx.y][threadIdx.x+BX] = b1;
        shu[2][kthm0][threadIdx.y][threadIdx.x+BX] = b2;
        sh_lambda[kthm0][threadIdx.y][threadIdx.x+BX] = la_next_k_val_1;
        sh_mu[kthm0][threadIdx.y][threadIdx.x+BX] = mu_next_k_val_1;
    }
}
if (threadIdx.y < 4) {
    if (loader && loady2) {
        shu[0][kthm0][threadIdx.y+BY][threadIdx.x] = c0;
        shu[1][kthm0][threadIdx.y+BY][threadIdx.x] = c1;
        shu[2][kthm0][threadIdx.y+BY][threadIdx.x] = c2;
        sh_lambda[kthm0][threadIdx.y+BY][threadIdx.x] = la_next_k_val_2;
        sh_mu[kthm0][threadIdx.y+BY][threadIdx.x] = mu_next_k_val_2;
    }
    if (threadIdx.x < 4) {
        if (loadx2 && loady2) {
            shu[0][kthm0][threadIdx.y+BY][threadIdx.x+BX] = d0;
            shu[1][kthm0][threadIdx.y+BY][threadIdx.x+BX] = d1;
            shu[2][kthm0][threadIdx.y+BY][threadIdx.x+BX] = d2;
            sh_lambda[kthm0][threadIdx.y+BY][threadIdx.x+BX] = la_next_k_val_3;
            sh_mu[kthm0][threadIdx.y+BY][threadIdx.x+BX] = mu_next_k_val_3;
        }
    }
}
__syncthreads();
if (k < klast) {
    int current_k_idx_u = index + nij; // index is for k+1, so +nij for k+2
    if (loader) {
        a0 = u(0,current_k_idx_u);
        a1 = u(1,current_k_idx_u);
        a2 = u(2,current_k_idx_u);
        la_next_k_val_0 = a_lambda[current_k_idx_u];
        mu_next_k_val_0 = a_mu[current_k_idx_u];
    }
    int current_k_idx_u_bx = current_k_idx_u + BX;
    if (loadx2) {
        b0 = u(0,current_k_idx_u_bx);
        b1 = u(1,current_k_idx_u_bx);
        b2 = u(2,current_k_idx_u_bx);
        la_next_k_val_1 = a_lambda[current_k_idx_u_bx];
        mu_next_k_val_1 = a_mu[current_k_idx_u_bx];
    }
    int current_k_idx_u_by = current_k_idx_u + BY * ni;
    if (loader & loady2) {
        c0 = u(0,current_k_idx_u_by);
        c1 = u(1,current_k_idx_u_by);
        c2 = u(2,current_k_idx_u_by);
        la_next_k_val_2 = a_lambda[current_k_idx_u_by];
        mu_next_k_val_2 = a_mu[current_k_idx_u_by];
    }
    int current_k_idx_u_bx_by = current_k_idx_u + BY * ni + BX;
    if (loadx2 && loady2) {
        d0 = u(0,current_k_idx_u_bx_by);
        d1 = u(1,current_k_idx_u_bx_by);
        d2 = u(2,current_k_idx_u_bx_by);
        la_next_k_val_3 = a_lambda[current_k_idx_u_bx_by];
        mu_next_k_val_3 = a_mu[current_k_idx_u_bx_by];
    }
}
if (active) {
    strz_kp2 = a_strz[k+2];
    {
    int idx = k * nij + j * ni + i; // This idx is for k,j,i
    la_kp2 = sh_lambda[kthm0][tj][ti]; // kthm0 holds k+2 layer
    mu_kp2 = sh_mu[kthm0][tj][ti];     // kthm0 holds k+2 layer

    // These are for the current k layer (kthm2) with i/j offsets
    la_im2 = sh_lambda[kthm2][tj][ti-2];
    la_im1 = sh_lambda[kthm2][tj][ti-1];
    la_ip1 = sh_lambda[kthm2][tj][ti+1];
    la_ip2 = sh_lambda[kthm2][tj][ti+2];
    la_jm2 = sh_lambda[kthm2][tj-2][ti];
    la_jm1 = sh_lambda[kthm2][tj-1][ti];
    la_jp1 = sh_lambda[kthm2][tj+1][ti];
    la_jp2 = sh_lambda[kthm2][tj+2][ti];
    mu_im2 = sh_mu[kthm2][tj][ti-2];
    mu_im1 = sh_mu[kthm2][tj][ti-1];
    mu_ip1 = sh_mu[kthm2][tj][ti+1];
    mu_ip2 = sh_mu[kthm2][tj][ti+2];
    mu_jm2 = sh_mu[kthm2][tj-2][ti];
    mu_jm1 = sh_mu[kthm2][tj-1][ti];
    mu_jp1 = sh_mu[kthm2][tj+1][ti];
    mu_jp2 = sh_mu[kthm2][tj+2][ti];
    fo0 = fo(0,idx);
    fo1 = fo(1,idx);
    fo2 = fo(2,idx);
    rho = a_rho[idx];
    if (pred) {
    um0 = um(0,idx);
    um1 = um(1,idx);
    um2 = um(2,idx);
    }
    else {
    up0 = up(0,idx);
    up1 = up(1,idx);
    up2 = up(2,idx);
    }
    }
    {
    float_sw4 mux1, mux2, mux3, mux4;
    mux1 = mu_im1 * strx_im1 - tf * (mu_ijk * strx_i + mu_im2 * strx_im2);
    mux2 = mu_im2 * strx_im2 + mu_ip1 * strx_ip1 + 3 * (mu_ijk * strx_i + mu_im1 * strx_im1);
    mux3 = mu_im1 * strx_im1 + mu_ip2 * strx_ip2 + 3 * (mu_ip1 * strx_ip1 + mu_ijk * strx_i );
    mux4 = mu_ip1 * strx_ip1 - tf * (mu_ijk * strx_i + mu_ip2 * strx_ip2);
    r1 = strx_i * ((2 * mux1 + la_im1 * strx_im1 - tf * (la_ijk * strx_i + la_im2 * strx_im2)) *
    (shu[0][kthm2][tj+0][ti-2] - shu[0][kthm2][tj+0][ti+0]) +
    (2 * mux2 + la_im2 * strx_im2 + la_ip1 * strx_ip1 +
    3 * (la_ijk * strx_i + la_im1 * strx_im1)) *
    (shu[0][kthm2][tj+0][ti-1] - shu[0][kthm2][tj+0][ti+0]) +
    (2 * mux3 + la_im1 * strx_im1 + la_ip2 * strx_ip2 +
    3 * (la_ip1 * strx_ip1 + la_ijk * strx_i )) *
    (shu[0][kthm2][tj+0][ti+1] - shu[0][kthm2][tj+0][ti+0]) +
    (2 * mux4 + la_ip1 * strx_ip1 - tf * (la_ijk * strx_i + la_ip2 * strx_ip2)) *
    (shu[0][kthm2][tj+0][ti+2] - shu[0][kthm2][tj+0][ti+0]));
    r2 = strx_i * (mux1 * (shu[1][kthm2][tj+0][ti-2] - shu[1][kthm2][tj+0][ti+0]) +
    mux2 * (shu[1][kthm2][tj+0][ti-1] - shu[1][kthm2][tj+0][ti+0]) +
    mux3 * (shu[1][kthm2][tj+0][ti+1] - shu[1][kthm2][tj+0][ti+0]) +
    mux4 * (shu[1][kthm2][tj+0][ti+2] - shu[1][kthm2][tj+0][ti+0]));
    r3 = strx_i * (mux1 * (shu[2][kthm2][tj+0][ti-2] - shu[2][kthm2][tj+0][ti+0]) +
    mux2 * (shu[2][kthm2][tj+0][ti-1] - shu[2][kthm2][tj+0][ti+0]) +
    mux3 * (shu[2][kthm2][tj+0][ti+1] - shu[2][kthm2][tj+0][ti+0]) +
    mux4 * (shu[2][kthm2][tj+0][ti+2] - shu[2][kthm2][tj+0][ti+0]));
    }
    {
    float_sw4 muy1, muy2, muy3, muy4;
    muy1 = mu_jm1 * stry_jm1 - tf * (mu_ijk * stry_j + mu_jm2 * stry_jm2);
    muy2 = mu_jm2 * stry_jm2 + mu_jp1 * stry_jp1 + 3 * (mu_ijk * stry_j + mu_jm1 * stry_jm1);
    muy3 = mu_jm1 * stry_jm1 + mu_jp2 * stry_jp2 + 3 * (mu_jp1 * stry_jp1 + mu_ijk * stry_j );
    muy4 = mu_jp1 * stry_jp1 - tf * (mu_ijk * stry_j + mu_jp2 * stry_jp2);
    r1 += stry_j * (muy1 * (shu[0][kthm2][tj-2][ti+0] - shu[0][kthm2][tj+0][ti+0]) +
    muy2 * (shu[0][kthm2][tj-1][ti+0] - shu[0][kthm2][tj+0][ti+0]) +
    muy3 * (shu[0][kthm2][tj+1][ti+0] - shu[0][kthm2][tj+0][ti+0]) +
    muy4 * (shu[0][kthm2][tj+2][ti+0] - shu[0][kthm2][tj+0][ti+0]));
    r2 += stry_j * ((2 * muy1 + la_jm1 * stry_jm1 - tf * (la_ijk * stry_j + la_jm2 * stry_jm2)) *
    (shu[1][kthm2][tj-2][ti+0] - shu[1][kthm2][tj+0][ti+0]) +
    (2 * muy2 + la_jm2 * stry_jm2 + la_jp1 * stry_jp1 +
    3 * (la_ijk * stry_j + la_jm1 * stry_jm1)) *
    (shu[1][kthm2][tj-1][ti+0] - shu[1][kthm2][tj+0][ti+0]) +
    (2 * muy3 + la_jm1 * stry_jm1 + la_jp2 * stry_jp2 +
    3 * (la_jp1 * stry_jp1 + la_ijk * stry_j )) *
    (shu[1][kthm2][tj+1][ti+0] - shu[1][kthm2][tj+0][ti+0]) +
    (2 * muy4 + la_jp1 * stry_jp1 - tf * (la_ijk * stry_j + la_jp2 * stry_jp2)) *
    (shu[1][kthm2][tj+2][ti+0] - shu[1][kthm2][tj+0][ti+0]));
    r3 += stry_j *(muy1 * (shu[2][kthm2][tj-2][ti+0] - shu[2][kthm2][tj+0][ti+0]) +
    muy2 * (shu[2][kthm2][tj-1][ti+0] - shu[2][kthm2][tj+0][ti+0]) +
    muy3 * (shu[2][kthm2][tj+1][ti+0] - shu[2][kthm2][tj+0][ti+0]) +
    muy4 * (shu[2][kthm2][tj+2][ti+0] - shu[2][kthm2][tj+0][ti+0]));
    }
    {
    float_sw4 muz1, muz2, muz3, muz4;
    muz1 = mu_km1 * strz_km1 - tf * (mu_ijk * strz_k + mu_km2 * strz_km2);
    muz2 = mu_km2 * strz_km2 + mu_kp1 * strz_kp1 + 3 * (mu_ijk * strz_k + mu_km1 * strz_km1);
    muz3 = mu_km1 * strz_km1 + mu_kp2 * strz_kp2 + 3 * (mu_kp1 * strz_kp1 + mu_ijk * strz_k);
    muz4 = mu_kp1 * strz_kp1 - tf * (mu_ijk * strz_k + mu_kp2 * strz_kp2);
    r1 += strz_k * (muz1 * (shu[0][kthm4][tj+0][ti+0] - shu[0][kthm2][tj+0][ti+0]) +
    muz2 * (shu[0][kthm3][tj+0][ti+0] - shu[0][kthm2][tj+0][ti+0]) +
    muz3 * (shu[0][kthm1][tj+0][ti+0] - shu[0][kthm2][tj+0][ti+0]) +
    muz4 * (shu[0][kthm0][tj+0][ti+0] - shu[0][kthm2][tj+0][ti+0]));
    r2 += strz_k * (muz1 * (shu[1][kthm4][tj+0][ti+0] - shu[1][kthm2][tj+0][ti+0]) +
    muz2 * (shu[1][kthm3][tj+0][ti+0] - shu[1][kthm2][tj+0][ti+0]) +
    muz3 * (shu[1][kthm1][tj+0][ti+0] - shu[1][kthm2][tj+0][ti+0]) +
    muz4 * (shu[1][kthm0][tj+0][ti+0] - shu[1][kthm2][tj+0][ti+0]));
    r3 += strz_k * ((2 * muz1 + la_km1 * strz_km1 - tf * (la_ijk * strz_k + la_km2 * strz_km2)) *
    (shu[2][kthm4][tj+0][ti+0] - shu[2][kthm2][tj+0][ti+0]) +
    (2 * muz2 + la_km2 * strz_km2 + la_kp1 * strz_kp1 +
    3 * (la_ijk * strz_k + la_km1 * strz_km1)) *
    (shu[2][kthm3][tj+0][ti+0] - shu[2][kthm2][tj+0][ti+0]) +
    (2 * muz3 + la_km1 * strz_km1 + la_kp2 * strz_kp2 +
    3 * (la_kp1 * strz_kp1 + la_ijk * strz_k )) *
    (shu[2][kthm1][tj+0][ti+0] - shu[2][kthm2][tj+0][ti+0]) +
    (2 * muz4 + la_kp1 * strz_kp1 - tf * (la_ijk * strz_k + la_kp2 * strz_kp2)) *
    (shu[2][kthm0][tj+0][ti+0] - shu[2][kthm2][tj+0][ti+0]));
    }
    r1 *= i6;
    r2 *= i6;
    r3 *= i6;
    r1 += strx_i * stry_j * i144 *
    (la_im2 * (shu[1][kthm2][tj-2][ti-2] - shu[1][kthm2][tj+2][ti-2] + 8 *
    (-shu[1][kthm2][tj-1][ti-2] + shu[1][kthm2][tj+1][ti-2])) - 8 *
    (la_im1 * (shu[1][kthm2][tj-2][ti-1] - shu[1][kthm2][tj+2][ti-1] + 8 *
    (-shu[1][kthm2][tj-1][ti-1] + shu[1][kthm2][tj+1][ti-1]))) + 8 *
    (la_ip1 * (shu[1][kthm2][tj-2][ti+1] - shu[1][kthm2][tj+2][ti+1] + 8 *
    (-shu[1][kthm2][tj-1][ti+1] + shu[1][kthm2][tj+1][ti+1]))) -
    (la_ip2 * (shu[1][kthm2][tj-2][ti+2] - shu[1][kthm2][tj+2][ti+2] + 8 *
    (-shu[1][kthm2][tj-1][ti+2] + shu[1][kthm2][tj+1][ti+2]))))
    + strx_i * strz_k * i144 *
    (la_im2 * (shu[2][kthm4][tj+0][ti-2] - shu[2][kthm0][tj+0][ti-2] + 8 *
    (-shu[2][kthm3][tj+0][ti-2] + shu[2][kthm1][tj+0][ti-2])) - 8 *
    (la_im1 * (shu[2][kthm4][tj+0][ti-1] - shu[2][kthm0][tj+0][ti-1] + 8 *
    (-shu[2][kthm3][tj+0][ti-1] + shu[2][kthm1][tj+0][ti-1]))) + 8 *
    (la_ip1 * (shu[2][kthm4][tj+0][ti+1] - shu[2][kthm0][tj+0][ti+1] + 8 *
    (-shu[2][kthm3][tj+0][ti+1] + shu[2][kthm1][tj+0][ti+1]))) -
    (la_ip2 * (shu[2][kthm4][tj+0][ti+2] - shu[2][kthm0][tj+0][ti+2] + 8 *
    (-shu[2][kthm3][tj+0][ti+2] + shu[2][kthm1][tj+0][ti+2]))))
    + strx_i * stry_j * i144 *
    (mu_jm2 * (shu[1][kthm2][tj-2][ti-2] - shu[1][kthm2][tj-2][ti+2] + 8 *
    (-shu[1][kthm2][tj-2][ti-1] + shu[1][kthm2][tj-2][ti+1])) - 8 *
    (mu_jm1 * (shu[1][kthm2][tj-1][ti-2] - shu[1][kthm2][tj-1][ti+2] + 8 *
    (-shu[1][kthm2][tj-1][ti-1] + shu[1][kthm2][tj-1][ti+1]))) + 8 *
    (mu_jp1 * (shu[1][kthm2][tj+1][ti-2] - shu[1][kthm2][tj+1][ti+2] + 8 *
    (-shu[1][kthm2][tj+1][ti-1] + shu[1][kthm2][tj+1][ti+1]))) -
    (mu_jp2 * (shu[1][kthm2][tj+2][ti-2] - shu[1][kthm2][tj+2][ti+2] + 8 *
    (-shu[1][kthm2][tj+2][ti-1] + shu[1][kthm2][tj+2][ti+1]))))
    + strx_i * strz_k * i144 *
    (mu_km2 * (shu[2][kthm4][tj+0][ti-2] - shu[2][kthm4][tj+0][ti+2] + 8 *
    (-shu[2][kthm4][tj+0][ti-1] + shu[2][kthm4][tj+0][ti+1])) - 8 *
    (mu_km1 * (shu[2][kthm3][tj+0][ti-2] - shu[2][kthm3][tj+0][ti+2] + 8 *
    (-shu[2][kthm3][tj+0][ti-1] + shu[2][kthm3][tj+0][ti+1]))) + 8 *
    (mu_kp1 * (shu[2][kthm1][tj+0][ti-2] - shu[2][kthm1][tj+0][ti+2] + 8 *
    (-shu[2][kthm1][tj+0][ti-1] + shu[2][kthm1][tj+0][ti+1]))) -
    (mu_kp2 * (shu[2][kthm0][tj+0][ti-2] - shu[2][kthm0][tj+0][ti+2] + 8 *
    (-shu[2][kthm0][tj+0][ti-1] + shu[2][kthm0][tj+0][ti+1]))));
    r2 += strx_i *stry_j *i144*
    (mu_im2 * (shu[0][kthm2][tj-2][ti-2] - shu[0][kthm2][tj+2][ti-2] + 8 *
    (-shu[0][kthm2][tj-1][ti-2] + shu[0][kthm2][tj+1][ti-2])) - 8 *
    (mu_im1 * (shu[0][kthm2][tj-2][ti-1] - shu[0][kthm2][tj+2][ti-1] + 8 *
    (-shu[0][kthm2][tj-1][ti-1] + shu[0][kthm2][tj+1][ti-1]))) + 8 *
    (mu_ip1 * (shu[0][kthm2][tj-2][ti+1] - shu[0][kthm2][tj+2][ti+1] + 8 *
    (-shu[0][kthm2][tj-1][ti+1] + shu[0][kthm2][tj+1][ti+1]))) -
    (mu_ip2 * (shu[0][kthm2][tj-2][ti+2] - shu[0][kthm2][tj+2][ti+2] + 8 *
    (-shu[0][kthm2][tj-1][ti+2] + shu[0][kthm2][tj+1][ti+2]))))
    + strx_i * stry_j * i144 *
    (la_jm2 * (shu[0][kthm2][tj-2][ti-2] - shu[0][kthm2][tj-2][ti+2] + 8 *
    (-shu[0][kthm2][tj-2][ti-1] + shu[0][kthm2][tj-2][ti+1])) - 8 *
    (la_jm1 * (shu[0][kthm2][tj-1][ti-2] - shu[0][kthm2][tj-1][ti+2] + 8 *
    (-shu[0][kthm2][tj-1][ti-1] + shu[0][kthm2][tj-1][ti+1]))) + 8 *
    (la_jp1 * (shu[0][kthm2][tj+1][ti-2] - shu[0][kthm2][tj+1][ti+2] + 8 *
    (-shu[0][kthm2][tj+1][ti-1] + shu[0][kthm2][tj+1][ti+1]))) -
    (la_jp2 * (shu[0][kthm2][tj+2][ti-2] - shu[0][kthm2][tj+2][ti+2] + 8 *
    (-shu[0][kthm2][tj+2][ti-1] + shu[0][kthm2][tj+2][ti+1]))))
    + stry_j * strz_k * i144 *
    (la_jm2 * (shu[2][kthm4][tj-2][ti+0] - shu[2][kthm0][tj-2][ti+0] + 8 *
    (-shu[2][kthm3][tj-2][ti+0] + shu[2][kthm1][tj-2][ti+0])) - 8 *
    (la_jm1 * (shu[2][kthm4][tj-1][ti+0] - shu[2][kthm0][tj-1][ti+0] + 8 *
    (-shu[2][kthm3][tj-1][ti+0] + shu[2][kthm1][tj-1][ti+0]))) + 8 *
    (la_jp1 * (shu[2][kthm4][tj+1][ti+0] - shu[2][kthm0][tj+1][ti+0] + 8 *
    (-shu[2][kthm3][tj+1][ti+0] + shu[2][kthm1][tj+1][ti+0]))) -
    (la_jp2 * (shu[2][kthm4][tj+2][ti+0] - shu[2][kthm0][tj+2][ti+0] + 8 *
    (-shu[2][kthm3][tj+2][ti+0] + shu[2][kthm1][tj+2][ti+0]))))
    + stry_j * strz_k * i144 *
    (mu_km2 * (shu[2][kthm4][tj-2][ti+0] - shu[2][kthm4][tj+2][ti+0] + 8 *
    (-shu[2][kthm4][tj-1][ti+0] + shu[2][kthm4][tj+1][ti+0])) - 8 *
    (mu_km1 * (shu[2][kthm3][tj-2][ti+0] - shu[2][kthm3][tj+2][ti+0] + 8 *
    (-shu[2][kthm3][tj-1][ti+0] + shu[2][kthm3][tj+1][ti+0]))) + 8 *
    (mu_kp1 * (shu[2][kthm1][tj-2][ti+0] - shu[2][kthm1][tj+2][ti+0] + 8 *
    (-shu[2][kthm1][tj-1][ti+0] + shu[2][kthm1][tj+1][ti+0]))) -
    (mu_kp2 * (shu[2][kthm0][tj-2][ti+0] - shu[2][kthm0][tj+2][ti+0] + 8 *
    (-shu[2][kthm0][tj-1][ti+0] + shu[2][kthm0][tj+1][ti+0]))));
    r3 += strx_i * strz_k * i144 *
    (mu_im2 * (shu[0][kthm4][tj+0][ti-2] - shu[0][kthm0][tj+0][ti-2] + 8 *
    (-shu[0][kthm3][tj+0][ti-2] + shu[0][kthm1][tj+0][ti-2])) - 8 *
    (mu_im1 * (shu[0][kthm4][tj+0][ti-1] - shu[0][kthm0][tj+0][ti-1] + 8 *
    (-shu[0][kthm3][tj+0][ti-1] + shu[0][kthm1][tj+0][ti-1]))) + 8 *
    (mu_ip1 * (shu[0][kthm4][tj+0][ti+1] - shu[0][kthm0][tj+0][ti+1] + 8 *
    (-shu[0][kthm3][tj+0][ti+1] + shu[0][kthm1][tj+0][ti+1]))) -
    (mu_ip2 * (shu[0][kthm4][tj+0][ti+2] - shu[0][kthm0][tj+0][ti+2] + 8 *
    (-shu[0][kthm3][tj+0][ti+2] + shu[0][kthm1][tj+0][ti+2]))))
    + stry_j *strz_k*i144*
    (mu_jm2 * (shu[1][kthm4][tj-2][ti+0] - shu[1][kthm0][tj-2][ti+0] + 8 *
    (- shu[1][kthm3][tj-2][ti+0]+shu[1][kthm1][tj-2][ti+0])) - 8 *
    (mu_jm1 * (shu[1][kthm4][tj-1][ti+0] - shu[1][kthm0][tj-1][ti+0] + 8 *
    (- shu[1][kthm3][tj-1][ti+0]+shu[1][kthm1][tj-1][ti+0]))) + 8 *
    (mu_jp1 * (shu[1][kthm4][tj+1][ti+0] - shu[1][kthm0][tj+1][ti+0] + 8 *
    (- shu[1][kthm3][tj+1][ti+0]+shu[1][kthm1][tj+1][ti+0]))) -
    (mu_jp2 * (shu[1][kthm4][tj+2][ti+0] - shu[1][kthm0][tj+2][ti+0] + 8 *
    (- shu[1][kthm3][tj+2][ti+0]+shu[1][kthm1][tj+2][ti+0]))))
    + strx_i *strz_k*i144*
    (la_km2*(shu[0][kthm4][tj+0][ti-2] - shu[0][kthm4][tj+0][ti+2] + 8 *
    (- shu[0][kthm4][tj+0][ti-1]+shu[0][kthm4][tj+0][ti+1])) - 8*
    (la_km1*(shu[0][kthm3][tj+0][ti-2] - shu[0][kthm3][tj+0][ti+2] + 8 *
    (- shu[0][kthm3][tj+0][ti-1]+shu[0][kthm3][tj+0][ti+1]))) + 8*
    (la_kp1*(shu[0][kthm1][tj+0][ti-2] - shu[0][kthm1][tj+0][ti+2] + 8 *
    (- shu[0][kthm1][tj+0][ti-1]+shu[0][kthm1][tj+0][ti+1]))) -
    (la_kp2*(shu[0][kthm0][tj+0][ti-2] - shu[0][kthm0][tj+0][ti+2] + 8 *
    (- shu[0][kthm0][tj+0][ti-1]+shu[0][kthm0][tj+0][ti+1]))))
    + stry_j *strz_k*i144*
    (la_km2*(shu[1][kthm4][tj-2][ti+0] - shu[1][kthm4][tj+2][ti+0] + 8 *
    (- shu[1][kthm4][tj-1][ti+0]+shu[1][kthm4][tj+1][ti+0])) - 8*
    (la_km1*(shu[1][kthm3][tj-2][ti+0] - shu[1][kthm3][tj+2][ti+0] + 8 *
    (- shu[1][kthm3][tj-1][ti+0]+shu[1][kthm3][tj+1][ti+0]))) + 8*
    (la_kp1*(shu[1][kthm1][tj-2][ti+0] - shu[1][kthm1][tj+2][ti+0] + 8 *
    (- shu[1][kthm1][tj-1][ti+0]+shu[1][kthm1][tj+1][ti+0]))) -
    (la_kp2*(shu[1][kthm0][tj-2][ti+0] - shu[1][kthm0][tj+2][ti+0] + 8 *
    (- shu[1][kthm0][tj-1][ti+0]+shu[1][kthm0][tj+1][ti+0]))));
    if (pred) {
    int idx = k * nij + j * ni + i;
    float_sw4 fact = dt*dt / rho;
    up(0,idx) = 2 * shu[0][kthm2][tj][ti] - um0 + fact * (cof * r1 + fo0);
    up(1,idx) = 2 * shu[1][kthm2][tj][ti] - um1 + fact * (cof * r2 + fo1);
    up(2,idx) = 2 * shu[2][kthm2][tj][ti] - um2 + fact * (cof * r3 + fo2);
    }
    else {
    int idx = k * nij + j * ni + i;
    float_sw4 fact = dt*dt*dt*dt / (12 * rho);
    up(0,idx) = up0 + fact * (cof * r1 + fo0);
    up(1,idx) = up1 + fact * (cof * r2 + fo1);
    up(2,idx) = up2 + fact * (cof * r3 + fo2);
    }
    strz_km2 = strz_km1;
    strz_km1 = strz_k;
    strz_k = strz_kp1;
    strz_kp1 = strz_kp2;
    la_km2 = la_km1;
    la_km1 = la_ijk;
    la_ijk = la_kp1;
    la_kp1 = la_kp2;
    mu_km2 = mu_km1;
    mu_km1 = mu_ijk;
    mu_ijk = mu_kp1;
    mu_kp1 = mu_kp2;
}
index += nij;
}
}