#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <string>
#include <iostream>
#include <cuda_runtime.h>

using namespace std;

#define SKINNY64_ROUNDS 36

// Loading the lookup tables in GPU constant memory
__constant__ uint8_t S_GPU[16]         = {0xc,0x6,0x9,0x0,0x1,0xa,0x2,0xb,0x3,0x8,0x5,0xd,0x4,0xe,0x7,0xf};
__constant__ uint8_t S_inverse_GPU[16] = {0x3,0x4,0x6,0x8,0xc,0xa,0x1,0xe,0x9,0x2,0x5,0x7,0x0,0xb,0xd,0xf};
__constant__ uint8_t P_GPU[16]         = {0x0,0x1,0x2,0x3,0x7,0x4,0x5,0x6,0xa,0xb,0x8,0x9,0xd,0xe,0xf,0xc};
__constant__ uint8_t P_inverse_GPU[16] = {0x0,0x1,0x2,0x3,0x5,0x6,0x7,0x4,0xa,0xb,0x8,0x9,0xf,0xc,0xd,0xe};
__constant__ uint8_t Q_GPU[16]         = {0x9,0xf,0x8,0xd,0xa,0xe,0xc,0xb,0x0,0x1,0x2,0x3,0x4,0x5,0x6,0x7};
__constant__ uint8_t RC_GPU[36]        = {
    0x01,0x03,0x07,0x0F,0x1F,0x3E,0x3D,0x3B,0x37,0x2F,0x1E,0x3C,
    0x39,0x33,0x27,0x0E,0x1D,0x3A,0x35,0x2B,0x16,0x2C,0x18,0x30,
    0x21,0x02,0x05,0x0B,0x17,0x2E,0x1C,0x38,0x31,0x23,0x06,0x0D
};
__constant__ uint8_t RTK_GPU[SKINNY64_ROUNDS][8];

// Host side CPU lookup tables
const uint8_t S[16]         = {0xc,0x6,0x9,0x0,0x1,0xa,0x2,0xb,0x3,0x8,0x5,0xd,0x4,0xe,0x7,0xf};
const uint8_t S_inverse[16] = {0x3,0x4,0x6,0x8,0xc,0xa,0x1,0xe,0x9,0x2,0x5,0x7,0x0,0xb,0xd,0xf};
const uint8_t P[16]         = {0x0,0x1,0x2,0x3,0x7,0x4,0x5,0x6,0xa,0xb,0x8,0x9,0xd,0xe,0xf,0xc};
const uint8_t P_inverse[16] = {0x0,0x1,0x2,0x3,0x5,0x6,0x7,0x4,0xa,0xb,0x8,0x9,0xf,0xc,0xd,0xe};
const uint8_t Q[16]         = {0x9,0xf,0x8,0xd,0xa,0xe,0xc,0xb,0x0,0x1,0x2,0x3,0x4,0x5,0x6,0x7};
const uint8_t Round_Constant[36] = {
    0x01,0x03,0x07,0x0F,0x1F,0x3E,0x3D,0x3B,0x37,0x2F,0x1E,0x3C,
    0x39,0x33,0x27,0x0E,0x1D,0x3A,0x35,0x2B,0x16,0x2C,0x18,0x30,
    0x21,0x02,0x05,0x0B,0x17,0x2E,0x1C,0x38,0x31,0x23,0x06,0x0D
};

// Macro to check for CUDA errors
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d - %s\n",                      \
                    __FILE__, __LINE__, cudaGetErrorString(err));               \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

// ===========================================================================
// CPU helpers
// ===========================================================================

uint8_t hex_char_to_val(char c)
{
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return 0;
}

void print_message(uint8_t *msg, int num_blocks)
{
    for (int i = 0; i < num_blocks * 16; i++) printf("%01x", msg[i]);
    printf("\n");
}

void convert_hex_string_to_statearray(string hex_string, uint8_t int_array[16], bool reversed = false)
{
    for (size_t i = 0; i < 16; i++) {
        uint8_t val = hex_char_to_val(hex_string[i]);
        if (reversed) int_array[15 - i] = val;
        else          int_array[i]       = val;
    }
}

uint8_t TK2_lfsr(uint8_t x)
{
    x = (x << 1) ^ ((x >> 3) & 0x1) ^ ((x >> 2) & 0x1);
    return x & 0xf;
}

void round_tweakey_schedule(int rounds, uint8_t TK1[][16], uint8_t TK2[][16],
                             uint8_t round_tweakey[][8])
{
    uint8_t TKp1[16], TKp2[16];
    for (uint8_t i = 0; i < 16; i++) {
        TK1[0][i] = TK1[0][i] & 0xf;
        TK2[0][i] = TK2[0][i] & 0xf;
    }
    for (uint8_t i = 0; i < 8; i++)
        round_tweakey[0][i] = TK1[0][i] ^ TK2[0][i];
    for (int r = 1; r < rounds; r++) {
        for (int i = 0; i < 16; i++) {
            TKp1[i] = TK1[r-1][Q[i]];
            TKp2[i] = TK2[r-1][Q[i]];
        }
        for (int i = 0; i < 16; i++) {
            TK1[r][i] = TKp1[i];
            TK2[r][i] = (i < 8) ? TK2_lfsr(TKp2[i]) : TKp2[i];
        }
        for (int i = 0; i < 8; i++)
            round_tweakey[r][i] = TK1[r][i] ^ TK2[r][i];
    }
}

// ===========================================================================
// GPU device functions
// ===========================================================================

__device__ void gpu_mix_columns(uint8_t state[16])
{
    uint8_t tmp;
    for (uint8_t j = 0; j < 4; j++) {
        state[j + 4*1] ^= state[j + 4*2];
        state[j + 4*2] ^= state[j + 4*0];
        state[j + 4*3] ^= state[j + 4*2];
        tmp             = state[j + 4*3];
        state[j + 4*3]  = state[j + 4*2];
        state[j + 4*2]  = state[j + 4*1];
        state[j + 4*1]  = state[j + 4*0];
        state[j + 4*0]  = tmp;
    }
}

__device__ void gpu_inverse_mix_columns(uint8_t state[16])
{
    uint8_t tmp;
    for (uint8_t j = 0; j < 4; j++) {
        tmp             = state[j + 4*3];
        state[j + 4*3]  = state[j + 4*0];
        state[j + 4*0]  = state[j + 4*1];
        state[j + 4*1]  = state[j + 4*2];
        state[j + 4*2]  = tmp;
        state[j + 4*3] ^= state[j + 4*2];
        state[j + 4*2] ^= state[j + 4*0];
        state[j + 4*1] ^= state[j + 4*2];
    }
}

__device__ void gpu_encryption_block(int R, const uint8_t in[16], uint8_t out[16])
{
    for (int i = 0; i < 16; i++) out[i] = in[i] & 0xf;
    for (int r = 0; r < R; r++) {
        for (int i = 0; i < 16; i++) out[i] = S_GPU[out[i]];
        out[0] ^= (RC_GPU[r] & 0xf);
        out[4] ^= ((RC_GPU[r] >> 4) & 0x3);
        out[8] ^= 0x2;
        for (int i = 0; i < 8; i++) out[i] ^= RTK_GPU[r][i];
        uint8_t tmp[16];
        for (int i = 0; i < 16; i++) tmp[i] = out[i];
        for (int i = 0; i < 16; i++) out[i] = tmp[P_GPU[i]];
        gpu_mix_columns(out);
    }
}

__device__ void gpu_decryption_block(int R, const uint8_t in[16], uint8_t out[16])
{
    for (int i = 0; i < 16; i++) out[i] = in[i] & 0xf;
    for (int r = 0; r < R; r++) {
        gpu_inverse_mix_columns(out);
        uint8_t tmp[16];
        for (int i = 0; i < 16; i++) tmp[i] = out[i];
        for (int i = 0; i < 16; i++) out[i] = tmp[P_inverse_GPU[i]];
        int index = R - r - 1;
        for (int i = 0; i < 8; i++) out[i] ^= RTK_GPU[index][i];
        out[0] ^= (RC_GPU[index] & 0xf);
        out[4] ^= ((RC_GPU[index] >> 4) & 0x3);
        out[8] ^= 0x2;
        for (int i = 0; i < 16; i++) out[i] = S_inverse_GPU[out[i]];
    }
}

// ===========================================================================
// Kernels
// ===========================================================================

__global__ void kernel_ctr_encrypt(int R,
    const uint8_t * __restrict__ plaintext,
          uint8_t * __restrict__ ciphertext,
    int num_blocks, const uint8_t * __restrict__ nonce)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= num_blocks) return;
    uint8_t counter[16];
    for (int i = 0; i < 16; i++) counter[i] = nonce[i];
    int carry = id;
    for (int i = 15; i >= 0 && carry; i--) {
        carry += counter[i]; counter[i] = carry & 0xf; carry >>= 4;
    }
    uint8_t keystream[16];
    gpu_encryption_block(R, counter, keystream);
    for (int i = 0; i < 16; i++)
        ciphertext[id*16+i] = (plaintext[id*16+i] ^ keystream[i]) & 0xf;
}

__global__ void kernel_ctr_decrypt(int R,
    const uint8_t * __restrict__ ciphertext,
          uint8_t * __restrict__ plaintext,
    int num_blocks, const uint8_t * __restrict__ nonce)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= num_blocks) return;
    uint8_t counter[16];
    for (int i = 0; i < 16; i++) counter[i] = nonce[i];
    int carry = id;
    for (int i = 15; i >= 0 && carry; i--) {
        carry += counter[i]; counter[i] = carry & 0xf; carry >>= 4;
    }
    uint8_t keystream[16];
    gpu_encryption_block(R, counter, keystream);
    for (int i = 0; i < 16; i++)
        plaintext[id*16+i] = (ciphertext[id*16+i] ^ keystream[i]) & 0xf;
}

__global__ void kernel_cbc_encrypt(int R,
    const uint8_t * __restrict__ plaintext,
          uint8_t * __restrict__ ciphertext,
    int num_blocks, const uint8_t * __restrict__ iv)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    uint8_t prev[16], block_in[16];
    for (int i = 0; i < 16; i++) prev[i] = iv[i];
    for (int b = 0; b < num_blocks; b++) {
        for (int i = 0; i < 16; i++)
            block_in[i] = (plaintext[b*16+i] ^ prev[i]) & 0xf;
        gpu_encryption_block(R, block_in, &ciphertext[b*16]);
        for (int i = 0; i < 16; i++) prev[i] = ciphertext[b*16+i];
    }
}

__global__ void kernel_cbc_decrypt(int R,
    const uint8_t * __restrict__ ciphertext,
          uint8_t * __restrict__ plaintext,
    int num_blocks, const uint8_t * __restrict__ iv)
{
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= num_blocks) return;
    uint8_t block_out[16];
    gpu_decryption_block(R, &ciphertext[b*16], block_out);
    const uint8_t *prev = (b == 0) ? iv : &ciphertext[(b-1)*16];
    for (int i = 0; i < 16; i++)
        plaintext[b*16+i] = (block_out[i] ^ prev[i]) & 0xf;
}

// ===========================================================================
// Host wrappers
// ===========================================================================

void gpu_ctr_encrypt(int R, uint8_t *h_plain, uint8_t *h_cipher,
                     int num_blocks, uint8_t nonce[16])
{
    size_t sz = num_blocks * 16;
    uint8_t *d_plain, *d_cipher, *d_nonce;
    CUDA_CHECK(cudaMalloc(&d_plain,  sz)); CUDA_CHECK(cudaMalloc(&d_cipher, sz));
    CUDA_CHECK(cudaMalloc(&d_nonce,  16));
    CUDA_CHECK(cudaMemcpy(d_plain, h_plain, sz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_nonce, nonce,   16, cudaMemcpyHostToDevice));
    int threads = 256, blocks = (num_blocks + threads - 1) / threads;
    kernel_ctr_encrypt<<<blocks, threads>>>(R, d_plain, d_cipher, num_blocks, d_nonce);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_cipher, d_cipher, sz, cudaMemcpyDeviceToHost));
    cudaFree(d_plain); cudaFree(d_cipher); cudaFree(d_nonce);
}

void gpu_ctr_decrypt(int R, uint8_t *h_cipher, uint8_t *h_plain,
                     int num_blocks, uint8_t nonce[16])
{
    size_t sz = num_blocks * 16;
    uint8_t *d_cipher, *d_plain, *d_nonce;
    CUDA_CHECK(cudaMalloc(&d_cipher, sz)); CUDA_CHECK(cudaMalloc(&d_plain, sz));
    CUDA_CHECK(cudaMalloc(&d_nonce, 16));
    CUDA_CHECK(cudaMemcpy(d_cipher, h_cipher, sz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_nonce,  nonce,    16, cudaMemcpyHostToDevice));
    int threads = 256, blocks = (num_blocks + threads - 1) / threads;
    kernel_ctr_decrypt<<<blocks, threads>>>(R, d_cipher, d_plain, num_blocks, d_nonce);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_plain, d_plain, sz, cudaMemcpyDeviceToHost));
    cudaFree(d_cipher); cudaFree(d_plain); cudaFree(d_nonce);
}

void gpu_cbc_encrypt(int R, uint8_t *h_plain, uint8_t *h_cipher,
                     int num_blocks, uint8_t iv[16])
{
    size_t sz = num_blocks * 16;
    uint8_t *d_plain, *d_cipher, *d_iv;
    CUDA_CHECK(cudaMalloc(&d_plain, sz)); CUDA_CHECK(cudaMalloc(&d_cipher, sz));
    CUDA_CHECK(cudaMalloc(&d_iv, 16));
    CUDA_CHECK(cudaMemcpy(d_plain, h_plain, sz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_iv,    iv,      16, cudaMemcpyHostToDevice));
    kernel_cbc_encrypt<<<1, 1>>>(R, d_plain, d_cipher, num_blocks, d_iv);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_cipher, d_cipher, sz, cudaMemcpyDeviceToHost));
    cudaFree(d_plain); cudaFree(d_cipher); cudaFree(d_iv);
}

void gpu_cbc_decrypt(int R, uint8_t *h_cipher, uint8_t *h_plain,
                     int num_blocks, uint8_t iv[16])
{
    size_t sz = num_blocks * 16;
    uint8_t *d_cipher, *d_plain, *d_iv;
    CUDA_CHECK(cudaMalloc(&d_cipher, sz)); CUDA_CHECK(cudaMalloc(&d_plain, sz));
    CUDA_CHECK(cudaMalloc(&d_iv, 16));
    CUDA_CHECK(cudaMemcpy(d_cipher, h_cipher, sz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_iv,     iv,       16, cudaMemcpyHostToDevice));
    int threads = 256, blocks = (num_blocks + threads - 1) / threads;
    kernel_cbc_decrypt<<<blocks, threads>>>(R, d_cipher, d_plain, num_blocks, d_iv);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_plain, d_plain, sz, cudaMemcpyDeviceToHost));
    cudaFree(d_cipher); cudaFree(d_plain); cudaFree(d_iv);
}

// ===========================================================================
// Validation
// ===========================================================================

int assert_equal(const char *label, uint8_t *got, uint8_t *expected, int len)
{
    if (memcmp(got, expected, len) == 0) {
        printf("The recovered plaintext matches the original plaintext %s\n", label);
        return 1;
    }
    printf("MISMATCH %s\n  got:      ", label);
    for (int i = 0; i < len; i++) printf("%01x", got[i]);
    printf("\n  expected: ");
    for (int i = 0; i < len; i++) printf("%01x", expected[i]);
    printf("\n");
    return 0;
}

// ===========================================================================
// Throughput benchmark (existing)
// ===========================================================================

#define BUF_BLOCKS 256
#define TRIALS     31

static float benchmark_kernel_ms(int R, int num_blocks, bool do_ctr)
{
    size_t sz = (size_t)num_blocks * 16;
    uint8_t *h_buf    = new uint8_t[sz];
    uint8_t *h_cipher = new uint8_t[sz];
    for (size_t i = 0; i < sz; i++) h_buf[i] = (uint8_t)(i & 0xf);
    uint8_t iv_nonce[16] = {0};
    if (!do_ctr) gpu_cbc_encrypt(R, h_buf, h_cipher, num_blocks, iv_nonce);

    size_t bsz = sz * sizeof(uint8_t);
    uint8_t *d_in, *d_out, *d_iv;
    CUDA_CHECK(cudaMalloc(&d_in, bsz)); CUDA_CHECK(cudaMalloc(&d_out, bsz));
    CUDA_CHECK(cudaMalloc(&d_iv, 16));
    CUDA_CHECK(cudaMemcpy(d_in, do_ctr ? h_buf : h_cipher, bsz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_iv, iv_nonce, 16, cudaMemcpyHostToDevice));
    int threads = 256, gblocks = (num_blocks + threads - 1) / threads;

    if (do_ctr) kernel_ctr_encrypt<<<gblocks, threads>>>(R, d_in, d_out, num_blocks, d_iv);
    else        kernel_cbc_decrypt<<<gblocks, threads>>>(R, d_in, d_out, num_blocks, d_iv);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start)); CUDA_CHECK(cudaEventCreate(&stop));
    float samples[TRIALS];
    for (int t = 0; t < TRIALS; t++) {
        CUDA_CHECK(cudaEventRecord(start));
        if (do_ctr) kernel_ctr_encrypt<<<gblocks, threads>>>(R, d_in, d_out, num_blocks, d_iv);
        else        kernel_cbc_decrypt<<<gblocks, threads>>>(R, d_in, d_out, num_blocks, d_iv);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0; CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        samples[t] = ms;
    }
    for (int i = 0; i < TRIALS-1; i++)
        for (int j = i+1; j < TRIALS; j++)
            if (samples[j] < samples[i]) { float t = samples[i]; samples[i] = samples[j]; samples[j] = t; }
    float median = samples[TRIALS/2];
    cudaEventDestroy(start); cudaEventDestroy(stop);
    cudaFree(d_in); cudaFree(d_out); cudaFree(d_iv);
    delete[] h_buf; delete[] h_cipher;
    return median;
}

// ===========================================================================
// NEW 1: Single 64-byte block latency
// ===========================================================================

static void measure_single_block_latency(int R)
{
    const int LAT_BLOCKS = 8;
    const int LAT_TRIALS = 101;

    size_t sz = LAT_BLOCKS * 16 * sizeof(uint8_t);
    uint8_t h_plain [LAT_BLOCKS * 16];
    uint8_t h_cipher[LAT_BLOCKS * 16];
    uint8_t h_recov [LAT_BLOCKS * 16];
    uint8_t nonce[16] = {0};
    uint8_t iv[16]    = {0};
    for (int i = 0; i < LAT_BLOCKS * 16; i++) h_plain[i] = (uint8_t)(i & 0xf);

    uint8_t *d_plain, *d_cipher, *d_nonce, *d_iv;
    CUDA_CHECK(cudaMalloc(&d_plain,  sz)); CUDA_CHECK(cudaMalloc(&d_cipher, sz));
    CUDA_CHECK(cudaMalloc(&d_nonce,  16)); CUDA_CHECK(cudaMalloc(&d_iv,     16));
    CUDA_CHECK(cudaMemcpy(d_nonce, nonce, 16, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_iv,    iv,    16, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start)); CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaMemcpy(d_plain, h_plain, sz, cudaMemcpyHostToDevice));
    kernel_ctr_encrypt<<<1, LAT_BLOCKS>>>(R, d_plain, d_cipher, LAT_BLOCKS, d_nonce);
    CUDA_CHECK(cudaMemcpy(h_cipher, d_cipher, sz, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());

    float ctr_samples[LAT_TRIALS];
    for (int t = 0; t < LAT_TRIALS; t++) {
        CUDA_CHECK(cudaEventRecord(start));
        CUDA_CHECK(cudaMemcpy(d_plain, h_plain, sz, cudaMemcpyHostToDevice));
        kernel_ctr_encrypt<<<1, LAT_BLOCKS>>>(R, d_plain, d_cipher, LAT_BLOCKS, d_nonce);
        CUDA_CHECK(cudaMemcpy(h_cipher, d_cipher, sz, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0; CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ctr_samples[t] = ms * 1000.0f;
    }

    gpu_cbc_encrypt(R, h_plain, h_cipher, LAT_BLOCKS, iv);

    float cbc_samples[LAT_TRIALS];
    for (int t = 0; t < LAT_TRIALS; t++) {
        CUDA_CHECK(cudaEventRecord(start));
        CUDA_CHECK(cudaMemcpy(d_cipher, h_cipher, sz, cudaMemcpyHostToDevice));
        kernel_cbc_decrypt<<<1, LAT_BLOCKS>>>(R, d_cipher, d_plain, LAT_BLOCKS, d_iv);
        CUDA_CHECK(cudaMemcpy(h_recov, d_plain, sz, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0; CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        cbc_samples[t] = ms * 1000.0f;
    }

    for (int i = 0; i < LAT_TRIALS-1; i++)
        for (int j = i+1; j < LAT_TRIALS; j++) {
            if (ctr_samples[j] < ctr_samples[i]) { float t=ctr_samples[i]; ctr_samples[i]=ctr_samples[j]; ctr_samples[j]=t; }
            if (cbc_samples[j] < cbc_samples[i]) { float t=cbc_samples[i]; cbc_samples[i]=cbc_samples[j]; cbc_samples[j]=t; }
        }

    printf("\n--------Single 64-Byte Block Latency--------\n");
    printf("  Input size : %d cipher blocks = 64 nibble-bytes\n", LAT_BLOCKS);
    printf("  Trials     : %d  \n\n", LAT_TRIALS);
    printf("  %-35s : %8.2f us\n", "CTR encrypt latency",  ctr_samples[LAT_TRIALS/2]);
    printf("  %-35s : %8.2f us\n", "CBC decrypt latency",  cbc_samples[LAT_TRIALS/2]);

    cudaEventDestroy(start); cudaEventDestroy(stop);
    cudaFree(d_plain); cudaFree(d_cipher); cudaFree(d_nonce); cudaFree(d_iv);
}


static void measure_key_schedule_cost(uint8_t tweakey_1[16], uint8_t tweakey_2[16], int R)
{
    const int KS_TRIALS = 101;

    uint8_t TK_1[SKINNY64_ROUNDS][16];
    uint8_t TK_2[SKINNY64_ROUNDS][16];
    uint8_t RTK [SKINNY64_ROUNDS][8];

    for (int j = 0; j < 16; j++) { TK_1[0][j] = tweakey_1[j]; TK_2[0][j] = tweakey_2[j]; }
    round_tweakey_schedule(R, TK_1, TK_2, RTK);
    CUDA_CHECK(cudaMemcpyToSymbol(RTK_GPU, RTK, sizeof(RTK)));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start)); CUDA_CHECK(cudaEventCreate(&stop));

    float ks_samples[KS_TRIALS];
    for (int t = 0; t < KS_TRIALS; t++) {
        for (int j = 0; j < 16; j++) { TK_1[0][j] = tweakey_1[j]; TK_2[0][j] = tweakey_2[j]; }
        CUDA_CHECK(cudaEventRecord(start));
        round_tweakey_schedule(R, TK_1, TK_2, RTK);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0; CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ks_samples[t] = ms * 1000.0f;
    }

    float up_samples[KS_TRIALS];
    for (int t = 0; t < KS_TRIALS; t++) {
        CUDA_CHECK(cudaEventRecord(start));
        CUDA_CHECK(cudaMemcpyToSymbol(RTK_GPU, RTK, sizeof(RTK)));
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0; CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        up_samples[t] = ms * 1000.0f;
    }

    for (int i = 0; i < KS_TRIALS-1; i++)
        for (int j = i+1; j < KS_TRIALS; j++) {
            if (ks_samples[j] < ks_samples[i]) { float t=ks_samples[i]; ks_samples[i]=ks_samples[j]; ks_samples[j]=t; }
            if (up_samples[j] < up_samples[i]) { float t=up_samples[i]; up_samples[i]=up_samples[j]; up_samples[j]=t; }
        }
    float ks_med = ks_samples[KS_TRIALS/2];
    float up_med = up_samples[KS_TRIALS/2];

    size_t rtk_bytes     = SKINNY64_ROUNDS * 8;

    printf("\n--------Key Schedule Cost--------\n");
    printf("\n  Time (median of %d trials)\n", KS_TRIALS);
    printf("  %-48s : %7.2f us\n", "CPU compute  (round_tweakey_schedule)",  ks_med);
    printf("  %-48s : %7.2f us\n", "GPU upload   (cudaMemcpyToSymbol, PCIe)", up_med);
    printf("  %-48s : %7.2f us\n", "Total key setup cost",                    ks_med + up_med);
    printf("\n  Space\n");
    printf("  %-48s : %3zu bytes  (%d rounds x 8 nibble-bytes)\n",
           "Expanded RTK in GPU constant memory", rtk_bytes, SKINNY64_ROUNDS);
    
   

    cudaEventDestroy(start); cudaEventDestroy(stop);
}


int main()
{
    printf("--------SKINNY-64-128 CUDA Implementation--------\n");
    printf("--------CTR and CBC Mode Tests--------\n\n");

    int R = SKINNY64_ROUNDS;

    uint8_t TK_1[SKINNY64_ROUNDS][16];
    uint8_t TK_2[SKINNY64_ROUNDS][16];
    uint8_t RTK[SKINNY64_ROUNDS][8];
    uint8_t tweakey_1[16], tweakey_2[16];

    string TK1_string = "9eb93640d088da63";
    string TK2_string = "76a39d1c8bea71e1";
    convert_hex_string_to_statearray(TK1_string, tweakey_1, false);
    convert_hex_string_to_statearray(TK2_string, tweakey_2, false);
    for (uint8_t i = 0; i < 16; i++) { TK_1[0][i] = tweakey_1[i]; TK_2[0][i] = tweakey_2[i]; }

    printf("Generating tweakey schedule\n");
    round_tweakey_schedule(R, TK_1, TK_2, RTK);
    CUDA_CHECK(cudaMemcpyToSymbol(RTK_GPU, RTK, sizeof(RTK)));
    printf("Tweakey schedule generated and uploaded to GPU\n\n");

    string plain_string = "cf16cfe8fd0f98aa";
    uint8_t plain_text_0[16];
    convert_hex_string_to_statearray(plain_string, plain_text_0, false);

    // CTR mode test
    printf("--------CTR Mode (GPU)--------\n\n");
    int num_blocks = 1;
    uint8_t ctr_plain[16], ctr_cipher[16], ctr_recovered[16];
    uint8_t nonce[16] = {0};
    for (int i = 0; i < 16; i++) ctr_plain[i] = plain_text_0[i];
    printf("Original Plaintext   : "); print_message(ctr_plain, num_blocks);
    gpu_ctr_encrypt(R, ctr_plain, ctr_cipher, num_blocks, nonce);
    printf("CTR Ciphertext       : "); print_message(ctr_cipher, num_blocks);
    gpu_ctr_decrypt(R, ctr_cipher, ctr_recovered, num_blocks, nonce);
    printf("Recovered Plaintext  : "); print_message(ctr_recovered, num_blocks);
    printf("\n");
    assert_equal("CTR mode", ctr_recovered, ctr_plain, num_blocks * 16);

    // CBC mode test
    printf("\n--------CBC Mode (GPU)--------\n\n");
    uint8_t cbc_plain[16], cbc_cipher[16], cbc_recovered[16];
    uint8_t iv[16] = {0};
    for (int i = 0; i < 16; i++) cbc_plain[i] = plain_text_0[i];
    printf("Original Plaintext   : "); print_message(cbc_plain, num_blocks);
    gpu_cbc_encrypt(R, cbc_plain, cbc_cipher, num_blocks, iv);
    printf("CBC Ciphertext       : "); print_message(cbc_cipher, num_blocks);
    gpu_cbc_decrypt(R, cbc_cipher, cbc_recovered, num_blocks, iv);
    printf("Recovered Plaintext  : "); print_message(cbc_recovered, num_blocks);
    printf("\n");
    assert_equal("CBC mode", cbc_recovered, cbc_plain, num_blocks * 16);

    printf("\n--------CBC and CTR modes are validated--------\n");

    // Throughput benchmark
    printf("\n--------GPU Throughput Benchmark--------\n");
    printf("  Median of %d trials, Buffer = %d blocks (%d nibble-bytes)\n",
           TRIALS, BUF_BLOCKS, BUF_BLOCKS * 8);
    float ctr_ms  = benchmark_kernel_ms(R, BUF_BLOCKS, true);
    float cbcd_ms = benchmark_kernel_ms(R, BUF_BLOCKS, false);
    double data_bytes = BUF_BLOCKS * 8.0;
    printf("  %-40s : %8.4f ms  (~%.6f GB/s)\n", "CTR encrypt",
           ctr_ms,  (data_bytes/1e9)/(ctr_ms/1e3));
    printf("  %-40s : %8.4f ms  (~%.6f GB/s)\n", "CBC decrypt",
           cbcd_ms, (data_bytes/1e9)/(cbcd_ms/1e3));

    // --- NEW benchmarks ---
    measure_single_block_latency(R);
    measure_key_schedule_cost(tweakey_1, tweakey_2, R);

    // Device info
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("\n--------GPU Device--------\n");
    printf("  Device        : %s\n", prop.name);
    printf("  SMs           : %d\n", prop.multiProcessorCount);
    printf("  Clock         : %.0f MHz\n", prop.clockRate / 1e3);
    printf("  Global memory : %.0f MB\n", prop.totalGlobalMem / 1.0e6);

    return 0;
}
