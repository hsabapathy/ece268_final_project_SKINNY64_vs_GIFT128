// GIFT-128 Block Cipher GPU CUDA Implementation

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <iostream>
#include <time.h>
#include <cuda_runtime.h>

using namespace std;

// Constants


#define GIFT_ROUNDS  40
#define BLOCK_BYTES  16
#define KEY_BYTES    16
#define TRIALS       31

// Host-side lookup tables (used to populate __constant__ memory)

static const uint8_t GIFT_S[16]     = {1,10, 4,12, 6,15, 3, 9, 2,13,11, 7, 5, 0, 8,14};
static const uint8_t GIFT_S_inv[16] = {13, 0, 8, 6, 2,12, 4,11,14, 7, 1,10, 3, 9,15, 5};

static const uint8_t GIFT_P[128] = {
     0, 33, 66, 99, 96,  1, 34, 67, 64, 97,  2, 35, 32, 65, 98,  3,
     4, 37, 70,103,100,  5, 38, 71, 68,101,  6, 39, 36, 69,102,  7,
     8, 41, 74,107,104,  9, 42, 75, 72,105, 10, 43, 40, 73,106, 11,
    12, 45, 78,111,108, 13, 46, 79, 76,109, 14, 47, 44, 77,110, 15,
    16, 49, 82,115,112, 17, 50, 83, 80,113, 18, 51, 48, 81,114, 19,
    20, 53, 86,119,116, 21, 54, 87, 84,117, 22, 55, 52, 85,118, 23,
    24, 57, 90,123,120, 25, 58, 91, 88,121, 26, 59, 56, 89,122, 27,
    28, 61, 94,127,124, 29, 62, 95, 92,125, 30, 63, 60, 93,126, 31
};

static const uint8_t GIFT_P_inv[128] = {
     0,  5, 10, 15, 16, 21, 26, 31, 32, 37, 42, 47, 48, 53, 58, 63,
    64, 69, 74, 79, 80, 85, 90, 95, 96,101,106,111,112,117,122,127,
    12,  1,  6, 11, 28, 17, 22, 27, 44, 33, 38, 43, 60, 49, 54, 59,
    76, 65, 70, 75, 92, 81, 86, 91,108, 97,102,107,124,113,118,123,
     8, 13,  2,  7, 24, 29, 18, 23, 40, 45, 34, 39, 56, 61, 50, 55,
    72, 77, 66, 71, 88, 93, 82, 87,104,109, 98,103,120,125,114,119,
     4,  9, 14,  3, 20, 25, 30, 19, 36, 41, 46, 35, 52, 57, 62, 51,
    68, 73, 78, 67, 84, 89, 94, 83,100,105,110, 99,116,121,126,115
};

static const uint8_t GIFT_RC[62] = {
    0x01,0x03,0x07,0x0F,0x1F,0x3E,0x3D,0x3B,0x37,0x2F,
    0x1E,0x3C,0x39,0x33,0x27,0x0E,0x1D,0x3A,0x35,0x2B,
    0x16,0x2C,0x18,0x30,0x21,0x02,0x05,0x0B,0x17,0x2E,
    0x1C,0x38,0x31,0x23,0x06,0x0D,0x1B,0x36,0x2D,0x1A,
    0x34,0x29,0x12,0x24,0x08,0x11,0x22,0x04,0x09,0x13,
    0x26,0x0c,0x19,0x32,0x25,0x0a,0x15,0x2a,0x14,0x28,
    0x10,0x20
};

// CUDA error checker

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d - %s\n",                      \
                    __FILE__, __LINE__, cudaGetErrorString(err));               \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

// GPU __constant__ memory tables

__constant__ uint8_t d_GIFT_S[16];
__constant__ uint8_t d_GIFT_S_inv[16];
__constant__ uint8_t d_GIFT_P[128];
__constant__ uint8_t d_GIFT_P_inv[128];
__constant__ uint8_t d_GIFT_RC[62];

static void upload_constant_tables()
{
    CUDA_CHECK(cudaMemcpyToSymbol(d_GIFT_S,     GIFT_S,     sizeof(GIFT_S)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_GIFT_S_inv, GIFT_S_inv, sizeof(GIFT_S_inv)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_GIFT_P,     GIFT_P,     sizeof(GIFT_P)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_GIFT_P_inv, GIFT_P_inv, sizeof(GIFT_P_inv)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_GIFT_RC,    GIFT_RC,    sizeof(GIFT_RC)));
}

// CPU helpers: pack / unpack nibbles

static void pack_nibbles(const uint8_t nibbles[32], uint8_t bytes[16])
{
    for (int i = 0; i < 16; i++)
        bytes[i] = ((nibbles[31 - 2*i] & 0xF) << 4) | (nibbles[30 - 2*i] & 0xF);
}

static void unpack_nibbles(const uint8_t bytes[16], uint8_t nibbles[32])
{
    for (int i = 0; i < 16; i++) {
        nibbles[31 - 2*i] = (bytes[i] >> 4) & 0xF;
        nibbles[30 - 2*i] =  bytes[i]        & 0xF;
    }
}

// GPU device helpers: pack / unpack nibbles

__device__ __forceinline__
static void d_pack_nibbles(const uint8_t nibbles[32], uint8_t bytes[16])
{
    for (int i = 0; i < 16; i++)
        bytes[i] = ((nibbles[31 - 2*i] & 0xF) << 4) | (nibbles[30 - 2*i] & 0xF);
}

__device__ __forceinline__
static void d_unpack_nibbles(const uint8_t bytes[16], uint8_t nibbles[32])
{
    for (int i = 0; i < 16; i++) {
        nibbles[31 - 2*i] = (bytes[i] >> 4) & 0xF;
        nibbles[30 - 2*i] =  bytes[i]        & 0xF;
    }
}

// CPU key-schedule helpers

static void gift128_precompute_round_keys(const uint8_t key[KEY_BYTES],
                                          uint8_t out_rkeys[GIFT_ROUNDS * 32])
{
    uint8_t k[32], temp_key[32];
    unpack_nibbles(key, k);

    for (int r = 0; r < GIFT_ROUNDS; r++) {
        for (int i = 0; i < 32; i++)
            out_rkeys[r * 32 + i] = k[i];

        for (int i = 0; i < 32; i++)
            temp_key[i] = k[(i+8) % 32];
        for (int i = 0; i < 24; i++) k[i] = temp_key[i];
        k[24] = temp_key[27];
        k[25] = temp_key[24];
        k[26] = temp_key[25];
        k[27] = temp_key[26];
        k[28] = ((temp_key[28]&0xC)>>2) ^ ((temp_key[29]&0x3)<<2);
        k[29] = ((temp_key[29]&0xC)>>2) ^ ((temp_key[30]&0x3)<<2);
        k[30] = ((temp_key[30]&0xC)>>2) ^ ((temp_key[31]&0x3)<<2);
        k[31] = ((temp_key[31]&0xC)>>2) ^ ((temp_key[28]&0x3)<<2);
    }
}

// GPU device functions: cipher core

__device__ __forceinline__
static void d_gift128_enc_nibbles(uint8_t input[32], uint8_t key[32])
{
    uint8_t bits[128], perm_bits[128], key_bits[128], temp_key[32];

    for (int r = 0; r < GIFT_ROUNDS; r++) {
        for (int i = 0; i < 32; i++)
            input[i] = d_GIFT_S[input[i]];

        for (int i = 0; i < 32; i++)
            for (int j = 0; j < 4; j++)
                bits[4*i+j] = (input[i] >> j) & 0x1;
        for (int i = 0; i < 128; i++)
            perm_bits[d_GIFT_P[i]] = bits[i];
        for (int i = 0; i < 32; i++) {
            input[i] = 0;
            for (int j = 0; j < 4; j++)
                input[i] ^= perm_bits[4*i+j] << j;
        }

        for (int i = 0; i < 32; i++)
            for (int j = 0; j < 4; j++)
                bits[4*i+j] = (input[i] >> j) & 0x1;
        for (int i = 0; i < 32; i++)
            for (int j = 0; j < 4; j++)
                key_bits[4*i+j] = (key[i] >> j) & 0x1;
        for (int i = 0; i < 32; i++) {
            bits[4*i+1] ^= key_bits[i];
            bits[4*i+2] ^= key_bits[i+64];
        }

        bits[3]   ^= (d_GIFT_RC[r]     ) & 0x1;
        bits[7]   ^= (d_GIFT_RC[r] >> 1) & 0x1;
        bits[11]  ^= (d_GIFT_RC[r] >> 2) & 0x1;
        bits[15]  ^= (d_GIFT_RC[r] >> 3) & 0x1;
        bits[19]  ^= (d_GIFT_RC[r] >> 4) & 0x1;
        bits[23]  ^= (d_GIFT_RC[r] >> 5) & 0x1;
        bits[127] ^= 1;

        for (int i = 0; i < 32; i++) {
            input[i] = 0;
            for (int j = 0; j < 4; j++)
                input[i] ^= bits[4*i+j] << j;
        }

        for (int i = 0; i < 32; i++)
            temp_key[i] = key[(i+8) % 32];
        for (int i = 0; i < 24; i++) key[i] = temp_key[i];
        key[24] = temp_key[27];
        key[25] = temp_key[24];
        key[26] = temp_key[25];
        key[27] = temp_key[26];
        key[28] = ((temp_key[28]&0xC)>>2) ^ ((temp_key[29]&0x3)<<2);
        key[29] = ((temp_key[29]&0xC)>>2) ^ ((temp_key[30]&0x3)<<2);
        key[30] = ((temp_key[30]&0xC)>>2) ^ ((temp_key[31]&0x3)<<2);
        key[31] = ((temp_key[31]&0xC)>>2) ^ ((temp_key[28]&0x3)<<2);
    }
}

__device__ __forceinline__
static void d_gift128_encrypt_block(const uint8_t plaintext[BLOCK_BYTES],
                                    const uint8_t key[KEY_BYTES],
                                          uint8_t ciphertext[BLOCK_BYTES])
{
    uint8_t state[32], key_nibs[32];
    d_unpack_nibbles(plaintext, state);
    d_unpack_nibbles(key,       key_nibs);
    d_gift128_enc_nibbles(state, key_nibs);
    d_pack_nibbles(state, ciphertext);
}

__device__ __forceinline__
static void d_gift128_decrypt_block_precomp(
    const uint8_t  ciphertext[BLOCK_BYTES],
    const uint8_t  precomp_rkeys[GIFT_ROUNDS * 32],
          uint8_t  plaintext[BLOCK_BYTES])
{
    uint8_t input[32];
    d_unpack_nibbles(ciphertext, input);

    uint8_t bits[128], perm_bits[128], key_bits[128];

    for (int r = GIFT_ROUNDS - 1; r >= 0; r--) {
        for (int i = 0; i < 32; i++)
            for (int j = 0; j < 4; j++)
                bits[4*i+j] = (input[i] >> j) & 0x1;
        for (int i = 0; i < 32; i++)
            for (int j = 0; j < 4; j++)
                key_bits[4*i+j] = (precomp_rkeys[r * 32 + i] >> j) & 0x1;
        for (int i = 0; i < 32; i++) {
            bits[4*i+1] ^= key_bits[i];
            bits[4*i+2] ^= key_bits[i+64];
        }

        bits[3]   ^= (d_GIFT_RC[r]     ) & 0x1;
        bits[7]   ^= (d_GIFT_RC[r] >> 1) & 0x1;
        bits[11]  ^= (d_GIFT_RC[r] >> 2) & 0x1;
        bits[15]  ^= (d_GIFT_RC[r] >> 3) & 0x1;
        bits[19]  ^= (d_GIFT_RC[r] >> 4) & 0x1;
        bits[23]  ^= (d_GIFT_RC[r] >> 5) & 0x1;
        bits[127] ^= 1;

        for (int i = 0; i < 32; i++) {
            input[i] = 0;
            for (int j = 0; j < 4; j++)
                input[i] ^= bits[4*i+j] << j;
        }

        for (int i = 0; i < 32; i++)
            for (int j = 0; j < 4; j++)
                bits[4*i+j] = (input[i] >> j) & 0x1;
        for (int i = 0; i < 128; i++)
            perm_bits[d_GIFT_P_inv[i]] = bits[i];
        for (int i = 0; i < 32; i++) {
            input[i] = 0;
            for (int j = 0; j < 4; j++)
                input[i] ^= perm_bits[4*i+j] << j;
        }

        for (int i = 0; i < 32; i++)
            input[i] = d_GIFT_S_inv[input[i]];
    }
    d_pack_nibbles(input, plaintext);
}

// GPU kernels

__global__
void gift128_ecb_encrypt_kernel(const uint8_t* __restrict__ d_key,
                                const uint8_t* __restrict__ d_input,
                                      uint8_t* __restrict__ d_output,
                                size_t num_blocks)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if ((size_t)tid >= num_blocks) return;

    uint8_t local_key[KEY_BYTES];
    for (int i = 0; i < KEY_BYTES; i++)
        local_key[i] = d_key[i];

    size_t offset = (size_t)tid * BLOCK_BYTES;
    d_gift128_encrypt_block(d_input + offset, local_key, d_output + offset);
}

__global__
void gift128_ecb_decrypt_kernel(const uint8_t* __restrict__ d_rkeys,
                                const uint8_t* __restrict__ d_input,
                                      uint8_t* __restrict__ d_output,
                                size_t num_blocks)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if ((size_t)tid >= num_blocks) return;

    size_t offset = (size_t)tid * BLOCK_BYTES;
    d_gift128_decrypt_block_precomp(d_input + offset, d_rkeys, d_output + offset);
}

__global__
void gift128_ctr_kernel(const uint8_t* __restrict__ d_key,
                        const uint8_t* __restrict__ d_nonce,
                        const uint8_t* __restrict__ d_input,
                              uint8_t* __restrict__ d_output,
                        size_t num_blocks)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if ((size_t)tid >= num_blocks) return;

    uint8_t counter_block[BLOCK_BYTES];
    for (int i = 0; i < 12; i++)
        counter_block[i] = d_nonce[i];
    counter_block[12] = (tid >> 24) & 0xFF;
    counter_block[13] = (tid >> 16) & 0xFF;
    counter_block[14] = (tid >>  8) & 0xFF;
    counter_block[15] =  tid        & 0xFF;

    uint8_t local_key[KEY_BYTES];
    for (int i = 0; i < KEY_BYTES; i++)
        local_key[i] = d_key[i];

    uint8_t keystream[BLOCK_BYTES];
    d_gift128_encrypt_block(counter_block, local_key, keystream);

    size_t offset = (size_t)tid * BLOCK_BYTES;
    for (int i = 0; i < BLOCK_BYTES; i++)
        d_output[offset + i] = d_input[offset + i] ^ keystream[i];
}

__global__
void gift128_ctr_decrypt_kernel(const uint8_t* __restrict__ d_key,
                                const uint8_t* __restrict__ d_nonce,
                                const uint8_t* __restrict__ d_input,
                                      uint8_t* __restrict__ d_output,
                                size_t num_blocks)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if ((size_t)tid >= num_blocks) return;

    uint8_t counter_block[BLOCK_BYTES];
    for (int i = 0; i < 12; i++)
        counter_block[i] = d_nonce[i];
    counter_block[12] = (tid >> 24) & 0xFF;
    counter_block[13] = (tid >> 16) & 0xFF;
    counter_block[14] = (tid >>  8) & 0xFF;
    counter_block[15] =  tid        & 0xFF;

    uint8_t local_key[KEY_BYTES];
    for (int i = 0; i < KEY_BYTES; i++)
        local_key[i] = d_key[i];

    uint8_t keystream[BLOCK_BYTES];
    d_gift128_encrypt_block(counter_block, local_key, keystream);

    size_t offset = (size_t)tid * BLOCK_BYTES;
    for (int i = 0; i < BLOCK_BYTES; i++)
        d_output[offset + i] = d_input[offset + i] ^ keystream[i];
}

__global__
void gift128_cbc_encrypt_kernel(const uint8_t* __restrict__ d_key,
                                const uint8_t* __restrict__ d_iv,
                                const uint8_t* __restrict__ d_input,
                                      uint8_t* __restrict__ d_output,
                                size_t num_blocks)
{
    // Sequential single-thread kernel — CBC encrypt is inherently serial
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    uint8_t chain[BLOCK_BYTES], xored[BLOCK_BYTES], local_key[KEY_BYTES];
    for (int i = 0; i < BLOCK_BYTES; i++) chain[i]     = d_iv[i];
    for (int i = 0; i < KEY_BYTES;   i++) local_key[i] = d_key[i];

    for (size_t b = 0; b < num_blocks; b++) {
        size_t off = b * BLOCK_BYTES;
        for (int i = 0; i < BLOCK_BYTES; i++)
            xored[i] = d_input[off + i] ^ chain[i];
        d_gift128_encrypt_block(xored, local_key, d_output + off);
        for (int i = 0; i < BLOCK_BYTES; i++)
            chain[i] = d_output[off + i];
    }
}

__global__
void gift128_cbc_decrypt_kernel(const uint8_t* __restrict__ d_rkeys,
                                const uint8_t* __restrict__ d_iv,
                                const uint8_t* __restrict__ d_input,
                                      uint8_t* __restrict__ d_output,
                                size_t num_blocks)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if ((size_t)tid >= num_blocks) return;

    size_t offset = (size_t)tid * BLOCK_BYTES;

    uint8_t decrypted[BLOCK_BYTES];
    d_gift128_decrypt_block_precomp(d_input + offset, d_rkeys, decrypted);

    const uint8_t *prev = (tid == 0)
                          ? d_iv
                          : d_input + offset - BLOCK_BYTES;
    for (int i = 0; i < BLOCK_BYTES; i++)
        d_output[offset + i] = decrypted[i] ^ prev[i];
}

// Host wrappers

void gpu_ecb_encrypt(const uint8_t *key,
                     const uint8_t *input,
                           uint8_t *output,
                           size_t   length)
{
    size_t num_blocks = (length + BLOCK_BYTES - 1) / BLOCK_BYTES;
    size_t padded     = num_blocks * BLOCK_BYTES;

    uint8_t *d_key, *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_key, KEY_BYTES));
    CUDA_CHECK(cudaMalloc(&d_in,  padded));
    CUDA_CHECK(cudaMalloc(&d_out, padded));

    CUDA_CHECK(cudaMemcpy(d_key, key, KEY_BYTES, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_in, 0, padded));
    CUDA_CHECK(cudaMemcpy(d_in, input, length, cudaMemcpyHostToDevice));

    int tpb  = 256;
    int grid = (num_blocks + tpb - 1) / tpb;
    gift128_ecb_encrypt_kernel<<<grid, tpb>>>(d_key, d_in, d_out, num_blocks);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(output, d_out, length, cudaMemcpyDeviceToHost));
    cudaFree(d_key); cudaFree(d_in); cudaFree(d_out);
}

void gpu_ecb_decrypt(const uint8_t *key,
                     const uint8_t *input,
                           uint8_t *output,
                           size_t   length)
{
    size_t num_blocks = (length + BLOCK_BYTES - 1) / BLOCK_BYTES;
    size_t padded     = num_blocks * BLOCK_BYTES;

    uint8_t host_rkeys[GIFT_ROUNDS * 32];
    gift128_precompute_round_keys(key, host_rkeys);

    uint8_t *d_rkeys, *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_rkeys, GIFT_ROUNDS * 32));
    CUDA_CHECK(cudaMalloc(&d_in,    padded));
    CUDA_CHECK(cudaMalloc(&d_out,   padded));

    CUDA_CHECK(cudaMemcpy(d_rkeys, host_rkeys, GIFT_ROUNDS * 32, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_in, 0, padded));
    CUDA_CHECK(cudaMemcpy(d_in, input, length, cudaMemcpyHostToDevice));

    int tpb  = 256;
    int grid = (num_blocks + tpb - 1) / tpb;
    gift128_ecb_decrypt_kernel<<<grid, tpb>>>(d_rkeys, d_in, d_out, num_blocks);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(output, d_out, length, cudaMemcpyDeviceToHost));
    cudaFree(d_rkeys); cudaFree(d_in); cudaFree(d_out);
}

void gpu_ctr_crypt(const uint8_t *key,
                   const uint8_t  nonce[12],
                   const uint8_t *input,
                         uint8_t *output,
                         size_t   length)
{
    size_t num_blocks = (length + BLOCK_BYTES - 1) / BLOCK_BYTES;
    size_t padded     = num_blocks * BLOCK_BYTES;

    uint8_t *d_key, *d_nonce, *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_key,   KEY_BYTES));
    CUDA_CHECK(cudaMalloc(&d_nonce, 12));
    CUDA_CHECK(cudaMalloc(&d_in,    padded));
    CUDA_CHECK(cudaMalloc(&d_out,   padded));

    CUDA_CHECK(cudaMemcpy(d_key,   key,   KEY_BYTES, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_nonce, nonce, 12,        cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_in, 0, padded));
    CUDA_CHECK(cudaMemcpy(d_in, input, length, cudaMemcpyHostToDevice));

    int tpb  = 256;
    int grid = (num_blocks + tpb - 1) / tpb;
    gift128_ctr_kernel<<<grid, tpb>>>(d_key, d_nonce, d_in, d_out, num_blocks);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(output, d_out, length, cudaMemcpyDeviceToHost));
    cudaFree(d_key); cudaFree(d_nonce); cudaFree(d_in); cudaFree(d_out);
}

void gpu_cbc_encrypt(const uint8_t *key,
                     const uint8_t  iv[BLOCK_BYTES],
                     const uint8_t *input,
                           uint8_t *output,
                           size_t   length)
{
    size_t num_blocks = (length + BLOCK_BYTES - 1) / BLOCK_BYTES;
    size_t padded     = num_blocks * BLOCK_BYTES;

    uint8_t *d_key, *d_iv, *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_key, KEY_BYTES));
    CUDA_CHECK(cudaMalloc(&d_iv,  BLOCK_BYTES));
    CUDA_CHECK(cudaMalloc(&d_in,  padded));
    CUDA_CHECK(cudaMalloc(&d_out, padded));

    CUDA_CHECK(cudaMemcpy(d_key, key,   KEY_BYTES,  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_iv,  iv,    BLOCK_BYTES, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_in, 0, padded));
    CUDA_CHECK(cudaMemcpy(d_in, input, length, cudaMemcpyHostToDevice));

    gift128_cbc_encrypt_kernel<<<1, 1>>>(d_key, d_iv, d_in, d_out, num_blocks);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(output, d_out, length, cudaMemcpyDeviceToHost));
    cudaFree(d_key); cudaFree(d_iv); cudaFree(d_in); cudaFree(d_out);
}

void gpu_cbc_decrypt(const uint8_t *key,
                     const uint8_t  iv[BLOCK_BYTES],
                     const uint8_t *input,
                           uint8_t *output,
                           size_t   length)
{
    size_t num_blocks = (length + BLOCK_BYTES - 1) / BLOCK_BYTES;
    size_t padded     = num_blocks * BLOCK_BYTES;

    uint8_t host_rkeys[GIFT_ROUNDS * 32];
    gift128_precompute_round_keys(key, host_rkeys);

    uint8_t *d_rkeys, *d_iv, *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_rkeys, GIFT_ROUNDS * 32));
    CUDA_CHECK(cudaMalloc(&d_iv,    BLOCK_BYTES));
    CUDA_CHECK(cudaMalloc(&d_in,    padded));
    CUDA_CHECK(cudaMalloc(&d_out,   padded));

    CUDA_CHECK(cudaMemcpy(d_rkeys, host_rkeys, GIFT_ROUNDS * 32, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_iv,    iv,         BLOCK_BYTES,       cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_in, 0, padded));
    CUDA_CHECK(cudaMemcpy(d_in, input, length, cudaMemcpyHostToDevice));

    int tpb  = 256;
    int grid = (num_blocks + tpb - 1) / tpb;
    gift128_cbc_decrypt_kernel<<<grid, tpb>>>(d_rkeys, d_iv, d_in, d_out, num_blocks);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(output, d_out, length, cudaMemcpyDeviceToHost));
    cudaFree(d_rkeys); cudaFree(d_iv); cudaFree(d_in); cudaFree(d_out);
}

// Validation
static int assert_equal(const char *label, const uint8_t *got,
                        const uint8_t *expected, int len)
{
    if (memcmp(got, expected, len) == 0) {
        printf("The recovered plaintext matches the original plaintext %s\n", label);
        return 1;
    }
    printf("MISMATCH %s\n  got:      ", label);
    for (int i = 0; i < len; i++) printf("%02x", got[i]);
    printf("\n  expected: ");
    for (int i = 0; i < len; i++) printf("%02x", expected[i]);
    printf("\n");
    return 0;
}

static void print_block(const char *label, const uint8_t *b, int n)
{
    printf("%-22s: ", label);
    for (int i = 0; i < n; i++) printf("%02x", b[i]);
    printf("\n");
}

// Throughput sweep

static void throughput_sweep(const uint8_t *key,
                             const uint8_t  nonce[12],
                             const uint8_t  iv[BLOCK_BYTES])
{
    printf("\n");
    printf("--------GIFT-128 GPU Throughput vs Input Size--------\n");
    printf("  Median of %d trials per measurement.\n\n", TRIALS);

    const size_t data_sizes[] = {
        1024,
        4   * 1024,
        16  * 1024,
        64  * 1024,
        256 * 1024,
        1   * 1024 * 1024,
        4   * 1024 * 1024,
        16  * 1024 * 1024,
        64  * 1024 * 1024,
    };
    const int NSIZES = (int)(sizeof(data_sizes) / sizeof(data_sizes[0]));
    const int REPS   = TRIALS;

    printf("  %-10s  %-14s  %12s  %12s\n",
           "Data Size", "Mode", "Wall (ms)", "Throughput");
    printf("  %-10s  %-14s  %12s  %12s\n",
           "----------", "--------------", "------------", "----------");

    cudaEvent_t ev_s, ev_e;
    CUDA_CHECK(cudaEventCreate(&ev_s));
    CUDA_CHECK(cudaEventCreate(&ev_e));

    // Pre-compute round keys once — reused across all sizes
    uint8_t host_rkeys[GIFT_ROUNDS * 32];
    gift128_precompute_round_keys(key, host_rkeys);

    for (int si = 0; si < NSIZES; si++) {
        size_t N          = data_sizes[si];
        size_t num_blocks = (N + BLOCK_BYTES - 1) / BLOCK_BYTES;
        size_t padded     = num_blocks * BLOCK_BYTES;

        // Pinned host buffers
        uint8_t *h_plain, *h_cipher;
        CUDA_CHECK(cudaMallocHost(&h_plain,  padded));
        CUDA_CHECK(cudaMallocHost(&h_cipher, padded));
        for (size_t i = 0; i < padded; i++) h_plain[i] = (uint8_t)(i & 0xFF);

        // Build valid ECB and CBC ciphertexts for decrypt benchmarks
        uint8_t *h_ecb_cipher;
        CUDA_CHECK(cudaMallocHost(&h_ecb_cipher, padded));
        gpu_ecb_encrypt(key, h_plain, h_ecb_cipher, padded);
        gpu_cbc_encrypt(key, iv, h_plain, h_cipher, padded);

        // Device buffers
        uint8_t *d_key_g, *d_nonce_g, *d_rkeys_g, *d_iv_g;
        uint8_t *d_in, *d_cipher_g, *d_ecb_cipher_g, *d_out;
        CUDA_CHECK(cudaMalloc(&d_key_g,       KEY_BYTES));
        CUDA_CHECK(cudaMalloc(&d_nonce_g,     12));
        CUDA_CHECK(cudaMalloc(&d_rkeys_g,     GIFT_ROUNDS * 32));
        CUDA_CHECK(cudaMalloc(&d_iv_g,        BLOCK_BYTES));
        CUDA_CHECK(cudaMalloc(&d_in,          padded));
        CUDA_CHECK(cudaMalloc(&d_cipher_g,    padded));
        CUDA_CHECK(cudaMalloc(&d_ecb_cipher_g,padded));
        CUDA_CHECK(cudaMalloc(&d_out,         padded));

        CUDA_CHECK(cudaMemcpy(d_key_g,        key,          KEY_BYTES,         cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_nonce_g,      nonce,        12,                cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_rkeys_g,      host_rkeys,   GIFT_ROUNDS * 32,  cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_iv_g,         iv,           BLOCK_BYTES,       cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_in,           h_plain,      padded,            cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_cipher_g,     h_cipher,     padded,            cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_ecb_cipher_g, h_ecb_cipher, padded,            cudaMemcpyHostToDevice));

        int tpb  = 256;
        int grid = (int)((num_blocks + tpb - 1) / tpb);

        // ── GPU ECB encrypt ─────────────────────────────────────────────────
        float ecb_enc_samples[TRIALS];
        for (int t = 0; t < REPS; t++) {
            CUDA_CHECK(cudaEventRecord(ev_s));
            gift128_ecb_encrypt_kernel<<<grid, tpb>>>(d_key_g, d_in, d_out, num_blocks);
            CUDA_CHECK(cudaEventRecord(ev_e));
            CUDA_CHECK(cudaEventSynchronize(ev_e));
            float ms = 0; CUDA_CHECK(cudaEventElapsedTime(&ms, ev_s, ev_e));
            ecb_enc_samples[t] = ms;
        }

        // ── GPU ECB decrypt ─────────────────────────────────────────────────
        float ecb_dec_samples[TRIALS];
        for (int t = 0; t < REPS; t++) {
            CUDA_CHECK(cudaEventRecord(ev_s));
            gift128_ecb_decrypt_kernel<<<grid, tpb>>>(d_rkeys_g, d_ecb_cipher_g, d_out, num_blocks);
            CUDA_CHECK(cudaEventRecord(ev_e));
            CUDA_CHECK(cudaEventSynchronize(ev_e));
            float ms = 0; CUDA_CHECK(cudaEventElapsedTime(&ms, ev_s, ev_e));
            ecb_dec_samples[t] = ms;
        }

        // ── GPU CTR encrypt ─────────────────────────────────────────────────
        float ctr_enc_samples[TRIALS];
        for (int t = 0; t < REPS; t++) {
            CUDA_CHECK(cudaEventRecord(ev_s));
            gift128_ctr_kernel<<<grid, tpb>>>(d_key_g, d_nonce_g, d_in, d_out, num_blocks);
            CUDA_CHECK(cudaEventRecord(ev_e));
            CUDA_CHECK(cudaEventSynchronize(ev_e));
            float ms = 0; CUDA_CHECK(cudaEventElapsedTime(&ms, ev_s, ev_e));
            ctr_enc_samples[t] = ms;
        }

        // ── GPU CTR decrypt ─────────────────────────────────────────────────
        float ctr_dec_samples[TRIALS];
        for (int t = 0; t < REPS; t++) {
            CUDA_CHECK(cudaEventRecord(ev_s));
            gift128_ctr_decrypt_kernel<<<grid, tpb>>>(d_key_g, d_nonce_g, d_cipher_g, d_out, num_blocks);
            CUDA_CHECK(cudaEventRecord(ev_e));
            CUDA_CHECK(cudaEventSynchronize(ev_e));
            float ms = 0; CUDA_CHECK(cudaEventElapsedTime(&ms, ev_s, ev_e));
            ctr_dec_samples[t] = ms;
        }

        // ── GPU CBC encrypt ─────────────────────────────────────────────────
        float cbc_enc_samples[TRIALS];
        for (int t = 0; t < REPS; t++) {
            CUDA_CHECK(cudaEventRecord(ev_s));
            gift128_cbc_encrypt_kernel<<<1, 1>>>(d_key_g, d_iv_g, d_in, d_out, num_blocks);
            CUDA_CHECK(cudaEventRecord(ev_e));
            CUDA_CHECK(cudaEventSynchronize(ev_e));
            float ms = 0; CUDA_CHECK(cudaEventElapsedTime(&ms, ev_s, ev_e));
            cbc_enc_samples[t] = ms;
        }

        // ── GPU CBC decrypt ─────────────────────────────────────────────────
        float cbc_dec_samples[TRIALS];
        for (int t = 0; t < REPS; t++) {
            CUDA_CHECK(cudaEventRecord(ev_s));
            gift128_cbc_decrypt_kernel<<<grid, tpb>>>(d_rkeys_g, d_iv_g, d_cipher_g, d_out, num_blocks);
            CUDA_CHECK(cudaEventRecord(ev_e));
            CUDA_CHECK(cudaEventSynchronize(ev_e));
            float ms = 0; CUDA_CHECK(cudaEventElapsedTime(&ms, ev_s, ev_e));
            cbc_dec_samples[t] = ms;
        }

        // Sort all six arrays for median
        for (int i = 0; i < TRIALS-1; i++)
            for (int j = i+1; j < TRIALS; j++) {
                #define SWAP(arr) if (arr[j] < arr[i]) { float _t = arr[i]; arr[i] = arr[j]; arr[j] = _t; }
                SWAP(ecb_enc_samples)
                SWAP(ecb_dec_samples)
                SWAP(ctr_enc_samples)
                SWAP(ctr_dec_samples)
                SWAP(cbc_enc_samples)
                SWAP(cbc_dec_samples)
                #undef SWAP
            }

        double ecb_enc_ms = ecb_enc_samples[TRIALS/2];
        double ecb_dec_ms = ecb_dec_samples[TRIALS/2];
        double ctr_enc_ms = ctr_enc_samples[TRIALS/2];
        double ctr_dec_ms = ctr_dec_samples[TRIALS/2];
        double cbc_enc_ms = cbc_enc_samples[TRIALS/2];
        double cbc_dec_ms = cbc_dec_samples[TRIALS/2];

        double ecb_enc_gbs = (double)N / (ecb_enc_ms * 1e-3) / 1e9;
        double ecb_dec_gbs = (double)N / (ecb_dec_ms * 1e-3) / 1e9;
        double ctr_enc_gbs = (double)N / (ctr_enc_ms * 1e-3) / 1e9;
        double ctr_dec_gbs = (double)N / (ctr_dec_ms * 1e-3) / 1e9;
        double cbc_enc_gbs = (double)N / (cbc_enc_ms * 1e-3) / 1e9;
        double cbc_dec_gbs = (double)N / (cbc_dec_ms * 1e-3) / 1e9;

        const char *unit = (N >= 1024*1024) ? "MB" : "KB";
        double      nd   = (N >= 1024*1024) ? N / 1048576.0 : N / 1024.0;

        printf("  %5.0f %-3s  GPU-ECB-E    %10.4f ms  %8.4f GB/s\n",
               nd, unit, ecb_enc_ms, ecb_enc_gbs);
        printf("  %5.0f %-3s  GPU-ECB-D    %10.4f ms  %8.4f GB/s\n",
               nd, unit, ecb_dec_ms, ecb_dec_gbs);
        printf("  %5.0f %-3s  GPU-CTR-E    %10.4f ms  %8.4f GB/s\n",
               nd, unit, ctr_enc_ms, ctr_enc_gbs);
        printf("  %5.0f %-3s  GPU-CTR-D    %10.4f ms  %8.4f GB/s\n",
               nd, unit, ctr_dec_ms, ctr_dec_gbs);
        printf("  %5.0f %-3s  GPU-CBC-E    %10.4f ms  %8.4f GB/s\n",
               nd, unit, cbc_enc_ms, cbc_enc_gbs);
        printf("  %5.0f %-3s  GPU-CBC-D    %10.4f ms  %8.4f GB/s\n",
               nd, unit, cbc_dec_ms, cbc_dec_gbs);
        printf("\n");

        CUDA_CHECK(cudaFree(d_key_g));        CUDA_CHECK(cudaFree(d_nonce_g));
        CUDA_CHECK(cudaFree(d_rkeys_g));      CUDA_CHECK(cudaFree(d_iv_g));
        CUDA_CHECK(cudaFree(d_in));           CUDA_CHECK(cudaFree(d_cipher_g));
        CUDA_CHECK(cudaFree(d_ecb_cipher_g)); CUDA_CHECK(cudaFree(d_out));
        CUDA_CHECK(cudaFreeHost(h_plain));    CUDA_CHECK(cudaFreeHost(h_cipher));
        CUDA_CHECK(cudaFreeHost(h_ecb_cipher));
    }

    CUDA_CHECK(cudaEventDestroy(ev_s));
    CUDA_CHECK(cudaEventDestroy(ev_e));
}

// Single-block latency

static void measure_single_block_latency(const uint8_t *key,
                                         const uint8_t  nonce[12],
                                         const uint8_t  iv[BLOCK_BYTES])
{
    const int LAT_BLOCKS = 4;
    const int LAT_TRIALS = 101;

    size_t sz = (size_t)LAT_BLOCKS * BLOCK_BYTES;

    uint8_t h_plain [LAT_BLOCKS * BLOCK_BYTES];
    uint8_t h_ecb_cipher[LAT_BLOCKS * BLOCK_BYTES];
    uint8_t h_cbc_cipher[LAT_BLOCKS * BLOCK_BYTES];
    uint8_t h_recov [LAT_BLOCKS * BLOCK_BYTES];
    for (int i = 0; i < LAT_BLOCKS * BLOCK_BYTES; i++) h_plain[i] = (uint8_t)(i & 0xFF);

    gpu_ecb_encrypt(key, h_plain, h_ecb_cipher, sz);
    gpu_cbc_encrypt(key, iv, h_plain, h_cbc_cipher, sz);

    uint8_t host_rkeys[GIFT_ROUNDS * 32];
    gift128_precompute_round_keys(key, host_rkeys);

    uint8_t *d_key, *d_nonce, *d_rkeys, *d_iv;
    uint8_t *d_plain, *d_ecb_cipher, *d_cbc_cipher, *d_out;
    CUDA_CHECK(cudaMalloc(&d_key,        KEY_BYTES));
    CUDA_CHECK(cudaMalloc(&d_nonce,      12));
    CUDA_CHECK(cudaMalloc(&d_rkeys,      GIFT_ROUNDS * 32));
    CUDA_CHECK(cudaMalloc(&d_iv,         BLOCK_BYTES));
    CUDA_CHECK(cudaMalloc(&d_plain,      sz));
    CUDA_CHECK(cudaMalloc(&d_ecb_cipher, sz));
    CUDA_CHECK(cudaMalloc(&d_cbc_cipher, sz));
    CUDA_CHECK(cudaMalloc(&d_out,        sz));

    CUDA_CHECK(cudaMemcpy(d_key,        key,          KEY_BYTES,        cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_nonce,      nonce,        12,               cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rkeys,      host_rkeys,   GIFT_ROUNDS * 32, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_iv,         iv,           BLOCK_BYTES,      cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_plain,      h_plain,      sz,               cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ecb_cipher, h_ecb_cipher, sz,               cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cbc_cipher, h_cbc_cipher, sz,               cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // ── ECB encrypt latency ──────────────────────────────────────────────────
    float ecb_enc_samples[LAT_TRIALS];
    for (int t = 0; t < LAT_TRIALS; t++) {
        CUDA_CHECK(cudaEventRecord(start));
        CUDA_CHECK(cudaMemcpy(d_plain, h_plain, sz, cudaMemcpyHostToDevice));
        gift128_ecb_encrypt_kernel<<<1, LAT_BLOCKS>>>(d_key, d_plain, d_out, LAT_BLOCKS);
        CUDA_CHECK(cudaMemcpy(h_recov, d_out, sz, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0; CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ecb_enc_samples[t] = ms * 1000.0f;
    }

    // ── ECB decrypt latency ──────────────────────────────────────────────────
    float ecb_dec_samples[LAT_TRIALS];
    for (int t = 0; t < LAT_TRIALS; t++) {
        CUDA_CHECK(cudaEventRecord(start));
        CUDA_CHECK(cudaMemcpy(d_ecb_cipher, h_ecb_cipher, sz, cudaMemcpyHostToDevice));
        gift128_ecb_decrypt_kernel<<<1, LAT_BLOCKS>>>(d_rkeys, d_ecb_cipher, d_out, LAT_BLOCKS);
        CUDA_CHECK(cudaMemcpy(h_recov, d_out, sz, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0; CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ecb_dec_samples[t] = ms * 1000.0f;
    }

    // ── CTR encrypt latency ──────────────────────────────────────────────────
    float ctr_samples[LAT_TRIALS];
    for (int t = 0; t < LAT_TRIALS; t++) {
        CUDA_CHECK(cudaEventRecord(start));
        CUDA_CHECK(cudaMemcpy(d_plain, h_plain, sz, cudaMemcpyHostToDevice));
        gift128_ctr_kernel<<<1, LAT_BLOCKS>>>(d_key, d_nonce, d_plain, d_out, LAT_BLOCKS);
        CUDA_CHECK(cudaMemcpy(h_recov, d_out, sz, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0; CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ctr_samples[t] = ms * 1000.0f;
    }

    // ── CBC decrypt latency ──────────────────────────────────────────────────
    float cbc_samples[LAT_TRIALS];
    for (int t = 0; t < LAT_TRIALS; t++) {
        CUDA_CHECK(cudaEventRecord(start));
        CUDA_CHECK(cudaMemcpy(d_cbc_cipher, h_cbc_cipher, sz, cudaMemcpyHostToDevice));
        gift128_cbc_decrypt_kernel<<<1, LAT_BLOCKS>>>(d_rkeys, d_iv, d_cbc_cipher, d_out, LAT_BLOCKS);
        CUDA_CHECK(cudaMemcpy(h_recov, d_out, sz, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0; CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        cbc_samples[t] = ms * 1000.0f;
    }

    for (int i = 0; i < LAT_TRIALS-1; i++)
        for (int j = i+1; j < LAT_TRIALS; j++) {
            #define SWAP(arr) if (arr[j] < arr[i]) { float _t = arr[i]; arr[i] = arr[j]; arr[j] = _t; }
            SWAP(ecb_enc_samples)
            SWAP(ecb_dec_samples)
            SWAP(ctr_samples)
            SWAP(cbc_samples)
            #undef SWAP
        }

    printf("\n--------Single 64-Byte Block Latency (side data point)--------\n");
    printf("  Input size : %d cipher blocks = %d bytes\n", LAT_BLOCKS, LAT_BLOCKS * BLOCK_BYTES);
    printf("  Trials     : %d\n\n", LAT_TRIALS);
    printf("  %-35s : %8.2f us\n", "ECB encrypt latency (H2D+K+D2H)", ecb_enc_samples[LAT_TRIALS/2]);
    printf("  %-35s : %8.2f us\n", "ECB decrypt latency (H2D+K+D2H)", ecb_dec_samples[LAT_TRIALS/2]);
    printf("  %-35s : %8.2f us\n", "CTR encrypt latency (H2D+K+D2H)", ctr_samples[LAT_TRIALS/2]);
    printf("  %-35s : %8.2f us\n", "CBC decrypt latency (H2D+K+D2H)", cbc_samples[LAT_TRIALS/2]);

    cudaEventDestroy(start); cudaEventDestroy(stop);
    cudaFree(d_key);  cudaFree(d_nonce); cudaFree(d_rkeys); cudaFree(d_iv);
    cudaFree(d_plain); cudaFree(d_ecb_cipher); cudaFree(d_cbc_cipher); cudaFree(d_out);
}

// Key schedule cost

static void measure_key_schedule_cost(const uint8_t *key)
{
    const int KS_TRIALS = 101;

    uint8_t host_rkeys[GIFT_ROUNDS * 32];
    gift128_precompute_round_keys(key, host_rkeys);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    float ks_samples[KS_TRIALS];
    for (int t = 0; t < KS_TRIALS; t++) {
        CUDA_CHECK(cudaEventRecord(start));
        gift128_precompute_round_keys(key, host_rkeys);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0; CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ks_samples[t] = ms * 1000.0f;
    }

    uint8_t *d_rkeys;
    CUDA_CHECK(cudaMalloc(&d_rkeys, GIFT_ROUNDS * 32));

    float up_samples[KS_TRIALS];
    for (int t = 0; t < KS_TRIALS; t++) {
        CUDA_CHECK(cudaEventRecord(start));
        CUDA_CHECK(cudaMemcpy(d_rkeys, host_rkeys, GIFT_ROUNDS * 32, cudaMemcpyHostToDevice));
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

    size_t rtk_bytes = (size_t)(GIFT_ROUNDS * 32);

    printf("\n--------Key Schedule Cost--------\n");
    printf("\n  Time (median of %d trials)\n", KS_TRIALS);
    printf("  %-48s : %7.2f us\n", "CPU compute  (gift128_precompute_round_keys)",  ks_med);
    printf("  %-48s : %7.2f us\n", "GPU upload   (cudaMemcpy H->D, PCIe)",           up_med);
    printf("  %-48s : %7.2f us\n", "Total key setup cost",                            ks_med + up_med);
    printf("\n  Space\n");
    printf("  %-48s : %3zu bytes  (%d rounds x 32 nibble-bytes)\n",
           "Expanded round keys in GPU global memory", rtk_bytes, GIFT_ROUNDS);

    cudaFree(d_rkeys);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// Main

int main()
{
    printf("--------GIFT-128 CUDA Implementation--------\n");
    printf("--------ECB, CTR and CBC Mode Tests--------\n\n");

    upload_constant_tables();

    uint8_t key[KEY_BYTES] = {
        0x9e,0xb9,0x36,0x40,0xd0,0x88,0xda,0x63,
        0x76,0xa3,0x9d,0x1c,0x8b,0xea,0x71,0xe1
    };
    uint8_t plain[BLOCK_BYTES] = {
        0xcf,0x16,0xcf,0xe8,0xfd,0x0f,0x98,0xaa,
        0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00
    };
    uint8_t nonce[12]        = {0};
    uint8_t iv[BLOCK_BYTES]  = {0};

    // ── ECB mode test ────────────────────────────────────────────────────────
    printf("--------ECB Mode (GPU)--------\n\n");

    uint8_t ecb_cipher[BLOCK_BYTES], ecb_recovered[BLOCK_BYTES];

    print_block("Original Plaintext",  plain,         BLOCK_BYTES);
    gpu_ecb_encrypt(key, plain,        ecb_cipher,    BLOCK_BYTES);
    print_block("ECB Ciphertext",      ecb_cipher,    BLOCK_BYTES);
    gpu_ecb_decrypt(key, ecb_cipher,   ecb_recovered, BLOCK_BYTES);
    print_block("Recovered Plaintext", ecb_recovered, BLOCK_BYTES);
    printf("\n");
    assert_equal("ECB mode", ecb_recovered, plain, BLOCK_BYTES);

    // ── CTR mode test ────────────────────────────────────────────────────────
    printf("\n--------CTR Mode (GPU)--------\n\n");

    uint8_t ctr_cipher[BLOCK_BYTES], ctr_recovered[BLOCK_BYTES];

    print_block("Original Plaintext",   plain,         BLOCK_BYTES);
    gpu_ctr_crypt(key, nonce, plain,       ctr_cipher,   BLOCK_BYTES);
    print_block("CTR Ciphertext",        ctr_cipher,    BLOCK_BYTES);
    gpu_ctr_crypt(key, nonce, ctr_cipher,  ctr_recovered, BLOCK_BYTES);
    print_block("Recovered Plaintext",   ctr_recovered, BLOCK_BYTES);
    printf("\n");
    assert_equal("CTR mode", ctr_recovered, plain, BLOCK_BYTES);

    // ── CBC mode test ────────────────────────────────────────────────────────
    printf("\n--------CBC Mode (GPU)--------\n\n");

    uint8_t cbc_cipher[BLOCK_BYTES], cbc_recovered[BLOCK_BYTES];

    print_block("Original Plaintext",   plain,          BLOCK_BYTES);
    gpu_cbc_encrypt(key, iv, plain,       cbc_cipher,   BLOCK_BYTES);
    print_block("CBC Ciphertext",        cbc_cipher,    BLOCK_BYTES);
    gpu_cbc_decrypt(key, iv, cbc_cipher,  cbc_recovered, BLOCK_BYTES);
    print_block("Recovered Plaintext",   cbc_recovered, BLOCK_BYTES);
    printf("\n");
    assert_equal("CBC mode", cbc_recovered, plain, BLOCK_BYTES);

    printf("\n--------ECB, CBC and CTR modes are validated--------\n");

    // ── Throughput sweep ─────────────────────────────────────────────────────
    throughput_sweep(key, nonce, iv);

    // ── Single block latency ─────────────────────────────────────────────────
    measure_single_block_latency(key, nonce, iv);

    // ── Key schedule cost ────────────────────────────────────────────────────
    measure_key_schedule_cost(key);

    // ── Device info ──────────────────────────────────────────────────────────
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("\n--------GPU Device--------\n");
    printf("  Device        : %s\n", prop.name);
    printf("  SMs           : %d\n", prop.multiProcessorCount);
    printf("  Clock         : %.0f MHz\n", prop.clockRate / 1e3);
    printf("  Global memory : %.0f MB\n", prop.totalGlobalMem / 1.0e6);

    return 0;
}
