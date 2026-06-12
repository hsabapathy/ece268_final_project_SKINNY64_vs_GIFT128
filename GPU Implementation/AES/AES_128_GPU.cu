#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <time.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d - %s\n",                      \
                    __FILE__, __LINE__, cudaGetErrorString(err));               \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

#define TRIALS 31

__constant__ uint8_t S_GPU[256] = {
    0x63,0x7c,0x77,0x7b,0xf2,0x6b,0x6f,0xc5,0x30,0x01,0x67,0x2b,0xfe,0xd7,0xab,0x76,
    0xca,0x82,0xc9,0x7d,0xfa,0x59,0x47,0xf0,0xad,0xd4,0xa2,0xaf,0x9c,0xa4,0x72,0xc0,
    0xb7,0xfd,0x93,0x26,0x36,0x3f,0xf7,0xcc,0x34,0xa5,0xe5,0xf1,0x71,0xd8,0x31,0x15,
    0x04,0xc7,0x23,0xc3,0x18,0x96,0x05,0x9a,0x07,0x12,0x80,0xe2,0xeb,0x27,0xb2,0x75,
    0x09,0x83,0x2c,0x1a,0x1b,0x6e,0x5a,0xa0,0x52,0x3b,0xd6,0xb3,0x29,0xe3,0x2f,0x84,
    0x53,0xd1,0x00,0xed,0x20,0xfc,0xb1,0x5b,0x6a,0xcb,0xbe,0x39,0x4a,0x4c,0x58,0xcf,
    0xd0,0xef,0xaa,0xfb,0x43,0x4d,0x33,0x85,0x45,0xf9,0x02,0x7f,0x50,0x3c,0x9f,0xa8,
    0x51,0xa3,0x40,0x8f,0x92,0x9d,0x38,0xf5,0xbc,0xb6,0xda,0x21,0x10,0xff,0xf3,0xd2,
    0xcd,0x0c,0x13,0xec,0x5f,0x97,0x44,0x17,0xc4,0xa7,0x7e,0x3d,0x64,0x5d,0x19,0x73,
    0x60,0x81,0x4f,0xdc,0x22,0x2a,0x90,0x88,0x46,0xee,0xb8,0x14,0xde,0x5e,0x0b,0xdb,
    0xe0,0x32,0x3a,0x0a,0x49,0x06,0x24,0x5c,0xc2,0xd3,0xac,0x62,0x91,0x95,0xe4,0x79,
    0xe7,0xc8,0x37,0x6d,0x8d,0xd5,0x4e,0xa9,0x6c,0x56,0xf4,0xea,0x65,0x7a,0xae,0x08,
    0xba,0x78,0x25,0x2e,0x1c,0xa6,0xb4,0xc6,0xe8,0xdd,0x74,0x1f,0x4b,0xbd,0x8b,0x8a,
    0x70,0x3e,0xb5,0x66,0x48,0x03,0xf6,0x0e,0x61,0x35,0x57,0xb9,0x86,0xc1,0x1d,0x9e,
    0xe1,0xf8,0x98,0x11,0x69,0xd9,0x8e,0x94,0x9b,0x1e,0x87,0xe9,0xce,0x55,0x28,0xdf,
    0x8c,0xa1,0x89,0x0d,0xbf,0xe6,0x42,0x68,0x41,0x99,0x2d,0x0f,0xb0,0x54,0xbb,0x16
};

__constant__ uint8_t S_INV_GPU[256] = {
    0x52,0x09,0x6a,0xd5,0x30,0x36,0xa5,0x38,0xbf,0x40,0xa3,0x9e,0x81,0xf3,0xd7,0xfb,
    0x7c,0xe3,0x39,0x82,0x9b,0x2f,0xff,0x87,0x34,0x8e,0x43,0x44,0xc4,0xde,0xe9,0xcb,
    0x54,0x7b,0x94,0x32,0xa6,0xc2,0x23,0x3d,0xee,0x4c,0x95,0x0b,0x42,0xfa,0xc3,0x4e,
    0x08,0x2e,0xa1,0x66,0x28,0xd9,0x24,0xb2,0x76,0x5b,0xa2,0x49,0x6d,0x8b,0xd1,0x25,
    0x72,0xf8,0xf6,0x64,0x86,0x68,0x98,0x16,0xd4,0xa4,0x5c,0xcc,0x5d,0x65,0xb6,0x92,
    0x6c,0x70,0x48,0x50,0xfd,0xed,0xb9,0xda,0x5e,0x15,0x46,0x57,0xa7,0x8d,0x9d,0x84,
    0x90,0xd8,0xab,0x00,0x8c,0xbc,0xd3,0x0a,0xf7,0xe4,0x58,0x05,0xb8,0xb3,0x45,0x06,
    0xd0,0x2c,0x1e,0x8f,0xca,0x3f,0x0f,0x02,0xc1,0xaf,0xbd,0x03,0x01,0x13,0x8a,0x6b,
    0x3a,0x91,0x11,0x41,0x4f,0x67,0xdc,0xea,0x97,0xf2,0xcf,0xce,0xf0,0xb4,0xe6,0x73,
    0x96,0xac,0x74,0x22,0xe7,0xad,0x35,0x85,0xe2,0xf9,0x37,0xe8,0x1c,0x75,0xdf,0x6e,
    0x47,0xf1,0x1a,0x71,0x1d,0x29,0xc5,0x89,0x6f,0xb7,0x62,0x0e,0xaa,0x18,0xbe,0x1b,
    0xfc,0x56,0x3e,0x4b,0xc6,0xd2,0x79,0x20,0x9a,0xdb,0xc0,0xfe,0x78,0xcd,0x5a,0xf4,
    0x1f,0xdd,0xa8,0x33,0x88,0x07,0xc7,0x31,0xb1,0x12,0x10,0x59,0x27,0x80,0xec,0x5f,
    0x60,0x51,0x7f,0xa9,0x19,0xb5,0x4a,0x0d,0x2d,0xe5,0x7a,0x9f,0x93,0xc9,0x9c,0xef,
    0xa0,0xe0,0x3b,0x4d,0xae,0x2a,0xf5,0xb0,0xc8,0xeb,0xbb,0x3c,0x83,0x53,0x99,0x61,
    0x17,0x2b,0x04,0x7e,0xba,0x77,0xd6,0x26,0xe1,0x69,0x14,0x63,0x55,0x21,0x0c,0x7d
};

__constant__ uint8_t RCON_GPU[10] = {
    0x01,0x02,0x04,0x08,0x10,0x20,0x40,0x80,0x1b,0x36
};

__constant__ uint8_t RK_GPU[176];
__constant__ uint8_t DRK_GPU[176];

namespace aes128 {

constexpr int Nb = 4;
constexpr int Nk = 4;
constexpr int Nr = 10;
constexpr int BLOCK_SIZE = 16;
constexpr int KEY_SIZE   = 16;
constexpr int EXPANDED_KEY_SIZE = 16 * (Nr + 1);

static const uint8_t sbox[256] = {
    0x63,0x7c,0x77,0x7b,0xf2,0x6b,0x6f,0xc5,0x30,0x01,0x67,0x2b,0xfe,0xd7,0xab,0x76,
    0xca,0x82,0xc9,0x7d,0xfa,0x59,0x47,0xf0,0xad,0xd4,0xa2,0xaf,0x9c,0xa4,0x72,0xc0,
    0xb7,0xfd,0x93,0x26,0x36,0x3f,0xf7,0xcc,0x34,0xa5,0xe5,0xf1,0x71,0xd8,0x31,0x15,
    0x04,0xc7,0x23,0xc3,0x18,0x96,0x05,0x9a,0x07,0x12,0x80,0xe2,0xeb,0x27,0xb2,0x75,
    0x09,0x83,0x2c,0x1a,0x1b,0x6e,0x5a,0xa0,0x52,0x3b,0xd6,0xb3,0x29,0xe3,0x2f,0x84,
    0x53,0xd1,0x00,0xed,0x20,0xfc,0xb1,0x5b,0x6a,0xcb,0xbe,0x39,0x4a,0x4c,0x58,0xcf,
    0xd0,0xef,0xaa,0xfb,0x43,0x4d,0x33,0x85,0x45,0xf9,0x02,0x7f,0x50,0x3c,0x9f,0xa8,
    0x51,0xa3,0x40,0x8f,0x92,0x9d,0x38,0xf5,0xbc,0xb6,0xda,0x21,0x10,0xff,0xf3,0xd2,
    0xcd,0x0c,0x13,0xec,0x5f,0x97,0x44,0x17,0xc4,0xa7,0x7e,0x3d,0x64,0x5d,0x19,0x73,
    0x60,0x81,0x4f,0xdc,0x22,0x2a,0x90,0x88,0x46,0xee,0xb8,0x14,0xde,0x5e,0x0b,0xdb,
    0xe0,0x32,0x3a,0x0a,0x49,0x06,0x24,0x5c,0xc2,0xd3,0xac,0x62,0x91,0x95,0xe4,0x79,
    0xe7,0xc8,0x37,0x6d,0x8d,0xd5,0x4e,0xa9,0x6c,0x56,0xf4,0xea,0x65,0x7a,0xae,0x08,
    0xba,0x78,0x25,0x2e,0x1c,0xa6,0xb4,0xc6,0xe8,0xdd,0x74,0x1f,0x4b,0xbd,0x8b,0x8a,
    0x70,0x3e,0xb5,0x66,0x48,0x03,0xf6,0x0e,0x61,0x35,0x57,0xb9,0x86,0xc1,0x1d,0x9e,
    0xe1,0xf8,0x98,0x11,0x69,0xd9,0x8e,0x94,0x9b,0x1e,0x87,0xe9,0xce,0x55,0x28,0xdf,
    0x8c,0xa1,0x89,0x0d,0xbf,0xe6,0x42,0x68,0x41,0x99,0x2d,0x0f,0xb0,0x54,0xbb,0x16
};
static const uint8_t inv_sbox[256] = {
    0x52,0x09,0x6a,0xd5,0x30,0x36,0xa5,0x38,0xbf,0x40,0xa3,0x9e,0x81,0xf3,0xd7,0xfb,
    0x7c,0xe3,0x39,0x82,0x9b,0x2f,0xff,0x87,0x34,0x8e,0x43,0x44,0xc4,0xde,0xe9,0xcb,
    0x54,0x7b,0x94,0x32,0xa6,0xc2,0x23,0x3d,0xee,0x4c,0x95,0x0b,0x42,0xfa,0xc3,0x4e,
    0x08,0x2e,0xa1,0x66,0x28,0xd9,0x24,0xb2,0x76,0x5b,0xa2,0x49,0x6d,0x8b,0xd1,0x25,
    0x72,0xf8,0xf6,0x64,0x86,0x68,0x98,0x16,0xd4,0xa4,0x5c,0xcc,0x5d,0x65,0xb6,0x92,
    0x6c,0x70,0x48,0x50,0xfd,0xed,0xb9,0xda,0x5e,0x15,0x46,0x57,0xa7,0x8d,0x9d,0x84,
    0x90,0xd8,0xab,0x00,0x8c,0xbc,0xd3,0x0a,0xf7,0xe4,0x58,0x05,0xb8,0xb3,0x45,0x06,
    0xd0,0x2c,0x1e,0x8f,0xca,0x3f,0x0f,0x02,0xc1,0xaf,0xbd,0x03,0x01,0x13,0x8a,0x6b,
    0x3a,0x91,0x11,0x41,0x4f,0x67,0xdc,0xea,0x97,0xf2,0xcf,0xce,0xf0,0xb4,0xe6,0x73,
    0x96,0xac,0x74,0x22,0xe7,0xad,0x35,0x85,0xe2,0xf9,0x37,0xe8,0x1c,0x75,0xdf,0x6e,
    0x47,0xf1,0x1a,0x71,0x1d,0x29,0xc5,0x89,0x6f,0xb7,0x62,0x0e,0xaa,0x18,0xbe,0x1b,
    0xfc,0x56,0x3e,0x4b,0xc6,0xd2,0x79,0x20,0x9a,0xdb,0xc0,0xfe,0x78,0xcd,0x5a,0xf4,
    0x1f,0xdd,0xa8,0x33,0x88,0x07,0xc7,0x31,0xb1,0x12,0x10,0x59,0x27,0x80,0xec,0x5f,
    0x60,0x51,0x7f,0xa9,0x19,0xb5,0x4a,0x0d,0x2d,0xe5,0x7a,0x9f,0x93,0xc9,0x9c,0xef,
    0xa0,0xe0,0x3b,0x4d,0xae,0x2a,0xf5,0xb0,0xc8,0xeb,0xbb,0x3c,0x83,0x53,0x99,0x61,
    0x17,0x2b,0x04,0x7e,0xba,0x77,0xd6,0x26,0xe1,0x69,0x14,0x63,0x55,0x21,0x0c,0x7d
};
static const uint8_t Rcon[10] = {
    0x01,0x02,0x04,0x08,0x10,0x20,0x40,0x80,0x1b,0x36
};

static inline uint8_t xtime(uint8_t x) {
    return (uint8_t)((x << 1) ^ ((x >> 7) * 0x1b));
}

void KeyExpansion(const uint8_t key[KEY_SIZE], uint8_t roundKeys[EXPANDED_KEY_SIZE]) {
    std::memcpy(roundKeys, key, KEY_SIZE);
    uint8_t t[4];
    for (int i = Nk; i < Nb * (Nr + 1); ++i) {
        t[0] = roundKeys[(i-1)*4+0]; t[1] = roundKeys[(i-1)*4+1];
        t[2] = roundKeys[(i-1)*4+2]; t[3] = roundKeys[(i-1)*4+3];
        if (i % Nk == 0) {
            uint8_t u = t[0]; t[0] = t[1]; t[1] = t[2]; t[2] = t[3]; t[3] = u;
            t[0] = sbox[t[0]]; t[1] = sbox[t[1]];
            t[2] = sbox[t[2]]; t[3] = sbox[t[3]];
            t[0] ^= Rcon[i / Nk - 1];
        }
        for (int j = 0; j < 4; ++j)
            roundKeys[i*4+j] = roundKeys[(i-Nk)*4+j] ^ t[j];
    }
}

static void InvMixColumns(uint8_t s[16]) {
    for (int c = 0; c < 4; ++c) {
        uint8_t a0=s[c*4+0], a1=s[c*4+1], a2=s[c*4+2], a3=s[c*4+3];
        uint8_t T = a0^a1^a2^a3;
        s[c*4+0] = T^a0^xtime(a0^a1);
        s[c*4+1] = T^a1^xtime(a1^a2);
        s[c*4+2] = T^a2^xtime(a2^a3);
        s[c*4+3] = T^a3^xtime(a3^a0);
        uint8_t u = xtime(xtime(a0^a2));
        uint8_t v = xtime(xtime(a1^a3));
        uint8_t w = xtime(u^v);
        s[c*4+0] ^= w^u; s[c*4+1] ^= w^v;
        s[c*4+2] ^= w^u; s[c*4+3] ^= w^v;
    }
}


void BuildDecryptRoundKeys(const uint8_t rk[EXPANDED_KEY_SIZE],
                            uint8_t drk[EXPANDED_KEY_SIZE]) {
    std::memcpy(drk, rk + Nr*16, 16);
    for (int r = 1; r < Nr; ++r) {
        uint8_t tmp[16];
        std::memcpy(tmp, rk + (Nr - r) * 16, 16);
        InvMixColumns(tmp);
        std::memcpy(drk + r*16, tmp, 16);
    }
    std::memcpy(drk + Nr*16, rk, 16);
}

} 


__device__ static inline uint8_t gpu_xtime(uint8_t x) {
    return (uint8_t)((x << 1) ^ ((x >> 7) * 0x1b));
}

__device__ static void gpu_SubBytes(uint8_t s[16]) {
    for (int i = 0; i < 16; ++i) s[i] = S_GPU[s[i]];
}
__device__ static void gpu_InvSubBytes(uint8_t s[16]) {
    for (int i = 0; i < 16; ++i) s[i] = S_INV_GPU[s[i]];
}

__device__ static void gpu_ShiftRows(uint8_t s[16]) {
    uint8_t t;
    t = s[1];  s[1]  = s[5];  s[5]  = s[9];  s[9]  = s[13]; s[13] = t;
    t = s[2];  s[2]  = s[10]; s[10] = t;
    t = s[6];  s[6]  = s[14]; s[14] = t;
    t = s[3];  s[3]  = s[15]; s[15] = s[11]; s[11] = s[7];  s[7]  = t;
}
__device__ static void gpu_InvShiftRows(uint8_t s[16]) {
    uint8_t t;
    t = s[13]; s[13] = s[9];  s[9]  = s[5];  s[5]  = s[1];  s[1]  = t;
    t = s[2];  s[2]  = s[10]; s[10] = t;
    t = s[6];  s[6]  = s[14]; s[14] = t;
    t = s[3];  s[3]  = s[7];  s[7]  = s[11]; s[11] = s[15]; s[15] = t;
}

__device__ static void gpu_MixColumns(uint8_t s[16]) {
    for (int c = 0; c < 4; ++c) {
        uint8_t a0=s[c*4+0], a1=s[c*4+1], a2=s[c*4+2], a3=s[c*4+3];
        uint8_t T = a0^a1^a2^a3;
        s[c*4+0] ^= T ^ gpu_xtime(a0^a1);
        s[c*4+1] ^= T ^ gpu_xtime(a1^a2);
        s[c*4+2] ^= T ^ gpu_xtime(a2^a3);
        s[c*4+3] ^= T ^ gpu_xtime(a3^a0);
    }
}


__device__ static void gpu_InvMixColumns(uint8_t s[16]) {
    for (int c = 0; c < 4; ++c) {
        uint8_t a0=s[c*4+0], a1=s[c*4+1], a2=s[c*4+2], a3=s[c*4+3];
        uint8_t T = a0^a1^a2^a3;
        s[c*4+0] = T^a0^gpu_xtime(a0^a1);
        s[c*4+1] = T^a1^gpu_xtime(a1^a2);
        s[c*4+2] = T^a2^gpu_xtime(a2^a3);
        s[c*4+3] = T^a3^gpu_xtime(a3^a0);
        uint8_t u = gpu_xtime(gpu_xtime(a0^a2));
        uint8_t v = gpu_xtime(gpu_xtime(a1^a3));
        uint8_t w = gpu_xtime(u^v);
        s[c*4+0] ^= w^u; s[c*4+1] ^= w^v;
        s[c*4+2] ^= w^u; s[c*4+3] ^= w^v;
    }
}

__device__ static void gpu_AddRoundKey(uint8_t s[16], int round, const uint8_t * __restrict__ rk) {
    const uint8_t *p = rk + round * 16;
    for (int i = 0; i < 16; ++i) s[i] ^= p[i];
}

//----GPU block cipher----

__device__ static void gpu_EncryptBlock(const uint8_t in[16], uint8_t out[16]) {
    uint8_t s[16];
    for (int i = 0; i < 16; ++i) s[i] = in[i];
    gpu_AddRoundKey(s, 0, RK_GPU);
    for (int r = 1; r < aes128::Nr; ++r) {
        gpu_SubBytes(s);
        gpu_ShiftRows(s);
        gpu_MixColumns(s);
        gpu_AddRoundKey(s, r, RK_GPU);
    }
    gpu_SubBytes(s);
    gpu_ShiftRows(s);
    gpu_AddRoundKey(s, aes128::Nr, RK_GPU);
    for (int i = 0; i < 16; ++i) out[i] = s[i];
}


__device__ static void gpu_DecryptBlock(const uint8_t in[16], uint8_t out[16]) {
    uint8_t s[16];
    for (int i = 0; i < 16; ++i) s[i] = in[i];

    gpu_AddRoundKey(s, 0, DRK_GPU);                      // drk[0] = rk[Nr]
    for (int r = 1; r < aes128::Nr; ++r) {
        gpu_InvSubBytes(s);
        gpu_InvShiftRows(s);
        gpu_InvMixColumns(s);                            // *** was missing ***
        gpu_AddRoundKey(s, r, DRK_GPU);                  // drk[r] = InvMixColumns(rk[Nr-r])
    }
    gpu_InvSubBytes(s);
    gpu_InvShiftRows(s);
    gpu_AddRoundKey(s, aes128::Nr, DRK_GPU);             // drk[Nr] = rk[0]

    for (int i = 0; i < 16; ++i) out[i] = s[i];
}


//----Kernels----

__global__ void kernel_ecb_encrypt(
    const uint8_t * __restrict__ plaintext,
          uint8_t * __restrict__ ciphertext,
    int num_blocks)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= num_blocks) return;
    gpu_EncryptBlock(&plaintext[id*16], &ciphertext[id*16]);
}

__global__ void kernel_ecb_decrypt(
    const uint8_t * __restrict__ ciphertext,
          uint8_t * __restrict__ plaintext,
    int num_blocks)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= num_blocks) return;
    gpu_DecryptBlock(&ciphertext[id*16], &plaintext[id*16]);
}

__global__ void kernel_ctr_encrypt(
    const uint8_t * __restrict__ plaintext,
          uint8_t * __restrict__ ciphertext,
    int num_blocks,
    const uint8_t * __restrict__ nonce)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= num_blocks) return;
    uint8_t counter[16];
    for (int i = 0; i < 16; ++i) counter[i] = nonce[i];
    int carry = id;
    for (int i = 15; i >= 0 && carry; --i) {
        carry += counter[i]; counter[i] = (uint8_t)(carry & 0xff); carry >>= 8;
    }
    uint8_t keystream[16];
    gpu_EncryptBlock(counter, keystream);
    for (int i = 0; i < 16; ++i)
        ciphertext[id*16+i] = plaintext[id*16+i] ^ keystream[i];
}

__global__ void kernel_ctr_decrypt(
    const uint8_t * __restrict__ ciphertext,
          uint8_t * __restrict__ plaintext,
    int num_blocks,
    const uint8_t * __restrict__ nonce)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= num_blocks) return;
    uint8_t counter[16];
    for (int i = 0; i < 16; ++i) counter[i] = nonce[i];
    int carry = id;
    for (int i = 15; i >= 0 && carry; --i) {
        carry += counter[i]; counter[i] = (uint8_t)(carry & 0xff); carry >>= 8;
    }
    uint8_t keystream[16];
    gpu_EncryptBlock(counter, keystream);
    for (int i = 0; i < 16; ++i)
        plaintext[id*16+i] = ciphertext[id*16+i] ^ keystream[i];
}

__global__ void kernel_cbc_encrypt(
    const uint8_t * __restrict__ plaintext,
          uint8_t * __restrict__ ciphertext,
    int num_blocks,
    const uint8_t * __restrict__ iv)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    uint8_t prev[16], block[16];
    for (int i = 0; i < 16; ++i) prev[i] = iv[i];
    for (int b = 0; b < num_blocks; ++b) {
        for (int i = 0; i < 16; ++i) block[i] = plaintext[b*16+i] ^ prev[i];
        gpu_EncryptBlock(block, &ciphertext[b*16]);
        for (int i = 0; i < 16; ++i) prev[i] = ciphertext[b*16+i];
    }
}

__global__ void kernel_cbc_decrypt(
    const uint8_t * __restrict__ ciphertext,
          uint8_t * __restrict__ plaintext,
    int num_blocks,
    const uint8_t * __restrict__ iv)
{
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= num_blocks) return;
    uint8_t block_out[16];
    gpu_DecryptBlock(&ciphertext[b*16], block_out);
    const uint8_t *prev = (b == 0) ? iv : &ciphertext[(b-1)*16];
    for (int i = 0; i < 16; ++i)
        plaintext[b*16+i] = block_out[i] ^ prev[i];
}



void gpu_ecb_encrypt(const uint8_t *h_plain, uint8_t *h_cipher, int num_blocks) {
    size_t sz = (size_t)num_blocks * 16;
    uint8_t *d_plain, *d_cipher;
    CUDA_CHECK(cudaMalloc(&d_plain,  sz));
    CUDA_CHECK(cudaMalloc(&d_cipher, sz));
    CUDA_CHECK(cudaMemcpy(d_plain, h_plain, sz, cudaMemcpyHostToDevice));
    int threads = 256, blocks = (num_blocks + threads - 1) / threads;
    kernel_ecb_encrypt<<<blocks, threads>>>(d_plain, d_cipher, num_blocks);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_cipher, d_cipher, sz, cudaMemcpyDeviceToHost));
    cudaFree(d_plain); cudaFree(d_cipher);
}

void gpu_ecb_decrypt(const uint8_t *h_cipher, uint8_t *h_plain, int num_blocks) {
    size_t sz = (size_t)num_blocks * 16;
    uint8_t *d_cipher, *d_plain;
    CUDA_CHECK(cudaMalloc(&d_cipher, sz));
    CUDA_CHECK(cudaMalloc(&d_plain,  sz));
    CUDA_CHECK(cudaMemcpy(d_cipher, h_cipher, sz, cudaMemcpyHostToDevice));
    int threads = 256, blocks = (num_blocks + threads - 1) / threads;
    kernel_ecb_decrypt<<<blocks, threads>>>(d_cipher, d_plain, num_blocks);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_plain, d_plain, sz, cudaMemcpyDeviceToHost));
    cudaFree(d_cipher); cudaFree(d_plain);
}

void gpu_ctr_encrypt(const uint8_t *h_plain, uint8_t *h_cipher,
                     int num_blocks, const uint8_t nonce[16]) {
    size_t sz = (size_t)num_blocks * 16;
    uint8_t *d_plain, *d_cipher, *d_nonce;
    CUDA_CHECK(cudaMalloc(&d_plain,  sz));
    CUDA_CHECK(cudaMalloc(&d_cipher, sz));
    CUDA_CHECK(cudaMalloc(&d_nonce,  16));
    CUDA_CHECK(cudaMemcpy(d_plain,  h_plain, sz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_nonce,  nonce,   16, cudaMemcpyHostToDevice));
    int threads = 256, blocks = (num_blocks + threads - 1) / threads;
    kernel_ctr_encrypt<<<blocks, threads>>>(d_plain, d_cipher, num_blocks, d_nonce);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_cipher, d_cipher, sz, cudaMemcpyDeviceToHost));
    cudaFree(d_plain); cudaFree(d_cipher); cudaFree(d_nonce);
}

void gpu_ctr_decrypt(const uint8_t *h_cipher, uint8_t *h_plain,
                     int num_blocks, const uint8_t nonce[16]) {
    size_t sz = (size_t)num_blocks * 16;
    uint8_t *d_cipher, *d_plain, *d_nonce;
    CUDA_CHECK(cudaMalloc(&d_cipher, sz));
    CUDA_CHECK(cudaMalloc(&d_plain,  sz));
    CUDA_CHECK(cudaMalloc(&d_nonce,  16));
    CUDA_CHECK(cudaMemcpy(d_cipher, h_cipher, sz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_nonce,  nonce,    16, cudaMemcpyHostToDevice));
    int threads = 256, blocks = (num_blocks + threads - 1) / threads;
    kernel_ctr_decrypt<<<blocks, threads>>>(d_cipher, d_plain, num_blocks, d_nonce);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_plain, d_plain, sz, cudaMemcpyDeviceToHost));
    cudaFree(d_cipher); cudaFree(d_plain); cudaFree(d_nonce);
}

void gpu_cbc_encrypt(const uint8_t *h_plain, uint8_t *h_cipher,
                     int num_blocks, const uint8_t iv[16]) {
    size_t sz = (size_t)num_blocks * 16;
    uint8_t *d_plain, *d_cipher, *d_iv;
    CUDA_CHECK(cudaMalloc(&d_plain,  sz));
    CUDA_CHECK(cudaMalloc(&d_cipher, sz));
    CUDA_CHECK(cudaMalloc(&d_iv,     16));
    CUDA_CHECK(cudaMemcpy(d_plain, h_plain, sz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_iv,    iv,      16, cudaMemcpyHostToDevice));
    kernel_cbc_encrypt<<<1, 1>>>(d_plain, d_cipher, num_blocks, d_iv);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_cipher, d_cipher, sz, cudaMemcpyDeviceToHost));
    cudaFree(d_plain); cudaFree(d_cipher); cudaFree(d_iv);
}

void gpu_cbc_decrypt(const uint8_t *h_cipher, uint8_t *h_plain,
                     int num_blocks, const uint8_t iv[16]) {
    size_t sz = (size_t)num_blocks * 16;
    uint8_t *d_cipher, *d_plain, *d_iv;
    CUDA_CHECK(cudaMalloc(&d_cipher, sz));
    CUDA_CHECK(cudaMalloc(&d_plain,  sz));
    CUDA_CHECK(cudaMalloc(&d_iv,     16));
    CUDA_CHECK(cudaMemcpy(d_cipher, h_cipher, sz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_iv,     iv,       16, cudaMemcpyHostToDevice));
    int threads = 256, blocks = (num_blocks + threads - 1) / threads;
    kernel_cbc_decrypt<<<blocks, threads>>>(d_cipher, d_plain, num_blocks, d_iv);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_plain, d_plain, sz, cudaMemcpyDeviceToHost));
    cudaFree(d_cipher); cudaFree(d_plain); cudaFree(d_iv);
}


static void print_message(const uint8_t *msg, int len) {
    for (int i = 0; i < len; i++) std::printf("%02x", msg[i]);
    std::printf("\n");
}

static int assert_equal(const char *label, const uint8_t *got,
                        const uint8_t *expected, int len) {
    if (std::memcmp(got, expected, len) == 0) {
        std::printf("The recovered plaintext matches the original plaintext %s\n", label);
        return 1;
    }
    std::printf("MISMATCH %s\n  got:      ", label);
    for (int i = 0; i < len; i++) std::printf("%02x", got[i]);
    std::printf("\n  expected: ");
    for (int i = 0; i < len; i++) std::printf("%02x", expected[i]);
    std::printf("\n");
    return 0;
}


//----Benchmarks----

static void throughput_sweep(const uint8_t nonce[16], const uint8_t iv[16])
{
    std::printf("\n--------AES-128 GPU Throughput vs Input Size--------\n");
    std::printf("  Median of %d trials per measurement.\n\n", TRIALS);

    const size_t data_sizes[] = {
        1024, 4*1024, 16*1024, 64*1024, 256*1024,
        1*1024*1024, 4*1024*1024, 16*1024*1024, 64*1024*1024
    };
    const int NSIZES = (int)(sizeof(data_sizes) / sizeof(data_sizes[0]));

    std::printf("  %-10s  %-14s  %12s  %12s\n", "Data Size", "Mode", "Wall (ms)", "Throughput");
    std::printf("  %-10s  %-14s  %12s  %12s\n", "----------", "--------------", "------------", "----------");

    cudaEvent_t ev_s, ev_e;
    CUDA_CHECK(cudaEventCreate(&ev_s));
    CUDA_CHECK(cudaEventCreate(&ev_e));

    for (int si = 0; si < NSIZES; si++) {
        size_t data_bytes = data_sizes[si];
        size_t num_blocks = (data_bytes + 15) / 16;
        size_t buf_size   = num_blocks * 16;

        uint8_t *h_plain, *h_cipher;
        CUDA_CHECK(cudaMallocHost(&h_plain,  buf_size));
        CUDA_CHECK(cudaMallocHost(&h_cipher, buf_size));
        for (size_t i = 0; i < buf_size; i++) h_plain[i] = (uint8_t)(i & 0xff);
        gpu_cbc_encrypt(h_plain, h_cipher, (int)num_blocks, iv);

        uint8_t *d_plain, *d_cipher, *d_out, *d_nonce_dev, *d_iv_dev;
        CUDA_CHECK(cudaMalloc(&d_plain,     buf_size));
        CUDA_CHECK(cudaMalloc(&d_cipher,    buf_size));
        CUDA_CHECK(cudaMalloc(&d_out,       buf_size));
        CUDA_CHECK(cudaMalloc(&d_nonce_dev, 16));
        CUDA_CHECK(cudaMalloc(&d_iv_dev,    16));
        CUDA_CHECK(cudaMemcpy(d_plain,     h_plain,  buf_size, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_cipher,    h_cipher, buf_size, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_nonce_dev, nonce,    16,       cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_iv_dev,    iv,       16,       cudaMemcpyHostToDevice));

        int threads = 256;
        int grid    = (int)((num_blocks + threads - 1) / threads);

#define BENCH(label, kernel_call, samples)                               \
        float samples[TRIALS];                                           \
        for (int t = 0; t < TRIALS; t++) {                              \
            CUDA_CHECK(cudaEventRecord(ev_s));                           \
            kernel_call;                                                 \
            CUDA_CHECK(cudaEventRecord(ev_e));                           \
            CUDA_CHECK(cudaEventSynchronize(ev_e));                      \
            float ms = 0; CUDA_CHECK(cudaEventElapsedTime(&ms, ev_s, ev_e)); \
            samples[t] = ms;                                             \
        }                                                                \
        for (int i = 0; i < TRIALS-1; i++)                              \
            for (int j = i+1; j < TRIALS; j++)                          \
                if (samples[j] < samples[i]) { float _t=samples[i]; samples[i]=samples[j]; samples[j]=_t; }

        BENCH("ecb_enc", (kernel_ecb_encrypt<<<grid,threads>>>(d_plain,  d_out, (int)num_blocks)),              ecb_enc_s)
        BENCH("ecb_dec", (kernel_ecb_decrypt<<<grid,threads>>>(d_cipher, d_out, (int)num_blocks)),              ecb_dec_s)
        BENCH("ctr_enc", (kernel_ctr_encrypt<<<grid,threads>>>(d_plain,  d_out, (int)num_blocks, d_nonce_dev)), ctr_enc_s)
        BENCH("ctr_dec", (kernel_ctr_decrypt<<<grid,threads>>>(d_cipher, d_out, (int)num_blocks, d_nonce_dev)), ctr_dec_s)
        BENCH("cbc_enc", (kernel_cbc_encrypt<<<1,1>>>(d_plain,  d_out, (int)num_blocks, d_iv_dev)),             cbc_enc_s)
        BENCH("cbc_dec", (kernel_cbc_decrypt<<<grid,threads>>>(d_cipher, d_out, (int)num_blocks, d_iv_dev)),    cbc_dec_s)
#undef BENCH

        auto gbs = [&](size_t b, double ms) { return (double)b / (ms * 1e-3) / 1e9; };
        const char *unit = (data_bytes >= 1024*1024) ? "MB" : "KB";
        double nd        = (data_bytes >= 1024*1024) ? data_bytes/1048576.0 : data_bytes/1024.0;

        std::printf("  %5.0f %-3s  GPU-ECB-E    %10.4f ms  %8.4f GB/s\n", nd, unit, (double)ecb_enc_s[TRIALS/2], gbs(data_bytes, ecb_enc_s[TRIALS/2]));
        std::printf("  %5.0f %-3s  GPU-ECB-D    %10.4f ms  %8.4f GB/s\n", nd, unit, (double)ecb_dec_s[TRIALS/2], gbs(data_bytes, ecb_dec_s[TRIALS/2]));
        std::printf("  %5.0f %-3s  GPU-CTR-E    %10.4f ms  %8.4f GB/s\n", nd, unit, (double)ctr_enc_s[TRIALS/2], gbs(data_bytes, ctr_enc_s[TRIALS/2]));
        std::printf("  %5.0f %-3s  GPU-CTR-D    %10.4f ms  %8.4f GB/s\n", nd, unit, (double)ctr_dec_s[TRIALS/2], gbs(data_bytes, ctr_dec_s[TRIALS/2]));
        std::printf("  %5.0f %-3s  GPU-CBC-E    %10.4f ms  %8.4f GB/s\n", nd, unit, (double)cbc_enc_s[TRIALS/2], gbs(data_bytes, cbc_enc_s[TRIALS/2]));
        std::printf("  %5.0f %-3s  GPU-CBC-D    %10.4f ms  %8.4f GB/s\n", nd, unit, (double)cbc_dec_s[TRIALS/2], gbs(data_bytes, cbc_dec_s[TRIALS/2]));
        std::printf("\n");

        CUDA_CHECK(cudaFree(d_plain));  CUDA_CHECK(cudaFree(d_cipher));
        CUDA_CHECK(cudaFree(d_out));
        CUDA_CHECK(cudaFree(d_nonce_dev)); CUDA_CHECK(cudaFree(d_iv_dev));
        CUDA_CHECK(cudaFreeHost(h_plain)); CUDA_CHECK(cudaFreeHost(h_cipher));
    }

    CUDA_CHECK(cudaEventDestroy(ev_s));
    CUDA_CHECK(cudaEventDestroy(ev_e));
}

static void measure_single_block_latency(const uint8_t nonce[16], const uint8_t iv[16])
{
    const int LAT_BLOCKS = 8;
    const int LAT_TRIALS = 101;
    const size_t sz = LAT_BLOCKS * 16;

    uint8_t h_plain[sz], h_cipher[sz], h_recov[sz];
    for (int i = 0; i < (int)sz; i++) h_plain[i] = (uint8_t)(i & 0xff);

    uint8_t *d_plain, *d_cipher, *d_nonce, *d_iv;
    CUDA_CHECK(cudaMalloc(&d_plain,  sz)); CUDA_CHECK(cudaMalloc(&d_cipher, sz));
    CUDA_CHECK(cudaMalloc(&d_nonce,  16)); CUDA_CHECK(cudaMalloc(&d_iv,     16));
    CUDA_CHECK(cudaMemcpy(d_nonce, nonce, 16, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_iv,    iv,    16, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start)); CUDA_CHECK(cudaEventCreate(&stop));

    float gpu_ctr_s[LAT_TRIALS];
    for (int t = 0; t < LAT_TRIALS; t++) {
        CUDA_CHECK(cudaEventRecord(start));
        CUDA_CHECK(cudaMemcpy(d_plain, h_plain, sz, cudaMemcpyHostToDevice));
        kernel_ctr_encrypt<<<1, LAT_BLOCKS>>>(d_plain, d_cipher, LAT_BLOCKS, d_nonce);
        CUDA_CHECK(cudaMemcpy(h_cipher, d_cipher, sz, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0; CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        gpu_ctr_s[t] = ms * 1000.0f;
    }

    gpu_cbc_encrypt(h_plain, h_cipher, LAT_BLOCKS, iv);
    float gpu_cbc_s[LAT_TRIALS];
    for (int t = 0; t < LAT_TRIALS; t++) {
        CUDA_CHECK(cudaEventRecord(start));
        CUDA_CHECK(cudaMemcpy(d_cipher, h_cipher, sz, cudaMemcpyHostToDevice));
        kernel_cbc_decrypt<<<1, LAT_BLOCKS>>>(d_cipher, d_plain, LAT_BLOCKS, d_iv);
        CUDA_CHECK(cudaMemcpy(h_recov, d_plain, sz, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0; CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        gpu_cbc_s[t] = ms * 1000.0f;
    }

    for (int i = 0; i < LAT_TRIALS-1; i++)
        for (int j = i+1; j < LAT_TRIALS; j++) {
            if (gpu_ctr_s[j] < gpu_ctr_s[i]) { float t=gpu_ctr_s[i]; gpu_ctr_s[i]=gpu_ctr_s[j]; gpu_ctr_s[j]=t; }
            if (gpu_cbc_s[j] < gpu_cbc_s[i]) { float t=gpu_cbc_s[i]; gpu_cbc_s[i]=gpu_cbc_s[j]; gpu_cbc_s[j]=t; }
        }

    std::printf("\n--------Single 128-Byte Block Latency (side data point)--------\n");
    std::printf("  Input size : %d cipher blocks = %d bytes\n", LAT_BLOCKS, LAT_BLOCKS * 16);
    std::printf("  Trials     : %d\n", LAT_TRIALS);
    std::printf("  %-35s : %8.2f us\n", "GPU CTR encrypt latency", (double)gpu_ctr_s[LAT_TRIALS/2]);
    std::printf("  %-35s : %8.2f us\n", "GPU CBC decrypt latency", (double)gpu_cbc_s[LAT_TRIALS/2]);

    cudaEventDestroy(start); cudaEventDestroy(stop);
    cudaFree(d_plain); cudaFree(d_cipher); cudaFree(d_nonce); cudaFree(d_iv);
}

static void measure_key_schedule_cost(const uint8_t key[aes128::KEY_SIZE])
{
    const int KS_TRIALS = 101;
    uint8_t roundKeys[aes128::EXPANDED_KEY_SIZE];
    uint8_t decryptKeys[aes128::EXPANDED_KEY_SIZE];
    aes128::KeyExpansion(key, roundKeys);
    aes128::BuildDecryptRoundKeys(roundKeys, decryptKeys);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start)); CUDA_CHECK(cudaEventCreate(&stop));

    float up_samples[KS_TRIALS];
    for (int t = 0; t < KS_TRIALS; t++) {
        CUDA_CHECK(cudaEventRecord(start));
        CUDA_CHECK(cudaMemcpyToSymbol(RK_GPU,  roundKeys,   sizeof(RK_GPU)));
        CUDA_CHECK(cudaMemcpyToSymbol(DRK_GPU, decryptKeys, sizeof(DRK_GPU)));
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0; CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        up_samples[t] = ms * 1000.0f;
    }
    for (int i = 0; i < KS_TRIALS-1; i++)
        for (int j = i+1; j < KS_TRIALS; j++)
            if (up_samples[j] < up_samples[i]) { float t=up_samples[i]; up_samples[i]=up_samples[j]; up_samples[j]=t; }

    std::printf("\n--------Key Schedule Cost--------\n");
    std::printf("\n  Time (median of %d trials)\n", KS_TRIALS);
    std::printf("  %-48s : %7.2f us\n", "GPU upload (RK_GPU + DRK_GPU)", (double)up_samples[KS_TRIALS/2]);
    std::printf("\n  Space\n");
    std::printf("  %-48s : %3d bytes  (%d rounds x 16 bytes)\n",
                "RK_GPU  in constant memory",  aes128::EXPANDED_KEY_SIZE, aes128::Nr + 1);
    std::printf("  %-48s : %3d bytes  (%d rounds x 16 bytes)\n",
                "DRK_GPU in constant memory",  aes128::EXPANDED_KEY_SIZE, aes128::Nr + 1);
    std::printf("  %-48s : %3d bytes  total\n",
                "Combined constant memory",  2 * aes128::EXPANDED_KEY_SIZE);

    cudaEventDestroy(start); cudaEventDestroy(stop);
}


int main()
{
    std::printf("--------AES-128 GPU Implementation--------\n");
    std::printf("--------ECB, CTR and CBC Mode Tests--------\n\n");

    const uint8_t key[16] = {
        0x2b,0x7e,0x15,0x16,0x28,0xae,0xd2,0xa6,
        0xab,0xf7,0x15,0x88,0x09,0xcf,0x4f,0x3c
    };
    const uint8_t plaintext[16] = {
        0xcf,0x16,0xcf,0xe8,0xfd,0x0f,0x98,0xaa,
        0xcf,0x16,0xcf,0xe8,0xfd,0x0f,0x98,0xaa
    };
    const uint8_t nonce[16] = {0};
    const uint8_t iv[16]    = {0};

    uint8_t round_keys[aes128::EXPANDED_KEY_SIZE];
    uint8_t decrypt_keys[aes128::EXPANDED_KEY_SIZE];

    std::printf("Generating key schedule\n");
    aes128::KeyExpansion(key, round_keys);
    aes128::BuildDecryptRoundKeys(round_keys, decrypt_keys);
    CUDA_CHECK(cudaMemcpyToSymbol(RK_GPU,  round_keys,   sizeof(RK_GPU)));
    CUDA_CHECK(cudaMemcpyToSymbol(DRK_GPU, decrypt_keys, sizeof(DRK_GPU)));
    std::printf("Key schedule generated and uploaded to GPU (RK_GPU + DRK_GPU)\n\n");

    uint8_t ecb_cipher[16], ecb_recovered[16];
    uint8_t ctr_cipher[16], ctr_recovered[16];
    uint8_t cbc_cipher[16], cbc_recovered[16];

    std::printf("--------ECB Mode (GPU)--------\n\n");
    std::printf("Original Plaintext   : "); print_message(plaintext, 16);
    gpu_ecb_encrypt(plaintext, ecb_cipher, 1);
    std::printf("ECB Ciphertext       : "); print_message(ecb_cipher, 16);
    gpu_ecb_decrypt(ecb_cipher, ecb_recovered, 1);
    std::printf("Recovered Plaintext  : "); print_message(ecb_recovered, 16);
    std::printf("\n");
    assert_equal("ECB mode (GPU)", ecb_recovered, plaintext, 16);

    std::printf("\n--------CTR Mode (GPU)--------\n\n");
    std::printf("Original Plaintext   : "); print_message(plaintext, 16);
    gpu_ctr_encrypt(plaintext, ctr_cipher, 1, nonce);
    std::printf("CTR Ciphertext       : "); print_message(ctr_cipher, 16);
    gpu_ctr_decrypt(ctr_cipher, ctr_recovered, 1, nonce);
    std::printf("Recovered Plaintext  : "); print_message(ctr_recovered, 16);
    std::printf("\n");
    assert_equal("CTR mode (GPU)", ctr_recovered, plaintext, 16);

    std::printf("\n--------CBC Mode (GPU)--------\n\n");
    std::printf("Original Plaintext   : "); print_message(plaintext, 16);
    gpu_cbc_encrypt(plaintext, cbc_cipher, 1, iv);
    std::printf("CBC Ciphertext       : "); print_message(cbc_cipher, 16);
    gpu_cbc_decrypt(cbc_cipher, cbc_recovered, 1, iv);
    std::printf("Recovered Plaintext  : "); print_message(cbc_recovered, 16);
    std::printf("\n");
    assert_equal("CBC mode (GPU)", cbc_recovered, plaintext, 16);

    std::printf("\n--------ECB, CBC and CTR modes validated (GPU)--------\n");

    throughput_sweep(nonce, iv);
    measure_single_block_latency(nonce, iv);
    measure_key_schedule_cost(key);

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::printf("\n--------GPU Device--------\n");
    std::printf("  Device        : %s\n", prop.name);
    std::printf("  SMs           : %d\n", prop.multiProcessorCount);
    std::printf("  Clock         : %.0f MHz\n", prop.clockRate / 1e3);
    std::printf("  Global memory : %.0f MB\n", prop.totalGlobalMem / 1.0e6);

    return 0;
}