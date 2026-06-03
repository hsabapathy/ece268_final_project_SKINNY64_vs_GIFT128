
// GIFT-128 Block Cipher with GPU implmentation (using CUDA)


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <iostream>
#include <iomanip>
#include <time.h>
#include <cuda_runtime.h>
#include <sys/stat.h> 

using namespace std;

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

#define GIFT_ROUNDS 40
#define BLOCK_BYTES 16   // 128-bit block
#define KEY_BYTES   16   // 128-bit key

// GPU Accelaration 1: 'Device' copies of all lookup tables in __constant__ memory
// __constant__ memory sits in a dedicated read-only GPU cache.
// All threads in a warp read the same address simultaneously,
// so there is zero bank-conflict overhead (faster than reading from
// global memory inside a kernel.)
// cudaMemcpyToSymbol() is used once before the first kernel launch to
// populate these from the host arrays above.

__constant__ uint8_t d_GIFT_S[16];       //S-box in constant memory
__constant__ uint8_t d_GIFT_S_inv[16];   //Inverse S-box in constant memory
__constant__ uint8_t d_GIFT_P[128];      //Bit permutation in constant memory
__constant__ uint8_t d_GIFT_P_inv[128];  //Inv permutation in constant memory
__constant__ uint8_t d_GIFT_RC[62];      //Round constants in constant memory

// Helper: upload all constant tables to the GPU.
// Call this once before any GPU cipher work.

static void upload_constant_tables()
{
    //Copy each host table into its matching __constant__ symbol.
    // cudaMemcpyToSymbol resolves the symbol name at compile time.

    cudaMemcpyToSymbol(d_GIFT_S,     GIFT_S,     sizeof(GIFT_S));
    cudaMemcpyToSymbol(d_GIFT_S_inv, GIFT_S_inv, sizeof(GIFT_S_inv));
    cudaMemcpyToSymbol(d_GIFT_P,     GIFT_P,     sizeof(GIFT_P));
    cudaMemcpyToSymbol(d_GIFT_P_inv, GIFT_P_inv, sizeof(GIFT_P_inv));
    cudaMemcpyToSymbol(d_GIFT_RC,    GIFT_RC,    sizeof(GIFT_RC));
}

//  Internal helpers: pack/unpack bytes to nibbles
static void pack_nibbles(const uint8_t nibbles[32], uint8_t bytes[16])
{
    for (int i = 0; i < 16; i++)
        bytes[i] = ((nibbles[31 - 2*i] & 0xF) << 4) | (nibbles[30 - 2*i] & 0xF);
}

//byte[0] -> nibbles[31] (high) and nibbles[30] (low)

static void unpack_nibbles(const uint8_t bytes[16], uint8_t nibbles[32])
{
    for (int i = 0; i < 16; i++) {
        nibbles[31 - 2*i] = (bytes[i] >> 4) & 0xF;
        nibbles[30 - 2*i] =  bytes[i]        & 0xF;
    }
}

// GPU-Accelaration: DEVICE versions of pack/unpack nibbles
// Identical logic to the CPU helpers, but marked __device__ so they
// can be called from inside CUDA kernels.  
// __forceinline__ avoids the overhead of a device-function call

__device__ __forceinline__
static void d_pack_nibbles(const uint8_t nibbles[32], uint8_t bytes[16])
// callable only from GPU kernels

{
    for (int i = 0; i < 16; i++)
        bytes[i] = ((nibbles[31 - 2*i] & 0xF) << 4) | (nibbles[30 - 2*i] & 0xF);
}

__device__ __forceinline__
static void d_unpack_nibbles(const uint8_t bytes[16], uint8_t nibbles[32])
// callable only from GPU kernels
{
    for (int i = 0; i < 16; i++) {
        nibbles[31 - 2*i] = (bytes[i] >> 4) & 0xF;
        nibbles[30 - 2*i] =  bytes[i]        & 0xF;
    }
}


//  GIFT-128 Core: single-block encrypt / decrypt

static void gift128_enc_nibbles(uint8_t input[32], uint8_t key[32])
{
    uint8_t bits[128], perm_bits[128];
    uint8_t key_bits[128];
    uint8_t temp_key[32];

    for (int r = 0; r < GIFT_ROUNDS; r++) {
        // SubCells
        for (int i = 0; i < 32; i++)
            input[i] = GIFT_S[input[i]];

        // PermBits
        for (int i = 0; i < 32; i++)
            for (int j = 0; j < 4; j++)
                bits[4*i+j] = (input[i] >> j) & 0x1;
        for (int i = 0; i < 128; i++)
            perm_bits[GIFT_P[i]] = bits[i];
        for (int i = 0; i < 32; i++) {
            input[i] = 0;
            for (int j = 0; j < 4; j++)
                input[i] ^= perm_bits[4*i+j] << j;
        }
        // AddRoundKey
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
        // Add round constant
        bits[3]   ^= (GIFT_RC[r]     ) & 0x1;
        bits[7]   ^= (GIFT_RC[r] >> 1) & 0x1;
        bits[11]  ^= (GIFT_RC[r] >> 2) & 0x1;
        bits[15]  ^= (GIFT_RC[r] >> 3) & 0x1;
        bits[19]  ^= (GIFT_RC[r] >> 4) & 0x1;
        bits[23]  ^= (GIFT_RC[r] >> 5) & 0x1;
        bits[127] ^= 1;

        for (int i = 0; i < 32; i++) {
            input[i] = 0;
            for (int j = 0; j < 4; j++)
                input[i] ^= bits[4*i+j] << j;
        }

        // Key update: entire key >> 32
        for (int i = 0; i < 32; i++)
            temp_key[i] = key[(i+8) % 32];
        for (int i = 0; i < 24; i++) key[i] = temp_key[i];
        // k0 >> 12 (nibble rotation)
        key[24] = temp_key[27];
        key[25] = temp_key[24];
        key[26] = temp_key[25];
        key[27] = temp_key[26];
        // k1 >> 2 (bit rotation within nibbles)
        key[28] = ((temp_key[28]&0xC)>>2) ^ ((temp_key[29]&0x3)<<2);
        key[29] = ((temp_key[29]&0xC)>>2) ^ ((temp_key[30]&0x3)<<2);
        key[30] = ((temp_key[30]&0xC)>>2) ^ ((temp_key[31]&0x3)<<2);
        key[31] = ((temp_key[31]&0xC)>>2) ^ ((temp_key[28]&0x3)<<2);
    }
}

static void gift128_dec_nibbles(uint8_t input[32], uint8_t key[32])
{
    // Pre-compute all round keys
    uint8_t round_key_state[GIFT_ROUNDS][32];
    uint8_t temp_key[32];

    for (int r = 0; r < GIFT_ROUNDS; r++) {
        for (int i = 0; i < 32; i++)
            round_key_state[r][i] = key[i];

        // Apply same key update as encryption
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

    uint8_t bits[128], perm_bits[128], key_bits[128];

    for (int r = GIFT_ROUNDS-1; r >= 0; r--) {
        // Inverse AddRoundKey
        for (int i = 0; i < 32; i++)
            for (int j = 0; j < 4; j++)
                bits[4*i+j] = (input[i] >> j) & 0x1;
        for (int i = 0; i < 32; i++)
            for (int j = 0; j < 4; j++)
                key_bits[4*i+j] = (round_key_state[r][i] >> j) & 0x1;

        for (int i = 0; i < 32; i++) {
            bits[4*i+1] ^= key_bits[i];
            bits[4*i+2] ^= key_bits[i+64];
        }

        bits[3]   ^= (GIFT_RC[r]     ) & 0x1;
        bits[7]   ^= (GIFT_RC[r] >> 1) & 0x1;
        bits[11]  ^= (GIFT_RC[r] >> 2) & 0x1;
        bits[15]  ^= (GIFT_RC[r] >> 3) & 0x1;
        bits[19]  ^= (GIFT_RC[r] >> 4) & 0x1;
        bits[23]  ^= (GIFT_RC[r] >> 5) & 0x1;
        bits[127] ^= 1;

        for (int i = 0; i < 32; i++) {
            input[i] = 0;
            for (int j = 0; j < 4; j++)
                input[i] ^= bits[4*i+j] << j;
        }

        // Inverse PermBits
        for (int i = 0; i < 32; i++)
            for (int j = 0; j < 4; j++)
                bits[4*i+j] = (input[i] >> j) & 0x1;
        for (int i = 0; i < 128; i++)
            perm_bits[GIFT_P_inv[i]] = bits[i];
        for (int i = 0; i < 32; i++) {
            input[i] = 0;
            for (int j = 0; j < 4; j++)
                input[i] ^= perm_bits[4*i+j] << j;
        }

        // Inverse SubCells
        for (int i = 0; i < 32; i++)
            input[i] = GIFT_S_inv[input[i]];
    }
}

/*
 gift128_encrypt_block
 plaintext: 16-byte input block
 key: 16-byte key
 ciphertext: 16-byte output block
 */
void gift128_encrypt_block(const uint8_t plaintext[BLOCK_BYTES],
                           const uint8_t key[KEY_BYTES],
                           uint8_t       ciphertext[BLOCK_BYTES])
{
    uint8_t state[32], key_nibs[32];
    unpack_nibbles(plaintext, state);
    unpack_nibbles(key,       key_nibs);
    gift128_enc_nibbles(state, key_nibs);
    pack_nibbles(state, ciphertext);
}

/*
 gift128_decrypt_block
 ciphertext: 16-byte input block
 key: 16-byte key
 plaintext: 16-byte output block
 */
void gift128_decrypt_block(const uint8_t ciphertext[BLOCK_BYTES],
                           const uint8_t key[KEY_BYTES],
                           uint8_t       plaintext[BLOCK_BYTES])
{
    uint8_t state[32], key_nibs[32];
    unpack_nibbles(ciphertext, state);
    unpack_nibbles(key,        key_nibs);
    gift128_dec_nibbles(state, key_nibs);
    pack_nibbles(state, plaintext);
}


//  CBC Mode
//  Encrypt: C[i] = E_K(P[i] XOR C[i-1]),   C[-1] = IV
//  Decrypt: P[i] = D_K(C[i]) XOR C[i-1],   C[-1] = IV

// GPU Accelarated DEVICE core: GIFT-128 single-block encrypt
// This is an exact copy of the CPU logic, but:
// -> uses d_GIFT_S, d_GIFT_P, d_GIFT_RC from __constant__ memory
// -> __forceinline__ tells the compiler to expand this at every call site,
//     avoiding the overhead of a device function call stack frame

__device__ __forceinline__
static void d_gift128_enc_nibbles(uint8_t input[32], uint8_t key[32])

// GPU encrypt core -> reads tables from __constant__ memory
{
    uint8_t bits[128], perm_bits[128];
    uint8_t key_bits[128];
    uint8_t temp_key[32];

    for (int r = 0; r < GIFT_ROUNDS; r++) {
        // SubCells —> uses d_GIFT_S from __constant__ memory
        for (int i = 0; i < 32; i++)
            input[i] = d_GIFT_S[input[i]];  // constant memory read

        // PermBits —> uses d_GIFT_P from __constant__ memory
        for (int i = 0; i < 32; i++)
            for (int j = 0; j < 4; j++)
                bits[4*i+j] = (input[i] >> j) & 0x1;
        for (int i = 0; i < 128; i++)
            perm_bits[d_GIFT_P[i]] = bits[i];  // constant memory read
        for (int i = 0; i < 32; i++) {
            input[i] = 0;
            for (int j = 0; j < 4; j++)
                input[i] ^= perm_bits[4*i+j] << j;
        }

        // AddRoundKey
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

        // Add round constant —> uses d_GIFT_RC from __constant__ memory
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

        // Key update
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

// DEVICE block-encrypt wrapper
// Mirrors gift128_encrypt_block() but runs on GPU

__device__ __forceinline__
static void d_gift128_encrypt_block(const uint8_t plaintext[BLOCK_BYTES],
                                    const uint8_t key[KEY_BYTES],
                                          uint8_t ciphertext[BLOCK_BYTES])
// Device wrapper for GPU: unpacks, encrypts, repacks
{
    uint8_t state[32], key_nibs[32];
    d_unpack_nibbles(plaintext, state);   // device unpack
    d_unpack_nibbles(key,       key_nibs);
    d_gift128_enc_nibbles(state, key_nibs); // device encrypt core
    d_pack_nibbles(state, ciphertext);    // device pack
}

// DEVICE block-decrypt (uses pre-computed round keys)
// the GPU version accepts a flat array of all 40 round-key nibble states 
// that were computed once on the CPU.
// This avoids repeating the key schedule inside every thread.

__device__ __forceinline__
static void d_gift128_decrypt_block_precomp(
    const uint8_t  ciphertext[BLOCK_BYTES],
    const uint8_t  precomp_rkeys[GIFT_ROUNDS * 32],  // pre-computed round keys
          uint8_t  plaintext[BLOCK_BYTES])
// Device decrypt using pre-computed round keys
{
    uint8_t input[32];
    d_unpack_nibbles(ciphertext, input);  // device unpack

    uint8_t bits[128], perm_bits[128], key_bits[128];

    for (int r = GIFT_ROUNDS - 1; r >= 0; r--) {
        // Inverse AddRoundKey
        for (int i = 0; i < 32; i++)
            for (int j = 0; j < 4; j++)
                bits[4*i+j] = (input[i] >> j) & 0x1;

        // Read round key from the pre-computed flat array instead of
        // re-deriving it. Offset = r * 32 nibbles.
        for (int i = 0; i < 32; i++)
            for (int j = 0; j < 4; j++)
                key_bits[4*i+j] = (precomp_rkeys[r * 32 + i] >> j) & 0x1;

        for (int i = 0; i < 32; i++) {
            bits[4*i+1] ^= key_bits[i];
            bits[4*i+2] ^= key_bits[i+64];
        }

        // Inverse round constant —> uses d_GIFT_RC from __constant__ memory
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

        // Inverse PermBits — uses d_GIFT_P_inv from __constant__ memory
        for (int i = 0; i < 32; i++)
            for (int j = 0; j < 4; j++)
                bits[4*i+j] = (input[i] >> j) & 0x1;
        for (int i = 0; i < 128; i++)
            perm_bits[d_GIFT_P_inv[i]] = bits[i];  // constant memory read
        for (int i = 0; i < 32; i++) {
            input[i] = 0;
            for (int j = 0; j < 4; j++)
                input[i] ^= perm_bits[4*i+j] << j;
        }

        // Inverse SubCells —> uses d_GIFT_S_inv from __constant__ memory
        for (int i = 0; i < 32; i++)
            input[i] = d_GIFT_S_inv[input[i]];  // constant memory read
    }
    d_pack_nibbles(input, plaintext);  // device pack
}

// CPU helper: pre-compute all 40 round-key nibble states
// The CPU decrypt already walked the key schedule once to fill
// round_key_state[40][32].
// defined it as a standalone function
// so callers can compute it once and upload it to GPU memory
// Output: flat array of GIFT_ROUNDS * 32 bytes  (round 0 first)

static void gift128_precompute_round_keys(const uint8_t key[KEY_BYTES],
                                          uint8_t out_rkeys[GIFT_ROUNDS * 32])
// Pre-compute round keys on CPU; result is uploaded to GPU once
{
    uint8_t k[32], temp_key[32];
    unpack_nibbles(key, k);  // convert 16-byte key to 32 nibbles

    for (int r = 0; r < GIFT_ROUNDS; r++) {
        // Store this round's key state into the flat output array
        for (int i = 0; i < 32; i++)
            out_rkeys[r * 32 + i] = k[i];  // write round r key nibbles

        // Same key update as gift128_dec_nibbles (unchanged logic)
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

/*
 gift128_cbc_encrypt
 key: 16-byte
 iv: 16-byte 
 input: plaintext
 output: ciphertext
 */
void gift128_cbc_encrypt(const uint8_t *key,
                         const uint8_t  iv[BLOCK_BYTES],
                         const uint8_t *input,
                               uint8_t *output,
                               size_t   length)
{
    uint8_t chain[BLOCK_BYTES];
    memcpy(chain, iv, BLOCK_BYTES);

    for (size_t offset = 0; offset < length; offset += BLOCK_BYTES) {
        // XOR plaintext block with previous ciphertext (or IV)
        uint8_t xored[BLOCK_BYTES];
        for (int i = 0; i < BLOCK_BYTES; i++)
            xored[i] = input[offset + i] ^ chain[i];

        // Encrypt
        gift128_encrypt_block(xored, key, output + offset);

        memcpy(chain, output + offset, BLOCK_BYTES);
    }
}
/*
 gift128_cbc_decrypt
 key: 16-byte
 iv: 16-byte
 input: ciphertext
 output: plaintext 

 */
void gift128_cbc_decrypt(const uint8_t *key,
                         const uint8_t  iv[BLOCK_BYTES],
                         const uint8_t *input,
                               uint8_t *output,
                               size_t   length)
{
    uint8_t chain[BLOCK_BYTES];
    memcpy(chain, iv, BLOCK_BYTES);

    for (size_t offset = 0; offset < length; offset += BLOCK_BYTES) {
        // Decrypt block
        uint8_t decrypted[BLOCK_BYTES];
        gift128_decrypt_block(input + offset, key, decrypted);

        // XOR with previous ciphertext block (or IV)
        for (int i = 0; i < BLOCK_BYTES; i++)
            output[offset + i] = decrypted[i] ^ chain[i];

        // Advance chain to current ciphertext block
        memcpy(chain, input + offset, BLOCK_BYTES);
    }
}


//  CTR Mode
//
//  Keystream block i: K[i] = E_K(nonce || counter_i)
//  Output: C[i] = P[i] XOR K[i]   (same for decrypt)
//  Counter starts at 0 and increments per block.

static void increment_counter(uint8_t counter_block[BLOCK_BYTES])
{
    for (int i = BLOCK_BYTES - 1; i >= BLOCK_BYTES - 4; i--) {
        if (++counter_block[i] != 0) break;
    }
}

/*
 gift128_ctr_crypt  (encryption and decryption are identical in CTR)
 key: 16-byte 
 nonce: 12-byte nonce (the remaining 4 bytes are the counter)
 input: plaintext/ ciphertext
 output: ciphertext/ plaintext
 */
void gift128_ctr_crypt(const uint8_t *key,
                       const uint8_t  nonce[12],
                       const uint8_t *input,
                             uint8_t *output,
                             size_t   length)
{
    // Building initial counter block
    uint8_t counter_block[BLOCK_BYTES];
    memcpy(counter_block, nonce, 12);
    memset(counter_block + 12, 0, 4);

    size_t offset = 0;
    while (offset < length) {
        uint8_t keystream[BLOCK_BYTES];
        gift128_encrypt_block(counter_block, key, keystream);

        // XOR as many bytes as remain
        size_t rem = (length - offset < BLOCK_BYTES) ? (length - offset) : BLOCK_BYTES;
        for (size_t i = 0; i < rem; i++)
            output[offset + i] = input[offset + i] ^ keystream[i];

        increment_counter(counter_block);
        offset += rem;
    }
}

// CUDA kernel: CTR mode encryption / decryption
// Each CUDA thread handles one 128-bit block.
// Because CTR keystream block i depends only on (nonce || i), blocks are
// fully independent
//
// Thread index  →  block index  →  counter value written into bytes 12-15
// of the counter block, then encrypted with GIFT-128 to produce keystream.
// Keystream is XOR'd with the corresponding input bytes to produce output.
//
// Parameters:
//   d_key        : 16-byte key already in GPU global memory
//   d_nonce      : 12-byte nonce already in GPU global memory
//   d_input      : plaintext (encrypt) or ciphertext (decrypt)
//   d_output     : ciphertext (encrypt) or plaintext (decrypt)
//   num_blocks   : total number of 16-byte blocks to process

__global__
void gift128_ctr_kernel(const uint8_t* __restrict__ d_key,    //key in device memory
                        const uint8_t* __restrict__ d_nonce,  //12-byte nonce
                        const uint8_t* __restrict__ d_input,  //input buffer
                              uint8_t* __restrict__ d_output, //output buffer
                        size_t num_blocks)                     //total block count
{
    // Computing a unique global thread index.
    // blockIdx.x  = which thread block this thread belongs to
    // blockDim.x  = threads per thread block (128)
    // threadIdx.x = this thread's index within its thread block
    // a unique id in [0, gridDim.x * blockDim.x).

    int tid = blockIdx.x * blockDim.x + threadIdx.x;  // one thread per cipher block

    // if we launched more threads than blocks, excess threads do nothing.
    if ((size_t)tid >= num_blocks) return;  // discard out-of-range threads

    // Build this thread's counter block:
    // bytes 0-11 = nonce (same for all threads)
    // bytes 12-15 = big-endian counter = tid
    uint8_t counter_block[BLOCK_BYTES];
    for (int i = 0; i < 12; i++)
        counter_block[i] = d_nonce[i];           // copy nonce bytes inline

    // Write counter as big-endian 32-bit integer into bytes 12-15.
    // Each thread's tid is unique, so each thread produces a unique
    // counter block and therefore a unique keystream block.
    counter_block[12] = (tid >> 24) & 0xFF;      // counter MSB
    counter_block[13] = (tid >> 16) & 0xFF;      
    counter_block[14] = (tid >>  8) & 0xFF;    
    counter_block[15] =  tid        & 0xFF;      // counter LSB

    // Copy the key into a local array (device function needs non-const ptr)
    uint8_t local_key[KEY_BYTES];
    for (int i = 0; i < KEY_BYTES; i++)
        local_key[i] = d_key[i];

    // Encrypt the counter block to produce this thread's keystream block.
    //         d_gift128_encrypt_block uses __constant__ tables
    uint8_t keystream[BLOCK_BYTES];
    d_gift128_encrypt_block(counter_block, local_key, keystream);

    // XOR input with keystream. The byte offset into the global buffers
    // is tid * 16 (each thread owns a 16-byte window).
    size_t offset = (size_t)tid * BLOCK_BYTES;   // this thread's byte offset
    for (int i = 0; i < BLOCK_BYTES; i++)
        d_output[offset + i] = d_input[offset + i] ^ keystream[i];  // XOR
}

// Host-side launcher for the CTR kernel
//
// Handling host to device transfers:
//   1. Allocate GPU buffers
//   2. Copy key, nonce, and input to GPU
//   3. Launch the kernel with the right grid/block dimensions
//   4. Copy output back to host
//   5. Free GPU buffers
void gift128_ctr_crypt_gpu(const uint8_t *key,
                           const uint8_t  nonce[12],
                           const uint8_t *input,
                                 uint8_t *output,
                                 size_t   length)
// GPU-accelerated CTR mode: one thread per 16-byte block
{
    // Number of 16-byte blocks to encrypt
    size_t num_blocks = (length + BLOCK_BYTES - 1) / BLOCK_BYTES; 

    // Allocate GPU (device) memory for key, nonce, input, output
    uint8_t *d_key, *d_nonce, *d_in, *d_out;
    cudaMalloc(&d_key,   KEY_BYTES);                      // 16-byte key 
    cudaMalloc(&d_nonce, 12);                             // 12-byte nonce
    cudaMalloc(&d_in,    num_blocks * BLOCK_BYTES);       // padded input
    cudaMalloc(&d_out,   num_blocks * BLOCK_BYTES);       // [output buffer

    // Copy key and nonce from host to GPU global memory
    cudaMemcpy(d_key,   key,   KEY_BYTES, cudaMemcpyHostToDevice); 
    cudaMemcpy(d_nonce, nonce, 12,        cudaMemcpyHostToDevice); 

    // Zero-pad the input buffer on GPU 
    cudaMemset(d_in, 0, num_blocks * BLOCK_BYTES);         // zero pad
    cudaMemcpy(d_in, input, length, cudaMemcpyHostToDevice); // copy actual data

    // Determine grid dimensions.
    //         threads_per_block = 128 (fills half a warp).
    //         blocks_in_grid = ceil(num_blocks / 128) so every cipher block
    //         gets exactly one thread.
    int threads_per_block = 128;                                        
    int blocks_in_grid    = (num_blocks + threads_per_block - 1)       
                            / threads_per_block;

    // Launch the CTR kernel. All threads run simultaneously on the GPU.
    //         <<<blocks_in_grid, threads_per_block>>> is the CUDA launch config.
    
    gift128_ctr_kernel<<<blocks_in_grid, threads_per_block>>>(
        d_key, d_nonce, d_in, d_out, num_blocks);
    cudaDeviceSynchronize();  //

    // Copy the result back from GPU to host 
    cudaMemcpy(output, d_out, length, cudaMemcpyDeviceToHost);  //

    // Free all GPU buffers
    cudaFree(d_key);   
    cudaFree(d_nonce); 
    cudaFree(d_in);    
    cudaFree(d_out);   
}

// CUDA kernel: CBC decrypt
//
// In CBC decryption, all ciphertext blocks C[0..N-1] are known before we
// start.  The decryption formula is:
//     P[i] = D_K( C[i] ) XOR C[i-1]     (C[-1] = IV)
//
// D_K(C[i]) for every i is independent. We therefore assign one thread per block and decrypt
// all blocks in parallel, then XOR with the appropriate predecessor.
//
// The pre-computed round keys (from gift128_precompute_round_keys) are
// passed in d_rkeys so each thread skips the key schedule entirely.
//
// Parameters:
//   d_rkeys      : GIFT_ROUNDS * 32 bytes of pre-computed round keys 
//   d_iv         : 16-byte IV (C[-1])
//   d_input      : ciphertext buffer
//   d_output     : plaintext output
//   num_blocks   : number of 16-byte blocks

__global__ void gift128_cbc_decrypt_kernel(
    const uint8_t* __restrict__ d_rkeys,   // pre-computed round keys
    const uint8_t* __restrict__ d_iv,      // IV = C[-1]
    const uint8_t* __restrict__ d_input,   // ciphertext
          uint8_t* __restrict__ d_output,  // plaintext
    size_t num_blocks)
{
    // Unique thread index
    int tid = blockIdx.x * blockDim.x + threadIdx.x;  // one thread per cipher block
    if ((size_t)tid >= num_blocks) return;              // discard excess threads

    size_t offset = (size_t)tid * BLOCK_BYTES;  // byte offset for this thread

    // Decrypt C[tid] using pre-computed round keys.
    //         This call reads from d_rkeys.
    uint8_t decrypted[BLOCK_BYTES];
    d_gift128_decrypt_block_precomp(            
        d_input + offset, d_rkeys, decrypted);

    // XOR with the previous ciphertext block (or IV for block 0).
    // The predecessor is always available in d_input / d_iv upfront (this is why CBC_decrypt parallelises easily unlike CBC encrypt)
    const uint8_t *prev = (tid == 0)
                          ? d_iv                              // block 0 uses IV
                          : d_input + offset - BLOCK_BYTES;  // others use C[i-1]

    for (int i = 0; i < BLOCK_BYTES; i++)
        d_output[offset + i] = decrypted[i] ^ prev[i];  // chain XOR
}

// Host-side launcher for the CBC decrypt kernel
//
// Key schedule is pre-computed once on the CPU, uploaded once, and
// reused by all threads
void gift128_cbc_decrypt_gpu(const uint8_t *key,
                             const uint8_t  iv[BLOCK_BYTES],
                             const uint8_t *input,
                                   uint8_t *output,
                                   size_t   length)
// GPU-accelerated CBC decrypt: fully parallel, one thread per block
{
    size_t num_blocks = (length + BLOCK_BYTES - 1) / BLOCK_BYTES;

    // Pre-compute all 40 round keys on the CPU once.
    // Cost: one key schedule walk. Benefit: every GPU thread avoids
    // running the key schedule itself (saves GIFT_ROUNDS * 32 nibble
    // operations per thread).
    uint8_t host_rkeys[GIFT_ROUNDS * 32];
    gift128_precompute_round_keys(key, host_rkeys);  // CPU pre-computation

    // Allocate GPU buffers
    uint8_t *d_rkeys, *d_iv_dev, *d_in, *d_out;
    cudaMalloc(&d_rkeys,  GIFT_ROUNDS * 32);            // round keys on GPU
    cudaMalloc(&d_iv_dev, BLOCK_BYTES);                 // IV on GPU
    cudaMalloc(&d_in,     num_blocks * BLOCK_BYTES);    // ciphertext on GPU
    cudaMalloc(&d_out,    num_blocks * BLOCK_BYTES);    // output buffer on GPU

    // Upload pre-computed round keys to GPU 
    cudaMemcpy(d_rkeys, host_rkeys, GIFT_ROUNDS * 32,
               cudaMemcpyHostToDevice);  // single upload, all threads read this

    // Upload IV and ciphertext
    cudaMemcpy(d_iv_dev, iv,    BLOCK_BYTES,            cudaMemcpyHostToDevice);
    cudaMemset(d_in, 0, num_blocks * BLOCK_BYTES);                             
    cudaMemcpy(d_in,     input, length,                 cudaMemcpyHostToDevice);  

    // Grid / block configuration 
    int threads_per_block = 128;                                        
    int blocks_in_grid    = (num_blocks + threads_per_block - 1)       
                            / threads_per_block;

    // Launch CBC decrypt kernel
    gift128_cbc_decrypt_kernel<<<blocks_in_grid, threads_per_block>>>(  
        d_rkeys, d_iv_dev, d_in, d_out, num_blocks);

    cudaDeviceSynchronize();  // wait for all threads to finish

    // Copy plaintext back to host
    cudaMemcpy(output, d_out, length, cudaMemcpyDeviceToHost);  

    //Free GPU buffers
    cudaFree(d_rkeys);   
    cudaFree(d_iv_dev);  
    cudaFree(d_in);      
    cudaFree(d_out);     
}

//Printing a byte array as hex
static void print_hex(const char *label, const uint8_t *data, size_t len)
{
    printf("%-20s: ", label);
    for (size_t i = 0; i < len; i++) printf("%02x", data[i]);
    printf("\n");
}

//Assert and printing pass/fail
 
static void assert_equal(const char *label, const uint8_t *a, const uint8_t *b, size_t len)
{
    bool ok = (memcmp(a, b, len) == 0);
    printf("%s: %s\n", label, ok ? "PASS" : "FAIL");
}

//Using the same test vectors as SKINNY64

static void run_gift128_core_test()
{
    printf("\n GIFT-128 Core Block Cipher Test\n");

    uint8_t key[16] = {
        0x9e,0xb9,0x36,0x40,0xd0,0x88,0xda,0x63,   
        0x76,0xa3,0x9d,0x1c,0x8b,0xea,0x71,0xe1    
    };
    uint8_t pt[16] = {
        0xcf,0x16,0xcf,0xe8,0xfd,0x0f,0x98,0xaa,   // SKINNY plaintext
        0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00    // zero-padded to 128 bits
    };
    uint8_t ct[16], recovered[16];
 
    gift128_encrypt_block(pt, key, ct);
    gift128_decrypt_block(ct, key, recovered);

    assert_equal("Encryption->Decryption round-trip", recovered, pt, 16);
}
static void run_cbc_cpu_test()
{
    printf("\n--------CBC Mode (CPU)--------\n\n");

    uint8_t key[16] = {
        0x9e,0xb9,0x36,0x40,0xd0,0x88,0xda,0x63,
        0x76,0xa3,0x9d,0x1c,0x8b,0xea,0x71,0xe1
    };
    uint8_t plain[16] = {
        0xcf,0x16,0xcf,0xe8,0xfd,0x0f,0x98,0xaa,
        0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00
    };
    uint8_t iv[16] = {0};
    uint8_t cbc_cipher[16], cbc_recovered[16];

    printf("Original Plaintext   : ");
    for (int i = 0; i < 16; i++) printf("%02x", plain[i]);
    printf("\n");

    gift128_cbc_encrypt(key, iv, plain, cbc_cipher, 16);
 
    gift128_cbc_decrypt(key, iv, cbc_cipher, cbc_recovered, 16);


    printf("Recovered Plaintext  : ");
    for (int i = 0; i < 16; i++) printf("%02x", cbc_recovered[i]);
    printf("\n\n");

    assert_equal("CPU CBC encrypt->decrypt", cbc_recovered, plain, 16);
}
 
// GPU CBC decrypt
static void run_cbc_gpu_decrypt_test()
{
    printf("\n--------CBC Decrypt (GPU)--------\n\n");

    uint8_t key[16] = {
        0x9e,0xb9,0x36,0x40,0xd0,0x88,0xda,0x63,
        0x76,0xa3,0x9d,0x1c,0x8b,0xea,0x71,0xe1
    };
    uint8_t plain[16] = {
        0xcf,0x16,0xcf,0xe8,0xfd,0x0f,0x98,0xaa,
        0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00
    };
    uint8_t iv[16] = {0};
    uint8_t cbc_cipher[16];
    uint8_t gpu_recovered[16];

    // Encrypt on CPU (CBC encrypt is sequential)
    gift128_cbc_encrypt(key, iv, plain, cbc_cipher, 16);

    printf("Ciphertext (CPU enc) : ");
    for (int i = 0; i < 16; i++) printf("%02x", cbc_cipher[i]);
    printf("\n");

    // Decrypt on GPU using the parallel CBC decrypt kernel
    gift128_cbc_decrypt_gpu(key, iv, cbc_cipher, gpu_recovered, 16); 

    printf("GPU Recovered Plain  : ");
    for (int i = 0; i < 16; i++) printf("%02x", gpu_recovered[i]);
    printf("\n\n");

    assert_equal("GPU CBC decrypt matches CPU plaintext", gpu_recovered, plain, 16);
}

static void run_ctr_cpu_test()
{
    printf("\n--------CTR Mode (CPU)--------\n\n");

    uint8_t key[16] = {
        0x9e,0xb9,0x36,0x40,0xd0,0x88,0xda,0x63,
        0x76,0xa3,0x9d,0x1c,0x8b,0xea,0x71,0xe1
    };
    uint8_t plain[16] = {
        0xcf,0x16,0xcf,0xe8,0xfd,0x0f,0x98,0xaa,
        0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00
    };
    uint8_t nonce[12] = {0};
    uint8_t ctr_cipher[16], ctr_recovered[16];

    printf("Original Plaintext   : ");
    for (int i = 0; i < 16; i++) printf("%02x", plain[i]);
    printf("\n");

    gift128_ctr_crypt(key, nonce, plain, ctr_cipher, 16);
    gift128_ctr_crypt(key, nonce, ctr_cipher, ctr_recovered, 16);

    printf("CPU CTR Ciphertext   : ");
    for (int i = 0; i < 16; i++) printf("%02x", ctr_cipher[i]);
    printf("\n");

    printf("CPU Recovered Plain  : ");
    for (int i = 0; i < 16; i++) printf("%02x", ctr_recovered[i]);
    printf("\n\n");

    assert_equal("CPU CTR encrypt->decrypt", ctr_recovered, plain, 16);
}

// GPU CTR
static void run_ctr_gpu_test()
{
    printf("\n--------CTR Mode (GPU) --------\n\n");

    uint8_t key[16] = {
        0x9e,0xb9,0x36,0x40,0xd0,0x88,0xda,0x63,
        0x76,0xa3,0x9d,0x1c,0x8b,0xea,0x71,0xe1
    };
    uint8_t plain[16] = {
        0xcf,0x16,0xcf,0xe8,0xfd,0x0f,0x98,0xaa,
        0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00
    };
    uint8_t nonce[12] = {0};

    // Reference: encrypt and decrypt on CPU
    uint8_t cpu_cipher[16], cpu_recovered[16];
    gift128_ctr_crypt(key, nonce, plain, cpu_cipher, 16);
    gift128_ctr_crypt(key, nonce, cpu_cipher, cpu_recovered, 16);

    // Encrypt on GPU
    uint8_t gpu_cipher[16], gpu_recovered[16];
    gift128_ctr_crypt_gpu(key, nonce, plain, gpu_cipher, 16);     

    printf("CPU CTR Ciphertext   : ");
    for (int i = 0; i < 16; i++) printf("%02x", cpu_cipher[i]);
    printf("\n");

    printf("GPU CTR Ciphertext   : ");
    for (int i = 0; i < 16; i++) printf("%02x", gpu_cipher[i]);
    printf("\n");

    assert_equal("GPU CTR matches CPU CTR", gpu_cipher, cpu_cipher, 16);

    // Decrypt on GPU (CTR decrypt == CTR encrypt with same key/nonce)
    gift128_ctr_crypt_gpu(key, nonce, gpu_cipher, gpu_recovered, 16); 

    printf("GPU Recovered Plain  : ");
    for (int i = 0; i < 16; i++) printf("%02x", gpu_recovered[i]);
    printf("\n\n");

    assert_equal("GPU CTR decrypt round-trip ", gpu_recovered, plain, 16);
}

//  Printing Quantitative Metrics

// RDTSC helper: to find the number of cycles

static inline uint64_t rdtsc()
{
    uint32_t lo, hi;
    __asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
    return ((uint64_t)hi << 32) | lo;
}
 
// Comparison function for qsort
static int cmp_u64(const void *a, const void *b)
{
    uint64_t x = *(const uint64_t*)a;
    uint64_t y = *(const uint64_t*)b;
    return (x > y) - (x < y);
}
 
#define BUF_BLOCKS 256      //256 blocks, fits in L1 cache
#define TRIALS     31       //we're using odd number so median is unambiguous

//Finding number of cycles needed to encrypt 1 byte in CTR
static double measure_cycles_per_byte_ctr()
{
    static uint8_t buf[BUF_BLOCKS * BLOCK_BYTES];
    static uint8_t out[BUF_BLOCKS * BLOCK_BYTES];
    uint8_t key[KEY_BYTES] = {0x9e,0xb9,0x36,0x40,0xd0,0x88,0xda,0x63,
                               0x76,0xa3,0x9d,0x1c,0x8b,0xea,0x71,0xe1};
    uint8_t nonce[12] = {0};
    memset(buf, 0xA5, sizeof(buf));
 
    gift128_ctr_crypt(key, nonce, buf, out, sizeof(buf));
 
    uint64_t samples[TRIALS];
    for (int t = 0; t < TRIALS; t++) {
        uint8_t n[12] = {0};
        uint64_t t0 = rdtsc();
        gift128_ctr_crypt(key, n, buf, out, sizeof(buf));
        uint64_t t1 = rdtsc();
        samples[t] = t1 - t0;
    }
    qsort(samples, TRIALS, sizeof(uint64_t), cmp_u64);
    return (double)samples[TRIALS / 2] / (BUF_BLOCKS * BLOCK_BYTES);
}

//Finding number of cycles needed to encrypt 1 byte in CBC
static double measure_cycles_per_byte_cbc()
{
    static uint8_t buf[BUF_BLOCKS * BLOCK_BYTES];
    static uint8_t out[BUF_BLOCKS * BLOCK_BYTES];
    uint8_t key[KEY_BYTES] = {0x9e,0xb9,0x36,0x40,0xd0,0x88,0xda,0x63,
                               0x76,0xa3,0x9d,0x1c,0x8b,0xea,0x71,0xe1};
    uint8_t iv[BLOCK_BYTES] = {0};
    memset(buf, 0xA5, sizeof(buf));
 
    gift128_cbc_encrypt(key, iv, buf, out, sizeof(buf));
 
    uint64_t samples[TRIALS];
    for (int t = 0; t < TRIALS; t++) {
        uint8_t v[BLOCK_BYTES] = {0};
        uint64_t t0 = rdtsc();
        gift128_cbc_encrypt(key, v, buf, out, sizeof(buf));
        uint64_t t1 = rdtsc();
        samples[t] = t1 - t0;
    }
    qsort(samples, TRIALS, sizeof(uint64_t), cmp_u64);
    return (double)samples[TRIALS / 2] / (BUF_BLOCKS * BLOCK_BYTES);
}
 
//Finding time taken for one full encrypt
static double measure_keyschedule_cycles()
{
    uint8_t key[KEY_BYTES]   = {0x9e,0xb9,0x36,0x40,0xd0,0x88,0xda,0x63,
                                  0x76,0xa3,0x9d,0x1c,0x8b,0xea,0x71,0xe1};
    uint8_t pt[BLOCK_BYTES]  = {0xcf,0x16,0xcf,0xe8,0xfd,0x0f,0x98,0xaa,
                                  0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00};
    uint8_t ct[BLOCK_BYTES];
 
    for (int i = 0; i < 100; i++) gift128_encrypt_block(pt, key, ct);
 
    uint64_t samples[TRIALS];
    for (int t = 0; t < TRIALS; t++) {
        uint64_t t0 = rdtsc();
        gift128_encrypt_block(pt, key, ct);
        uint64_t t1 = rdtsc();
        samples[t] = t1 - t0;
    }
    qsort(samples, TRIALS, sizeof(uint64_t), cmp_u64);
    return (double)samples[TRIALS / 2];
}

//Printing code size
/*
static void print_code_size()
{
    printf("\n--------- Code Size ----------\n");
 
    struct TableSize { const char *name; size_t bytes; };
    TableSize tables[] = {
        { "GIFT_S         (4-bit Sbox, 16 entries)",          sizeof(GIFT_S)     },
        { "GIFT_S_inv     (4-bit Sbox inverse, 16 entries)",  sizeof(GIFT_S_inv) },
        { "GIFT_P         (bit permutation, 128 entries)",     sizeof(GIFT_P)     },
        { "GIFT_P_inv     (inverse permutation, 128 entries)", sizeof(GIFT_P_inv) },
        { "GIFT_RC        (round constants, 62 entries)",      sizeof(GIFT_RC)    },
    };
    size_t total = 0;
    int ntables = sizeof(tables) / sizeof(tables[0]);
    for (int i = 0; i < ntables; i++) {
        printf("  %-52s : %3zu bytes\n", tables[i].name, tables[i].bytes);
        total += tables[i].bytes;
    }
    printf("  %-50s : %3zu bytes\n", "Total table footprint", total);
    // Report GPU constant memory usage
    printf("\n  Same tables mirrored in __constant__ memory on GPU\n");
    printf("  %-50s : %3zu bytes\n","GPU constant memory (d_GIFT_S + d_GIFT_S_inv + d_GIFT_P + d_GIFT_P_inv + d_GIFT_RC)",total);
}
*/

static void print_performance_metrics()
{
    printf("\n-------- Performance Metrics (CPU) ---------\n");
    printf(" Median of %d trials, buffer = %d blocks\n", TRIALS, BUF_BLOCKS);

    double cpb_ctr = measure_cycles_per_byte_ctr();
    double cpb_cbc = measure_cycles_per_byte_cbc();
    double ks_cyc  = measure_keyschedule_cycles();

    printf("  %-40s : %8.2f  cycles/byte\n", "CTR encrypt (CPU)", cpb_ctr);
    printf("  %-40s : %8.2f  cycles/byte\n", "CBC encrypt (CPU, chained)", cpb_cbc);
    printf("\n  CBC overhead vs CTR                    : %+.2f cycles/byte\n",
           cpb_cbc - cpb_ctr);

    printf("\n--------- Key-Schedule Cost (CPU) ------\n");
    printf("  Rounds             : %d\n", GIFT_ROUNDS);
    printf("  Per-round update   : 1 word rotation (nibble shift) +\n");
    printf("                       2 sub-word rotations (bit-level)\n");
    printf("  Cycles / key setup : %.0f  (gift128_encrypt_block, median of %d trials)\n",
           ks_cyc, TRIALS);
    printf("  Cycles / round     : %.1f\n", ks_cyc / GIFT_ROUNDS);
    printf("  RAM cost           : 0 bytes (keys derived inline)\n");
    printf("  Re-use across blocks: NO (key schedule re-run each block for encryption)\n");

    printf("\n  CPU Decryption schedule: PRE-COMPUTED (%d round keys derived upfront)\n", GIFT_ROUNDS);
    printf("  RAM cost           : %d rounds x 32 nibbles = %d bytes on stack\n",
           GIFT_ROUNDS, GIFT_ROUNDS * 32);

    // Report GPU key schedule cost
    printf("\n--------- Key-Schedule Cost (GPU) ------\n");
    printf("  Round keys pre-computed ONCE on CPU: %d x 32 = %d bytes\n",
           GIFT_ROUNDS, GIFT_ROUNDS * 32);
    printf("  Uploaded to GPU global memory: 1 cudaMemcpy call\n");
    printf("  Each GPU thread reads pre-computed keys: 0 key-schedule work per thread\n");
    printf("  Saving vs per-thread schedule: ~%d nibble ops per thread\n",
           GIFT_ROUNDS * 32);
}


//  Metric helpers

// Wall-clock timer using CUDA events (GPU) or clock_gettime (CPU).
// Returns elapsed seconds.
static double cuda_event_seconds(cudaEvent_t start, cudaEvent_t stop)
{
    float ms = 0.f;
    cudaEventElapsedTime(&ms, start, stop);
    return (double)ms * 1e-3;
}

// Metric: throughput + wall-clock as a function of input size
//
// Tests both CTR and CBC-decrypt kernels across a range of sizes.
// Also reports key-setup host-side timing once.

static void print_gpu_throughput_sweep()
{
    printf("\n\n");
    printf("  GPU Throughput vs Input Size\n");
    printf("\n");

    uint8_t key[KEY_BYTES]  = {0x9e,0xb9,0x36,0x40,0xd0,0x88,0xda,0x63,
                                0x76,0xa3,0x9d,0x1c,0x8b,0xea,0x71,0xe1};
    uint8_t nonce[12]       = {0};
    uint8_t iv[BLOCK_BYTES] = {0};

    // Input sizes to sweep (bytes)
    const size_t sizes[] = {
        1024,                    //   1 KB
        4  * 1024,               //   4 KB
        16 * 1024,               //  16 KB
        64 * 1024,               //  64 KB
        256* 1024,               // 256 KB
        1  * 1024 * 1024,        //   1 MB
        4  * 1024 * 1024,        //   4 MB
        16 * 1024 * 1024,        //  16 MB
        64 * 1024 * 1024,        //  64 MB
    };
    const int NSIZES = sizeof(sizes) / sizeof(sizes[0]);
    const int WARMUP = 3, REPS = 7;

    printf("size_bytes,mode,wall_sec,throughput_GBs\n");

    for (int si = 0; si < NSIZES; si++) {
        size_t N = sizes[si];
        size_t num_blocks = (N + BLOCK_BYTES - 1) / BLOCK_BYTES;
        size_t padded     = num_blocks * BLOCK_BYTES;

        // Allocate pinned host buffers
        uint8_t *h_in, *h_out;
        cudaMallocHost(&h_in,  padded);
        cudaMallocHost(&h_out, padded);
        memset(h_in, 0xA5, padded);

        // Allocate device buffers
        uint8_t *d_key_g, *d_nonce_g, *d_in_g, *d_out_g;
        cudaMalloc(&d_key_g,   KEY_BYTES);
        cudaMalloc(&d_nonce_g, 12);
        cudaMalloc(&d_in_g,    padded);
        cudaMalloc(&d_out_g,   padded);

        cudaMemcpy(d_key_g,   key,   KEY_BYTES, cudaMemcpyHostToDevice);
        cudaMemcpy(d_nonce_g, nonce, 12,        cudaMemcpyHostToDevice);
        cudaMemcpy(d_in_g,    h_in,  padded,    cudaMemcpyHostToDevice);

        int tpb  = 128;
        int grid = (num_blocks + tpb - 1) / tpb;

        cudaEvent_t ev_start, ev_stop;
        cudaEventCreate(&ev_start);
        cudaEventCreate(&ev_stop);

        // CTR kernel 
        // Warm-up
        for (int w = 0; w < WARMUP; w++)
            gift128_ctr_kernel<<<grid, tpb>>>(d_key_g, d_nonce_g,
                                              d_in_g, d_out_g, num_blocks);
        cudaDeviceSynchronize();

        double ctr_total = 0.0;
        for (int r = 0; r < REPS; r++) {
            cudaEventRecord(ev_start);
            gift128_ctr_kernel<<<grid, tpb>>>(d_key_g, d_nonce_g,
                                              d_in_g, d_out_g, num_blocks);
            cudaEventRecord(ev_stop);
            cudaEventSynchronize(ev_stop);
            ctr_total += cuda_event_seconds(ev_start, ev_stop);
        }
        double ctr_sec  = ctr_total / REPS;
        double ctr_gbs  = (double)N / ctr_sec / 1e9;

        // CBC-decrypt kernel
        // Pre-compute round keys (done once per key, not per block)
        uint8_t host_rkeys[GIFT_ROUNDS * 32];
        gift128_precompute_round_keys(key, host_rkeys);

        uint8_t *d_rkeys_g, *d_iv_g;
        cudaMalloc(&d_rkeys_g, GIFT_ROUNDS * 32);
        cudaMalloc(&d_iv_g,    BLOCK_BYTES);
        cudaMemcpy(d_rkeys_g, host_rkeys, GIFT_ROUNDS * 32,
                   cudaMemcpyHostToDevice);
        cudaMemcpy(d_iv_g, iv, BLOCK_BYTES, cudaMemcpyHostToDevice);

        for (int w = 0; w < WARMUP; w++)
            gift128_cbc_decrypt_kernel<<<grid, tpb>>>(d_rkeys_g, d_iv_g,
                                                      d_in_g, d_out_g,
                                                      num_blocks);
        cudaDeviceSynchronize();

        double cbc_total = 0.0;
        for (int r = 0; r < REPS; r++) {
            cudaEventRecord(ev_start);
            gift128_cbc_decrypt_kernel<<<grid, tpb>>>(d_rkeys_g, d_iv_g,
                                                      d_in_g, d_out_g,
                                                      num_blocks);
            cudaEventRecord(ev_stop);
            cudaEventSynchronize(ev_stop);
            cbc_total += cuda_event_seconds(ev_start, ev_stop);
        }
        double cbc_sec  = cbc_total / REPS;
        double cbc_gbs  = (double)N / cbc_sec / 1e9;

        // Print human-readable row
        const char *unit  = (N >= 1024*1024) ? "MB" : "KB";
        double      ndisp = (N >= 1024*1024) ? N/1048576.0 : N/1024.0;
        printf("  %6.0f %-3s  CTR   wall=%8.4f ms  throughput=%7.3f GB/s\n",
               ndisp, unit, ctr_sec*1e3, ctr_gbs);
        printf("  %6.0f %-3s  CBC-D wall=%8.4f ms  throughput=%7.3f GB/s\n",
               ndisp, unit, cbc_sec*1e3, cbc_gbs);

        printf("%zu,CTR,%.6f,%.6f\n",  N, ctr_sec, ctr_gbs);
        printf("%zu,CBC-D,%.6f,%.6f\n", N, cbc_sec, cbc_gbs);

        // Clean up this iteration
        cudaFree(d_key_g); cudaFree(d_nonce_g);
        cudaFree(d_in_g);  cudaFree(d_out_g);
        cudaFree(d_rkeys_g); cudaFree(d_iv_g);
        cudaFreeHost(h_in);  cudaFreeHost(h_out);
        cudaEventDestroy(ev_start); cudaEventDestroy(ev_stop);
    }
}

// Metric: key-setup timing
// Measures the CPU-side gift128_precompute_round_keys() cost
// and a single cudaMemcpy of those keys to the GPU.
// This is the total per-key cost before any GPU decryption can start.
static void print_key_schedule_timing()
{
    printf("  Key-Schedule Timing (Metric 5)\n");

    uint8_t key[KEY_BYTES] = {0x9e,0xb9,0x36,0x40,0xd0,0x88,0xda,0x63,
                               0x76,0xa3,0x9d,0x1c,0x8b,0xea,0x71,0xe1};
    uint8_t rkeys[GIFT_ROUNDS * 32];

    const int KSREPS = 10000;

    // CPU key-schedule cost 
    // Warm-up
    for (int i = 0; i < 1000; i++)
        gift128_precompute_round_keys(key, rkeys);

    uint64_t ks_samples[TRIALS];
    for (int t = 0; t < TRIALS; t++) {
        uint64_t t0 = rdtsc();
        for (int i = 0; i < KSREPS; i++)
            gift128_precompute_round_keys(key, rkeys);
        uint64_t t1 = rdtsc();
        ks_samples[t] = (t1 - t0) / KSREPS;
    }
    qsort(ks_samples, TRIALS, sizeof(uint64_t), cmp_u64);
    printf("  CPU: gift128_precompute_round_keys()  median = %llu cycles\n",
           (unsigned long long)ks_samples[TRIALS/2]);

    // Wall-clock version (nanoseconds)
    struct timespec ts0, ts1;
    clock_gettime(CLOCK_MONOTONIC, &ts0);
    for (int i = 0; i < KSREPS; i++)
        gift128_precompute_round_keys(key, rkeys);
    clock_gettime(CLOCK_MONOTONIC, &ts1);
    double ns_per_ks = ((ts1.tv_sec - ts0.tv_sec)*1e9 +
                        (ts1.tv_nsec - ts0.tv_nsec)) / KSREPS;
    printf("  CPU: gift128_precompute_round_keys()  wall   = %.1f ns/key\n",
           ns_per_ks);

    //  GPU upload cost (cudaMemcpy host to device for the round-key block) 
    uint8_t *d_rkeys;
    cudaMalloc(&d_rkeys, GIFT_ROUNDS * 32);

    // Warm-up
    for (int i = 0; i < 5; i++)
        cudaMemcpy(d_rkeys, rkeys, GIFT_ROUNDS*32, cudaMemcpyHostToDevice);

    cudaEvent_t ev_s, ev_e;
    cudaEventCreate(&ev_s); cudaEventCreate(&ev_e);

    const int GPU_REPS = 1000;
    cudaEventRecord(ev_s);
    for (int i = 0; i < GPU_REPS; i++)
        cudaMemcpy(d_rkeys, rkeys, GIFT_ROUNDS*32, cudaMemcpyHostToDevice);
    cudaEventRecord(ev_e);
    cudaEventSynchronize(ev_e);

    float gpu_ms = 0.f;
    cudaEventElapsedTime(&gpu_ms, ev_s, ev_e);
    printf("  GPU: cudaMemcpy round-keys H→D        wall   = %.2f µs/key\n",
           (double)gpu_ms * 1e3 / GPU_REPS);   

    printf("\n  Strategy: key schedule computed ONCE on CPU (%d bytes),\n",
           GIFT_ROUNDS * 32);
    printf("  then uploaded to GPU global memory via 1 cudaMemcpy.\n");
    printf("  Each GPU thread reads pre-computed keys — 0 schedule work per thread.\n");

    cudaFree(d_rkeys);
    cudaEventDestroy(ev_s); cudaEventDestroy(ev_e);
}

// Metric: key-schedule space 
static void print_key_schedule_space()
{
    printf("  Key-Schedule Space  \n");

    printf("  GIFT-128 key size                    : %d bytes\n", KEY_BYTES);
    printf("  Rounds                               : %d\n",       GIFT_ROUNDS);
    printf("  Nibbles per round-key state          : 32\n");
    printf("  Expanded key schedule (CPU/GPU)      : %d rounds × 32 bytes = %d bytes\n",
           GIFT_ROUNDS, GIFT_ROUNDS * 32);
    printf("  CTR encryption                       : 0 bytes (key schedule re-run inline)\n");
    printf("  CBC decryption (CPU)                 : %d bytes on stack (round_key_state)\n",
           GIFT_ROUNDS * 32);
    printf("  CBC decryption (GPU)                 : %d bytes in GPU global memory\n",
           GIFT_ROUNDS * 32);
}

// Metric: compiled binary / PTX size 
// We can only query the final binary size at runtime via stat().
// PTX must be extracted with  nvcc --ptx  at build time.
static void print_binary_size(const char *argv0)
{
    printf("  Compiled Binary Size \n");

    struct stat st;
    if (argv0 && stat(argv0, &st) == 0) {
        printf("  Executable  : %s\n", argv0);
        printf("  Size on disk: %.1f KB (%ld bytes)\n",
               st.st_size / 1024.0, (long)st.st_size);
    } else {
        printf("  (Pass argv[0] to main() to auto-detect binary path)\n");
    }
    printf("\n  For PTX size, build with:\n");
    printf("    nvcc --ptx -O2 gift128.cu -o gift128.ptx\n");
    printf("    wc -l gift128.ptx       # PTX line count\n");
    printf("    wc -c gift128.ptx       # PTX byte size\n");
    printf("  Then grep for the kernel function names to isolate kernel PTX.\n");
}

// Metric: static memory footprint 
static void print_static_memory()
{
    printf("  Static Memory Footprint (Metric 4)\n");

    size_t s_box    = sizeof(GIFT_S);
    size_t s_box_i  = sizeof(GIFT_S_inv);
    size_t perm     = sizeof(GIFT_P);
    size_t perm_i   = sizeof(GIFT_P_inv);
    size_t rc       = sizeof(GIFT_RC);
    size_t total    = s_box + s_box_i + perm + perm_i + rc;

    printf("  Table                              CPU (ROM)   GPU (__constant__)\n");
    printf("  ─────────────────────────────────────────────────────────────────\n");
    printf("  GIFT_S        (S-box, 16 entries)  %3zu bytes   %3zu bytes\n",  s_box, s_box);
    printf("  GIFT_S_inv    (inv S-box, 16)      %3zu bytes   %3zu bytes\n",  s_box_i, s_box_i);
    printf("  GIFT_P        (bit perm, 128)      %3zu bytes   %3zu bytes\n",  perm, perm);
    printf("  GIFT_P_inv    (inv perm, 128)      %3zu bytes   %3zu bytes\n",  perm_i, perm_i);
    printf("  GIFT_RC       (round consts, 62)   %3zu bytes   %3zu bytes\n",  rc, rc);
    printf("  ─────────────────────────────────────────────────────────────────\n");
    printf("  Total                              %3zu bytes   %3zu bytes\n",  total, total);
    printf("  GIFT-128 total table footprint is %.0f%% of AES-128's tables.\n",
           100.0 * total / 512.0);
    printf("\n  GPU: all tables live in __constant__ memory (64 KB dedicated\n");
    printf("  read-only cache — zero bank-conflict broadcast to all threads).\n");
}

// Main
int main(int argc, char **argv)
{
    // Upload all lookup tables to GPU __constant__ memory once,
    // before any GPU cipher operation is performed.
    upload_constant_tables(); 

    // CPU correctness tests
    run_gift128_core_test();
    run_cbc_cpu_test();
    run_ctr_cpu_test();

    // GPU correctness tests
    run_ctr_gpu_test();         
    run_cbc_gpu_decrypt_test(); 

    printf("\n--------All tests have completed--------\n");

    //print_code_size();
    print_performance_metrics();
    print_static_memory();          // Metric 4
    print_key_schedule_space();     // Metric 6
    print_key_schedule_timing();    // Metric 5
    print_performance_metrics();    // existing CPU cycles/byte
    print_gpu_throughput_sweep();   // Metric 1  (also emits CSV)
    print_binary_size(argv[0]);

    return 0;
}
