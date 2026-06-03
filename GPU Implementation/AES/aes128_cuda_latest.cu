#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CUDA_CHECK(call) do { \
    cudaError_t err__ = (call); \
    if (err__ != cudaSuccess) { \
        std::fprintf(stderr, "CUDA error %s:%d: %s\n", \
                     __FILE__, __LINE__, cudaGetErrorString(err__)); \
        std::exit(1); \
    } \
} while (0)

namespace aes128 {

constexpr int NR = 10;
constexpr int KEY_SIZE = 16;
constexpr int EXPANDED_KEY_SIZE = 176;

static const uint8_t SBOX[256] = {
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

static const uint8_t INV_SBOX[256] = {
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

static const uint8_t RCON[11] = {
    0x00,0x01,0x02,0x04,0x08,0x10,0x20,0x40,0x80,0x1b,0x36
};

static void KeyExpansion(const uint8_t key[KEY_SIZE],
                         uint8_t round_keys[EXPANDED_KEY_SIZE]) {
    std::memcpy(round_keys, key, KEY_SIZE);
    uint8_t t[4];
    for (int i = 4; i < 44; ++i) {
        t[0] = round_keys[(i - 1) * 4 + 0];
        t[1] = round_keys[(i - 1) * 4 + 1];
        t[2] = round_keys[(i - 1) * 4 + 2];
        t[3] = round_keys[(i - 1) * 4 + 3];
        if (i % 4 == 0) {
            const uint8_t u = t[0];
            t[0] = SBOX[t[1]] ^ RCON[i / 4];
            t[1] = SBOX[t[2]];
            t[2] = SBOX[t[3]];
            t[3] = SBOX[u];
        }
        for (int j = 0; j < 4; ++j) {
            round_keys[i * 4 + j] = round_keys[(i - 4) * 4 + j] ^ t[j];
        }
    }
}

} // namespace aes128

__constant__ uint8_t D_SBOX[256];
__constant__ uint8_t D_INV_SBOX[256];
__constant__ uint8_t D_ROUND_KEYS[aes128::EXPANDED_KEY_SIZE];

__device__ uint8_t xtime_device(uint8_t x) {
    return static_cast<uint8_t>((x << 1) ^ ((x >> 7) * 0x1b));
}

__device__ void add_round_key(uint8_t state[16], int round) {
    for (int i = 0; i < 16; ++i) {
        state[i] ^= D_ROUND_KEYS[round * 16 + i];
    }
}

__device__ void sub_bytes(uint8_t state[16]) {
    for (int i = 0; i < 16; ++i) {
        state[i] = D_SBOX[state[i]];
    }
}

__device__ void inv_sub_bytes(uint8_t state[16]) {
    for (int i = 0; i < 16; ++i) {
        state[i] = D_INV_SBOX[state[i]];
    }
}

__device__ void shift_rows(uint8_t s[16]) {
    uint8_t t;
    t = s[1];  s[1] = s[5];  s[5] = s[9];  s[9] = s[13]; s[13] = t;
    t = s[2];  s[2] = s[10]; s[10] = t;
    t = s[6];  s[6] = s[14]; s[14] = t;
    t = s[3];  s[3] = s[15]; s[15] = s[11]; s[11] = s[7]; s[7] = t;
}

__device__ void inv_shift_rows(uint8_t s[16]) {
    uint8_t t;
    t = s[13]; s[13] = s[9];  s[9] = s[5];  s[5] = s[1]; s[1] = t;
    t = s[2];  s[2] = s[10]; s[10] = t;
    t = s[6];  s[6] = s[14]; s[14] = t;
    t = s[3];  s[3] = s[7];  s[7] = s[11]; s[11] = s[15]; s[15] = t;
}

__device__ uint8_t gmul_device(uint8_t a, uint8_t b) {
    uint8_t r = 0;
    while (b) {
        if (b & 1) r ^= a;
        a = xtime_device(a);
        b >>= 1;
    }
    return r;
}

__device__ void mix_columns(uint8_t s[16]) {
    for (int c = 0; c < 4; ++c) {
        const int i = c * 4;
        uint8_t a0 = s[i], a1 = s[i + 1], a2 = s[i + 2], a3 = s[i + 3];
        uint8_t t = a0 ^ a1 ^ a2 ^ a3;
        s[i]     ^= t ^ xtime_device(a0 ^ a1);
        s[i + 1] ^= t ^ xtime_device(a1 ^ a2);
        s[i + 2] ^= t ^ xtime_device(a2 ^ a3);
        s[i + 3] ^= t ^ xtime_device(a3 ^ a0);
    }
}

__device__ void inv_mix_columns(uint8_t s[16]) {
    for (int c = 0; c < 4; ++c) {
        const int i = c * 4;
        uint8_t a0 = s[i], a1 = s[i + 1], a2 = s[i + 2], a3 = s[i + 3];
        s[i]     = gmul_device(a0,0x0e) ^ gmul_device(a1,0x0b) ^ gmul_device(a2,0x0d) ^ gmul_device(a3,0x09);
        s[i + 1] = gmul_device(a0,0x09) ^ gmul_device(a1,0x0e) ^ gmul_device(a2,0x0b) ^ gmul_device(a3,0x0d);
        s[i + 2] = gmul_device(a0,0x0d) ^ gmul_device(a1,0x09) ^ gmul_device(a2,0x0e) ^ gmul_device(a3,0x0b);
        s[i + 3] = gmul_device(a0,0x0b) ^ gmul_device(a1,0x0d) ^ gmul_device(a2,0x09) ^ gmul_device(a3,0x0e);
    }
}

__device__ void aes_encrypt_block_device(const uint8_t in[16], uint8_t out[16]) {
    uint8_t s[16];
    for (int i = 0; i < 16; ++i) s[i] = in[i];
    add_round_key(s, 0);
    for (int round = 1; round < aes128::NR; ++round) {
        sub_bytes(s);
        shift_rows(s);
        mix_columns(s);
        add_round_key(s, round);
    }
    sub_bytes(s);
    shift_rows(s);
    add_round_key(s, aes128::NR);
    for (int i = 0; i < 16; ++i) out[i] = s[i];
}

__device__ void aes_decrypt_block_device(const uint8_t in[16], uint8_t out[16]) {
    uint8_t s[16];
    for (int i = 0; i < 16; ++i) s[i] = in[i];
    add_round_key(s, aes128::NR);
    for (int round = aes128::NR - 1; round >= 1; --round) {
        inv_shift_rows(s);
        inv_sub_bytes(s);
        add_round_key(s, round);
        inv_mix_columns(s);
    }
    inv_shift_rows(s);
    inv_sub_bytes(s);
    add_round_key(s, 0);
    for (int i = 0; i < 16; ++i) out[i] = s[i];
}

__device__ void add_block_index_to_counter(const uint8_t nonce_counter[16],
                                           unsigned long long block_index,
                                           uint8_t out[16]) {
    for (int i = 0; i < 16; ++i) out[i] = nonce_counter[i];
    for (int i = 15; i >= 0 && block_index != 0; --i) {
        unsigned int sum = static_cast<unsigned int>(out[i]) +
                           static_cast<unsigned int>(block_index & 0xffULL);
        out[i] = static_cast<uint8_t>(sum & 0xffU);
        block_index = (block_index >> 8) + (sum >> 8);
    }
}

__global__ void kernel_ctr_crypt(const uint8_t* plaintext,
                                 uint8_t* ciphertext,
                                 size_t num_bytes,
                                 const uint8_t* nonce_counter) {
    size_t block_id = blockIdx.x * blockDim.x + threadIdx.x;
    size_t off = block_id * 16;
    if (off >= num_bytes) return;

    uint8_t counter[16];
    uint8_t stream[16];
    add_block_index_to_counter(nonce_counter, block_id, counter);
    aes_encrypt_block_device(counter, stream);

    size_t take = (num_bytes - off < 16) ? (num_bytes - off) : 16;
    for (size_t i = 0; i < take; ++i) {
        ciphertext[off + i] = plaintext[off + i] ^ stream[i];
    }
}

__global__ void kernel_cbc_encrypt(const uint8_t* plaintext,
                                   uint8_t* ciphertext,
                                   size_t num_bytes,
                                   const uint8_t* iv) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    uint8_t prev[16];
    uint8_t block[16];
    for (int i = 0; i < 16; ++i) prev[i] = iv[i];
    for (size_t off = 0; off < num_bytes; off += 16) {
        for (int i = 0; i < 16; ++i) block[i] = plaintext[off + i] ^ prev[i];
        aes_encrypt_block_device(block, &ciphertext[off]);
        for (int i = 0; i < 16; ++i) prev[i] = ciphertext[off + i];
    }
}

__global__ void kernel_cbc_decrypt(const uint8_t* ciphertext,
                                   uint8_t* plaintext,
                                   size_t num_bytes,
                                   const uint8_t* iv) {
    size_t block_id = blockIdx.x * blockDim.x + threadIdx.x;
    size_t off = block_id * 16;
    if (off >= num_bytes) return;
    uint8_t tmp[16];
    aes_decrypt_block_device(&ciphertext[off], tmp);
    const uint8_t* prev = (block_id == 0) ? iv : &ciphertext[off - 16];
    for (int i = 0; i < 16; ++i) plaintext[off + i] = tmp[i] ^ prev[i];
}

static uint8_t xtime_host(uint8_t x) {
    return static_cast<uint8_t>((x << 1) ^ ((x >> 7) * 0x1b));
}

static void shift_rows_host(uint8_t s[16]) {
    uint8_t t;
    t = s[1];  s[1] = s[5];  s[5] = s[9];  s[9] = s[13]; s[13] = t;
    t = s[2];  s[2] = s[10]; s[10] = t;
    t = s[6];  s[6] = s[14]; s[14] = t;
    t = s[3];  s[3] = s[15]; s[15] = s[11]; s[11] = s[7]; s[7] = t;
}

static void mix_columns_host(uint8_t s[16]) {
    for (int c = 0; c < 4; ++c) {
        int i = c * 4;
        uint8_t a0 = s[i], a1 = s[i + 1], a2 = s[i + 2], a3 = s[i + 3];
        uint8_t t = a0 ^ a1 ^ a2 ^ a3;
        s[i]     ^= t ^ xtime_host(a0 ^ a1);
        s[i + 1] ^= t ^ xtime_host(a1 ^ a2);
        s[i + 2] ^= t ^ xtime_host(a2 ^ a3);
        s[i + 3] ^= t ^ xtime_host(a3 ^ a0);
    }
}

static void aes_encrypt_block_host(const uint8_t in[16],
                                   uint8_t out[16],
                                   const uint8_t round_keys[aes128::EXPANDED_KEY_SIZE]) {
    uint8_t s[16];
    std::memcpy(s, in, 16);
    for (int i = 0; i < 16; ++i) s[i] ^= round_keys[i];
    for (int round = 1; round < aes128::NR; ++round) {
        for (int i = 0; i < 16; ++i) s[i] = aes128::SBOX[s[i]];
        shift_rows_host(s);
        mix_columns_host(s);
        for (int i = 0; i < 16; ++i) s[i] ^= round_keys[round * 16 + i];
    }
    for (int i = 0; i < 16; ++i) s[i] = aes128::SBOX[s[i]];
    shift_rows_host(s);
    for (int i = 0; i < 16; ++i) out[i] = s[i] ^ round_keys[aes128::NR * 16 + i];
}

static void increment_counter_host(uint8_t counter[16]) {
    for (int i = 15; i >= 0; --i) {
        counter[i] = static_cast<uint8_t>(counter[i] + 1);
        if (counter[i] != 0) break;
    }
}

static void ctr_crypt_host(const uint8_t* in,
                           uint8_t* out,
                           size_t num_bytes,
                           const uint8_t round_keys[aes128::EXPANDED_KEY_SIZE],
                           const uint8_t nonce_counter[16]) {
    uint8_t counter[16];
    uint8_t stream[16];
    std::memcpy(counter, nonce_counter, 16);
    for (size_t off = 0; off < num_bytes; off += 16) {
        aes_encrypt_block_host(counter, stream, round_keys);
        size_t take = (num_bytes - off < 16) ? (num_bytes - off) : 16;
        for (size_t i = 0; i < take; ++i) {
            out[off + i] = in[off + i] ^ stream[i];
        }
        increment_counter_host(counter);
    }
}

struct GpuTiming {
    double total_ms;
    float kernel_ms;
};

enum class GpuMode {
    CTR,
    CBC_ENCRYPT,
    CBC_DECRYPT
};

static void upload_aes_tables(const uint8_t round_keys[aes128::EXPANDED_KEY_SIZE]) {
    CUDA_CHECK(cudaMemcpyToSymbol(D_SBOX, aes128::SBOX, sizeof(aes128::SBOX)));
    CUDA_CHECK(cudaMemcpyToSymbol(D_INV_SBOX, aes128::INV_SBOX, sizeof(aes128::INV_SBOX)));
    CUDA_CHECK(cudaMemcpyToSymbol(D_ROUND_KEYS, round_keys, aes128::EXPANDED_KEY_SIZE));
}

static void launch_mode(GpuMode mode, uint8_t* d_in, uint8_t* d_out,
                        size_t num_bytes, uint8_t* d_iv_or_nonce) {
    size_t aes_blocks = (num_bytes + 15) / 16;
    int threads = 256;
    int blocks = static_cast<int>((aes_blocks + threads - 1) / threads);
    if (mode == GpuMode::CTR) {
        kernel_ctr_crypt<<<blocks, threads>>>(d_in, d_out, num_bytes, d_iv_or_nonce);
    } else if (mode == GpuMode::CBC_ENCRYPT) {
        kernel_cbc_encrypt<<<1, 1>>>(d_in, d_out, num_bytes, d_iv_or_nonce);
    } else {
        kernel_cbc_decrypt<<<blocks, threads>>>(d_in, d_out, num_bytes, d_iv_or_nonce);
    }
}

static GpuTiming run_gpu_mode(GpuMode mode,
                              const std::vector<uint8_t>& input,
                              std::vector<uint8_t>& output,
                              const uint8_t round_keys[aes128::EXPANDED_KEY_SIZE],
                              const uint8_t iv_or_nonce[16]) {
    uint8_t *d_in = nullptr;
    uint8_t *d_out = nullptr;
    uint8_t *d_iv_or_nonce = nullptr;
    output.resize(input.size());

    upload_aes_tables(round_keys);
    CUDA_CHECK(cudaMalloc(&d_in, input.size()));
    CUDA_CHECK(cudaMalloc(&d_out, input.size()));
    CUDA_CHECK(cudaMalloc(&d_iv_or_nonce, 16));

    auto total_start = std::chrono::high_resolution_clock::now();
    CUDA_CHECK(cudaMemcpy(d_in, input.data(), input.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_iv_or_nonce, iv_or_nonce, 16, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    launch_mode(mode, d_in, d_out, input.size(), d_iv_or_nonce);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float kernel_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, start, stop));
    CUDA_CHECK(cudaMemcpy(output.data(), d_out, input.size(), cudaMemcpyDeviceToHost));
    auto total_end = std::chrono::high_resolution_clock::now();

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_iv_or_nonce));

    double total_ms =
        std::chrono::duration<double, std::milli>(total_end - total_start).count();
    return {total_ms, kernel_ms};
}

static GpuTiming ctr_crypt_gpu(const std::vector<uint8_t>& input,
                               std::vector<uint8_t>& output,
                               const uint8_t round_keys[aes128::EXPANDED_KEY_SIZE],
                               const uint8_t nonce_counter[16]) {
    return run_gpu_mode(GpuMode::CTR, input, output, round_keys, nonce_counter);
}

static double cpu_ctr_time_ms(const std::vector<uint8_t>& input,
                              std::vector<uint8_t>& output,
                              const uint8_t round_keys[aes128::EXPANDED_KEY_SIZE],
                              const uint8_t nonce_counter[16]) {
    output.resize(input.size());
    auto start = std::chrono::high_resolution_clock::now();
    ctr_crypt_host(input.data(), output.data(), input.size(), round_keys, nonce_counter);
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

static double mb_per_second(size_t bytes, double ms) {
    double mb = static_cast<double>(bytes) / (1024.0 * 1024.0);
    return mb / (ms / 1000.0);
}

static void print_code_size() {
    std::printf("\n--------Code Size--------\n");

    std::printf("\n  (a) Static memory footprint (GPU constant memory)\n");
    std::printf("    SBOX          (Sbox,          256 entries x 1 B)       : %3zu bytes\n", sizeof(aes128::SBOX));
    std::printf("    INV_SBOX      (Sbox inverse,  256 entries x 1 B)       : %3zu bytes\n", sizeof(aes128::INV_SBOX));
    std::printf("    RCON          (round consts,   11 entries x 1 B)       : %3zu bytes\n", sizeof(aes128::RCON));
    std::printf("    ROUND_KEYS    (expanded key,  176 entries x 1 B)       : %3d bytes\n", aes128::EXPANDED_KEY_SIZE);
    std::printf("    TOTAL (with precomputed round keys)                    : %3zu bytes\n",
                sizeof(aes128::SBOX) + sizeof(aes128::INV_SBOX) + sizeof(aes128::RCON) +
                static_cast<size_t>(aes128::EXPANDED_KEY_SIZE));
    std::printf("    TOTAL (without round keys, on-the-fly)                 : %3zu bytes\n",
                sizeof(aes128::SBOX) + sizeof(aes128::INV_SBOX) + sizeof(aes128::RCON));

    std::printf("\n  (b) Device/kernel lines of code (approx, excl. blank lines & comments)\n");
    std::printf("    aes_encrypt_block_device :  15 LoC\n");
    std::printf("    aes_decrypt_block_device :  15 LoC\n");
    std::printf("    kernel_ctr_crypt         :  12 LoC\n");
    std::printf("    kernel_cbc_encrypt       :  12 LoC\n");
    std::printf("    kernel_cbc_decrypt       :   8 LoC\n");
    std::printf("                              ----\n");
    std::printf("    Total device code        :  64 LoC\n");

    std::printf("\n  (c) PTX and compiled binary size\n");
    std::printf("    Run these commands after compiling to inspect sizes:\n\n");
    std::printf("    # PTX (human-readable GPU assembly):\n");
    std::printf("    nvcc -O3 -arch=sm_75 -ptx -o aes128_cuda.ptx aes128_cuda_updated.cu\n");
    std::printf("    wc -l aes128_cuda.ptx\n");
    std::printf("\n    # CUBIN (compiled binary for your GPU):\n");
    std::printf("    nvcc -O3 -arch=sm_75 -cubin -o aes128_cuda.cubin aes128_cuda_updated.cu\n");
    std::printf("    ls -lh aes128_cuda.cubin\n");
    std::printf("\n    # Per-kernel register and shared memory usage:\n");
    std::printf("    nvcc -O3 -arch=sm_75 --ptxas-options=-v aes128_cuda_updated.cu\n");
}

static void measure_key_schedule_cost(const uint8_t key[16]) {
    constexpr int TRIALS = 101;
    uint8_t round_keys[aes128::EXPANDED_KEY_SIZE];
    std::vector<double> cpu_samples;
    std::vector<float> upload_samples;
    cpu_samples.reserve(TRIALS);
    upload_samples.reserve(TRIALS);

    for (int i = 0; i < TRIALS; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        aes128::KeyExpansion(key, round_keys);
        auto end = std::chrono::high_resolution_clock::now();
        cpu_samples.push_back(std::chrono::duration<double, std::micro>(end - start).count());
    }

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    for (int i = 0; i < TRIALS; ++i) {
        CUDA_CHECK(cudaEventRecord(start));
        CUDA_CHECK(cudaMemcpyToSymbol(D_ROUND_KEYS, round_keys, aes128::EXPANDED_KEY_SIZE));
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        upload_samples.push_back(ms * 1000.0f);
    }
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    std::sort(cpu_samples.begin(), cpu_samples.end());
    std::sort(upload_samples.begin(), upload_samples.end());
    double cpu_us = cpu_samples[TRIALS / 2];
    double upload_us = upload_samples[TRIALS / 2];

    std::printf("\n--------Key Schedule Cost--------\n");
    std::printf("\n  Time (median of %d trials)\n", TRIALS);
    std::printf("  %-48s : %7.2f us\n", "CPU compute  (KeyExpansion)", cpu_us);
    std::printf("  %-48s : %7.2f us\n", "GPU upload   (cudaMemcpyToSymbol, PCIe)", upload_us);
    std::printf("  %-48s : %7.2f us\n", "Total key setup cost", cpu_us + upload_us);
    std::printf("\n  Space\n");
    std::printf("  %-48s : %3d bytes  (11 round keys x 16 bytes)\n",
                "Expanded key in GPU constant memory", aes128::EXPANDED_KEY_SIZE);
}

static float benchmark_kernel_ms(GpuMode mode,
                                 const uint8_t round_keys[aes128::EXPANDED_KEY_SIZE],
                                 const uint8_t iv_or_nonce[16]) {
    constexpr int BUF_BLOCKS = 256;
    constexpr int TRIALS = 31;
    const size_t sz = BUF_BLOCKS * 16;
    std::vector<uint8_t> input(sz), temp, output;
    for (size_t i = 0; i < sz; ++i) input[i] = static_cast<uint8_t>(i & 0xffu);
    if (mode == GpuMode::CBC_DECRYPT) {
        run_gpu_mode(GpuMode::CBC_ENCRYPT, input, temp, round_keys, iv_or_nonce);
        input = temp;
    }

    std::vector<float> samples;
    samples.reserve(TRIALS);
    for (int t = 0; t < TRIALS; ++t) {
        GpuTiming timing = run_gpu_mode(mode, input, output, round_keys, iv_or_nonce);
        samples.push_back(timing.kernel_ms);
    }
    std::sort(samples.begin(), samples.end());
    return samples[TRIALS / 2];
}

static void measure_single_64_byte_latency(
    const uint8_t round_keys[aes128::EXPANDED_KEY_SIZE],
    const uint8_t nonce_counter[16]) {
    const size_t n = 64;
    const int trials = 101;
    std::vector<uint8_t> input(n), output, cbc_cipher;
    for (size_t i = 0; i < n; ++i) input[i] = static_cast<uint8_t>(i & 0xffu);
    run_gpu_mode(GpuMode::CBC_ENCRYPT, input, cbc_cipher, round_keys, nonce_counter);

    std::vector<double> ctr_samples, cbc_samples;
    ctr_samples.reserve(trials);
    cbc_samples.reserve(trials);
    for (int t = 0; t < trials; ++t) {
        GpuTiming ctr = run_gpu_mode(GpuMode::CTR, input, output, round_keys, nonce_counter);
        GpuTiming cbc = run_gpu_mode(GpuMode::CBC_DECRYPT, cbc_cipher, output, round_keys, nonce_counter);
        ctr_samples.push_back(ctr.total_ms * 1000.0);
        cbc_samples.push_back(cbc.total_ms * 1000.0);
    }
    std::sort(ctr_samples.begin(), ctr_samples.end());
    std::sort(cbc_samples.begin(), cbc_samples.end());

    std::printf("\n--------Single 64-Byte Block Latency--------\n");
    std::printf("  Input size : 4 cipher blocks = 64 bytes\n");
    std::printf("  Trials     : %d  \n\n", trials);
    std::printf("  %-35s : %8.2f us\n", "CTR encrypt latency", ctr_samples[trials / 2]);
    std::printf("  %-35s : %8.2f us\n", "CBC decrypt latency", cbc_samples[trials / 2]);
}

int main() {
    std::printf("--------AES-128 CUDA Implementation--------\n");
    std::printf("--------CTR and CBC Mode Tests--------\n\n");

    const uint8_t key[16] = {
        0x2b,0x7e,0x15,0x16,0x28,0xae,0xd2,0xa6,
        0xab,0xf7,0x15,0x88,0x09,0xcf,0x4f,0x3c
    };
    const uint8_t nonce_counter[16] = {
        0x0f,0x0e,0x0d,0x0c,0x0b,0x0a,0x09,0x08,
        0x07,0x06,0x05,0x04,0x03,0x02,0x01,0x00
    };

    uint8_t round_keys[aes128::EXPANDED_KEY_SIZE];
    std::printf("Generating key schedule\n");
    aes128::KeyExpansion(key, round_keys);
    upload_aes_tables(round_keys);
    std::printf("Key schedule generated and uploaded to GPU\n\n");

    std::vector<uint8_t> plain(16), cpu_cipher, gpu_cipher, recovered;
    const uint8_t test_plain[16] = {
        0x00,0x11,0x22,0x33,0x44,0x55,0x66,0x77,
        0x88,0x99,0xaa,0xbb,0xcc,0xdd,0xee,0xff
    };
    std::memcpy(plain.data(), test_plain, 16);

    std::printf("--------CTR Mode (GPU)--------\n\n");
    std::printf("Original Plaintext   : ");
    for (uint8_t b : plain) std::printf("%02x", b);
    std::printf("\n");

    cpu_cipher.resize(plain.size());
    ctr_crypt_host(plain.data(), cpu_cipher.data(), plain.size(), round_keys, nonce_counter);
    ctr_crypt_gpu(plain, gpu_cipher, round_keys, nonce_counter);
    ctr_crypt_gpu(gpu_cipher, recovered, round_keys, nonce_counter);

    std::printf("CTR Ciphertext       : ");
    for (uint8_t b : gpu_cipher) std::printf("%02x", b);
    std::printf("\n");
    std::printf("Recovered Plaintext  : ");
    for (uint8_t b : recovered) std::printf("%02x", b);
    std::printf("\n\n");

    bool correctness_ok = (gpu_cipher == cpu_cipher) && (recovered == plain);
    if (correctness_ok) {
        std::printf("The recovered plaintext matches the original plaintext CTR mode\n");
    } else {
        std::printf("MISMATCH CTR mode\n");
    }

    std::vector<uint8_t> cbc_cipher, cbc_recovered;
    uint8_t iv[16] = {0};
    std::printf("\n--------CBC Mode (GPU)--------\n\n");
    std::printf("Original Plaintext   : ");
    for (uint8_t b : plain) std::printf("%02x", b);
    std::printf("\n");
    run_gpu_mode(GpuMode::CBC_ENCRYPT, plain, cbc_cipher, round_keys, iv);
    run_gpu_mode(GpuMode::CBC_DECRYPT, cbc_cipher, cbc_recovered, round_keys, iv);
    std::printf("CBC Ciphertext       : ");
    for (uint8_t b : cbc_cipher) std::printf("%02x", b);
    std::printf("\n");
    std::printf("Recovered Plaintext  : ");
    for (uint8_t b : cbc_recovered) std::printf("%02x", b);
    std::printf("\n\n");
    bool cbc_ok = (cbc_recovered == plain);
    if (cbc_ok) {
        std::printf("The recovered plaintext matches the original plaintext CBC mode\n");
    } else {
        std::printf("MISMATCH CBC mode\n");
    }
    correctness_ok = correctness_ok && cbc_ok;

    if (correctness_ok) {
        std::printf("\n--------CBC and CTR modes are validated--------\n");
    } else {
        std::printf("\n--------CBC and CTR validation failed--------\n");
    }

    constexpr int BUF_BLOCKS = 256;
    constexpr int TRIALS = 31;
    float ctr_ms = benchmark_kernel_ms(GpuMode::CTR, round_keys, nonce_counter);
    float cbc_ms = benchmark_kernel_ms(GpuMode::CBC_DECRYPT, round_keys, iv);
    double data_bytes = BUF_BLOCKS * 16.0;
    std::printf("\n--------GPU Throughput Benchmark--------\n");
    std::printf("  Median of %d trials, Buffer = %d blocks (%d bytes)\n",
                TRIALS, BUF_BLOCKS, BUF_BLOCKS * 16);
    std::printf("  %-40s : %8.4f ms  (~%.6f GB/s)\n", "CTR encrypt",
                ctr_ms, (data_bytes / 1e9) / (ctr_ms / 1e3));
    std::printf("  %-40s : %8.4f ms  (~%.6f GB/s)\n", "CBC decrypt",
                cbc_ms, (data_bytes / 1e9) / (cbc_ms / 1e3));

    measure_single_64_byte_latency(round_keys, nonce_counter);
    measure_key_schedule_cost(key);

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::printf("\n--------GPU Device--------\n");
    std::printf("  Device        : %s\n", prop.name);
    std::printf("  SMs           : %d\n", prop.multiProcessorCount);
    std::printf("  Clock         : %.0f MHz\n", prop.clockRate / 1e3);
    std::printf("  Global memory : %.0f MB\n", prop.totalGlobalMem / 1.0e6);

    return correctness_ok ? 0 : 1;
}
