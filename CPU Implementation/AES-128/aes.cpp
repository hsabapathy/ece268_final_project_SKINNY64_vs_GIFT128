//AES-128 CPU implmentation

#include <cstdint>
#include <cstring>
#include <cstdio>
#include <time.h>
#include <x86intrin.h>

#define TRIALS     31
#define BUF_BLOCKS 256

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

static void SubBytes(uint8_t s[16])    { for (int i = 0; i < 16; ++i) s[i] = sbox[s[i]]; }
static void InvSubBytes(uint8_t s[16]) { for (int i = 0; i < 16; ++i) s[i] = inv_sbox[s[i]]; }

static void ShiftRows(uint8_t s[16]) {
    uint8_t t;
    t = s[1];  s[1]  = s[5];  s[5]  = s[9];  s[9]  = s[13]; s[13] = t;
    t = s[2];  s[2]  = s[10]; s[10] = t;
    t = s[6];  s[6]  = s[14]; s[14] = t;
    t = s[3];  s[3]  = s[15]; s[15] = s[11]; s[11] = s[7];  s[7]  = t;
}
static void InvShiftRows(uint8_t s[16]) {
    uint8_t t;
    t = s[13]; s[13] = s[9];  s[9]  = s[5];  s[5]  = s[1];  s[1]  = t;
    t = s[2];  s[2]  = s[10]; s[10] = t;
    t = s[6];  s[6]  = s[14]; s[14] = t;
    t = s[3];  s[3]  = s[7];  s[7]  = s[11]; s[11] = s[15]; s[15] = t;
}

static void MixColumns(uint8_t s[16]) {
    for (int c = 0; c < 4; ++c) {
        uint8_t a0=s[c*4+0], a1=s[c*4+1], a2=s[c*4+2], a3=s[c*4+3];
        uint8_t T = a0^a1^a2^a3;
        s[c*4+0] ^= T ^ xtime(a0^a1);
        s[c*4+1] ^= T ^ xtime(a1^a2);
        s[c*4+2] ^= T ^ xtime(a2^a3);
        s[c*4+3] ^= T ^ xtime(a3^a0);
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

static void AddRoundKey(uint8_t s[16], const uint8_t* rk) {
    for (int i = 0; i < 16; ++i) s[i] ^= rk[i];
}

void EncryptBlock(const uint8_t in[16], uint8_t out[16],
                  const uint8_t roundKeys[EXPANDED_KEY_SIZE]) {
    uint8_t s[16]; std::memcpy(s, in, 16);
    AddRoundKey(s, roundKeys);
    for (int r = 1; r < Nr; ++r) {
        SubBytes(s); ShiftRows(s); MixColumns(s);
        AddRoundKey(s, roundKeys + r*16);
    }
    SubBytes(s); ShiftRows(s);
    AddRoundKey(s, roundKeys + Nr*16);
    std::memcpy(out, s, 16);
}

void DecryptBlock(const uint8_t in[16], uint8_t out[16],
                  const uint8_t roundKeys[EXPANDED_KEY_SIZE]) {
    uint8_t s[16]; std::memcpy(s, in, 16);
    AddRoundKey(s, roundKeys + Nr*16);
    InvShiftRows(s); InvSubBytes(s);
    for (int r = Nr-1; r >= 1; --r) {
        AddRoundKey(s, roundKeys + r*16);
        InvMixColumns(s); InvShiftRows(s); InvSubBytes(s);
    }
    AddRoundKey(s, roundKeys);
    std::memcpy(out, s, 16);
}

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
    std::printf("The recovered plaintext doesnt match the original plaintext %s\n  got:      ", label);
    for (int i = 0; i < len; i++) std::printf("%02x", got[i]);
    std::printf("\n  expected: ");
    for (int i = 0; i < len; i++) std::printf("%02x", expected[i]);
    std::printf("\n");
    return 0;
}


static inline uint64_t rdtsc() {
    uint32_t lo, hi;
    __asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
    return ((uint64_t)hi << 32) | lo;
}

static double cpu_ghz() {
    uint64_t t0 = rdtsc();
    struct timespec ts = {0, 100000000};
    nanosleep(&ts, NULL);
    uint64_t t1 = rdtsc();
    return (double)(t1 - t0) / 1e8;
}

static int cmp_u64(const void *a, const void *b) {
    uint64_t x = *(const uint64_t*)a, y = *(const uint64_t*)b;
    return (x > y) - (x < y);
}


static void ctr_encrypt(const uint8_t *plaintext, uint8_t *ciphertext, int num_blocks,
                        const uint8_t nonce[16],
                        const uint8_t roundKeys[aes128::EXPANDED_KEY_SIZE]) {
    uint8_t counter[16], keystream[16];
    std::memcpy(counter, nonce, 16);
    for (int b = 0; b < num_blocks; b++) {
        aes128::EncryptBlock(counter, keystream, roundKeys);
        for (int i = 0; i < 16; i++) ciphertext[b*16+i] = plaintext[b*16+i] ^ keystream[i];
        for (int i = 15; i >= 0; i--) { counter[i]++; if (counter[i]) break; }
    }
}

static void ctr_decrypt(const uint8_t *ciphertext, uint8_t *plaintext, int num_blocks,
                        const uint8_t nonce[16],
                        const uint8_t roundKeys[aes128::EXPANDED_KEY_SIZE]) {
    ctr_encrypt(ciphertext, plaintext, num_blocks, nonce, roundKeys);
}

static void cbc_encrypt(const uint8_t *plaintext, uint8_t *ciphertext, int num_blocks,
                        const uint8_t iv[16],
                        const uint8_t roundKeys[aes128::EXPANDED_KEY_SIZE]) {
    uint8_t prev[16], block[16];
    std::memcpy(prev, iv, 16);
    for (int b = 0; b < num_blocks; b++) {
        for (int i = 0; i < 16; i++) block[i] = plaintext[b*16+i] ^ prev[i];
        aes128::EncryptBlock(block, &ciphertext[b*16], roundKeys);
        std::memcpy(prev, &ciphertext[b*16], 16);
    }
}

static void cbc_decrypt(const uint8_t *ciphertext, uint8_t *plaintext, int num_blocks,
                        const uint8_t iv[16],
                        const uint8_t roundKeys[aes128::EXPANDED_KEY_SIZE]) {
    uint8_t prev[16], tmp[16], saved[16];
    std::memcpy(prev, iv, 16);
    for (int b = 0; b < num_blocks; b++) {
        std::memcpy(saved, &ciphertext[b*16], 16);
        aes128::DecryptBlock(&ciphertext[b*16], tmp, roundKeys);
        for (int i = 0; i < 16; i++) plaintext[b*16+i] = tmp[i] ^ prev[i];
        std::memcpy(prev, saved, 16);
    }
}

static void ecb_encrypt(const uint8_t *plaintext, uint8_t *ciphertext, int num_blocks,
                        const uint8_t roundKeys[aes128::EXPANDED_KEY_SIZE]) {
    for (int b = 0; b < num_blocks; b++)
        aes128::EncryptBlock(&plaintext[b*16], &ciphertext[b*16], roundKeys);
}

static void ecb_decrypt(const uint8_t *ciphertext, uint8_t *plaintext, int num_blocks,
                        const uint8_t roundKeys[aes128::EXPANDED_KEY_SIZE]) {
    for (int b = 0; b < num_blocks; b++)
        aes128::DecryptBlock(&ciphertext[b*16], &plaintext[b*16], roundKeys);
}

static void print_code_size() {
    std::printf("\n--------Code Size--------\n");
    struct TableSize { const char *name; size_t bytes; };
    TableSize tables[] = {
        { "sbox         (AES Sbox, 256 entries)",         sizeof(aes128::sbox)     },
        { "inv_sbox     (AES inverse Sbox, 256 entries)", sizeof(aes128::inv_sbox) },
        { "Rcon         (round constants, 10 entries)",   sizeof(aes128::Rcon)     },
    };
    size_t total = 0;
    int n = sizeof(tables) / sizeof(tables[0]);
    for (int i = 0; i < n; i++) {
        std::printf("  %-52s : %3zu bytes\n", tables[i].name, tables[i].bytes);
        total += tables[i].bytes;
    }
    std::printf("  %-52s : %3zu bytes\n", "Total table footprint", total);
}

static double measure_cycles_per_byte_ctr(const uint8_t roundKeys[aes128::EXPANDED_KEY_SIZE]) {
    static uint8_t buf[BUF_BLOCKS * 16];
    static uint8_t out[BUF_BLOCKS * 16];
    uint8_t nonce[16] = {0};
    for (int i = 0; i < BUF_BLOCKS * 16; i++) buf[i] = (uint8_t)(i & 0xff);

    uint64_t samples[TRIALS];
    for (int t = 0; t < TRIALS; t++) {
        uint8_t n[16] = {0};
        uint64_t t0 = rdtsc();
        ctr_encrypt(buf, out, BUF_BLOCKS, n, roundKeys);
        uint64_t t1 = rdtsc();
        samples[t] = t1 - t0;
    }
    qsort(samples, TRIALS, sizeof(uint64_t), cmp_u64);
    return (double)samples[TRIALS/2] / (BUF_BLOCKS * 16);
}

static double measure_cycles_per_byte_ecb(const uint8_t roundKeys[aes128::EXPANDED_KEY_SIZE]) {
    static uint8_t buf[BUF_BLOCKS * 16];
    static uint8_t out[BUF_BLOCKS * 16];
    for (int i = 0; i < BUF_BLOCKS * 16; i++) buf[i] = (uint8_t)(i & 0xff);

    uint64_t samples[TRIALS];
    for (int t = 0; t < TRIALS; t++) {
        uint64_t t0 = rdtsc();
        ecb_encrypt(buf, out, BUF_BLOCKS, roundKeys);
        uint64_t t1 = rdtsc();
        samples[t] = t1 - t0;
    }
    qsort(samples, TRIALS, sizeof(uint64_t), cmp_u64);
    return (double)samples[TRIALS/2] / (BUF_BLOCKS * 16);
}

static double measure_cycles_per_byte_cbc(const uint8_t roundKeys[aes128::EXPANDED_KEY_SIZE]) {
    static uint8_t buf[BUF_BLOCKS * 16];
    static uint8_t out[BUF_BLOCKS * 16];
    uint8_t iv[16] = {0};
    for (int i = 0; i < BUF_BLOCKS * 16; i++) buf[i] = (uint8_t)(i & 0xff);

    uint64_t samples[TRIALS];
    for (int t = 0; t < TRIALS; t++) {
        uint8_t v[16] = {0};
        uint64_t t0 = rdtsc();
        cbc_encrypt(buf, out, BUF_BLOCKS, v, roundKeys);
        uint64_t t1 = rdtsc();
        samples[t] = t1 - t0;
    }
    qsort(samples, TRIALS, sizeof(uint64_t), cmp_u64);
    return (double)samples[TRIALS/2] / (BUF_BLOCKS * 16);
}

static void print_performance_metrics(const uint8_t roundKeys[aes128::EXPANDED_KEY_SIZE]) {
    std::printf("\n--------Performance Metrics (cycles/byte)--------\n");
    std::printf("  Median of %d trials, Buffer = %d blocks (%d bytes)\n",
                TRIALS, BUF_BLOCKS, BUF_BLOCKS * 16);
    double cpb_ecb = measure_cycles_per_byte_ecb(roundKeys);
    double cpb_ctr = measure_cycles_per_byte_ctr(roundKeys);
    double cpb_cbc = measure_cycles_per_byte_cbc(roundKeys);
    std::printf("  %-40s : %8.2f  cycles/byte\n", "ECB encrypt", cpb_ecb);
    std::printf("  %-40s : %8.2f  cycles/byte\n", "CTR encrypt", cpb_ctr);
    std::printf("  %-40s : %8.2f  cycles/byte\n", "CBC encrypt (chained)", cpb_cbc);
    std::printf("\n  CTR overhead vs ECB                    : %+.2f cycles/byte\n",
                cpb_ctr - cpb_ecb);
    std::printf("  CBC overhead vs ECB                    : %+.2f cycles/byte\n",
                cpb_cbc - cpb_ecb);
    std::printf("  CBC overhead vs CTR                    : %+.2f cycles/byte\n",
                cpb_cbc - cpb_ctr);
}

static void throughput_sweep(const uint8_t roundKeys[aes128::EXPANDED_KEY_SIZE],
                              const uint8_t nonce[16], const uint8_t iv[16], double ghz) {
    std::printf("\n--------AES-128 CPU Throughput vs Input Size--------\n");
    std::printf("  Median of %d trials per measurement.\n\n", TRIALS);

    const size_t data_sizes[] = {
        1024, 4*1024, 16*1024, 64*1024, 256*1024,
        1*1024*1024, 4*1024*1024, 16*1024*1024, 64*1024*1024
    };
    const int NSIZES = (int)(sizeof(data_sizes) / sizeof(data_sizes[0]));

    std::printf("  %-10s  %-14s  %12s  %12s\n", "Data Size", "Mode", "Wall (ms)", "Throughput");
    std::printf("  %-10s  %-14s  %12s  %12s\n", "----------", "--------------", "------------", "----------");

    for (int si = 0; si < NSIZES; si++) {
        size_t data_bytes = data_sizes[si];
        size_t num_blocks = (data_bytes + 15) / 16;
        size_t buf_size   = num_blocks * 16;

        uint8_t *plain  = (uint8_t*)malloc(buf_size);
        uint8_t *cipher = (uint8_t*)malloc(buf_size);
        if (!plain || !cipher) { std::fprintf(stderr, "malloc failed\n"); exit(1); }
        for (size_t i = 0; i < buf_size; i++) plain[i] = (uint8_t)(i & 0xff);

        uint64_t ecb_enc_samples[TRIALS], ecb_dec_samples[TRIALS];
        uint64_t ctr_enc_samples[TRIALS], ctr_dec_samples[TRIALS];
        uint64_t cbc_enc_samples[TRIALS], cbc_dec_samples[TRIALS];
        
        for (int t = 0; t < TRIALS; t++) {
            uint64_t t0 = rdtsc(); ecb_encrypt(plain, cipher, (int)num_blocks, roundKeys); uint64_t t1 = rdtsc();
            ecb_enc_samples[t] = t1 - t0;
        }
        for (int t = 0; t < TRIALS; t++) {
            uint64_t t0 = rdtsc(); ecb_decrypt(cipher, plain, (int)num_blocks, roundKeys); uint64_t t1 = rdtsc();
            ecb_dec_samples[t] = t1 - t0;
        }
        
        for (int t = 0; t < TRIALS; t++) {
            uint8_t n[16]; std::memcpy(n, nonce, 16);
            uint64_t t0 = rdtsc(); ctr_encrypt(plain, cipher, (int)num_blocks, n, roundKeys); uint64_t t1 = rdtsc();
            ctr_enc_samples[t] = t1 - t0;
        }
        for (int t = 0; t < TRIALS; t++) {
            uint8_t n[16]; std::memcpy(n, nonce, 16);
            uint64_t t0 = rdtsc(); ctr_decrypt(cipher, plain, (int)num_blocks, n, roundKeys); uint64_t t1 = rdtsc();
            ctr_dec_samples[t] = t1 - t0;
        }
        for (int t = 0; t < TRIALS; t++) {
            uint8_t v[16]; std::memcpy(v, iv, 16);
            uint64_t t0 = rdtsc(); cbc_encrypt(plain, cipher, (int)num_blocks, v, roundKeys); uint64_t t1 = rdtsc();
            cbc_enc_samples[t] = t1 - t0;
        }
        for (int t = 0; t < TRIALS; t++) {
            uint8_t v[16]; std::memcpy(v, iv, 16);
            uint64_t t0 = rdtsc(); cbc_decrypt(cipher, plain, (int)num_blocks, v, roundKeys); uint64_t t1 = rdtsc();
            cbc_dec_samples[t] = t1 - t0;
        }

        qsort(ecb_enc_samples, TRIALS, sizeof(uint64_t), cmp_u64);
        qsort(ecb_dec_samples, TRIALS, sizeof(uint64_t), cmp_u64);
        qsort(ctr_enc_samples, TRIALS, sizeof(uint64_t), cmp_u64);
        qsort(ctr_dec_samples, TRIALS, sizeof(uint64_t), cmp_u64);
        qsort(cbc_enc_samples, TRIALS, sizeof(uint64_t), cmp_u64);
        qsort(cbc_dec_samples, TRIALS, sizeof(uint64_t), cmp_u64);

        double ecb_enc_ms = (double)ecb_enc_samples[TRIALS/2] / (ghz * 1e6);
        double ecb_dec_ms = (double)ecb_dec_samples[TRIALS/2] / (ghz * 1e6);
        double ctr_enc_ms = (double)ctr_enc_samples[TRIALS/2] / (ghz * 1e6);
        double ctr_dec_ms = (double)ctr_dec_samples[TRIALS/2] / (ghz * 1e6);
        double cbc_enc_ms = (double)cbc_enc_samples[TRIALS/2] / (ghz * 1e6);
        double cbc_dec_ms = (double)cbc_dec_samples[TRIALS/2] / (ghz * 1e6);

        double ecb_enc_gbs = (double)data_bytes / (ecb_enc_ms * 1e-3) / 1e9;
        double ecb_dec_gbs = (double)data_bytes / (ecb_dec_ms * 1e-3) / 1e9;
        double ctr_enc_gbs = (double)data_bytes / (ctr_enc_ms * 1e-3) / 1e9;
        double ctr_dec_gbs = (double)data_bytes / (ctr_dec_ms * 1e-3) / 1e9;
        double cbc_enc_gbs = (double)data_bytes / (cbc_enc_ms * 1e-3) / 1e9;
        double cbc_dec_gbs = (double)data_bytes / (cbc_dec_ms * 1e-3) / 1e9;

        const char *unit = (data_bytes >= 1024*1024) ? "MB" : "KB";
        double nd = (data_bytes >= 1024*1024) ? data_bytes/1048576.0 : data_bytes/1024.0;

        std::printf("  %5.0f %-3s  CPU-ECB-E    %10.4f ms  %8.4f GB/s\n", nd, unit, ecb_enc_ms, ecb_enc_gbs);
        std::printf("  %5.0f %-3s  CPU-ECB-D    %10.4f ms  %8.4f GB/s\n", nd, unit, ecb_dec_ms, ecb_dec_gbs);
        std::printf("  %5.0f %-3s  CPU-CTR-E    %10.4f ms  %8.4f GB/s\n", nd, unit, ctr_enc_ms, ctr_enc_gbs);
        std::printf("  %5.0f %-3s  CPU-CTR-D    %10.4f ms  %8.4f GB/s\n", nd, unit, ctr_dec_ms, ctr_dec_gbs);
        std::printf("  %5.0f %-3s  CPU-CBC-E    %10.4f ms  %8.4f GB/s\n", nd, unit, cbc_enc_ms, cbc_enc_gbs);
        std::printf("  %5.0f %-3s  CPU-CBC-D    %10.4f ms  %8.4f GB/s\n", nd, unit, cbc_dec_ms, cbc_dec_gbs);
        std::printf("\n");

        free(plain); free(cipher);
    }
}

static void measure_single_block_latency(const uint8_t roundKeys[aes128::EXPANDED_KEY_SIZE],
                                          const uint8_t nonce[16], const uint8_t iv[16],
                                          double ghz) {
    const int LAT_BLOCKS = 8;
    const int LAT_TRIALS = 101;

    uint8_t plain [LAT_BLOCKS * 16];
    uint8_t cipher[LAT_BLOCKS * 16];
    uint8_t recov [LAT_BLOCKS * 16];
    for (int i = 0; i < LAT_BLOCKS * 16; i++) plain[i] = (uint8_t)(i & 0xff);

    uint64_t ecb_samples[LAT_TRIALS];
    for (int t = 0; t < LAT_TRIALS; t++) {
        uint64_t t0 = rdtsc();
        ecb_encrypt(plain, cipher, LAT_BLOCKS, roundKeys);
        uint64_t t1 = rdtsc();
        ecb_samples[t] = t1 - t0;
    }
    
    uint64_t ctr_samples[LAT_TRIALS];
    for (int t = 0; t < LAT_TRIALS; t++) {
        uint8_t n[16]; std::memcpy(n, nonce, 16);
        uint64_t t0 = rdtsc();
        ctr_encrypt(plain, cipher, LAT_BLOCKS, n, roundKeys);
        uint64_t t1 = rdtsc();
        ctr_samples[t] = t1 - t0;
    }

    uint8_t v0[16]; std::memcpy(v0, iv, 16);
    cbc_encrypt(plain, cipher, LAT_BLOCKS, v0, roundKeys);

    uint64_t cbc_samples[LAT_TRIALS];
    for (int t = 0; t < LAT_TRIALS; t++) {
        uint8_t v[16]; std::memcpy(v, iv, 16);
        uint64_t t0 = rdtsc();
        cbc_decrypt(cipher, recov, LAT_BLOCKS, v, roundKeys);
        uint64_t t1 = rdtsc();
        cbc_samples[t] = t1 - t0;
    }

    for (int i = 0; i < LAT_TRIALS-1; i++)
        for (int j = i+1; j < LAT_TRIALS; j++) {
            if (ecb_samples[j] < ecb_samples[i]) { uint64_t t=ecb_samples[i]; ecb_samples[i]=ecb_samples[j]; ecb_samples[j]=t; }
            if (ctr_samples[j] < ctr_samples[i]) { uint64_t t=ctr_samples[i]; ctr_samples[i]=ctr_samples[j]; ctr_samples[j]=t; }
            if (cbc_samples[j] < cbc_samples[i]) { uint64_t t=cbc_samples[i]; cbc_samples[i]=cbc_samples[j]; cbc_samples[j]=t; }
        }
    double ecb_us = (double)ecb_samples[LAT_TRIALS/2] / (ghz * 1e3);
    double ctr_us = (double)ctr_samples[LAT_TRIALS/2] / (ghz * 1e3);
    double cbc_us = (double)cbc_samples[LAT_TRIALS/2] / (ghz * 1e3);

    std::printf("\n--------Single 128-Byte Block Latency (side data point)--------\n");
    std::printf("  Input size : %d cipher blocks = %d bytes\n", LAT_BLOCKS, LAT_BLOCKS * 16);
    std::printf("  Trials     : %d\n", LAT_TRIALS);
    std::printf("  %-35s : %8.2f us\n", "ECB encrypt latency", ecb_us);
    std::printf("  %-35s : %8.2f us\n", "CTR encrypt latency", ctr_us);
    std::printf("  %-35s : %8.2f us\n", "CBC decrypt latency", cbc_us);
}

static void measure_key_schedule_cost(const uint8_t key[aes128::KEY_SIZE], double ghz) {
    const int KS_TRIALS = 101;
    uint8_t roundKeys[aes128::EXPANDED_KEY_SIZE];

    uint64_t ks_samples[KS_TRIALS];
    for (int t = 0; t < KS_TRIALS; t++) {
        uint64_t t0 = rdtsc();
        aes128::KeyExpansion(key, roundKeys);
        uint64_t t1 = rdtsc();
        ks_samples[t] = t1 - t0;
    }
    qsort(ks_samples, KS_TRIALS, sizeof(uint64_t), cmp_u64);
    double ks_med = (double)ks_samples[KS_TRIALS/2] / (ghz * 1e3);

    std::printf("\n--------Key Schedule Cost--------\n");
    std::printf("\n  Time (median of %d trials)\n", KS_TRIALS);
    std::printf("  %-48s : %7.2f us\n", "CPU compute", ks_med);
    std::printf("  %-48s : %7.2f us\n", "GPU upload", 0.0);
    std::printf("  %-48s : %7.2f us\n", "Total key setup cost", ks_med);
    std::printf("\n  Space\n");
    std::printf("  %-48s : %3d bytes  (%d rounds x 16 bytes)\n",
                "Expanded round keys in RAM", aes128::EXPANDED_KEY_SIZE, aes128::Nr + 1);
}

int main() {
    std::printf("--------AES-128 CPU Implementation--------\n");
    std::printf("--------CTR and CBC Mode Tests--------\n\n");

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

    std::printf("Generating key schedule\n");
    aes128::KeyExpansion(key, round_keys);
    std::printf("Key schedule generated\n\n");

    double ghz = cpu_ghz();

    uint8_t ctr_cipher[16], ctr_recovered[16];
    uint8_t cbc_cipher[16], cbc_recovered[16];

    uint8_t ecb_cipher[16], ecb_recovered[16];

    std::printf("--------ECB Mode--------\n\n");
    std::printf("Original Plaintext   : "); print_message(plaintext, 16);
    ecb_encrypt(plaintext, ecb_cipher, 1, round_keys);
    std::printf("ECB Ciphertext       : "); print_message(ecb_cipher, 16);
    ecb_decrypt(ecb_cipher, ecb_recovered, 1, round_keys);
    std::printf("Recovered Plaintext  : "); print_message(ecb_recovered, 16);
    std::printf("\n");
    assert_equal("ECB mode", ecb_recovered, plaintext, 16);
    std::printf("\n");
    
    std::printf("--------CTR Mode--------\n\n");
    std::printf("Original Plaintext   : "); print_message(plaintext, 16);
    ctr_encrypt(plaintext, ctr_cipher, 1, nonce, round_keys);
    std::printf("CTR Ciphertext       : "); print_message(ctr_cipher, 16);
    ctr_decrypt(ctr_cipher, ctr_recovered, 1, nonce, round_keys);
    std::printf("Recovered Plaintext  : "); print_message(ctr_recovered, 16);
    std::printf("\n");
    assert_equal("CTR mode", ctr_recovered, plaintext, 16);

    std::printf("\n--------CBC Mode--------\n\n");
    std::printf("Original Plaintext   : "); print_message(plaintext, 16);
    cbc_encrypt(plaintext, cbc_cipher, 1, iv, round_keys);
    std::printf("CBC Ciphertext       : "); print_message(cbc_cipher, 16);
    cbc_decrypt(cbc_cipher, cbc_recovered, 1, iv, round_keys);
    std::printf("Recovered Plaintext  : "); print_message(cbc_recovered, 16);
    std::printf("\n");
    assert_equal("CBC mode", cbc_recovered, plaintext, 16);

    std::printf("\n--------ECB, CBC and CTR modes are validated--------\n");
    print_code_size();
    print_performance_metrics(round_keys);
    throughput_sweep(round_keys, nonce, iv, ghz);
    measure_single_block_latency(round_keys, nonce, iv, ghz);
    measure_key_schedule_cost(key, ghz);

    return 0;
}