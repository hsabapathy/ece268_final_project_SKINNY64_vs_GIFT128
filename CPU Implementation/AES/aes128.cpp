#include <cstdint>
#include <cstring>
#include <cstdio>
#include <chrono>
#include <vector>

namespace aes128 {

constexpr int Nb = 4;       // block size in 32-bit words
constexpr int Nk = 4;       // key size in 32-bit words (128-bit key)
constexpr int Nr = 10;      // number of rounds
constexpr int BLOCK_SIZE = 16;
constexpr int KEY_SIZE   = 16;
constexpr int EXPANDED_KEY_SIZE = 16 * (Nr + 1); // 176 bytes

// ---------- AES S-box ----------
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

// ---------- Inverse S-box ----------
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

// ---------- Round constants ----------
static const uint8_t Rcon[11] = {
    0x00, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36
};

// xtime: multiply by 2 (i.e. by x) in GF(2^8) with reduction poly 0x1b
static inline uint8_t xtime(uint8_t x) {
    return (uint8_t)((x << 1) ^ ((x >> 7) * 0x1b));
}

// General GF(2^8) multiplication, used only by InvMixColumns
static inline uint8_t gmul(uint8_t a, uint8_t b) {
    uint8_t r = 0;
    while (b) {
        if (b & 1) r ^= a;
        a = xtime(a);
        b >>= 1;
    }
    return r;
}

// =============================================================================
//  Key schedule (FIPS-197 §5.2)
//  Produces 11 round keys (176 bytes total) from the 16-byte master key.
// =============================================================================
void KeyExpansion(const uint8_t key[KEY_SIZE], uint8_t roundKeys[EXPANDED_KEY_SIZE]) {
    std::memcpy(roundKeys, key, KEY_SIZE);

    uint8_t t[4];
    for (int i = Nk; i < Nb * (Nr + 1); ++i) {
        // temp = previous word
        t[0] = roundKeys[(i - 1) * 4 + 0];
        t[1] = roundKeys[(i - 1) * 4 + 1];
        t[2] = roundKeys[(i - 1) * 4 + 2];
        t[3] = roundKeys[(i - 1) * 4 + 3];

        if (i % Nk == 0) {
            // RotWord: rotate left by one byte
            uint8_t u = t[0]; t[0] = t[1]; t[1] = t[2]; t[2] = t[3]; t[3] = u;
            // SubWord
            t[0] = sbox[t[0]];
            t[1] = sbox[t[1]];
            t[2] = sbox[t[2]];
            t[3] = sbox[t[3]];
            // XOR with round constant
            t[0] ^= Rcon[i / Nk];
        }
        for (int j = 0; j < 4; ++j) {
            roundKeys[i * 4 + j] = roundKeys[(i - Nk) * 4 + j] ^ t[j];
        }
    }
}

// =============================================================================
//  Round transformations
//  State layout (column-major):
//        s[0]  s[4]  s[8]  s[12]
//        s[1]  s[5]  s[9]  s[13]
//        s[2]  s[6]  s[10] s[14]
//        s[3]  s[7]  s[11] s[15]
// =============================================================================
static void SubBytes(uint8_t s[16]) {
    for (int i = 0; i < 16; ++i) s[i] = sbox[s[i]];
}
static void InvSubBytes(uint8_t s[16]) {
    for (int i = 0; i < 16; ++i) s[i] = inv_sbox[s[i]];
}

static void ShiftRows(uint8_t s[16]) {
    uint8_t t;
    // row 1: left rotate by 1
    t = s[1];  s[1]  = s[5];  s[5]  = s[9];  s[9]  = s[13]; s[13] = t;
    // row 2: left rotate by 2
    t = s[2];  s[2]  = s[10]; s[10] = t;
    t = s[6];  s[6]  = s[14]; s[14] = t;
    // row 3: left rotate by 3 (== right rotate by 1)
    t = s[3];  s[3]  = s[15]; s[15] = s[11]; s[11] = s[7]; s[7] = t;
}
static void InvShiftRows(uint8_t s[16]) {
    uint8_t t;
    // row 1: right rotate by 1
    t = s[13]; s[13] = s[9];  s[9]  = s[5];  s[5]  = s[1]; s[1] = t;
    // row 2: right rotate by 2
    t = s[2];  s[2]  = s[10]; s[10] = t;
    t = s[6];  s[6]  = s[14]; s[14] = t;
    // row 3: right rotate by 3 (== left rotate by 1)
    t = s[3];  s[3]  = s[7];  s[7]  = s[11]; s[11] = s[15]; s[15] = t;
}

static void MixColumns(uint8_t s[16]) {
    for (int c = 0; c < 4; ++c) {
        uint8_t a0 = s[c*4+0], a1 = s[c*4+1], a2 = s[c*4+2], a3 = s[c*4+3];
        uint8_t T  = a0 ^ a1 ^ a2 ^ a3;
        s[c*4+0] ^= T ^ xtime(a0 ^ a1);
        s[c*4+1] ^= T ^ xtime(a1 ^ a2);
        s[c*4+2] ^= T ^ xtime(a2 ^ a3);
        s[c*4+3] ^= T ^ xtime(a3 ^ a0);
    }
}
static void InvMixColumns(uint8_t s[16]) {
    for (int c = 0; c < 4; ++c) {
        uint8_t a0 = s[c*4+0], a1 = s[c*4+1], a2 = s[c*4+2], a3 = s[c*4+3];
        s[c*4+0] = gmul(a0,0x0e) ^ gmul(a1,0x0b) ^ gmul(a2,0x0d) ^ gmul(a3,0x09);
        s[c*4+1] = gmul(a0,0x09) ^ gmul(a1,0x0e) ^ gmul(a2,0x0b) ^ gmul(a3,0x0d);
        s[c*4+2] = gmul(a0,0x0d) ^ gmul(a1,0x09) ^ gmul(a2,0x0e) ^ gmul(a3,0x0b);
        s[c*4+3] = gmul(a0,0x0b) ^ gmul(a1,0x0d) ^ gmul(a2,0x09) ^ gmul(a3,0x0e);
    }
}

static void AddRoundKey(uint8_t s[16], const uint8_t* rk) {
    for (int i = 0; i < 16; ++i) s[i] ^= rk[i];
}

// =============================================================================
//  Block encrypt / decrypt (FIPS-197 §5.1, §5.3)
// =============================================================================
void EncryptBlock(const uint8_t in[16], uint8_t out[16],
                  const uint8_t roundKeys[EXPANDED_KEY_SIZE]) {
    uint8_t s[16];
    std::memcpy(s, in, 16);

    AddRoundKey(s, roundKeys);
    for (int r = 1; r < Nr; ++r) {
        SubBytes(s);
        ShiftRows(s);
        MixColumns(s);
        AddRoundKey(s, roundKeys + r * 16);
    }
    SubBytes(s);
    ShiftRows(s);
    AddRoundKey(s, roundKeys + Nr * 16);

    std::memcpy(out, s, 16);
}

void DecryptBlock(const uint8_t in[16], uint8_t out[16],
                  const uint8_t roundKeys[EXPANDED_KEY_SIZE]) {
    uint8_t s[16];
    std::memcpy(s, in, 16);

    AddRoundKey(s, roundKeys + Nr * 16);
    for (int r = Nr - 1; r >= 1; --r) {
        InvShiftRows(s);
        InvSubBytes(s);
        AddRoundKey(s, roundKeys + r * 16);
        InvMixColumns(s);
    }
    InvShiftRows(s);
    InvSubBytes(s);
    AddRoundKey(s, roundKeys);

    std::memcpy(out, s, 16);
}

} // namespace aes128

// =============================================================================
//  Test + benchmark harness
// =============================================================================
static void hexdump(const char* label, const uint8_t* p, size_t n) {
    std::printf("%-12s", label);
    for (size_t i = 0; i < n; ++i) std::printf("%02x", p[i]);
    std::printf("\n");
}

static bool eq(const uint8_t* a, const uint8_t* b, size_t n) {
    return std::memcmp(a, b, n) == 0;
}

static bool run_test(const char* name,
                     const uint8_t key[16],
                     const uint8_t pt[16],
                     const uint8_t expected_ct[16]) {
    uint8_t rk[aes128::EXPANDED_KEY_SIZE];
    uint8_t ct[16], dec[16];

    aes128::KeyExpansion(key, rk);
    aes128::EncryptBlock(pt, ct, rk);
    aes128::DecryptBlock(ct, dec, rk);

    bool enc_ok = eq(ct, expected_ct, 16);
    bool dec_ok = eq(dec, pt, 16);

    std::printf("[%s] %s\n", name, (enc_ok && dec_ok) ? "PASS" : "FAIL");
    hexdump("  key:",       key, 16);
    hexdump("  plaintext:", pt,  16);
    hexdump("  expected:",  expected_ct, 16);
    hexdump("  got ct:",    ct,  16);
    hexdump("  decrypted:", dec, 16);
    std::printf("\n");
    return enc_ok && dec_ok;
}

int main() {
    std::printf("AES-128 reference (byte-oriented, no AES-NI)\n");
    std::printf("============================================\n\n");

    // -------- FIPS-197 Appendix B test vector --------
    {
        uint8_t key[16] = {0x2b,0x7e,0x15,0x16,0x28,0xae,0xd2,0xa6,
                           0xab,0xf7,0x15,0x88,0x09,0xcf,0x4f,0x3c};
        uint8_t pt [16] = {0x32,0x43,0xf6,0xa8,0x88,0x5a,0x30,0x8d,
                           0x31,0x31,0x98,0xa2,0xe0,0x37,0x07,0x34};
        uint8_t ct [16] = {0x39,0x25,0x84,0x1d,0x02,0xdc,0x09,0xfb,
                           0xdc,0x11,0x85,0x97,0x19,0x6a,0x0b,0x32};
        run_test("FIPS-197 App.B", key, pt, ct);
    }

    // -------- FIPS-197 Appendix C.1 test vector --------
    {
        uint8_t key[16] = {0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07,
                           0x08,0x09,0x0a,0x0b,0x0c,0x0d,0x0e,0x0f};
        uint8_t pt [16] = {0x00,0x11,0x22,0x33,0x44,0x55,0x66,0x77,
                           0x88,0x99,0xaa,0xbb,0xcc,0xdd,0xee,0xff};
        uint8_t ct [16] = {0x69,0xc4,0xe0,0xd8,0x6a,0x7b,0x04,0x30,
                           0xd8,0xcd,0xb7,0x80,0x70,0xb4,0xc5,0x5a};
        run_test("FIPS-197 App.C.1", key, pt, ct);
    }

    // -------- Benchmark: key schedule and bulk encryption --------
    std::printf("Benchmarks (single thread, wall clock)\n");
    std::printf("--------------------------------------\n");

    using clock = std::chrono::high_resolution_clock;

    // Key schedule cost
    {
        uint8_t key[16] = {0};
        uint8_t rk[aes128::EXPANDED_KEY_SIZE];
        const int N = 1'000'000;
        auto t0 = clock::now();
        for (int i = 0; i < N; ++i) {
            key[0] = (uint8_t)i;  // vary to prevent constant-folding
            aes128::KeyExpansion(key, rk);
        }
        auto t1 = clock::now();
        double ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
        std::printf("  KeyExpansion:   %.1f ns/call (%d calls)\n", ns / N, N);
    }

    // Encryption throughput
    {
        uint8_t key[16] = {0};
        uint8_t rk[aes128::EXPANDED_KEY_SIZE];
        aes128::KeyExpansion(key, rk);

        const size_t NBLOCKS = 1'000'000;
        std::vector<uint8_t> in(NBLOCKS * 16, 0xa5);
        std::vector<uint8_t> out(NBLOCKS * 16);

        auto t0 = clock::now();
        for (size_t i = 0; i < NBLOCKS; ++i) {
            aes128::EncryptBlock(&in[i*16], &out[i*16], rk);
        }
        auto t1 = clock::now();
        double sec   = std::chrono::duration<double>(t1 - t0).count();
        double bytes = (double)NBLOCKS * 16.0;
        double mbps  = bytes / sec / (1024.0 * 1024.0);
        double cpb   = (sec * 1e9) / bytes; // ns per byte; rename for cycles if you measure freq
        std::printf("  EncryptBlock:   %.2f ns/byte   %.1f MiB/s   (%zu blocks)\n",
                    cpb, mbps, NBLOCKS);
    }

    // Decryption throughput
    {
        uint8_t key[16] = {0};
        uint8_t rk[aes128::EXPANDED_KEY_SIZE];
        aes128::KeyExpansion(key, rk);

        const size_t NBLOCKS = 1'000'000;
        std::vector<uint8_t> in(NBLOCKS * 16, 0xa5);
        std::vector<uint8_t> out(NBLOCKS * 16);

        auto t0 = clock::now();
        for (size_t i = 0; i < NBLOCKS; ++i) {
            aes128::DecryptBlock(&in[i*16], &out[i*16], rk);
        }
        auto t1 = clock::now();
        double sec   = std::chrono::duration<double>(t1 - t0).count();
        double bytes = (double)NBLOCKS * 16.0;
        double mbps  = bytes / sec / (1024.0 * 1024.0);
        double cpb   = (sec * 1e9) / bytes;
        std::printf("  DecryptBlock:   %.2f ns/byte   %.1f MiB/s   (%zu blocks)\n",
                    cpb, mbps, NBLOCKS);
    }

    std::printf("\nROM footprint reference:\n");
    std::printf("  sbox + inv_sbox:  512 bytes\n");
    std::printf("  Rcon:              11 bytes\n");
    std::printf("  Round-key buffer: %d bytes (per key)\n", aes128::EXPANDED_KEY_SIZE);
    return 0;
}
