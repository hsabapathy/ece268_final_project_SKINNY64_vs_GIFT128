// GIFT-128 Block Cipher with CBC, CTR, and ECB Modes, CPU Implementation


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <iostream>
#include <iomanip>
#include <time.h>

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
#define TRIALS      31
#define BUF_BLOCKS  256  // fits in L1 cache for cycles-per-byte measurements

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


//  GIFT-128 Core
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


//  ECB Mode
//  Encrypt: C[i] = E_K(P[i])   (each block independently)
//  Decrypt: P[i] = D_K(C[i])   (each block independently)

/*
 gift128_ecb_encrypt
 key: 16-byte
 input: plaintext (must be a multiple of BLOCK_BYTES)
 output: ciphertext
 length: byte length of input (must be a multiple of BLOCK_BYTES)
 */
void gift128_ecb_encrypt(const uint8_t *key,
                         const uint8_t *input,
                               uint8_t *output,
                               size_t   length)
{
    for (size_t offset = 0; offset < length; offset += BLOCK_BYTES)
        gift128_encrypt_block(input + offset, key, output + offset);
}

/*
 gift128_ecb_decrypt
 key: 16-byte
 input: ciphertext (must be a multiple of BLOCK_BYTES)
 output: plaintext
 length: byte length of input (must be a multiple of BLOCK_BYTES)
 */
void gift128_ecb_decrypt(const uint8_t *key,
                         const uint8_t *input,
                               uint8_t *output,
                               size_t   length)
{
    for (size_t offset = 0; offset < length; offset += BLOCK_BYTES)
        gift128_decrypt_block(input + offset, key, output + offset);
}


//  CBC Mode
//  Encrypt: C[i] = E_K(P[i] XOR C[i-1]),   C[-1] = IV
//  Decrypt: P[i] = D_K(C[i]) XOR C[i-1],   C[-1] = IV

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


static void print_hex(const char *label, const uint8_t *data, size_t len)
{
    printf("%-20s: ", label);
    for (size_t i = 0; i < len; i++) printf("%02x", data[i]);
    printf("\n");
}

static void assert_equal(const char *label, const uint8_t *a, const uint8_t *b, size_t len)
{
    bool ok = (memcmp(a, b, len) == 0);
    printf("%s: %s\n", label, ok ? "PASS" : "FAIL");
}


static void run_gift128_core_test()
{
    printf("\n GIFT-128 Core Block Cipher Test\n");

    uint8_t key[16] = {
        0x9e,0xb9,0x36,0x40,0xd0,0x88,0xda,0x63,
        0x76,0xa3,0x9d,0x1c,0x8b,0xea,0x71,0xe1
    };
    uint8_t pt[16] = {
        0xcf,0x16,0xcf,0xe8,0xfd,0x0f,0x98,0xaa,
        0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00
    };
    uint8_t ct[16], recovered[16];

    gift128_encrypt_block(pt, key, ct);
    gift128_decrypt_block(ct, key, recovered);

    assert_equal("Encryption->Decryption round-trip", recovered, pt, 16);
}

static void run_ecb_test()
{
    printf("\n--------ECB Mode--------\n\n");

    uint8_t key[16] = {
        0x9e,0xb9,0x36,0x40,0xd0,0x88,0xda,0x63,
        0x76,0xa3,0x9d,0x1c,0x8b,0xea,0x71,0xe1
    };
    uint8_t plain[16] = {
        0xcf,0x16,0xcf,0xe8,0xfd,0x0f,0x98,0xaa,
        0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00
    };

    uint8_t ecb_cipher[16];
    uint8_t ecb_recovered[16];

    printf("Original Plaintext   : ");
    for (int i = 0; i < 16; i++) printf("%02x", plain[i]);
    printf("\n");

    gift128_ecb_encrypt(key, plain,       ecb_cipher,    16);
    gift128_ecb_decrypt(key, ecb_cipher,  ecb_recovered, 16);

    printf("ECB Ciphertext       : ");
    for (int i = 0; i < 16; i++) printf("%02x", ecb_cipher[i]);
    printf("\n");

    printf("Recovered Plaintext  : ");
    for (int i = 0; i < 16; i++) printf("%02x", ecb_recovered[i]);
    printf("\n\n");

    assert_equal("ECB mode is validated ", ecb_recovered, plain, 16);
}

static void run_cbc_test()
{
    printf("\n--------CBC Mode--------\n\n");

    uint8_t key[16] = {
        0x9e,0xb9,0x36,0x40,0xd0,0x88,0xda,0x63,
        0x76,0xa3,0x9d,0x1c,0x8b,0xea,0x71,0xe1
    };
    uint8_t plain[16] = {
        0xcf,0x16,0xcf,0xe8,0xfd,0x0f,0x98,0xaa,
        0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00
    };
    uint8_t iv[16] = {0};   // all-zero IV

    uint8_t cbc_cipher[16];
    uint8_t cbc_recovered[16];

    printf("Original Plaintext   : ");
    for (int i = 0; i < 16; i++) printf("%02x", plain[i]);
    printf("\n");

    gift128_cbc_encrypt(key, iv, plain,       cbc_cipher,    16);
    gift128_cbc_decrypt(key, iv, cbc_cipher,  cbc_recovered, 16);

    printf("CBC Ciphertext       : ");
    for (int i = 0; i < 16; i++) printf("%02x", cbc_cipher[i]);
    printf("\n");

    printf("Recovered Plaintext  : ");
    for (int i = 0; i < 16; i++) printf("%02x", cbc_recovered[i]);
    printf("\n\n");

    assert_equal("CBC mode is validated ", cbc_recovered, plain, 16);
}


static void run_ctr_test()
{
    printf("\n--------CTR Mode--------\n\n");

    //  Same key and plaintext as core test
    uint8_t key[16] = {
        0x9e,0xb9,0x36,0x40,0xd0,0x88,0xda,0x63,
        0x76,0xa3,0x9d,0x1c,0x8b,0xea,0x71,0xe1
    };
    uint8_t plain[16] = {
        0xcf,0x16,0xcf,0xe8,0xfd,0x0f,0x98,0xaa,
        0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00
    };
    // Nonce = 12 zero bytes
    uint8_t nonce[12] = {0};

    uint8_t ctr_cipher[16];
    uint8_t ctr_recovered[16];

    printf("Original Plaintext   : ");
    for (int i = 0; i < 16; i++) printf("%02x", plain[i]);
    printf("\n");

    gift128_ctr_crypt(key, nonce, plain,       ctr_cipher,    16);
    gift128_ctr_crypt(key, nonce, ctr_cipher,  ctr_recovered, 16);

    printf("CTR Ciphertext       : ");
    for (int i = 0; i < 16; i++) printf("%02x", ctr_cipher[i]);
    printf("\n");

    printf("Recovered Plaintext  : ");
    for (int i = 0; i < 16; i++) printf("%02x", ctr_recovered[i]);
    printf("\n\n");

    assert_equal("CTR mode is validated", ctr_recovered, plain, 16);
}


//  Printing Quantitative Metrics

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

static int cmp_dbl(const void *a, const void *b)
{
    double x = *(const double*)a;
    double y = *(const double*)b;
    return (x > y) - (x < y);
}

static inline double elapsed_ms(struct timespec t0, struct timespec t1)
{
    return (t1.tv_sec - t0.tv_sec) * 1e3 +
           (t1.tv_nsec - t0.tv_nsec) * 1e-6;
}

//Finding number of cycles needed to encrypt 1 byte in CTR
static double measure_cycles_per_byte_ctr()
{
    static uint8_t buf[BUF_BLOCKS * BLOCK_BYTES];
    static uint8_t out[BUF_BLOCKS * BLOCK_BYTES];
    uint8_t key[KEY_BYTES] = {0x9e,0xb9,0x36,0x40,0xd0,0x88,0xda,0x63,
                               0x76,0xa3,0x9d,0x1c,0x8b,0xea,0x71,0xe1};
    uint8_t nonce[12] = {0};
    memset(buf, 0xA5, sizeof(buf));

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

static double measure_cycles_per_byte_cbc()
{
    static uint8_t buf[BUF_BLOCKS * BLOCK_BYTES];
    static uint8_t out[BUF_BLOCKS * BLOCK_BYTES];
    uint8_t key[KEY_BYTES] = {0x9e,0xb9,0x36,0x40,0xd0,0x88,0xda,0x63,
                               0x76,0xa3,0x9d,0x1c,0x8b,0xea,0x71,0xe1};
    uint8_t iv[BLOCK_BYTES] = {0};
    memset(buf, 0xA5, sizeof(buf));

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

static double measure_cycles_per_byte_ecb()
{
    static uint8_t buf[BUF_BLOCKS * BLOCK_BYTES];
    static uint8_t out[BUF_BLOCKS * BLOCK_BYTES];
    uint8_t key[KEY_BYTES] = {0x9e,0xb9,0x36,0x40,0xd0,0x88,0xda,0x63,
                               0x76,0xa3,0x9d,0x1c,0x8b,0xea,0x71,0xe1};
    memset(buf, 0xA5, sizeof(buf));

    uint64_t samples[TRIALS];
    for (int t = 0; t < TRIALS; t++) {
        uint64_t t0 = rdtsc();
        gift128_ecb_encrypt(key, buf, out, sizeof(buf));
        uint64_t t1 = rdtsc();
        samples[t] = t1 - t0;
    }
    qsort(samples, TRIALS, sizeof(uint64_t), cmp_u64);
    return (double)samples[TRIALS / 2] / (BUF_BLOCKS * BLOCK_BYTES);
}

//Printing code size
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
}

static void print_performance_metrics()
{
    printf("\n--------Performance Metrics (cycles/byte)--------\n");
    printf("  Median of %d trials, Buffer = %d blocks (%d bytes)\n",
           TRIALS, BUF_BLOCKS, BUF_BLOCKS * BLOCK_BYTES);

    double cpb_ctr = measure_cycles_per_byte_ctr();
    double cpb_cbc = measure_cycles_per_byte_cbc();
    double cpb_ecb = measure_cycles_per_byte_ecb();

    printf("  %-40s : %8.2f  cycles/byte\n", "CTR encrypt", cpb_ctr);
    printf("  %-40s : %8.2f  cycles/byte\n", "CBC encrypt (chained)", cpb_cbc);
    printf("  %-40s : %8.2f  cycles/byte\n", "ECB encrypt (independent blocks)", cpb_ecb);
    printf("\n  CBC overhead vs CTR                    : %+.2f cycles/byte\n",
           cpb_cbc - cpb_ctr);
    printf("  ECB overhead vs CTR                    : %+.2f cycles/byte\n",
           cpb_ecb - cpb_ctr);
}

// Throughput sweep

static void throughput_sweep(const uint8_t *key,
                             const uint8_t  nonce[12],
                             const uint8_t  iv[BLOCK_BYTES])
{
    printf("\n");
    printf("--------GIFT-128 CPU Throughput vs Input Size--------\n");
    printf("  Median of %d trials per measurement.\n\n", TRIALS);

    const size_t data_sizes[] = {
        1024,
        4   * 1024,
        16  * 1024,

    };
    const int NSIZES = (int)(sizeof(data_sizes) / sizeof(data_sizes[0]));

    printf("  %-10s  %-14s  %12s  %12s\n",
           "Data Size", "Mode", "Wall (ms)", "Throughput");
    printf("  %-10s  %-14s  %12s  %12s\n",
           "----------", "--------------", "------------", "----------");

    for (int si = 0; si < NSIZES; si++) {
        size_t N      = data_sizes[si];
        size_t padded = ((N + BLOCK_BYTES - 1) / BLOCK_BYTES) * BLOCK_BYTES;

        uint8_t *plain  = (uint8_t*)malloc(padded);
        uint8_t *cipher = (uint8_t*)malloc(padded);
        uint8_t *recov  = (uint8_t*)malloc(padded);
        if (!plain || !cipher || !recov) { fprintf(stderr, "malloc failed\n"); exit(1); }
        for (size_t i = 0; i < padded; i++) plain[i] = (uint8_t)(i & 0xFF);

        // Pre-compute valid CBC ciphertext for CBC decrypt benchmark
        gift128_cbc_encrypt(key, iv, plain, cipher, padded);

        struct timespec t0, t1;

        // CPU CTR encrypt
        double ctr_enc_samples[TRIALS];
        for (int t = 0; t < TRIALS; t++) {
            uint8_t n[12]; memcpy(n, nonce, 12);
            clock_gettime(CLOCK_MONOTONIC, &t0);
            gift128_ctr_crypt(key, n, plain, cipher, padded);
            clock_gettime(CLOCK_MONOTONIC, &t1);
            ctr_enc_samples[t] = elapsed_ms(t0, t1);
        }

        // CPU CTR decrypt
        double ctr_dec_samples[TRIALS];
        for (int t = 0; t < TRIALS; t++) {
            uint8_t n[12]; memcpy(n, nonce, 12);
            clock_gettime(CLOCK_MONOTONIC, &t0);
            gift128_ctr_crypt(key, n, cipher, recov, padded);
            clock_gettime(CLOCK_MONOTONIC, &t1);
            ctr_dec_samples[t] = elapsed_ms(t0, t1);
        }

        // CPU CBC encrypt
        double cbc_enc_samples[TRIALS];
        for (int t = 0; t < TRIALS; t++) {
            uint8_t v[BLOCK_BYTES]; memcpy(v, iv, BLOCK_BYTES);
            clock_gettime(CLOCK_MONOTONIC, &t0);
            gift128_cbc_encrypt(key, v, plain, cipher, padded);
            clock_gettime(CLOCK_MONOTONIC, &t1);
            cbc_enc_samples[t] = elapsed_ms(t0, t1);
        }

        // CPU CBC decrypt 
        double cbc_dec_samples[TRIALS];
        for (int t = 0; t < TRIALS; t++) {
            uint8_t v[BLOCK_BYTES]; memcpy(v, iv, BLOCK_BYTES);
            clock_gettime(CLOCK_MONOTONIC, &t0);
            gift128_cbc_decrypt(key, v, cipher, recov, padded);
            clock_gettime(CLOCK_MONOTONIC, &t1);
            cbc_dec_samples[t] = elapsed_ms(t0, t1);
        }

        // CPU ECB encrypt 
        double ecb_enc_samples[TRIALS];
        for (int t = 0; t < TRIALS; t++) {
            clock_gettime(CLOCK_MONOTONIC, &t0);
            gift128_ecb_encrypt(key, plain, cipher, padded);
            clock_gettime(CLOCK_MONOTONIC, &t1);
            ecb_enc_samples[t] = elapsed_ms(t0, t1);
        }

        // CPU ECB decrypt 
        // Pre-compute valid ECB ciphertext for decrypt benchmark
        gift128_ecb_encrypt(key, plain, cipher, padded);

        double ecb_dec_samples[TRIALS];
        for (int t = 0; t < TRIALS; t++) {
            clock_gettime(CLOCK_MONOTONIC, &t0);
            gift128_ecb_decrypt(key, cipher, recov, padded);
            clock_gettime(CLOCK_MONOTONIC, &t1);
            ecb_dec_samples[t] = elapsed_ms(t0, t1);
        }

        qsort(ctr_enc_samples, TRIALS, sizeof(double), cmp_dbl);
        qsort(ctr_dec_samples, TRIALS, sizeof(double), cmp_dbl);
        qsort(cbc_enc_samples, TRIALS, sizeof(double), cmp_dbl);
        qsort(cbc_dec_samples, TRIALS, sizeof(double), cmp_dbl);
        qsort(ecb_enc_samples, TRIALS, sizeof(double), cmp_dbl);
        qsort(ecb_dec_samples, TRIALS, sizeof(double), cmp_dbl);

        double ctr_enc_ms = ctr_enc_samples[TRIALS/2];
        double ctr_dec_ms = ctr_dec_samples[TRIALS/2];
        double cbc_enc_ms = cbc_enc_samples[TRIALS/2];
        double cbc_dec_ms = cbc_dec_samples[TRIALS/2];
        double ecb_enc_ms = ecb_enc_samples[TRIALS/2];
        double ecb_dec_ms = ecb_dec_samples[TRIALS/2];

        double ctr_enc_gbs = (double)N / (ctr_enc_ms * 1e-3) / 1e9;
        double ctr_dec_gbs = (double)N / (ctr_dec_ms * 1e-3) / 1e9;
        double cbc_enc_gbs = (double)N / (cbc_enc_ms * 1e-3) / 1e9;
        double cbc_dec_gbs = (double)N / (cbc_dec_ms * 1e-3) / 1e9;
        double ecb_enc_gbs = (double)N / (ecb_enc_ms * 1e-3) / 1e9;
        double ecb_dec_gbs = (double)N / (ecb_dec_ms * 1e-3) / 1e9;

        const char *unit = (N >= 1024*1024) ? "MB" : "KB";
        double      nd   = (N >= 1024*1024) ? N / 1048576.0 : N / 1024.0;

        printf("  %5.0f %-3s  CPU-CTR-E    %10.4f ms  %8.4f GB/s\n",
               nd, unit, ctr_enc_ms, ctr_enc_gbs);
        printf("  %5.0f %-3s  CPU-CTR-D    %10.4f ms  %8.4f GB/s\n",
               nd, unit, ctr_dec_ms, ctr_dec_gbs);
        printf("  %5.0f %-3s  CPU-CBC-E    %10.4f ms  %8.4f GB/s\n",
               nd, unit, cbc_enc_ms, cbc_enc_gbs);
        printf("  %5.0f %-3s  CPU-CBC-D    %10.4f ms  %8.4f GB/s\n",
               nd, unit, cbc_dec_ms, cbc_dec_gbs);
        printf("  %5.0f %-3s  CPU-ECB-E    %10.4f ms  %8.4f GB/s\n",
               nd, unit, ecb_enc_ms, ecb_enc_gbs);
        printf("  %5.0f %-3s  CPU-ECB-D    %10.4f ms  %8.4f GB/s\n",
               nd, unit, ecb_dec_ms, ecb_dec_gbs);
        printf("\n");

        free(plain); free(cipher); free(recov);
    }
}

static void measure_single_block_latency(const uint8_t *key,
                                         const uint8_t  nonce[12],
                                         const uint8_t  iv[BLOCK_BYTES])
{
    const int LAT_BLOCKS = 4;
    const int LAT_TRIALS = 101;

    size_t sz = (size_t)LAT_BLOCKS * BLOCK_BYTES;

    uint8_t plain [LAT_BLOCKS * BLOCK_BYTES];
    uint8_t cipher[LAT_BLOCKS * BLOCK_BYTES];
    uint8_t recov [LAT_BLOCKS * BLOCK_BYTES];
    for (int i = 0; i < LAT_BLOCKS * BLOCK_BYTES; i++) plain[i] = (uint8_t)(i & 0xFF);

    struct timespec t0, t1;

    float ctr_samples[LAT_TRIALS];
    for (int t = 0; t < LAT_TRIALS; t++) {
        uint8_t n[12]; memcpy(n, nonce, 12);
        clock_gettime(CLOCK_MONOTONIC, &t0);
        gift128_ctr_crypt(key, n, plain, cipher, sz);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        ctr_samples[t] = (float)(elapsed_ms(t0, t1) * 1000.0);
    }

    // Pre-compute valid CBC ciphertext for decrypt
    uint8_t v0[BLOCK_BYTES]; memcpy(v0, iv, BLOCK_BYTES);
    gift128_cbc_encrypt(key, v0, plain, cipher, sz);

    float cbc_samples[LAT_TRIALS];
    for (int t = 0; t < LAT_TRIALS; t++) {
        uint8_t v[BLOCK_BYTES]; memcpy(v, iv, BLOCK_BYTES);
        clock_gettime(CLOCK_MONOTONIC, &t0);
        gift128_cbc_decrypt(key, v, cipher, recov, sz);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        cbc_samples[t] = (float)(elapsed_ms(t0, t1) * 1000.0);
    }

    gift128_ecb_encrypt(key, plain, cipher, sz);

    float ecb_samples[LAT_TRIALS];
    for (int t = 0; t < LAT_TRIALS; t++) {
        clock_gettime(CLOCK_MONOTONIC, &t0);
        gift128_ecb_decrypt(key, cipher, recov, sz);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        ecb_samples[t] = (float)(elapsed_ms(t0, t1) * 1000.0);
    }

    for (int i = 0; i < LAT_TRIALS-1; i++)
        for (int j = i+1; j < LAT_TRIALS; j++) {
            if (ctr_samples[j] < ctr_samples[i]) { float tmp=ctr_samples[i]; ctr_samples[i]=ctr_samples[j]; ctr_samples[j]=tmp; }
            if (cbc_samples[j] < cbc_samples[i]) { float tmp=cbc_samples[i]; cbc_samples[i]=cbc_samples[j]; cbc_samples[j]=tmp; }
            if (ecb_samples[j] < ecb_samples[i]) { float tmp=ecb_samples[i]; ecb_samples[i]=ecb_samples[j]; ecb_samples[j]=tmp; }
        }

    printf("\n--------Single 64-Byte Block Latency--------\n");
    printf("  Input size : %d cipher blocks = %d bytes\n", LAT_BLOCKS, LAT_BLOCKS * BLOCK_BYTES);
    printf("  Trials     : %d\n\n", LAT_TRIALS);
    printf("  %-35s : %8.2f us\n", "CTR encrypt latency", ctr_samples[LAT_TRIALS/2]);
    printf("  %-35s : %8.2f us\n", "CBC decrypt latency", cbc_samples[LAT_TRIALS/2]);
    printf("  %-35s : %8.2f us\n", "ECB decrypt latency", ecb_samples[LAT_TRIALS/2]);
}

// Key schedule cost

static void measure_key_schedule_cost(const uint8_t *key)
{
    const int KS_TRIALS = 101;

    uint8_t pt[BLOCK_BYTES] = {0xcf,0x16,0xcf,0xe8,0xfd,0x0f,0x98,0xaa,
                                0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00};
    uint8_t ct[BLOCK_BYTES];

    struct timespec t0, t1;
    double ks_samples[KS_TRIALS];
    for (int t = 0; t < KS_TRIALS; t++) {
        clock_gettime(CLOCK_MONOTONIC, &t0);
        gift128_encrypt_block(pt, key, ct);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        ks_samples[t] = elapsed_ms(t0, t1) * 1000.0;
    }
    qsort(ks_samples, KS_TRIALS, sizeof(double), cmp_dbl);
    double ks_med = ks_samples[KS_TRIALS/2];

    printf("\n--------Key Schedule Cost--------\n");
    printf("\n  Time (median of %d trials)\n", KS_TRIALS);
    printf("  %-48s : %7.2f us\n", "CPU compute  (gift128_precompute_round_keys)", ks_med);
    printf("  %-48s : %7.2f us\n", "GPU upload   (cudaMemcpy H->D, PCIe)",          0.0);
    printf("  %-48s : %7.2f us\n", "Total key setup cost",                           ks_med);
    printf("\n  Space\n");
    printf("  %-48s : %3zu bytes  (%d rounds x 32 nibble-bytes)\n",
           "Expanded round keys in RAM",
           (size_t)(GIFT_ROUNDS * 32), GIFT_ROUNDS);
}

//  Main

int main()
{
    run_gift128_core_test();
    run_ecb_test();
    run_cbc_test();
    run_ctr_test();

    printf("\n--------All tests have completed--------\n");

    uint8_t key[KEY_BYTES] = {
        0x9e,0xb9,0x36,0x40,0xd0,0x88,0xda,0x63,
        0x76,0xa3,0x9d,0x1c,0x8b,0xea,0x71,0xe1
    };
    uint8_t nonce[12]       = {0};
    uint8_t iv[BLOCK_BYTES] = {0};

    print_code_size();
    print_performance_metrics();
    throughput_sweep(key, nonce, iv);
    measure_single_block_latency(key, nonce, iv);
    measure_key_schedule_cost(key);

    return 0;
}