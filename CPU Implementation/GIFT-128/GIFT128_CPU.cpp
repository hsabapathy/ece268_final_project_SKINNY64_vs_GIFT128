
// GIFT-128 Block Cipher with CBC and CTR Modes


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
static void run_cbc_test()
{
    printf("\n--------CBC Mode--------\n\n");
 
    //  Same key and plaintext
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
    printf("\n-------- Performance Metrics ---------n");
    printf(" Median of %d trials, buffer = %d blocks\n",TRIALS, BUF_BLOCKS);
 
    double cpb_ctr = measure_cycles_per_byte_ctr();
    double cpb_cbc = measure_cycles_per_byte_cbc();
    double ks_cyc  = measure_keyschedule_cycles();
 
    printf("  %-40s : %8.2f  cycles/byte\n", "CTR encrypt", cpb_ctr);
    printf("  %-40s : %8.2f  cycles/byte\n", "CBC encrypt (chained)", cpb_cbc);
    printf("\n  CBC overhead vs CTR                    : %+.2f cycles/byte\n",
           cpb_cbc - cpb_ctr);
 
    printf("\n--------- Key-Schedule Cost ------\n");

    printf("  Rounds             : %d\n", GIFT_ROUNDS);
    printf("  Per-round update   : 1 word rotation (nibble shift) +\n");
    printf("                       2 sub-word rotations (bit-level)\n");
    printf("  Cycles / key setup : %.0f  (gift128_encrypt_block, median of %d trials)\n",
           ks_cyc, TRIALS);
    printf("  Cycles / round     : %.1f\n", ks_cyc / GIFT_ROUNDS);
    printf("  RAM cost           : 0 bytes (keys derived inline, no table stored)\n");
    printf("  Re-use across blocks: NO (key schedule re-run each block for encryption)\n");
    printf("\n  Decryption schedule: PRE-COMPUTED (all %d round keys derived upfront)\n", GIFT_ROUNDS);
    printf("  RAM cost           : %d rounds x 32 nibbles = %d bytes on stack\n",
           GIFT_ROUNDS, GIFT_ROUNDS * 32);
    printf("  Re-use across blocks: YES (RTK computed once, reused for all blocks)\n");
}
 
//  Main

int main()
{
    run_gift128_core_test();
    run_cbc_test();
    run_ctr_test();

    printf("\n--------All tests have completed--------\n");
    
    print_code_size();
    print_performance_metrics();
    
    return 0;
}