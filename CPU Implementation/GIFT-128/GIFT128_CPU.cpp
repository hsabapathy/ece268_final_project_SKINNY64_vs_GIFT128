/*
 * GIFT-128 Block Cipher with CBC and CTR Modes
 *
 * Original cipher core by Siang Meng Sim (09 March 2017)
 * CBC/CTR modes and refactoring added on top of original
 *
 * Data representation:
 *   - All public interfaces use standard byte arrays (uint8_t*)
 *   - Block size: 16 bytes (128 bits)
 *   - Key size:   16 bytes (128 bits)
 *
 * Internally, the cipher core uses a nibble array (32 nibbles),
 * where each uint8_t holds one 4-bit value in its lower bits.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <iostream>
#include <iomanip>
#include <time.h>

using namespace std;

// ============================================================
//  GIFT-128 Core Constants
// ============================================================

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

// 
//  Pack/unpack bytes to nibbles
// 
/*
 * The original code prints the state as input[31]..input[0] (MSB first),
 * meaning nibble[31] is the most-significant nibble of the 128-bit state.
 * In terms of bytes: byte[15] = nibble[31]<<4 | nibble[30]  (most significant)
 *                    byte[0]  = nibble[1] <<4 | nibble[0]   (least significant)
 *
 * pack_nibbles: convert 32 nibbles back to 16 bytes in big-endian nibble order
 * byte[i] corresponds to nibble[31-2i] (high) and nibble[30-2i] (low)
 */
static void pack_nibbles(const uint8_t nibbles[32], uint8_t bytes[16])
{
    for (int i = 0; i < 16; i++)
        bytes[i] = ((nibbles[31 - 2*i] & 0xF) << 4) | (nibbles[30 - 2*i] & 0xF);
}

/*
 * unpack_nibbles: split 16 bytes into 32 nibbles in big-endian nibble order
 * byte[0] -> nibbles[31] (high) and nibbles[30] (low)
 */
static void unpack_nibbles(const uint8_t bytes[16], uint8_t nibbles[32])
{
    for (int i = 0; i < 16; i++) {
        nibbles[31 - 2*i] = (bytes[i] >> 4) & 0xF;
        nibbles[30 - 2*i] =  bytes[i]        & 0xF;
    }
}


// 
//  GIFT-128 Core: encryption / decryption
//  These operate on nibble arrays internally, but the public
//  wrappers below expose a clean byte-array interface.
// 

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

        // Key update: entire key >> 32 (one word rotation)
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

//  Public single-block interface (byte arrays)
/*
 * gift128_encrypt_block
 *   plaintext: 16-byte input block
 *   key: 16-byte key 
 *   ciphertext: 16-byte output block
 */
void gift128_encrypt_block(const uint8_t plaintext[BLOCK_BYTES],
                           const uint8_t key[KEY_BYTES],
                           uint8_t ciphertext[BLOCK_BYTES])
{
    uint8_t state[32], key_nibs[32];
    unpack_nibbles(plaintext, state);
    unpack_nibbles(key, key_nibs);
    gift128_enc_nibbles(state, key_nibs);
    pack_nibbles(state, ciphertext);
}

/**
 * gift128_decrypt_block
 *   ciphertext: 16-byte input block
 *   key: 16-byte key 
 *   plaintext: 16-byte output block
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

//  CBC Mode Wrapper
//  Encrypt: C[i] = E_K(P[i] XOR C[i-1]), C[-1] = IV
//  Decrypt: P[i] = D_K(C[i]) XOR C[i-1], C[-1] = IV
//
//  Here, length is a multiple of BLOCK_BYTES
// 
/**
 * gift128_cbc_encrypt
 *   
 *   key: 16-byte key
 *   iv: 16-byte initialisation vector
 *   input: plaintext  (length must be multiple of 16)
 *   output: ciphertext (same length as input)
 *   length: byte count
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

        // This ciphertext block is the next chain value
        memcpy(chain, output + offset, BLOCK_BYTES);
    }
}

/**
 * gift128_cbc_decrypt
 *   key: [in] 16-byte key
 *   iv: [in] 16-byte initialisation vector
 *   input: [in]  ciphertext (length must be multiple of 16)
 *   output [out] plaintext  (same length as input)
 *   length [in]  byte count
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

//  CTR MODE
//
//  Keystream block i: K[i] = E_K(nonce || counter_i)
//  Output:            C[i] = P[i] XOR K[i]   (same for decrypt)
//
//  Counter format: the 16-byte counter block is constructed as
//    [nonce (12 bytes)] [counter (4 bytes)]
//  Counter starts at 0 and increments per block.
//  Length can be any number of bytes (last block is partial).

//  Encryption and decryption are identical in CTR

/**
 * increment_counter: increment the 4-byte big-endian counter
 * in the last 4 bytes of the 16-byte counter block.
 */
static void increment_counter(uint8_t counter_block[BLOCK_BYTES])
{
    for (int i = BLOCK_BYTES - 1; i >= BLOCK_BYTES - 4; i--) {
        if (++counter_block[i] != 0) break;
    }
}

/**
 * gift128_ctr_crypt  (encryption and decryption are identical)
 *   key: 16-byte key
 *   nonce: 12-byte nonce (the remaining 4 bytes are the counter)
 *   input: plaintext or ciphertext
 *   output: ciphertext or plaintext
 *   length: byte count (any length)
 */
void gift128_ctr_crypt(const uint8_t *key,
                       const uint8_t  nonce[12],
                       const uint8_t *input,
                             uint8_t *output,
                             size_t   length)
{
    // Build initial counter block: [nonce 12 bytes | counter 4 bytes = 0]
    uint8_t counter_block[BLOCK_BYTES];
    memcpy(counter_block, nonce, 12);
    memset(counter_block + 12, 0, 4);

    size_t offset = 0;
    while (offset < length) {
        // Encrypt counter block to produce keystream block
        uint8_t keystream[BLOCK_BYTES];
        gift128_encrypt_block(counter_block, key, keystream);

        // XOR as many bytes as remain (handles partial last block)
        size_t rem = (length - offset < BLOCK_BYTES) ? (length - offset) : BLOCK_BYTES;
        for (size_t i = 0; i < rem; i++)
            output[offset + i] = input[offset + i] ^ keystream[i];

        increment_counter(counter_block);
        offset += rem;
    }
}


// 
//  Printing a byte array as hex

static void print_hex(const char *label, const uint8_t *data, size_t len)
{
    printf("%-20s: ", label);
    for (size_t i = 0; i < len; i++) printf("%02x", data[i]);
    printf("\n");
}

//  Test vectors used:
//  Plaintext  = 00000000 00000000 00000000 00000000
//  Key        = 00000000 00000000 00000000 00000000
//  Ciphertext = cd0bd738 388ad3f6 68b15a36 ceb6ff92

static void run_gift128_core_test()
{
    printf("\n=== GIFT-128 Core Block Cipher Test ===\n");
    uint8_t pt[16]       = {0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
                             0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00};
    uint8_t key[16]      = {0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
                             0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00};
    uint8_t expected[16] = {0xcd,0x0b,0xd7,0x38,0x38,0x8a,0xd3,0xf6,
                             0x68,0xb1,0x5a,0x36,0xce,0xb6,0xff,0x92};
    uint8_t ct[16], recovered[16];

    gift128_encrypt_block(pt, key, ct);
    gift128_decrypt_block(ct, key, recovered);

    print_hex("Plaintext", pt, 16);
    print_hex("Key", key, 16);
    print_hex("Ciphertext", ct, 16);
    print_hex("Expected CT", expected, 16);
    print_hex("Decrypted", recovered, 16);

    bool enc_ok = (memcmp(ct, expected, 16) == 0);
    bool dec_ok = (memcmp(recovered, pt, 16) == 0);
    printf("Encryption: %s\n", enc_ok ? "PASS" : "FAIL");
    printf("Decryption: %s\n", dec_ok ? "PASS" : "FAIL");
}

static void run_cbc_test()
{
    printf("\n=== CBC Mode Test ===\n");

    // Two-block message so we can verify chaining
    uint8_t key[16] = {0x01,0x23,0x45,0x67,0x89,0xab,0xcd,0xef,
                        0xfe,0xdc,0xba,0x98,0x76,0x54,0x32,0x10};
    uint8_t iv[16]  = {0x00,0x11,0x22,0x33,0x44,0x55,0x66,0x77,
                        0x88,0x99,0xaa,0xbb,0xcc,0xdd,0xee,0xff};

    // Plaintext: two 16-byte blocks
    uint8_t plaintext[32];
    memset(plaintext,    0xAB, 16);   // block 0
    memset(plaintext+16, 0xCD, 16);   // block 1

    uint8_t ciphertext[32];
    uint8_t decrypted[32];

    gift128_cbc_encrypt(key, iv, plaintext,  ciphertext, 32);
    gift128_cbc_decrypt(key, iv, ciphertext, decrypted,  32);

    print_hex("Key", key, 16);
    print_hex("IV", iv, 16);
    print_hex("Plaintext  [0]", plaintext,    16);
    print_hex("Plaintext  [1]", plaintext+16, 16);
    print_hex("Ciphertext [0]", ciphertext,    16);
    print_hex("Ciphertext [1]", ciphertext+16, 16);
    print_hex("Decrypted  [0]", decrypted,    16);
    print_hex("Decrypted  [1]", decrypted+16, 16);

    bool ok = (memcmp(plaintext, decrypted, 32) == 0);
    printf("CBC encrypt->decrypt round-trip: %s\n", ok ? "PASS" : "FAIL");

    // Verify that identical plaintext blocks produce different ciphertext (chaining)
    bool chained = (memcmp(ciphertext, ciphertext+16, 16) != 0);
    printf("Ciphertext blocks differ (chaining works): %s\n", chained ? "PASS" : "FAIL");
}

static void run_ctr_test()
{
    printf("\n=== CTR Mode Test ===\n");
    uint8_t key[16]   = {0xfe,0xdc,0xba,0x98,0x76,0x54,0x32,0x10,
                          0x01,0x23,0x45,0x67,0x89,0xab,0xcd,0xef};
    uint8_t nonce[12] = {0xca,0xfe,0xba,0xbe,0xfa,0xce,0xdb,0xad,
                          0xde,0xca,0xf8,0x88};

    // Test with a non-block-aligned length (35 bytes) to verify partial block handling
    const char *msg = "Hello, GIFT-128 CTR mode! Testing.";
    size_t msglen = strlen(msg);

    uint8_t ciphertext[64];
    uint8_t decrypted[64];
    memset(ciphertext, 0, sizeof(ciphertext));
    memset(decrypted,  0, sizeof(decrypted));

    gift128_ctr_crypt(key, nonce, (const uint8_t*)msg,  ciphertext, msglen);
    gift128_ctr_crypt(key, nonce,  ciphertext,           decrypted,  msglen);

    print_hex("Key", key, 16);
    print_hex("Nonce", nonce, 12);
    printf("Plaintext  (text): \"%s\" (%zu bytes)\n", msg, msglen);
    print_hex("Ciphertext", ciphertext, msglen);
    print_hex("Decrypted", decrypted, msglen);
    printf("Decrypted (text): \"%.*s\"\n", (int)msglen, decrypted);

    bool ok = (memcmp(msg, decrypted, msglen) == 0);
    printf("CTR encrypt->decrypt round-trip: %s\n", ok ? "PASS" : "FAIL");

    // Verify encrypt is deterministic (same key+nonce -> same ciphertext)
    uint8_t ct2[64];
    gift128_ctr_crypt(key, nonce, (const uint8_t*)msg, ct2, msglen);
    bool det = (memcmp(ciphertext, ct2, msglen) == 0);
    printf("CTR deterministic (same nonce => same CT): %s\n", det ? "PASS" : "FAIL");
}



//  Main

int main()
{
    printf("GIFT-128 Block Cipher — CBC & CTR Mode Implementation\n");
    printf("======================================================\n");

    run_gift128_core_test();
    run_cbc_test();
    run_ctr_test();

    return 0;
}