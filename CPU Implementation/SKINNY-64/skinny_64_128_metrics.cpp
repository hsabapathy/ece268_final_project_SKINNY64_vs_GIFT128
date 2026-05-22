#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <string>
#include <iostream>
#include <x86intrin.h> 

using namespace std;

#define SKINNY64_ROUNDS 36

// 4-bit Sbox 
const uint8_t S[16] = {0xc, 0x6, 0x9, 0x0, 0x1, 0xa, 0x2, 0xb, 0x3, 0x8, 0x5, 0xd, 0x4, 0xe, 0x7, 0xf};

// 4-bit Sbox inverse
const uint8_t S_inverse[16] = {0x3, 0x4, 0x6, 0x8, 0xc, 0xa, 0x1, 0xe, 0x9, 0x2, 0x5, 0x7, 0x0, 0xb, 0xd, 0xf};

// Permutation P[i] = x
const uint8_t P[16] = {0x0, 0x1, 0x2, 0x3, 0x7, 0x4, 0x5, 0x6, 0xa, 0xb, 0x8, 0x9, 0xd, 0xe, 0xf, 0xc};

// Inverse Permutation P_inverse[x] = i
const uint8_t P_inverse[16] = {0x0, 0x1, 0x2, 0x3, 0x5, 0x6, 0x7, 0x4, 0xa, 0xb, 0x8, 0x9, 0xf, 0xc, 0xd, 0xe};

// Tweakey Permutation PT
const uint8_t Q[16] = {0x9, 0xf, 0x8, 0xd, 0xa, 0xe, 0xc, 0xb, 0x0, 0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7};

// Round Constants (6-bit LFSR)
const uint8_t Round_Constant[36] = {
    0x01, 0x03, 0x07, 0x0F, 0x1F, 0x3E, 0x3D, 0x3B, 0x37, 0x2F, 0x1E, 0x3C,
    0x39, 0x33, 0x27, 0x0E, 0x1D, 0x3A, 0x35, 0x2B, 0x16, 0x2C, 0x18, 0x30,
    0x21, 0x02, 0x05, 0x0B, 0x17, 0x2E, 0x1C, 0x38, 0x31, 0x23, 0x06, 0x0D
};



uint8_t hex_char_to_val(char c)
{
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return 0;
}

void print_state(uint8_t state[16])
{
    for (int i = 0; i < 16; i++)
        printf("%01x", state[i]);
    printf("\n");
}

void print_message(uint8_t *msg, int num_blocks)
{
    for (int i = 0; i < num_blocks * 16; i++)
        printf("%01x", msg[i]);
    printf("\n");
}

void convert_hex_string_to_statearray(string hex_string, uint8_t int_array[16], bool reversed = false)
{
    for (size_t i = 0; i < 16; i++)
    {
        uint8_t val = hex_char_to_val(hex_string[i]);
        if (reversed)
            int_array[15 - i] = val;
        else
            int_array[i] = val;
    }
}

uint8_t TK2_lfsr(uint8_t x)
{
    x = (x << 1) ^ ((x >> 3) & 0x1) ^ ((x >> 2) & 0x1);
    x = x & 0xf;
    return x;
}


void mix_columns(uint8_t state[16])
{
    uint8_t tmp;
    for (uint8_t j = 0; j < 4; j++)
    {
        state[j + 4 * 1] ^= state[j + 4 * 2];
        state[j + 4 * 2] ^= state[j + 4 * 0];
        state[j + 4 * 3] ^= state[j + 4 * 2];
        tmp = state[j + 4 * 3];
        state[j + 4 * 3] = state[j + 4 * 2];
        state[j + 4 * 2] = state[j + 4 * 1];
        state[j + 4 * 1] = state[j + 4 * 0];
        state[j + 4 * 0] = tmp;
    }
}

void inverse_mix_columns(uint8_t state[16])
{
    uint8_t tmp;
    for (uint8_t j = 0; j < 4; j++)
    {
        tmp = state[j + 4 * 3];
        state[j + 4 * 3] = state[j + 4 * 0];
        state[j + 4 * 0] = state[j + 4 * 1];
        state[j + 4 * 1] = state[j + 4 * 2];
        state[j + 4 * 2] = tmp;
        state[j + 4 * 3] ^= state[j + 4 * 2];
        state[j + 4 * 2] ^= state[j + 4 * 0];
        state[j + 4 * 1] ^= state[j + 4 * 2];
    }
}

void round_tweakey_schedule(int rounds, uint8_t TK1[][16], uint8_t TK2[][16], uint8_t round_tweakey[][8])
{
    uint8_t TKp1[16];
    uint8_t TKp2[16];

    // Masking to 4 bits
    for (uint8_t i = 0; i < 16; i++)
    {
        TK1[0][i] = (TK1[0][i] & 0xf);
        TK2[0][i] = (TK2[0][i] & 0xf);
    }

    // Computing Round 0 tweakey RTK = TK1 ^ TK2
    for (uint8_t i = 0; i < 8; i++)
        round_tweakey[0][i] = TK1[0][i] ^ TK2[0][i];

    for (int r = 1; r < rounds; r++)
    {
        // Tweakey permutation of TK1 and TK2
        for (int i = 0; i < 16; i++)
        {
            TKp1[i] = TK1[r - 1][Q[i]];
            TKp2[i] = TK2[r - 1][Q[i]];
        }

        // Apply LFSR2 to first two rows of permuted TK2
        for (int i = 0; i < 16; i++)
        {
            TK1[r][i] = TKp1[i];

            if (i < 8)
                TK2[r][i] = TK2_lfsr(TKp2[i]);
            else
                TK2[r][i] = TKp2[i];
        }

        // Compute round tweakey
        for (int i = 0; i < 8; i++)
            round_tweakey[r][i] = (TK1[r][i] ^ TK2[r][i]);
    }
}

//Encryption main block which is called in CTR and CBC modes
void encryption_block(int R, uint8_t in[16], uint8_t out[16], uint8_t TK[][8])
{
    for (uint8_t i = 0; i < 16; i++)
        out[i] = in[i] & 0xf;

    for (uint8_t r = 0; r < R; r++)
    {
        // S-box substitution
        for (uint8_t i = 0; i < 16; i++)
            out[i] = S[out[i]];

        // Add round constants
        out[0] ^= (Round_Constant[r] & 0xf);
        out[4] ^= ((Round_Constant[r] >> 4) & 0x3);
        out[8] ^= 0x2;

        // Add round tweakey
        for (uint8_t i = 0; i < 8; i++)
            out[i] ^= TK[r][i];

        // Permutation (ShiftRows)
        uint8_t temp[16];
        for (uint8_t i = 0; i < 16; i++)
            temp[i] = out[i];
        for (uint8_t i = 0; i < 16; i++)
            out[i] = temp[P[i]];

        // MixColumns
        mix_columns(out);
    }
}

//Decryption main block which is called in CTR and CBC modes
void decryption_block(int R, uint8_t in[16], uint8_t out[16], uint8_t TK[][8])
{
    for (uint8_t i = 0; i < 16; i++)
        out[i] = in[i] & 0xf;

    for (int r = 0; r < R; r++)
    {
        inverse_mix_columns(out);

        uint8_t temp[16];
        for (uint8_t i = 0; i < 16; i++)
            temp[i] = out[i];
        for (uint8_t i = 0; i < 16; i++)
            out[i] = temp[P_inverse[i]];

        int index = R - r - 1;
        // Remove tweakey
        for (uint8_t i = 0; i < 8; i++)
            out[i] ^= TK[index][i];

        // Remove round constants
        out[0] ^= (Round_Constant[index] & 0xf);
        out[4] ^= ((Round_Constant[index] >> 4) & 0x3);
        out[8] ^= 0x2;

        // Inverse S-box
        for (uint8_t i = 0; i < 16; i++)
            out[i] = S_inverse[out[i]];
    }
}

// Increment counter ; nibble wise
void increment_counter(uint8_t counter[16])
{
    for (int i = 15; i >= 0; i--)
    {
        counter[i] = (counter[i] + 1) & 0xf;
        if (counter[i] != 0) break;
    }
}

//CTR mode encryption
void ctr_encrypt(int R, uint8_t *plaintext, uint8_t *ciphertext,
                 int num_blocks, uint8_t nonce[16], uint8_t TK[][8])
{
    uint8_t counter[16];
    uint8_t keystream[16];

    // Initialize counter with nonce
    for (int i = 0; i < 16; i++)
        counter[i] = nonce[i];

    for (int b = 0; b < num_blocks; b++)
    {
        // Encrypt counter block to produce keystream
        encryption_block(R, counter, keystream, TK);

        // XOR plaintext nibbles with keystream
        for (int i = 0; i < 16; i++)
            ciphertext[b * 16 + i] = (plaintext[b * 16 + i] ^ keystream[i]) & 0xf;

        // Increment counter for next block
        increment_counter(counter);
    }
}

// CTR mode decryption
void ctr_decrypt(int R, uint8_t *ciphertext, uint8_t *plaintext,
                 int num_blocks, uint8_t nonce[16], uint8_t TK[][8])
{
    ctr_encrypt(R, ciphertext, plaintext, num_blocks, nonce, TK);
}

//CBC mode encryption
void cbc_encrypt(int R, uint8_t *plaintext, uint8_t *ciphertext,
                 int num_blocks, uint8_t iv[16], uint8_t TK[][8])
{
    uint8_t prev[16];
    uint8_t block_in[16];

    // Initialize with IV
    for (int i = 0; i < 16; i++)
        prev[i] = iv[i];

    for (int b = 0; b < num_blocks; b++)
    {
        // XOR plaintext with previous ciphertext
        for (int i = 0; i < 16; i++)
            block_in[i] = (plaintext[b * 16 + i] ^ prev[i]) & 0xf;

        // Encrypt the XOR-ed block
        encryption_block(R, block_in, &ciphertext[b * 16], TK);

        // Update chaining value
        for (int i = 0; i < 16; i++)
            prev[i] = ciphertext[b * 16 + i];
    }
}

// CBC mode decryption
void cbc_decrypt(int R, uint8_t *ciphertext, uint8_t *plaintext,
                 int num_blocks, uint8_t iv[16], uint8_t TK[][8])
{
    uint8_t prev[16];
    uint8_t block_out[16];
    uint8_t saved_ct[16];

    // Initialize with IV
    for (int i = 0; i < 16; i++)
        prev[i] = iv[i];

    for (int b = 0; b < num_blocks; b++)
    {
        // Saving the ciphertext block before decryption
        for (int i = 0; i < 16; i++)
            saved_ct[i] = ciphertext[b * 16 + i];

        // Decrypting the ciphertext block
        decryption_block(R, &ciphertext[b * 16], block_out, TK);

        // XOR with previous ciphertext (or IV) to recover plaintext
        for (int i = 0; i < 16; i++)
            plaintext[b * 16 + i] = (block_out[i] ^ prev[i]) & 0xf;

        // Update chaining value
        for (int i = 0; i < 16; i++)
            prev[i] = saved_ct[i];
    }
}

//Validating the results
int assert_equal(const char *label, uint8_t *got, uint8_t *expected, int len)
{
    if (memcmp(got, expected, len) == 0)
    {
        printf("The recovered plaintext matches the original plaintext %s\n", label);
        return 1;
    }
    printf("The recovered plaintext doesnt match the original plaintext %s\n       got:      ", label);
    for (int i = 0; i < len; i++) printf("%01x", got[i]);
    printf("\n       expected: ");
    for (int i = 0; i < len; i++) printf("%01x", expected[i]);
    printf("\n");
    return 0;
}

// Using a rdtsc helper inorder to determine the number of cycles
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


#define BUF_BLOCKS 256      //since 256 blocks fits in L1 cache
#define TRIALS     31       //we are using odd number so that median is unambiguous

//Finding number of cycles needed to encrypt 1 byte in CTR
static double measure_cycles_per_byte_ctr(int R, uint8_t RTK[][8])
{
    static uint8_t buf[BUF_BLOCKS * 16];
    static uint8_t out[BUF_BLOCKS * 16];
    uint8_t nonce[16] = {0};
    for (int i = 0; i < BUF_BLOCKS * 16; i++) buf[i] = (uint8_t)(i & 0xf);

    
    ctr_encrypt(R, buf, out, BUF_BLOCKS, nonce, RTK);

    uint64_t samples[TRIALS];
    for (int t = 0; t < TRIALS; t++) {
        uint8_t n[16] = {0};
        uint64_t t0 = rdtsc();
        ctr_encrypt(R, buf, out, BUF_BLOCKS, n, RTK);
        uint64_t t1 = rdtsc();
        samples[t] = t1 - t0;
    }
    qsort(samples, TRIALS, sizeof(uint64_t), cmp_u64);
    uint64_t median_cycles = samples[TRIALS / 2];
    
    return (double)median_cycles / (BUF_BLOCKS * 8);
}
//Finding number of cycles needed to encrypt 1 byte in CTR
static double measure_cycles_per_byte_cbc(int R, uint8_t RTK[][8])
{
    static uint8_t buf[BUF_BLOCKS * 16];
    static uint8_t out[BUF_BLOCKS * 16];
    uint8_t iv[16] = {0};
    for (int i = 0; i < BUF_BLOCKS * 16; i++) buf[i] = (uint8_t)(i & 0xf);

    cbc_encrypt(R, buf, out, BUF_BLOCKS, iv, RTK);

    uint64_t samples[TRIALS];
    for (int t = 0; t < TRIALS; t++) {
        uint8_t v[16] = {0};
        uint64_t t0 = rdtsc();
        cbc_encrypt(R, buf, out, BUF_BLOCKS, v, RTK);
        uint64_t t1 = rdtsc();
        samples[t] = t1 - t0;
    }
    qsort(samples, TRIALS, sizeof(uint64_t), cmp_u64);
    uint64_t median_cycles = samples[TRIALS / 2];
    return (double)median_cycles / (BUF_BLOCKS * 8);
}

// Finding key schedule cost
static double measure_keyschedule_cycles(int R, uint8_t tweakey_1[16], uint8_t tweakey_2[16])
{

    uint8_t TK_1[SKINNY64_ROUNDS][16];
    uint8_t TK_2[SKINNY64_ROUNDS][16];
    uint8_t RTK [SKINNY64_ROUNDS][8];

    for (int i = 0; i < 100; i++) {
        for (int j = 0; j < 16; j++) { TK_1[0][j] = tweakey_1[j]; TK_2[0][j] = tweakey_2[j]; }
        round_tweakey_schedule(R, TK_1, TK_2, RTK);
    }

    uint64_t samples[TRIALS];
    for (int t = 0; t < TRIALS; t++) {
        for (int j = 0; j < 16; j++) { TK_1[0][j] = tweakey_1[j]; TK_2[0][j] = tweakey_2[j]; }
        uint64_t t0 = rdtsc();
        round_tweakey_schedule(R, TK_1, TK_2, RTK);
        uint64_t t1 = rdtsc();
        samples[t] = t1 - t0;
    }
    qsort(samples, TRIALS, sizeof(uint64_t), cmp_u64);
    return (double)samples[TRIALS / 2];   
}

// Printing the code size
static void print_code_size()
{
    printf("\n--------Code Size--------\n");
  

    struct TableSize { const char *name; size_t bytes; };
    TableSize tables[] = {
        { "S          (4-bit Sbox, 16 entries)",          sizeof(S)             },
        { "S_inverse  (4-bit Sbox inverse, 16 entries)",  sizeof(S_inverse)     },
        { "P          (ShiftRows permutation, 16 entries)",sizeof(P)             },
        { "P_inverse  (inverse permutation, 16 entries)", sizeof(P_inverse)     },
        { "Q          (tweakey permutation, 16 entries)",  sizeof(Q)             },
        { "Round_Constant (6-bit LFSR, 36 entries)",       sizeof(Round_Constant)},
    };
    size_t total = 0;
    int ntables = sizeof(tables) / sizeof(tables[0]);
    for (int i = 0; i < ntables; i++) {
        printf("  %-50s : %3zu bytes\n", tables[i].name, tables[i].bytes);
        total += tables[i].bytes;
    }
    
    printf("  %-50s : %3zu bytes\n", "Total table footprint", total);

}

static void print_performance_metrics(int R, uint8_t RTK[][8],
                                      uint8_t tweakey_1[16], uint8_t tweakey_2[16])
{
    printf("\n--------Performance Metrics--------\n");
    printf("  Median of %d trials, Buffer = %d blocks\n",
           TRIALS, BUF_BLOCKS);

    double cpb_ctr = measure_cycles_per_byte_ctr(R, RTK);
    double cpb_cbc = measure_cycles_per_byte_cbc(R, RTK);
    double ks_cyc  = measure_keyschedule_cycles(R, tweakey_1, tweakey_2);

    printf("  %-40s : %8.2f  cycles/byte\n", "CTR encrypt", cpb_ctr);
    printf("  %-40s : %8.2f  cycles/byte\n", "CBC encrypt (chained)", cpb_cbc);
    printf("\n  CBC overhead vs CTR                    : %+.2f cycles/byte\n",
           cpb_cbc - cpb_ctr);

    printf("\n--------Key Schedule Cost--------\n");
    printf("  Cycles / key setup : %.0f  (round_tweakey_schedule, median of %d trials)\n",ks_cyc, TRIALS);
    printf("  Cycles / round     : %.1f\n", ks_cyc / R);
    printf("  RAM cost           : %d rounds x 8 nibble-bytes = %d bytes \n", R, R * 8);

}

int main()
{
   
    printf("--------SKINNY-64-128 Implementation--------\n");
    printf("--------CTR and CBC Mode Tests--------\n");
   
    int R = SKINNY64_ROUNDS;

    uint8_t TK_1[R][16];
    uint8_t TK_2[R][16];
    uint8_t RTK[R][8];
    uint8_t tweakey_1[16];
    uint8_t tweakey_2[16];

    
    //Test vectors
    string TK1_string = "9eb93640d088da63";
    string TK2_string = "76a39d1c8bea71e1";

    convert_hex_string_to_statearray(TK1_string, tweakey_1, false);
    convert_hex_string_to_statearray(TK2_string, tweakey_2, false);

    for (uint8_t i = 0; i < 16; i++)
    {
        TK_1[0][i] = tweakey_1[i];
        TK_2[0][i] = tweakey_2[i];
    }

    printf("Generating tweakey schedule\n");
    round_tweakey_schedule(R, TK_1, TK_2, RTK);
    printf("Tweakey schedule generated\n\n");

    //CTR mode
    printf("--------CTR Mode--------\n\n");

    int num_blocks = 1;
    uint8_t ctr_plain[16];
    uint8_t ctr_cipher[16];
    uint8_t ctr_recovered[16];

    string plain_string = "cf16cfe8fd0f98aa";
    uint8_t plain_text_0[16];
    convert_hex_string_to_statearray(plain_string, plain_text_0, false);

    for (int i = 0; i < 16; i++) ctr_plain[i] = plain_text_0[i];

    uint8_t nonce[16] = {0};

    printf("Original Plaintext   : ");
    print_message(ctr_plain, num_blocks);

    ctr_encrypt(R, ctr_plain, ctr_cipher, num_blocks, nonce, RTK);

    printf("CTR Ciphertext       : ");
    print_message(ctr_cipher, num_blocks);

    ctr_decrypt(R, ctr_cipher, ctr_recovered, num_blocks, nonce, RTK);

    printf("Recovered Plaintext  : ");
    print_message(ctr_recovered, num_blocks);

    printf("\n");
    assert_equal("\nCTR mode is validated",
                 ctr_recovered, ctr_plain, num_blocks * 16);

    // CBC mode
    printf("\n--------CBC Mode--------\n\n");

    uint8_t cbc_plain[16];
    uint8_t cbc_cipher[16];
    uint8_t cbc_recovered[16];

   
    for (int i = 0; i < 16; i++) cbc_plain[i] = plain_text_0[i];

    // IV is set to all zeros
    uint8_t iv[16] = {0};

    printf("Original Plaintext   : ");
    print_message(cbc_plain, num_blocks);

    cbc_encrypt(R, cbc_plain, cbc_cipher, num_blocks, iv, RTK);

    printf("CBC Ciphertext       : ");
    print_message(cbc_cipher, num_blocks);

    cbc_decrypt(R, cbc_cipher, cbc_recovered, num_blocks, iv, RTK);

    printf("Recovered Plaintext  : ");
    print_message(cbc_recovered, num_blocks);

    printf("\n");
    assert_equal("\nCBC mode is validated ",
                 cbc_recovered, cbc_plain, num_blocks * 16);

   
    printf("--------All tests have completed--------\n");

    print_code_size();
    print_performance_metrics(R, RTK, tweakey_1, tweakey_2);

    return 0;
}
