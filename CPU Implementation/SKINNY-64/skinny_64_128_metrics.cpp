#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <string>
#include <iostream>
#include <time.h>
#include <x86intrin.h>

using namespace std;

#define SKINNY64_ROUNDS 36
#define TRIALS          31
#define BUF_BLOCKS      256

const uint8_t S[16]         = {0xc,0x6,0x9,0x0,0x1,0xa,0x2,0xb,0x3,0x8,0x5,0xd,0x4,0xe,0x7,0xf};
const uint8_t S_inverse[16] = {0x3,0x4,0x6,0x8,0xc,0xa,0x1,0xe,0x9,0x2,0x5,0x7,0x0,0xb,0xd,0xf};
const uint8_t P[16]         = {0x0,0x1,0x2,0x3,0x7,0x4,0x5,0x6,0xa,0xb,0x8,0x9,0xd,0xe,0xf,0xc};
const uint8_t P_inverse[16] = {0x0,0x1,0x2,0x3,0x5,0x6,0x7,0x4,0xa,0xb,0x8,0x9,0xf,0xc,0xd,0xe};
const uint8_t Q[16]         = {0x9,0xf,0x8,0xd,0xa,0xe,0xc,0xb,0x0,0x1,0x2,0x3,0x4,0x5,0x6,0x7};
const uint8_t Round_Constant[36] = {
    0x01,0x03,0x07,0x0F,0x1F,0x3E,0x3D,0x3B,0x37,0x2F,0x1E,0x3C,
    0x39,0x33,0x27,0x0E,0x1D,0x3A,0x35,0x2B,0x16,0x2C,0x18,0x30,
    0x21,0x02,0x05,0x0B,0x17,0x2E,0x1C,0x38,0x31,0x23,0x06,0x0D
};


uint8_t hex_char_to_val(char c)
{
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return 0;
}

void print_message(uint8_t *msg, int num_blocks)
{
    for (int i = 0; i < num_blocks * 16; i++) printf("%01x", msg[i]);
    printf("\n");
}

void convert_hex_string_to_statearray(string hex_string, uint8_t int_array[16],
                                      bool reversed = false)
{
    for (size_t i = 0; i < 16; i++) {
        uint8_t val = hex_char_to_val(hex_string[i]);
        if (reversed) int_array[15 - i] = val;
        else          int_array[i]       = val;
    }
}

uint8_t TK2_lfsr(uint8_t x)
{
    x = (x << 1) ^ ((x >> 3) & 0x1) ^ ((x >> 2) & 0x1);
    return x & 0xf;
}

void mix_columns(uint8_t state[16])
{
    uint8_t tmp;
    for (uint8_t j = 0; j < 4; j++) {
        state[j+4*1] ^= state[j+4*2];
        state[j+4*2] ^= state[j+4*0];
        state[j+4*3] ^= state[j+4*2];
        tmp            = state[j+4*3];
        state[j+4*3]   = state[j+4*2];
        state[j+4*2]   = state[j+4*1];
        state[j+4*1]   = state[j+4*0];
        state[j+4*0]   = tmp;
    }
}

void inverse_mix_columns(uint8_t state[16])
{
    uint8_t tmp;
    for (uint8_t j = 0; j < 4; j++) {
        tmp            = state[j+4*3];
        state[j+4*3]   = state[j+4*0];
        state[j+4*0]   = state[j+4*1];
        state[j+4*1]   = state[j+4*2];
        state[j+4*2]   = tmp;
        state[j+4*3]  ^= state[j+4*2];
        state[j+4*2]  ^= state[j+4*0];
        state[j+4*1]  ^= state[j+4*2];
    }
}

void round_tweakey_schedule(int rounds, uint8_t TK1[][16], uint8_t TK2[][16],
                             uint8_t round_tweakey[][8])
{
    uint8_t TKp1[16], TKp2[16];
    for (uint8_t i = 0; i < 16; i++) {
        TK1[0][i] = TK1[0][i] & 0xf;
        TK2[0][i] = TK2[0][i] & 0xf;
    }
    for (uint8_t i = 0; i < 8; i++)
        round_tweakey[0][i] = TK1[0][i] ^ TK2[0][i];
    for (int r = 1; r < rounds; r++) {
        for (int i = 0; i < 16; i++) { TKp1[i] = TK1[r-1][Q[i]]; TKp2[i] = TK2[r-1][Q[i]]; }
        for (int i = 0; i < 16; i++) {
            TK1[r][i] = TKp1[i];
            TK2[r][i] = (i < 8) ? TK2_lfsr(TKp2[i]) : TKp2[i];
        }
        for (int i = 0; i < 8; i++)
            round_tweakey[r][i] = TK1[r][i] ^ TK2[r][i];
    }
}

void encryption_block(int R, uint8_t in[16], uint8_t out[16], uint8_t TK[][8])
{
    for (uint8_t i = 0; i < 16; i++) out[i] = in[i] & 0xf;
    for (uint8_t r = 0; r < R; r++) {
        for (uint8_t i = 0; i < 16; i++) out[i] = S[out[i]];
        out[0] ^= (Round_Constant[r] & 0xf);
        out[4] ^= ((Round_Constant[r] >> 4) & 0x3);
        out[8] ^= 0x2;
        for (uint8_t i = 0; i < 8; i++) out[i] ^= TK[r][i];
        uint8_t temp[16];
        for (uint8_t i = 0; i < 16; i++) temp[i] = out[i];
        for (uint8_t i = 0; i < 16; i++) out[i] = temp[P[i]];
        mix_columns(out);
    }
}

void decryption_block(int R, uint8_t in[16], uint8_t out[16], uint8_t TK[][8])
{
    for (uint8_t i = 0; i < 16; i++) out[i] = in[i] & 0xf;
    for (int r = 0; r < R; r++) {
        inverse_mix_columns(out);
        uint8_t temp[16];
        for (uint8_t i = 0; i < 16; i++) temp[i] = out[i];
        for (uint8_t i = 0; i < 16; i++) out[i] = temp[P_inverse[i]];
        int index = R - r - 1;
        for (uint8_t i = 0; i < 8; i++) out[i] ^= TK[index][i];
        out[0] ^= (Round_Constant[index] & 0xf);
        out[4] ^= ((Round_Constant[index] >> 4) & 0x3);
        out[8] ^= 0x2;
        for (uint8_t i = 0; i < 16; i++) out[i] = S_inverse[out[i]];
    }
}

//---ECB mode---
void ecb_encrypt(int R, uint8_t *plaintext, uint8_t *ciphertext,
                 int num_blocks, uint8_t TK[][8])
{
    for (int b = 0; b < num_blocks; b++)
        encryption_block(R, &plaintext[b*16], &ciphertext[b*16], TK);
}

void ecb_decrypt(int R, uint8_t *ciphertext, uint8_t *plaintext,
                 int num_blocks, uint8_t TK[][8])
{
    for (int b = 0; b < num_blocks; b++)
        decryption_block(R, &ciphertext[b*16], &plaintext[b*16], TK);
}

//---CTR mode---
void increment_counter(uint8_t counter[16])
{
    for (int i = 15; i >= 0; i--) {
        counter[i] = (counter[i] + 1) & 0xf;
        if (counter[i] != 0) break;
    }
}

void ctr_encrypt(int R, uint8_t *plaintext, uint8_t *ciphertext,
                 int num_blocks, uint8_t nonce[16], uint8_t TK[][8])
{
    uint8_t counter[16], keystream[16];
    for (int i = 0; i < 16; i++) counter[i] = nonce[i];
    for (int b = 0; b < num_blocks; b++) {
        encryption_block(R, counter, keystream, TK);
        for (int i = 0; i < 16; i++)
            ciphertext[b*16+i] = (plaintext[b*16+i] ^ keystream[i]) & 0xf;
        increment_counter(counter);
    }
}

void ctr_decrypt(int R, uint8_t *ciphertext, uint8_t *plaintext,
                 int num_blocks, uint8_t nonce[16], uint8_t TK[][8])
{
    ctr_encrypt(R, ciphertext, plaintext, num_blocks, nonce, TK);
}

//---CBC mode---
void cbc_encrypt(int R, uint8_t *plaintext, uint8_t *ciphertext,
                 int num_blocks, uint8_t iv[16], uint8_t TK[][8])
{
    uint8_t prev[16], block_in[16];
    for (int i = 0; i < 16; i++) prev[i] = iv[i];
    for (int b = 0; b < num_blocks; b++) {
        for (int i = 0; i < 16; i++)
            block_in[i] = (plaintext[b*16+i] ^ prev[i]) & 0xf;
        encryption_block(R, block_in, &ciphertext[b*16], TK);
        for (int i = 0; i < 16; i++) prev[i] = ciphertext[b*16+i];
    }
}

void cbc_decrypt(int R, uint8_t *ciphertext, uint8_t *plaintext,
                 int num_blocks, uint8_t iv[16], uint8_t TK[][8])
{
    uint8_t prev[16], block_out[16], saved_ct[16];
    for (int i = 0; i < 16; i++) prev[i] = iv[i];
    for (int b = 0; b < num_blocks; b++) {
        for (int i = 0; i < 16; i++) saved_ct[i] = ciphertext[b*16+i];
        decryption_block(R, &ciphertext[b*16], block_out, TK);
        for (int i = 0; i < 16; i++)
            plaintext[b*16+i] = (block_out[i] ^ prev[i]) & 0xf;
        for (int i = 0; i < 16; i++) prev[i] = saved_ct[i];
    }
}

//---Validation---
int assert_equal(const char *label, uint8_t *got, uint8_t *expected, int len)
{
    if (memcmp(got, expected, len) == 0) {
        printf("The recovered plaintext matches the original plaintext %s\n", label);
        return 1;
    }
    printf("The recovered plaintext doesnt match the original plaintext %s\n  got:      ", label);
    for (int i = 0; i < len; i++) printf("%01x", got[i]);
    printf("\n  expected: ");
    for (int i = 0; i < len; i++) printf("%01x", expected[i]);
    printf("\n");
    return 0;
}


static inline uint64_t rdtsc()
{
    uint32_t lo, hi;
    __asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
    return ((uint64_t)hi << 32) | lo;
}

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

//----Computing Throughput----

static void throughput_sweep(int R, uint8_t RTK[][8],
                              uint8_t nonce[16], uint8_t iv[16])
{
    printf("\n");
    printf("--------SKINNY-64 CPU Throughput vs Input Size--------\n");
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
        64  * 1024 * 1024
    };
    const int NSIZES = (int)(sizeof(data_sizes) / sizeof(data_sizes[0]));

    printf("  %-10s  %-14s  %12s  %12s\n",
           "Data Size", "Mode", "Wall (ms)", "Throughput");
    printf("  %-10s  %-14s  %12s  %12s\n",
           "----------", "--------------", "------------", "----------");

    for (int si = 0; si < NSIZES; si++) {
        size_t data_bytes = data_sizes[si];

        size_t num_blocks = (data_bytes + 7) / 8;
        size_t buf_size   = num_blocks * 16;

        uint8_t *plain  = (uint8_t*)malloc(buf_size);
        uint8_t *cipher = (uint8_t*)malloc(buf_size);
        if (!plain || !cipher) { fprintf(stderr, "malloc failed\n"); exit(1); }
        for (size_t i = 0; i < buf_size; i++) plain[i] = (uint8_t)(i & 0xf);

  
        ecb_encrypt(R, plain, cipher, (int)num_blocks, RTK);

        struct timespec t0, t1;

        //----ECB encryption---
        double ecb_enc_samples[TRIALS];
        for (int t = 0; t < TRIALS; t++) {
            clock_gettime(CLOCK_MONOTONIC, &t0);
            ecb_encrypt(R, plain, cipher, (int)num_blocks, RTK);
            clock_gettime(CLOCK_MONOTONIC, &t1);
            ecb_enc_samples[t] = elapsed_ms(t0, t1);
        }

        //---ECB decryption---
        double ecb_dec_samples[TRIALS];
        for (int t = 0; t < TRIALS; t++) {
            clock_gettime(CLOCK_MONOTONIC, &t0);
            ecb_decrypt(R, cipher, plain, (int)num_blocks, RTK);
            clock_gettime(CLOCK_MONOTONIC, &t1);
            ecb_dec_samples[t] = elapsed_ms(t0, t1);
        }

        //----CTR encryption----
        double ctr_enc_samples[TRIALS];
        for (int t = 0; t < TRIALS; t++) {
            uint8_t n[16]; memcpy(n, nonce, 16);
            clock_gettime(CLOCK_MONOTONIC, &t0);
            ctr_encrypt(R, plain, cipher, (int)num_blocks, n, RTK);
            clock_gettime(CLOCK_MONOTONIC, &t1);
            ctr_enc_samples[t] = elapsed_ms(t0, t1);
        }

        //----CTR decryption----
        double ctr_dec_samples[TRIALS];
        for (int t = 0; t < TRIALS; t++) {
            uint8_t n[16]; memcpy(n, nonce, 16);
            clock_gettime(CLOCK_MONOTONIC, &t0);
            ctr_decrypt(R, cipher, plain, (int)num_blocks, n, RTK);
            clock_gettime(CLOCK_MONOTONIC, &t1);
            ctr_dec_samples[t] = elapsed_ms(t0, t1);
        }

        //----CBC encryption----
        double cbc_enc_samples[TRIALS];
        for (int t = 0; t < TRIALS; t++) {
            uint8_t v[16]; memcpy(v, iv, 16);
            clock_gettime(CLOCK_MONOTONIC, &t0);
            cbc_encrypt(R, plain, cipher, (int)num_blocks, v, RTK);
            clock_gettime(CLOCK_MONOTONIC, &t1);
            cbc_enc_samples[t] = elapsed_ms(t0, t1);
        }

        //----CBC decryption----
        double cbc_dec_samples[TRIALS];
        for (int t = 0; t < TRIALS; t++) {
            uint8_t v[16]; memcpy(v, iv, 16);
            clock_gettime(CLOCK_MONOTONIC, &t0);
            cbc_decrypt(R, cipher, plain, (int)num_blocks, v, RTK);
            clock_gettime(CLOCK_MONOTONIC, &t1);
            cbc_dec_samples[t] = elapsed_ms(t0, t1);
        }

      
        qsort(ecb_enc_samples, TRIALS, sizeof(double), cmp_dbl);
        qsort(ecb_dec_samples, TRIALS, sizeof(double), cmp_dbl);
        qsort(ctr_enc_samples, TRIALS, sizeof(double), cmp_dbl);
        qsort(ctr_dec_samples, TRIALS, sizeof(double), cmp_dbl);
        qsort(cbc_enc_samples, TRIALS, sizeof(double), cmp_dbl);
        qsort(cbc_dec_samples, TRIALS, sizeof(double), cmp_dbl);

        double ecb_enc_ms = ecb_enc_samples[TRIALS/2];
        double ecb_dec_ms = ecb_dec_samples[TRIALS/2];
        double ctr_enc_ms = ctr_enc_samples[TRIALS/2];
        double ctr_dec_ms = ctr_dec_samples[TRIALS/2];
        double cbc_enc_ms = cbc_enc_samples[TRIALS/2];
        double cbc_dec_ms = cbc_dec_samples[TRIALS/2];

        double ecb_enc_gbs = (double)data_bytes / (ecb_enc_ms * 1e-3) / 1e9;
        double ecb_dec_gbs = (double)data_bytes / (ecb_dec_ms * 1e-3) / 1e9;
        double ctr_enc_gbs = (double)data_bytes / (ctr_enc_ms * 1e-3) / 1e9;
        double ctr_dec_gbs = (double)data_bytes / (ctr_dec_ms * 1e-3) / 1e9;
        double cbc_enc_gbs = (double)data_bytes / (cbc_enc_ms * 1e-3) / 1e9;
        double cbc_dec_gbs = (double)data_bytes / (cbc_dec_ms * 1e-3) / 1e9;

        const char *unit = (data_bytes >= 1024*1024) ? "MB" : "KB";
        double      nd   = (data_bytes >= 1024*1024) ? data_bytes/1048576.0
                                                      : data_bytes/1024.0;

        printf("  %5.0f %-3s  CPU-ECB-E    %10.4f ms  %8.4f GB/s\n",
               nd, unit, ecb_enc_ms, ecb_enc_gbs);
        printf("  %5.0f %-3s  CPU-ECB-D    %10.4f ms  %8.4f GB/s\n",
               nd, unit, ecb_dec_ms, ecb_dec_gbs);
        printf("  %5.0f %-3s  CPU-CTR-E    %10.4f ms  %8.4f GB/s\n",
               nd, unit, ctr_enc_ms, ctr_enc_gbs);
        printf("  %5.0f %-3s  CPU-CTR-D    %10.4f ms  %8.4f GB/s\n",
               nd, unit, ctr_dec_ms, ctr_dec_gbs);
        printf("  %5.0f %-3s  CPU-CBC-E    %10.4f ms  %8.4f GB/s\n",
               nd, unit, cbc_enc_ms, cbc_enc_gbs);
        printf("  %5.0f %-3s  CPU-CBC-D    %10.4f ms  %8.4f GB/s\n",
               nd, unit, cbc_dec_ms, cbc_dec_gbs);
        printf("\n");

        free(plain);
        free(cipher);
    }
}

//----Computing Single Block Latency----
static void measure_single_block_latency(int R, uint8_t RTK[][8],
                                          uint8_t nonce[16], uint8_t iv[16])
{
    const int LAT_BLOCKS = 8;
    const int LAT_TRIALS = 101;

    uint8_t plain [LAT_BLOCKS * 16];
    uint8_t cipher[LAT_BLOCKS * 16];
    uint8_t recov [LAT_BLOCKS * 16];
    for (int i = 0; i < LAT_BLOCKS * 16; i++) plain[i] = (uint8_t)(i & 0xf);

    struct timespec t0, t1;

    //----Computing ECB encrypt latency---
    float ecb_enc_samples[LAT_TRIALS];
    for (int t = 0; t < LAT_TRIALS; t++) {
        clock_gettime(CLOCK_MONOTONIC, &t0);
        ecb_encrypt(R, plain, cipher, LAT_BLOCKS, RTK);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        ecb_enc_samples[t] = (float)(elapsed_ms(t0, t1) * 1000.0);
    }

    //----Computing ECB decrypt latency----
    ecb_encrypt(R, plain, cipher, LAT_BLOCKS, RTK);   

    float ecb_dec_samples[LAT_TRIALS];
    for (int t = 0; t < LAT_TRIALS; t++) {
        clock_gettime(CLOCK_MONOTONIC, &t0);
        ecb_decrypt(R, cipher, recov, LAT_BLOCKS, RTK);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        ecb_dec_samples[t] = (float)(elapsed_ms(t0, t1) * 1000.0);
    }

    //---Computing CTR Encrypt Latency---
    float ctr_samples[LAT_TRIALS];
    for (int t = 0; t < LAT_TRIALS; t++) {
        uint8_t n[16]; memcpy(n, nonce, 16);
        clock_gettime(CLOCK_MONOTONIC, &t0);
        ctr_encrypt(R, plain, cipher, LAT_BLOCKS, n, RTK);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        ctr_samples[t] = (float)(elapsed_ms(t0, t1) * 1000.0);
    }

    //---Computing CBC decrypt latency---
    uint8_t v0[16]; memcpy(v0, iv, 16);
    cbc_encrypt(R, plain, cipher, LAT_BLOCKS, v0, RTK);  

    float cbc_samples[LAT_TRIALS];
    for (int t = 0; t < LAT_TRIALS; t++) {
        uint8_t v[16]; memcpy(v, iv, 16);
        clock_gettime(CLOCK_MONOTONIC, &t0);
        cbc_decrypt(R, cipher, recov, LAT_BLOCKS, v, RTK);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        cbc_samples[t] = (float)(elapsed_ms(t0, t1) * 1000.0);
    }

    
    for (int i = 0; i < LAT_TRIALS-1; i++)
        for (int j = i+1; j < LAT_TRIALS; j++) {
#define Sorting(a) if ((a)[j] < (a)[i]) { float _t=(a)[i]; (a)[i]=(a)[j]; (a)[j]=_t; }
            Sorting(ecb_enc_samples)
            Sorting(ecb_dec_samples)
            Sorting(ctr_samples)
            Sorting(cbc_samples)
#undef Sorting
        }

    printf("\n--------Single 64-Byte Block Latency (side data point)--------\n");
    printf("  Input size : %d cipher blocks = 64 nibble-bytes\n", LAT_BLOCKS);
    printf("  Trials     : %d\n", LAT_TRIALS);
    printf("  %-35s : %8.2f us\n", "ECB encrypt latency", ecb_enc_samples[LAT_TRIALS/2]);
    printf("  %-35s : %8.2f us\n", "ECB decrypt latency", ecb_dec_samples[LAT_TRIALS/2]);
    printf("  %-35s : %8.2f us\n", "CTR encrypt latency", ctr_samples[LAT_TRIALS/2]);
    printf("  %-35s : %8.2f us\n", "CBC decrypt latency", cbc_samples[LAT_TRIALS/2]);
}

//---Computing Key Schedule Cost---
static void measure_key_schedule_cost(uint8_t tweakey_1[16], uint8_t tweakey_2[16], int R)
{
    const int KS_TRIALS = 101;

    uint8_t TK_1[SKINNY64_ROUNDS][16];
    uint8_t TK_2[SKINNY64_ROUNDS][16];
    uint8_t RTK [SKINNY64_ROUNDS][8];

    struct timespec t0, t1;
    double ks_samples[KS_TRIALS];
    for (int t = 0; t < KS_TRIALS; t++) {
        for (int j = 0; j < 16; j++) { TK_1[0][j] = tweakey_1[j]; TK_2[0][j] = tweakey_2[j]; }
        clock_gettime(CLOCK_MONOTONIC, &t0);
        round_tweakey_schedule(R, TK_1, TK_2, RTK);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        ks_samples[t] = elapsed_ms(t0, t1) * 1000.0;
    }
    qsort(ks_samples, KS_TRIALS, sizeof(double), cmp_dbl);
    double ks_med = ks_samples[KS_TRIALS/2];

    printf("\n--------Key Schedule Cost--------\n");
    printf("\n  Time (median of %d trials)\n", KS_TRIALS);
    printf("  %-48s : %7.2f us\n", "CPU compute", ks_med);
    printf("  %-48s : %7.2f us\n", "GPU upload", 0.0);
    printf("  %-48s : %7.2f us\n", "Total key setup cost", ks_med);
    printf("\n  Space\n");
    printf("  %-48s : %3zu bytes  (%d rounds x 8 nibble-bytes)\n",
           "Expanded RTK in RAM",
           (size_t)(SKINNY64_ROUNDS * 8), SKINNY64_ROUNDS);
}

//Computing Cycles per bytes

static double measure_cycles_per_byte_ecb(int R, uint8_t RTK[][8])
{
    static uint8_t buf[BUF_BLOCKS * 16];
    static uint8_t out[BUF_BLOCKS * 16];
    for (int i = 0; i < BUF_BLOCKS * 16; i++) buf[i] = (uint8_t)(i & 0xf);

    uint64_t samples[TRIALS];
    for (int t = 0; t < TRIALS; t++) {
        uint64_t t0 = rdtsc();
        ecb_encrypt(R, buf, out, BUF_BLOCKS, RTK);
        uint64_t t1 = rdtsc();
        samples[t] = t1 - t0;
    }
    qsort(samples, TRIALS, sizeof(uint64_t), cmp_u64);
    return (double)samples[TRIALS/2] / (BUF_BLOCKS * 8);
}

static double measure_cycles_per_byte_ctr(int R, uint8_t RTK[][8])
{
    static uint8_t buf[BUF_BLOCKS * 16];
    static uint8_t out[BUF_BLOCKS * 16];
    uint8_t nonce[16] = {0};
    for (int i = 0; i < BUF_BLOCKS * 16; i++) buf[i] = (uint8_t)(i & 0xf);

    uint64_t samples[TRIALS];
    for (int t = 0; t < TRIALS; t++) {
        uint8_t n[16] = {0};
        uint64_t t0 = rdtsc();
        ctr_encrypt(R, buf, out, BUF_BLOCKS, n, RTK);
        uint64_t t1 = rdtsc();
        samples[t] = t1 - t0;
    }
    qsort(samples, TRIALS, sizeof(uint64_t), cmp_u64);
    return (double)samples[TRIALS/2] / (BUF_BLOCKS * 8);
}

static double measure_cycles_per_byte_cbc(int R, uint8_t RTK[][8])
{
    static uint8_t buf[BUF_BLOCKS * 16];
    static uint8_t out[BUF_BLOCKS * 16];
    uint8_t iv[16] = {0};
    for (int i = 0; i < BUF_BLOCKS * 16; i++) buf[i] = (uint8_t)(i & 0xf);

    uint64_t samples[TRIALS];
    for (int t = 0; t < TRIALS; t++) {
        uint8_t v[16] = {0};
        uint64_t t0 = rdtsc();
        cbc_encrypt(R, buf, out, BUF_BLOCKS, v, RTK);
        uint64_t t1 = rdtsc();
        samples[t] = t1 - t0;
    }
    qsort(samples, TRIALS, sizeof(uint64_t), cmp_u64);
    return (double)samples[TRIALS/2] / (BUF_BLOCKS * 8);
}


static void print_code_size()
{
    printf("\n--------Code Size--------\n");
    struct TableSize { const char *name; size_t bytes; };
    TableSize tables[] = {
        { "S              (4-bit Sbox, 16 entries)",           sizeof(S)             },
        { "S_inverse      (4-bit Sbox inverse, 16 entries)",   sizeof(S_inverse)     },
        { "P              (ShiftRows permutation, 16 entries)", sizeof(P)             },
        { "P_inverse      (inverse permutation, 16 entries)",  sizeof(P_inverse)     },
        { "Q              (tweakey permutation, 16 entries)",   sizeof(Q)             },
        { "Round_Constant (6-bit LFSR, 36 entries)",            sizeof(Round_Constant)},
    };
    size_t total = 0;
    int ntables = sizeof(tables) / sizeof(tables[0]);
    for (int i = 0; i < ntables; i++) {
        printf("  %-52s : %3zu bytes\n", tables[i].name, tables[i].bytes);
        total += tables[i].bytes;
    }
    printf("  %-52s : %3zu bytes\n", "Total table footprint", total);
}

//---Performance Metrics----

static void print_performance_metrics(int R, uint8_t RTK[][8],
                                       uint8_t tweakey_1[16], uint8_t tweakey_2[16])
{
    printf("\n--------Performance Metrics (cycles/byte)--------\n");
    printf("  Median of %d trials, Buffer = %d blocks (%d nibble-bytes, %d actual bytes)\n",
           TRIALS, BUF_BLOCKS, BUF_BLOCKS * 16, BUF_BLOCKS * 8);

    double cpb_ecb = measure_cycles_per_byte_ecb(R, RTK);
    double cpb_ctr = measure_cycles_per_byte_ctr(R, RTK);
    double cpb_cbc = measure_cycles_per_byte_cbc(R, RTK);

    printf("  %-40s : %8.2f  cycles/byte\n", "ECB encrypt", cpb_ecb);
    printf("  %-40s : %8.2f  cycles/byte\n", "CTR encrypt", cpb_ctr);
    printf("  %-40s : %8.2f  cycles/byte\n", "CBC encrypt (chained)", cpb_cbc);
    printf("\n  CTR overhead vs ECB                    : %+.2f cycles/byte\n",
           cpb_ctr - cpb_ecb);
    printf("  CBC overhead vs ECB                    : %+.2f cycles/byte\n",
           cpb_cbc - cpb_ecb);
    printf("  CBC overhead vs CTR                    : %+.2f cycles/byte\n",
           cpb_cbc - cpb_ctr);
}


int main()
{
    printf("--------SKINNY-64-128 CPU Implementation--------\n");
    printf("--------ECB, CTR and CBC Mode Tests--------\n\n");

    int R = SKINNY64_ROUNDS;

    uint8_t TK_1[SKINNY64_ROUNDS][16];
    uint8_t TK_2[SKINNY64_ROUNDS][16];
    uint8_t RTK[SKINNY64_ROUNDS][8];
    uint8_t tweakey_1[16], tweakey_2[16];

    string TK1_string = "9eb93640d088da63";
    string TK2_string = "76a39d1c8bea71e1";
    convert_hex_string_to_statearray(TK1_string, tweakey_1, false);
    convert_hex_string_to_statearray(TK2_string, tweakey_2, false);
    for (uint8_t i = 0; i < 16; i++) { TK_1[0][i] = tweakey_1[i]; TK_2[0][i] = tweakey_2[i]; }

    printf("Generating tweakey schedule\n");
    round_tweakey_schedule(R, TK_1, TK_2, RTK);
    printf("Tweakey schedule generated\n\n");

    string plain_string = "cf16cfe8fd0f98aa";
    uint8_t plain_text_0[16];
    convert_hex_string_to_statearray(plain_string, plain_text_0, false);

    int num_blocks = 1;
    uint8_t nonce[16] = {0};
    uint8_t iv[16]    = {0};

    //---ECB Mode Evaluation---
    printf("--------ECB Mode--------\n\n");

    uint8_t ecb_plain[16], ecb_cipher[16], ecb_recovered[16];
    for (int i = 0; i < 16; i++) ecb_plain[i] = plain_text_0[i];

    printf("Original Plaintext   : "); print_message(ecb_plain, num_blocks);
    ecb_encrypt(R, ecb_plain, ecb_cipher, num_blocks, RTK);
    printf("ECB Ciphertext       : "); print_message(ecb_cipher, num_blocks);
    ecb_decrypt(R, ecb_cipher, ecb_recovered, num_blocks, RTK);
    printf("Recovered Plaintext  : "); print_message(ecb_recovered, num_blocks);
    printf("\n");
    assert_equal("ECB mode", ecb_recovered, ecb_plain, num_blocks * 16);

    //---CTR Mode Evaluation---
    printf("\n--------CTR Mode--------\n\n");

    uint8_t ctr_plain[16], ctr_cipher[16], ctr_recovered[16];
    for (int i = 0; i < 16; i++) ctr_plain[i] = plain_text_0[i];

    printf("Original Plaintext   : "); print_message(ctr_plain, num_blocks);
    ctr_encrypt(R, ctr_plain, ctr_cipher, num_blocks, nonce, RTK);
    printf("CTR Ciphertext       : "); print_message(ctr_cipher, num_blocks);
    ctr_decrypt(R, ctr_cipher, ctr_recovered, num_blocks, nonce, RTK);
    printf("Recovered Plaintext  : "); print_message(ctr_recovered, num_blocks);
    printf("\n");
    assert_equal("CTR mode", ctr_recovered, ctr_plain, num_blocks * 16);

    //---CBC Mode Evaluation---
    printf("\n--------CBC Mode--------\n\n");

    uint8_t cbc_plain[16], cbc_cipher[16], cbc_recovered[16];
    for (int i = 0; i < 16; i++) cbc_plain[i] = plain_text_0[i];

    printf("Original Plaintext   : "); print_message(cbc_plain, num_blocks);
    cbc_encrypt(R, cbc_plain, cbc_cipher, num_blocks, iv, RTK);
    printf("CBC Ciphertext       : "); print_message(cbc_cipher, num_blocks);
    cbc_decrypt(R, cbc_cipher, cbc_recovered, num_blocks, iv, RTK);
    printf("Recovered Plaintext  : "); print_message(cbc_recovered, num_blocks);
    printf("\n");
    assert_equal("CBC mode", cbc_recovered, cbc_plain, num_blocks * 16);

    printf("\n--------ECB, CBC and CTR modes are validated--------\n");

    print_code_size();
    print_performance_metrics(R, RTK, tweakey_1, tweakey_2);
    throughput_sweep(R, RTK, nonce, iv);
    measure_single_block_latency(R, RTK, nonce, iv);
    measure_key_schedule_cost(tweakey_1, tweakey_2, R);

    return 0;
}