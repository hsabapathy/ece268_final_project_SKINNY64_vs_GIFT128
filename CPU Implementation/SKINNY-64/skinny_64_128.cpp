#include <stdio.h>
#include <stdint.h>
#include <string>
#include <iostream>

using namespace std;

//4-bit Sbox
const uint8_t S[16] = {0xc, 0x6, 0x9, 0x0, 0x1, 0xa, 0x2, 0xb, 0x3, 0x8, 0x5, 0xd, 0x4, 0xe, 0x7, 0xf};

//4-bit Sbox inverse
const uint8_t S_inverse[16] = {0x3, 0x4, 0x6, 0x8, 0xc, 0xa, 0x1, 0xe, 0x9, 0x2, 0x5, 0x7, 0x0, 0xb, 0xd, 0xf};


//Permutation P[i] = x
const uint8_t P[16] = {0x0, 0x1, 0x2, 0x3, 0x7, 0x4, 0x5, 0x6, 0xa, 0xb, 0x8, 0x9, 0xd, 0xe, 0xf, 0xc};

//Inverse Permutation P_inverse[x] = i
const uint8_t P_inverse[16] = {0x0, 0x1, 0x2, 0x3, 0x5, 0x6, 0x7, 0x4, 0xa, 0xb, 0x8, 0x9, 0xf, 0xc, 0xd, 0xe};

// Tweakey Permutation
const uint8_t Q[16] = {0x9, 0xf, 0x8, 0xd, 0xa, 0xe, 0xc, 0xb, 0x0, 0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7};

// Round Constants
const uint8_t Round_Constant[36] = {0x01, 0x03, 0x07, 0x0F, 0x1F, 0x3E, 0x3D, 0x3B, 0x37, 0x2F, 0x1E, 0x3C, 0x39, 0x33, 0x27, 0x0E, 0x1D, 0x3A, 0x35, 0x2B, 0x16, 0x2C, 0x18, 0x30, 0x21, 0x02, 0x05, 0x0B, 0x17, 0x2E, 0x1C, 0x38, 0x31, 0x23, 0x06, 0x0D};

uint8_t hex_char_to_val(char c)
{
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return 0;
}

//Function to print the state
void print_state(uint8_t state[16])
{
    for (int i = 0; i < 16; i++)
        printf("%01x", state[i]);
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

// Function to create diffusion - Mixing 4 element column of state
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
    //Variables to store permutated version of TK1 and TK2 
    uint8_t TKp1[rounds - 1][16];
    uint8_t TKp2[rounds - 1][16];

    // Masking to 4 bits
    for (uint8_t i = 0; i < 16; i++)
        TK1[0][i] = (TK1[0][i] & 0xf);
    
    for (uint8_t i = 0; i < 16; i++)
        TK2[0][i] = (TK2[0][i] & 0xf);

    printf("TK1 Round 0 :");
    print_state(TK1[0]);

    printf("TK2 Round 0 :");
    print_state(TK2[0]);

    //Computing Round 0 tweaky RTK = TK1 ^ TK2
    for (uint8_t i = 0; i < 8; i++)
        round_tweakey[0][i] = TK1[0][i] ^ TK2[0][i];

    printf("RTK Round 0: ");
    print_state(round_tweakey[0]);
    
    for (int r = 1; r < rounds; r++)
    {
        // Tweakey permutation of TK1 and TK2
        for (int i = 0; i < 16; i++)
        {
            TKp1[r - 1][i] = TK1[r - 1][Q[i]];
            TKp2[r - 1][i] = TK2[r - 1][Q[i]];
        }

        printf("After Permutation TK1: ");
        print_state(TKp1[r-1]);

        printf("After Permuatation TK2: ");
        print_state(TKp2[r -1]);
        
        // Apply LFSR to TK2
        for (int i = 0; i < 16; i++)
        {
            TK1[r][i] = TKp1[r - 1][i];
            
            if (i < 8)
            {
                TK2[r][i] = TK2_lfsr(TKp2[r - 1][i]);
            }
            else
            {
                TK2[r][i] = TKp2[r - 1][i];
            }
            printf("After LFSR TK2: ");
            print_state(TK2[r]);

            printf("Updated TK1 :");
            print_state(TK1[r]);
            
        }
        for (int i = 0; i < 8; i++)
            round_tweakey[r][i] = (TK1[r][i] ^ TK2[r][i]);

        printf("Round Tweakey RTK :");
        print_state(round_tweakey[r]);
    }
}

void encryption(int R, uint8_t plain_text[16], uint8_t cipher_text[16], uint8_t TK[][8])
{
    for (uint8_t i = 0; i < 16; i++)
    {
        cipher_text[i] = plain_text[i] & 0xf;
    }
    printf("Initial State: ");
    print_state(cipher_text);

    for (uint8_t r = 0; r < R; r++)
    {
        // S - Box
        for (uint8_t i = 0; i < 16; i++)
            {
                cipher_text[i] = S[cipher_text[i]];
            }
        printf(" After SBox: ");
        print_state(cipher_text);

        //Adding round constants
        cipher_text[0] ^= (Round_Constant[r] & 0xf);
        cipher_text[4] ^= ((Round_Constant[r] >> 4) & 0x3);
        cipher_text[8] ^= 0x2;

        printf("After adding constants: ");
        print_state(cipher_text);

        //Adding round tweaky
        for (uint8_t i = 0; i < 8; i++)
            cipher_text[i] ^= TK[r][i];

        printf("After Round Tweaky: ");
        print_state(cipher_text);
        
        
        uint8_t temp[16];
        for (uint8_t i = 0; i < 16; i++)
            {
            temp[i] = cipher_text[i];
            }
        for (uint8_t i = 0; i < 16; i++)
            {
            cipher_text[i] = temp[P[i]];
            }
        
        printf("After permutation: ");
        print_state(cipher_text);
        
        mix_columns(cipher_text);

        printf("After mix_columns: ");
        print_state(cipher_text);
    }
    printf("Final Ciphertext:");
    print_state(cipher_text);
}

void decryption(int R, uint8_t plain_text[16], uint8_t cipher_text[16], uint8_t TK[][8])
{
    // Initializing the state
    for (uint8_t i = 0; i < 16; i++)
    {
        plain_text[i] = cipher_text[i] & 0xf;
    }

    printf("Initial Cipher State: ");
    print_state(plain_text);
    
    int index;
    uint8_t temp[16];
    
    for (int r = 0; r < R; r++)
    {
        // Inverse Mix Columns
        inverse_mix_columns(plain_text);
        printf("After perfomring inverse mix columns: ");
        print_state(plain_text);
        
        for (uint8_t i = 0; i < 16; i++)
            temp[i] = plain_text[i];
        
        //Performing inverse permuatation
        for (uint8_t i = 0; i < 16; i++)
            plain_text[i] = temp[P_inverse[i]];

        printf("After performing inverse permuatation :");
        print_state(plain_text);
        
        index = R - r - 1;
        //Removing Tweakey
        for (uint8_t i = 0; i < 8; i++)
            plain_text[i] ^= TK[index][i];

        printf("After Tweaky XOR :");
        print_state(plain_text);
        
        // Removing the constants
        plain_text[0] ^= (Round_Constant[index] & 0xf);
        plain_text[4] ^= ((Round_Constant[index] >> 4) & 0x3);
        plain_text[8] ^= 0x2;

        printf("After removal of constants: ");
        print_state(plain_text);
        
        // Inverse S-Box
        for (uint8_t i = 0; i < 16; i++)
            plain_text[i] = S_inverse[plain_text[i]];

        printf("After performing inverse S-Box :");
        print_state(plain_text);
    
    }
    printf("Final recovered Plain_text :");
    print_state(plain_text);
}

int main()
{
    uint8_t plain_text[16];
    uint8_t cipher_text[16];

    //Number of rounds
    int R = 36;
    
    uint8_t TK_1[R][16];
    uint8_t TK_2[R][16];
    uint8_t RTK[R][8];
    uint8_t tweakey_1[16];
    uint8_t tweakey_2[16];
    
    // Test vectors
    string TK1_string = "9eb93640d088da63";
    string TK2_string = "76a39d1c8bea71e1";
    string plain_string = "cf16cfe8fd0f98aa";
    string cipher_string = "6ceda1f43de92b9e";
    
    bool reversed = false;
    
    convert_hex_string_to_statearray(TK1_string, tweakey_1, reversed);
    convert_hex_string_to_statearray(TK2_string, tweakey_2, reversed);
    convert_hex_string_to_statearray(plain_string, plain_text, reversed);
    
    for (uint8_t i = 0; i < 16; i++)
    {
        TK_1[0][i] = tweakey_1[i];
        TK_2[0][i] = tweakey_2[i];
    }
    printf("\n-------Tweaky Schedule---------------\n");
   round_tweakey_schedule(R, TK_1, TK_2, RTK);
    
    printf("\n------Encryption---------------------\n");
    encryption(R, plain_text, cipher_text, RTK);
   
    printf("\n------Decryption---------------------\n");
    decryption(R, plain_text, cipher_text, RTK);

    printf("\n------Final Results------------------\n");
    printf("Original Plaintext   : %s\n", plain_string.c_str());

    printf("Final Ciphertext     : ");
    print_state(cipher_text);

    printf("Recovered Plaintext  : ");
    print_state(plain_text);   

    return 0;
}