/*
 * lenet5-qvdot4.c - Standalone bare-metal 4-bit quantized LeNet-5 CNN
 * benchmark for CV32E40P in core-v-verif.
 */

#include <stdint.h>
#include <stdio.h>

#define CONV1_OUT_DIM   28
#define CONV1_FILTERS   6
#define CONV1_WORDS     4    /* ceil(25 / 8)  = 4 packed 32-bit words  */

#define POOL1_OUT_DIM   14

#define CONV2_OUT_DIM   10
#define CONV2_FILTERS   16
#define CONV2_WORDS     19   /* ceil(150 / 8) = 19 packed 32-bit words */

#define POOL2_OUT_DIM   5
#define FC1_IN_FEATURES 400
#define FC1_WORDS       50   /* 400 / 8 = 50 packed 32-bit words       */
#define FC1_OUT         120

#define FC2_WORDS       15   /* 120 / 8 = 15 packed 32-bit words       */
#define FC2_OUT         84

#define FC3_WORDS       11   /* ceil(84 / 8)  = 11 packed 32-bit words */
#define FC3_OUT         10

static inline uint32_t read_mcycle_lo(void)
{
    uint32_t value;
    __asm__ volatile ("csrr %0, mcycle" : "=r"(value));
    return value;
}

static inline int32_t sext4(uint32_t nibble)
{
    return (int32_t)(int8_t)((uint8_t)(nibble << 4)) >> 4;
}

static int32_t qvdot4_sw(uint32_t a, uint32_t b)
{
    int32_t acc = 0;
    for (int lane = 0; lane < 8; ++lane) {
        int shift = lane * 4;
        acc += sext4((a >> shift) & 0xFu) * sext4((b >> shift) & 0xFu);
    }
    return acc;
}

static inline int32_t qvdot4_hw(uint32_t a, uint32_t b)
{
    int32_t res;
    __asm__ volatile (
        ".insn r 0x0b, 0, 0, %0, %1, %2"
        : "=r"(res)
        : "r"(a), "r"(b)
    );
    return res;
}

static inline int8_t clamp_s4(int32_t v)
{
    if (v > 7)  return 7;
    if (v < -8) return -8;
    return (int8_t)v;
}

static inline uint32_t next_rand(uint32_t *state)
{
    *state = (*state * 1664525u) + 1013904223u;
    return *state;
}

static int8_t   input_img[32][32];
static uint32_t conv1_patches[CONV1_OUT_DIM * CONV1_OUT_DIM][CONV1_WORDS];
static uint32_t conv1_weights[CONV1_FILTERS][CONV1_WORDS];
static int32_t  conv1_bias[CONV1_FILTERS];
static int8_t   conv1_out[CONV1_OUT_DIM][CONV1_OUT_DIM][CONV1_FILTERS];

static int8_t   pool1_out[POOL1_OUT_DIM][POOL1_OUT_DIM][CONV1_FILTERS];
static uint32_t conv2_patches[CONV2_OUT_DIM * CONV2_OUT_DIM][CONV2_WORDS];
static uint32_t conv2_weights[CONV2_FILTERS][CONV2_WORDS];
static int32_t  conv2_bias[CONV2_FILTERS];
static int8_t   conv2_out[CONV2_OUT_DIM][CONV2_OUT_DIM][CONV2_FILTERS];

static int8_t   pool2_out[FC1_IN_FEATURES];
static uint32_t fc1_in_packed[FC1_WORDS];
static uint32_t fc1_weights[FC1_OUT][FC1_WORDS];
static int32_t  fc1_bias[FC1_OUT];
static int8_t   fc1_out[FC1_OUT];

static uint32_t fc2_in_packed[FC2_WORDS];
static uint32_t fc2_weights[FC2_OUT][FC2_WORDS];
static int32_t  fc2_bias[FC2_OUT];
static int8_t   fc2_out[FC2_OUT];

static uint32_t fc3_in_packed[FC3_WORDS];
static uint32_t fc3_weights[FC3_OUT][FC3_WORDS];
static int32_t  fc3_bias[FC3_OUT];

struct lenet5_profile {
    uint32_t conv1_cycles;
    uint32_t pool1_cycles;
    uint32_t conv2_cycles;
    uint32_t pool2_cycles;
    uint32_t fc1_cycles;
    uint32_t fc2_cycles;
    uint32_t fc3_cycles;
    uint32_t total_cycles;
    int32_t  logits[FC3_OUT];
    int32_t  checksum;
};

static uint32_t pack8_s4(const int8_t *vals, int count)
{
    uint32_t word = 0u;
    for (int i = 0; i < 8; ++i) {
        uint32_t nib = (i < count) ? ((uint32_t)vals[i] & 0xFu) : 0u;
        word |= (nib << (i * 4));
    }
    return word;
}

static void init_model_and_input(void)
{
    uint32_t seed = 0x12345678u;

    for (int r = 0; r < 32; ++r) {
        for (int c = 0; c < 32; ++c) {
            int center = (r >= 6 && r < 26 && c >= 6 && c < 26);
            int32_t v = center ? (int32_t)((next_rand(&seed) & 0x7u)) : -2;
            input_img[r][c] = clamp_s4(v);
        }
    }

    for (int r = 0; r < CONV1_OUT_DIM; ++r) {
        for (int c = 0; c < CONV1_OUT_DIM; ++c) {
            int8_t patch[25];
            int idx = 0;
            int p_idx = r * CONV1_OUT_DIM + c;
            for (int kr = 0; kr < 5; ++kr) {
                for (int kc = 0; kc < 5; ++kc) {
                    patch[idx++] = input_img[r + kr][c + kc];
                }
            }
            for (int w = 0; w < CONV1_WORDS; ++w) {
                int rem = 25 - w * 8;
                conv1_patches[p_idx][w] = pack8_s4(&patch[w * 8], rem > 8 ? 8 : rem);
            }
        }
    }

    for (int f = 0; f < CONV1_FILTERS; ++f) {
        for (int w = 0; w < CONV1_WORDS; ++w) {
            uint32_t raw = next_rand(&seed);
            if (w == CONV1_WORDS - 1) {
                raw &= 0x0000000Fu;
            }
            conv1_weights[f][w] = raw;
        }
        conv1_bias[f] = (int32_t)(next_rand(&seed) & 0x7u);
    }

    for (int f = 0; f < CONV2_FILTERS; ++f) {
        for (int w = 0; w < CONV2_WORDS; ++w) {
            uint32_t raw = next_rand(&seed);
            if (w == CONV2_WORDS - 1) {
                raw &= 0x00FFFFFFu;
            }
            conv2_weights[f][w] = raw;
        }
        conv2_bias[f] = (int32_t)(next_rand(&seed) & 0x7u);
    }

    for (int n = 0; n < FC1_OUT; ++n) {
        for (int w = 0; w < FC1_WORDS; ++w) {
            fc1_weights[n][w] = next_rand(&seed);
        }
        fc1_bias[n] = (int32_t)(next_rand(&seed) & 0x7u);
    }

    for (int n = 0; n < FC2_OUT; ++n) {
        for (int w = 0; w < FC2_WORDS; ++w) {
            fc2_weights[n][w] = next_rand(&seed);
        }
        fc2_bias[n] = (int32_t)(next_rand(&seed) & 0x7u);
    }

    for (int n = 0; n < FC3_OUT; ++n) {
        for (int w = 0; w < FC3_WORDS; ++w) {
            uint32_t raw = next_rand(&seed);
            if (w == FC3_WORDS - 1) {
                raw &= 0x0000FFFFu;
            }
            fc3_weights[n][w] = raw;
        }
        fc3_bias[n] = (int32_t)(next_rand(&seed) & 0xFu) - 8;
    }
}

static void pack_conv2_patches(void)
{
    for (int r = 0; r < CONV2_OUT_DIM; ++r) {
        for (int c = 0; c < CONV2_OUT_DIM; ++c) {
            int8_t patch[150];
            int idx = 0;
            int p_idx = r * CONV2_OUT_DIM + c;
            for (int kr = 0; kr < 5; ++kr) {
                for (int kc = 0; kc < 5; ++kc) {
                    for (int ch = 0; ch < CONV1_FILTERS; ++ch) {
                        patch[idx++] = pool1_out[r + kr][c + kc][ch];
                    }
                }
            }
            for (int w = 0; w < CONV2_WORDS; ++w) {
                int rem = 150 - w * 8;
                conv2_patches[p_idx][w] = pack8_s4(&patch[w * 8], rem > 8 ? 8 : rem);
            }
        }
    }
}

static inline __attribute__((always_inline)) void run_lenet5(
    int use_hw, struct lenet5_profile *prof)
{
    uint32_t t0, t1;
    int flat_idx = 0;
    int32_t checksum = 0;

    t0 = read_mcycle_lo();
    for (int r = 0; r < CONV1_OUT_DIM; ++r) {
        for (int c = 0; c < CONV1_OUT_DIM; ++c) {
            int p_idx = r * CONV1_OUT_DIM + c;
            for (int f = 0; f < CONV1_FILTERS; ++f) {
                int32_t acc = conv1_bias[f];
                for (int w = 0; w < CONV1_WORDS; ++w) {
                    uint32_t a = conv1_patches[p_idx][w];
                    uint32_t b = conv1_weights[f][w];
                    acc += use_hw ? qvdot4_hw(a, b) : qvdot4_sw(a, b);
                }
                if (acc < 0) acc = 0;
                conv1_out[r][c][f] = clamp_s4(acc >> 4);
            }
        }
    }
    t1 = read_mcycle_lo();
    prof->conv1_cycles = t1 - t0;

    t0 = read_mcycle_lo();
    for (int r = 0; r < POOL1_OUT_DIM; ++r) {
        for (int c = 0; c < POOL1_OUT_DIM; ++c) {
            for (int f = 0; f < CONV1_FILTERS; ++f) {
                int32_t sum = (int32_t)conv1_out[2*r][2*c][f] +
                              (int32_t)conv1_out[2*r][2*c + 1][f] +
                              (int32_t)conv1_out[2*r + 1][2*c][f] +
                              (int32_t)conv1_out[2*r + 1][2*c + 1][f];
                pool1_out[r][c][f] = clamp_s4(sum >> 2);
            }
        }
    }
    t1 = read_mcycle_lo();
    prof->pool1_cycles = t1 - t0;

    pack_conv2_patches();

    t0 = read_mcycle_lo();
    for (int r = 0; r < CONV2_OUT_DIM; ++r) {
        for (int c = 0; c < CONV2_OUT_DIM; ++c) {
            int p_idx = r * CONV2_OUT_DIM + c;
            for (int f = 0; f < CONV2_FILTERS; ++f) {
                int32_t acc = conv2_bias[f];
                for (int w = 0; w < CONV2_WORDS; ++w) {
                    uint32_t a = conv2_patches[p_idx][w];
                    uint32_t b = conv2_weights[f][w];
                    acc += use_hw ? qvdot4_hw(a, b) : qvdot4_sw(a, b);
                }
                if (acc < 0) acc = 0;
                conv2_out[r][c][f] = clamp_s4(acc >> 5);
            }
        }
    }
    t1 = read_mcycle_lo();
    prof->conv2_cycles = t1 - t0;

    t0 = read_mcycle_lo();
    for (int r = 0; r < POOL2_OUT_DIM; ++r) {
        for (int c = 0; c < POOL2_OUT_DIM; ++c) {
            for (int f = 0; f < CONV2_FILTERS; ++f) {
                int32_t sum = (int32_t)conv2_out[2*r][2*c][f] +
                              (int32_t)conv2_out[2*r][2*c + 1][f] +
                              (int32_t)conv2_out[2*r + 1][2*c][f] +
                              (int32_t)conv2_out[2*r + 1][2*c + 1][f];
                pool2_out[flat_idx++] = clamp_s4(sum >> 2);
            }
        }
    }
    for (int w = 0; w < FC1_WORDS; ++w) {
        fc1_in_packed[w] = pack8_s4(&pool2_out[w * 8], 8);
    }
    t1 = read_mcycle_lo();
    prof->pool2_cycles = t1 - t0;

    t0 = read_mcycle_lo();
    for (int n = 0; n < FC1_OUT; ++n) {
        int32_t acc = fc1_bias[n];
        for (int w = 0; w < FC1_WORDS; ++w) {
            uint32_t a = fc1_in_packed[w];
            uint32_t b = fc1_weights[n][w];
            acc += use_hw ? qvdot4_hw(a, b) : qvdot4_sw(a, b);
        }
        if (acc < 0) acc = 0;
        fc1_out[n] = clamp_s4(acc >> 5);
    }
    for (int w = 0; w < FC2_WORDS; ++w) {
        fc2_in_packed[w] = pack8_s4(&fc1_out[w * 8], 8);
    }
    t1 = read_mcycle_lo();
    prof->fc1_cycles = t1 - t0;

    t0 = read_mcycle_lo();
    for (int n = 0; n < FC2_OUT; ++n) {
        int32_t acc = fc2_bias[n];
        for (int w = 0; w < FC2_WORDS; ++w) {
            uint32_t a = fc2_in_packed[w];
            uint32_t b = fc2_weights[n][w];
            acc += use_hw ? qvdot4_hw(a, b) : qvdot4_sw(a, b);
        }
        if (acc < 0) acc = 0;
        fc2_out[n] = clamp_s4(acc >> 4);
    }
    for (int w = 0; w < FC3_WORDS; ++w) {
        int rem = FC2_OUT - w * 8;
        fc3_in_packed[w] = pack8_s4(&fc2_out[w * 8], rem > 8 ? 8 : rem);
    }
    t1 = read_mcycle_lo();
    prof->fc2_cycles = t1 - t0;

    t0 = read_mcycle_lo();
    for (int n = 0; n < FC3_OUT; ++n) {
        int32_t acc = fc3_bias[n];
        for (int w = 0; w < FC3_WORDS; ++w) {
            uint32_t a = fc3_in_packed[w];
            uint32_t b = fc3_weights[n][w];
            acc += use_hw ? qvdot4_hw(a, b) : qvdot4_sw(a, b);
        }
        prof->logits[n] = acc;
        checksum += acc * (n + 1);
    }
    t1 = read_mcycle_lo();
    prof->fc3_cycles = t1 - t0;

    prof->total_cycles = prof->conv1_cycles + prof->pool1_cycles +
                         prof->conv2_cycles + prof->pool2_cycles +
                         prof->fc1_cycles   + prof->fc2_cycles   +
                         prof->fc3_cycles;
    prof->checksum = checksum;
}

int main(void)
{
    struct lenet5_profile sw_prof, hw_prof;
    int failures = 0;

    printf("LeNet-5 4-bit CNN benchmark (SW baseline vs. qvdot4 custom ISA)\n");
    init_model_and_input();

    run_lenet5(0, &sw_prof);
    run_lenet5(1, &hw_prof);

    for (int i = 0; i < FC3_OUT; ++i) {
        int ok = (sw_prof.logits[i] == hw_prof.logits[i]);
        if (!ok) {
            ++failures;
        }
        printf("Class %d: sw=%ld hw=%ld %s\n",
               i,
               (long)sw_prof.logits[i],
               (long)hw_prof.logits[i],
               ok ? "PASS" : "FAIL");
    }

    if (sw_prof.checksum != hw_prof.checksum) {
        ++failures;
    }

    printf("LAYER Conv1 (18816 dot4): sw_cycles=%lu hw_cycles=%lu\n",
           (unsigned long)sw_prof.conv1_cycles, (unsigned long)hw_prof.conv1_cycles);
    printf("LAYER Pool1 (14x14x6):    sw_cycles=%lu hw_cycles=%lu\n",
           (unsigned long)sw_prof.pool1_cycles, (unsigned long)hw_prof.pool1_cycles);
    printf("LAYER Conv2 (30400 dot4): sw_cycles=%lu hw_cycles=%lu\n",
           (unsigned long)sw_prof.conv2_cycles, (unsigned long)hw_prof.conv2_cycles);
    printf("LAYER Pool2 (5x5x16):     sw_cycles=%lu hw_cycles=%lu\n",
           (unsigned long)sw_prof.pool2_cycles, (unsigned long)hw_prof.pool2_cycles);
    printf("LAYER FC1   (6000 dot4):  sw_cycles=%lu hw_cycles=%lu\n",
           (unsigned long)sw_prof.fc1_cycles, (unsigned long)hw_prof.fc1_cycles);
    printf("LAYER FC2   (1260 dot4):  sw_cycles=%lu hw_cycles=%lu\n",
           (unsigned long)sw_prof.fc2_cycles, (unsigned long)hw_prof.fc2_cycles);
    printf("LAYER FC3   (110 dot4):   sw_cycles=%lu hw_cycles=%lu\n",
           (unsigned long)sw_prof.fc3_cycles, (unsigned long)hw_prof.fc3_cycles);

    printf("LENET5_TOTAL dot4_calls=56586 sw_cycles=%lu hw_cycles=%lu checksum=%ld\n",
           (unsigned long)sw_prof.total_cycles,
           (unsigned long)hw_prof.total_cycles,
           (long)hw_prof.checksum);

    printf("%s failures=%d\n", failures == 0 ? "RESULT PASS" : "RESULT FAIL", failures);
    return failures == 0 ? 0 : 1;
}
