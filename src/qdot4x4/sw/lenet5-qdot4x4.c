/*
 * lenet5-qdot4x4.c - Three-way 4-bit LeNet-5 benchmark for CV32E40P (core-v-verif).
 *
 * Derived from lenet5-qvdot4.c. Same synthetic network, weights, and input.
 * Compares, on ONE modified core and ONE ELF:
 *   mode 0  SW   : software dot product (qvdot4_sw)
 *   mode 1  Q8   : qvdot4   - 8-lane signed 4-bit reduction (.insn r 0x0b, 0, 0)
 *   mode 2  Q4x2 : qdot4x4  - RI-MAC-inspired 4-lane reduction (.insn r 0x0b, 1, 1),
 *                  issued twice per packed word (low and high halfwords) and summed.
 *
 * Build with -DSCALAR_ONLY to produce an ELF that contains only mode 0 and no
 * custom instructions (for the unmodified-core vs. modified-core check).
 *
 * Timing scopes reported per mode:
 *   layer_sum      : sum of the seven layer timers (same scope as the original benchmark)
 *   pack           : Conv1 and Conv2 patch packing (previously outside the timers)
 *   inference      : one mcycle interval from Conv1 packing through FC3, including packing
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
#if defined(__riscv)
    uint32_t value;
    __asm__ volatile ("csrr %0, mcycle" : "=r"(value));
    return value;
#else
    return 0u;
#endif
}

static inline int32_t sext4(uint32_t nibble)
{
    return (int32_t)(int8_t)((uint8_t)(nibble << 4)) >> 4;
}

/* Software baseline dot4 kernel (same definition as dot4_baseline.c) */
static int32_t qvdot4_sw(uint32_t a, uint32_t b)
{
    int32_t acc = 0;
    for (int lane = 0; lane < 8; ++lane) {
        int shift = lane * 4;
        acc += sext4((a >> shift) & 0xFu) * sext4((b >> shift) & 0xFu);
    }
    return acc;
}

/* Custom hardware instruction: qvdot4 rd, rs1, rs2 */
static inline int32_t qvdot4_hw(uint32_t a, uint32_t b)
{
#if defined(__riscv) && !defined(SCALAR_ONLY)
    int32_t res;
    __asm__ volatile (
        ".insn r 0x0b, 0, 0, %0, %1, %2"
        : "=r"(res)
        : "r"(a), "r"(b)
    );
    return res;
#else
    return qvdot4_sw(a, b);
#endif
}

#ifndef SCALAR_ONLY
/* RI-MAC-inspired instruction: qdot4x4 rd, rs1, rs2 (uses rs1[15:0], rs2[15:0]) */
static inline int32_t qdot4x4_hw(uint32_t a, uint32_t b)
{
#if defined(__riscv)
    int32_t res;
    __asm__ volatile (".insn r 0x0b, 1, 1, %0, %1, %2" : "=r"(res) : "r"(a), "r"(b));
    return res;
#else
    /* Host fallback: reference semantics of the instruction */
    int32_t acc = 0;
    for (int lane = 0; lane < 4; ++lane)
        acc += sext4((a >> (4 * lane)) & 0xFu) * sext4((b >> (4 * lane)) & 0xFu);
    return acc;
#endif
}

/* Eight lanes via two four-lane reductions: low halfword + high halfword */
static inline int32_t qdot4x4_pair(uint32_t a, uint32_t b)
{
    return qdot4x4_hw(a, b) + qdot4x4_hw(a >> 16, b >> 16);
}
#endif

#define MODE_SW   0
#define MODE_Q8   1
#define MODE_Q4X2 2

#ifdef SCALAR_ONLY
#define DOT(mode, a, b) qvdot4_sw((a), (b))
#else
#define DOT(mode, a, b) ((mode) == MODE_Q8   ? qvdot4_hw((a), (b)) : \
                         (mode) == MODE_Q4X2 ? qdot4x4_pair((a), (b)) : \
                                               qvdot4_sw((a), (b)))
#endif

/* Clamp signed integer to signed 4-bit range [-8, +7] */
static inline int8_t clamp_s4(int32_t v)
{
    if (v > 7)  return 7;
    if (v < -8) return -8;
    return (int8_t)v;
}

/* Deterministic PRNG for reproducible 4-bit weights and test image */
static inline uint32_t next_rand(uint32_t *state)
{
    *state = (*state * 1664525u) + 1013904223u;
    return *state;
}

/* Static network buffers in RAM */
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
    uint32_t total_cycles;      /* layer_sum */
    uint32_t pack_cycles;       /* Conv1 + Conv2 patch packing */
    uint32_t inference_cycles;  /* single interval incl. packing */
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

    /* Initialize synthetic 32x32 4-bit input image (digit-like center pattern) */
    for (int r = 0; r < 32; ++r) {
        for (int c = 0; c < 32; ++c) {
            int center = (r >= 6 && r < 26 && c >= 6 && c < 26);
            int32_t v = center ? (int32_t)((next_rand(&seed) & 0x7u)) : -2;
            input_img[r][c] = clamp_s4(v);
        }
    }

    /* Conv1 weights (25 lanes active; mask top 7 lanes of word 3 to zero) */
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

    /* Conv2 weights (150 lanes active; mask top 2 lanes of word 18 to zero) */
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

    /* FC1 weights (400 lanes = 50 full words) */
    for (int n = 0; n < FC1_OUT; ++n) {
        for (int w = 0; w < FC1_WORDS; ++w) {
            fc1_weights[n][w] = next_rand(&seed);
        }
        fc1_bias[n] = (int32_t)(next_rand(&seed) & 0x7u);
    }

    /* FC2 weights (120 lanes = 15 full words) */
    for (int n = 0; n < FC2_OUT; ++n) {
        for (int w = 0; w < FC2_WORDS; ++w) {
            fc2_weights[n][w] = next_rand(&seed);
        }
        fc2_bias[n] = (int32_t)(next_rand(&seed) & 0x7u);
    }

    /* FC3 weights (84 lanes -> 11 words; mask top 4 lanes of word 10 to zero) */
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

static void pack_conv1_patches(void)
{
    /* Pre-pack 28x28 Conv1 5x5 input patches (25 lanes -> 4 words, last 7 lanes 0) */
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
    int mode, struct lenet5_profile *prof)
{
    uint32_t t0, t1, t_start, p0, p1;
    (void)mode;
    int flat_idx = 0;
    int32_t checksum = 0;

    t_start = read_mcycle_lo();
    p0 = t_start;
    pack_conv1_patches();
    p1 = read_mcycle_lo();
    prof->pack_cycles = p1 - p0;

    /* 1. Conv1 (28x28x6, 4 packed dot4 calls per output = 18,816 dot4 calls) */
    t0 = read_mcycle_lo();
    for (int r = 0; r < CONV1_OUT_DIM; ++r) {
        for (int c = 0; c < CONV1_OUT_DIM; ++c) {
            int p_idx = r * CONV1_OUT_DIM + c;
            for (int f = 0; f < CONV1_FILTERS; ++f) {
                int32_t acc = conv1_bias[f];
                for (int w = 0; w < CONV1_WORDS; ++w) {
                    uint32_t a = conv1_patches[p_idx][w];
                    uint32_t b = conv1_weights[f][w];
                    acc += DOT(mode, a, b);
                }
                if (acc < 0) acc = 0; /* ReLU */
                conv1_out[r][c][f] = clamp_s4(acc >> 4);
            }
        }
    }
    t1 = read_mcycle_lo();
    prof->conv1_cycles = t1 - t0;

    /* 2. AvgPool1 (28x28x6 -> 14x14x6) */
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

    p0 = read_mcycle_lo();
    pack_conv2_patches();
    p1 = read_mcycle_lo();
    prof->pack_cycles += p1 - p0;

    /* 3. Conv2 (10x10x16, 19 packed dot4 calls per output = 30,400 dot4 calls) */
    t0 = read_mcycle_lo();
    for (int r = 0; r < CONV2_OUT_DIM; ++r) {
        for (int c = 0; c < CONV2_OUT_DIM; ++c) {
            int p_idx = r * CONV2_OUT_DIM + c;
            for (int f = 0; f < CONV2_FILTERS; ++f) {
                int32_t acc = conv2_bias[f];
                for (int w = 0; w < CONV2_WORDS; ++w) {
                    uint32_t a = conv2_patches[p_idx][w];
                    uint32_t b = conv2_weights[f][w];
                    acc += DOT(mode, a, b);
                }
                if (acc < 0) acc = 0; /* ReLU */
                conv2_out[r][c][f] = clamp_s4(acc >> 5);
            }
        }
    }
    t1 = read_mcycle_lo();
    prof->conv2_cycles = t1 - t0;

    /* 4. AvgPool2 (10x10x16 -> 5x5x16 = 400) + pack into 50 words */
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

    /* 5. FC1 (400 -> 120, 50 packed dot4 calls per neuron = 6,000 dot4 calls) */
    t0 = read_mcycle_lo();
    for (int n = 0; n < FC1_OUT; ++n) {
        int32_t acc = fc1_bias[n];
        for (int w = 0; w < FC1_WORDS; ++w) {
            uint32_t a = fc1_in_packed[w];
            uint32_t b = fc1_weights[n][w];
            acc += DOT(mode, a, b);
        }
        if (acc < 0) acc = 0; /* ReLU */
        fc1_out[n] = clamp_s4(acc >> 5);
    }
    for (int w = 0; w < FC2_WORDS; ++w) {
        fc2_in_packed[w] = pack8_s4(&fc1_out[w * 8], 8);
    }
    t1 = read_mcycle_lo();
    prof->fc1_cycles = t1 - t0;

    /* 6. FC2 (120 -> 84, 15 packed dot4 calls per neuron = 1,260 dot4 calls) */
    t0 = read_mcycle_lo();
    for (int n = 0; n < FC2_OUT; ++n) {
        int32_t acc = fc2_bias[n];
        for (int w = 0; w < FC2_WORDS; ++w) {
            uint32_t a = fc2_in_packed[w];
            uint32_t b = fc2_weights[n][w];
            acc += DOT(mode, a, b);
        }
        if (acc < 0) acc = 0; /* ReLU */
        fc2_out[n] = clamp_s4(acc >> 4);
    }
    for (int w = 0; w < FC3_WORDS; ++w) {
        int rem = FC2_OUT - w * 8;
        fc3_in_packed[w] = pack8_s4(&fc2_out[w * 8], rem > 8 ? 8 : rem);
    }
    t1 = read_mcycle_lo();
    prof->fc2_cycles = t1 - t0;

    /* 7. FC3 Output Layer (84 -> 10, 11 packed dot4 calls per class = 110 dot4 calls) */
    t0 = read_mcycle_lo();
    for (int n = 0; n < FC3_OUT; ++n) {
        int32_t acc = fc3_bias[n];
        for (int w = 0; w < FC3_WORDS; ++w) {
            uint32_t a = fc3_in_packed[w];
            uint32_t b = fc3_weights[n][w];
            acc += DOT(mode, a, b);
        }
        prof->logits[n] = acc;
        checksum += acc * (n + 1);
    }
    t1 = read_mcycle_lo();
    prof->fc3_cycles = t1 - t0;
    prof->inference_cycles = t1 - t_start;

    prof->total_cycles = prof->conv1_cycles + prof->pool1_cycles +
                         prof->conv2_cycles + prof->pool2_cycles +
                         prof->fc1_cycles   + prof->fc2_cycles   +
                         prof->fc3_cycles;
    prof->checksum = checksum;
}

static void print_profile(const char *name, const struct lenet5_profile *p)
{
    printf("MODE %-4s conv1=%lu pool1=%lu conv2=%lu pool2=%lu fc1=%lu fc2=%lu fc3=%lu "
           "layer_sum=%lu pack=%lu inference=%lu checksum=%ld\n", name,
           (unsigned long)p->conv1_cycles, (unsigned long)p->pool1_cycles,
           (unsigned long)p->conv2_cycles, (unsigned long)p->pool2_cycles,
           (unsigned long)p->fc1_cycles, (unsigned long)p->fc2_cycles,
           (unsigned long)p->fc3_cycles, (unsigned long)p->total_cycles,
           (unsigned long)p->pack_cycles, (unsigned long)p->inference_cycles,
           (long)p->checksum);
}

#ifndef SCALAR_ONLY
static int compare(const char *name, const struct lenet5_profile *ref,
                   const struct lenet5_profile *p)
{
    int failures = 0;
    for (int i = 0; i < FC3_OUT; ++i) {
        if (ref->logits[i] != p->logits[i]) {
            ++failures;
            printf("Class %d: sw=%ld %s=%ld FAIL\n", i, (long)ref->logits[i], name, (long)p->logits[i]);
        }
    }
    if (ref->checksum != p->checksum) ++failures;
    printf("CHECK %-4s vs SW: %s\n", name, failures == 0 ? "PASS (10/10 logits, checksum)" : "FAIL");
    return failures;
}
#endif

int main(void)
{
    struct lenet5_profile sw_prof;
    int failures = 0;

    init_model_and_input();
#ifdef SCALAR_ONLY
    printf("LeNet-5 4-bit benchmark, SCALAR_ONLY build (no custom instructions)\n");
    run_lenet5(MODE_SW, &sw_prof);
    print_profile("SW", &sw_prof);
#else
    struct lenet5_profile q8_prof, q4_prof;
    printf("LeNet-5 4-bit benchmark: SW vs qvdot4 (8-lane) vs qdot4x4 pair (RI-MAC-inspired 4-lane)\n");
    run_lenet5(MODE_SW,   &sw_prof);
    run_lenet5(MODE_Q8,   &q8_prof);
    run_lenet5(MODE_Q4X2, &q4_prof);
    for (int i = 0; i < FC3_OUT; ++i)
        printf("Class %d: sw=%ld qvdot4=%ld qdot4x4=%ld\n", i, (long)sw_prof.logits[i],
               (long)q8_prof.logits[i], (long)q4_prof.logits[i]);
    failures += compare("Q8", &sw_prof, &q8_prof);
    failures += compare("Q4X2", &sw_prof, &q4_prof);
    print_profile("SW", &sw_prof);
    print_profile("Q8", &q8_prof);
    print_profile("Q4X2", &q4_prof);
#endif
    printf("%s failures=%d\n", failures == 0 ? "RESULT PASS" : "RESULT FAIL", failures);
    return failures == 0 ? 0 : 1;
}
