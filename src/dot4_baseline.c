/*
 * dot4_baseline.c - software baseline for the packed signed 4-bit dot product
 * that will later be replaced by the custom RISC-V `qvdot4` instruction.
 *
 * Each 32-bit word holds eight signed 4-bit lanes (lane 0 = bits [3:0]).
 * Standalone C: no TensorFlow Lite Micro, no floating point, no OS timing.
 * Cycle counts come from gem5 / CV32E40P statistics.
 */

#include <stdint.h>
#include <stdio.h>

#ifndef DOT4_ITERATIONS
#define DOT4_ITERATIONS 100000
#endif

/* Set to 0 to measure the kernel loop alone, without the verification loop. */
#ifndef DOT4_VERIFY
#define DOT4_VERIFY 1
#endif

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

/* Kernel under study: the whole word is consumed in place, lane by lane. */
static int32_t qvdot4_sw(uint32_t a, uint32_t b)
{
    int32_t acc = 0;
    for (int lane = 0; lane < 8; ++lane) {
        int shift = lane * 4;
        acc += sext4((a >> shift) & 0xFu) * sext4((b >> shift) & 0xFu);
    }
    return acc;
}

#if DOT4_VERIFY
/* Independent reference: unpack both words first, then multiply-accumulate. */
static int32_t qvdot4_unpacked(uint32_t a, uint32_t b)
{
    int8_t la[8], lb[8];
    int32_t acc = 0;
    for (int lane = 0; lane < 8; ++lane) {
        la[lane] = (int8_t)sext4((a >> (lane * 4)) & 0xFu);
        lb[lane] = (int8_t)sext4((b >> (lane * 4)) & 0xFu);
    }
    for (int lane = 0; lane < 8; ++lane) {
        acc += (int32_t)la[lane] * (int32_t)lb[lane];
    }
    return acc;
}
#endif

struct dot4_case {
    uint32_t a;
    uint32_t b;
    int32_t expected;
    const char *name;
};

static const struct dot4_case kCases[] = {
    { 0x00000000u, 0x11111111u,     0, "zero"          },
    { 0x11111111u, 0x11111111u,     8, "all_ones"      },
    { 0xFFFFFFFFu, 0x11111111u,    -8, "all_minus_one" },
    { 0x77777777u, 0x11111111u,    56, "max_positive"  },
    { 0x88888888u, 0x11111111u,   -64, "min_negative"  },
    { 0x88888888u, 0x88888888u,   512, "min_squared"   },
    { 0x89ABCDEFu, 0x01234567u,   -84, "mixed"         },
};

#define NUM_CASES ((int)(sizeof(kCases) / sizeof(kCases[0])))

static int run_directed_tests(void)
{
    int failures = 0;
    for (int i = 0; i < NUM_CASES; ++i) {
        int32_t got = qvdot4_sw(kCases[i].a, kCases[i].b);
        int ok = (got == kCases[i].expected);
        if (!ok) {
            ++failures;
        }
        printf("%s test %-14s a=0x%08lX b=0x%08lX got=%ld expected=%ld\n",
               ok ? "PASS" : "FAIL",
               kCases[i].name,
               (unsigned long)kCases[i].a,
               (unsigned long)kCases[i].b,
               (long)got,
               (long)kCases[i].expected);
    }
    return failures;
}

/* Deterministic 32-bit LCG so the benchmark inputs never depend on the host. */
static inline uint32_t next_rand(uint32_t *state)
{
    *state = (*state * 1664525u) + 1013904223u;
    return *state;
}

static int run_benchmark(void)
{
    uint32_t state = 0x12345678u;
    int32_t checksum = 0;

    uint32_t cycle_start = read_mcycle_lo();

    for (uint32_t i = 0; i < (uint32_t)DOT4_ITERATIONS; ++i) {
        uint32_t a = next_rand(&state);
        uint32_t b = next_rand(&state);
        checksum += qvdot4_sw(a, b);
    }

    uint32_t cycle_end = read_mcycle_lo();
    uint32_t cycle_delta = cycle_end - cycle_start;

    uint32_t cycles_whole = cycle_delta / (uint32_t)DOT4_ITERATIONS;
    uint32_t cycles_frac_x1000 =
        ((cycle_delta % (uint32_t)DOT4_ITERATIONS) * 1000u) /
        (uint32_t)DOT4_ITERATIONS;

    printf("BENCH iterations=%ld checksum=%ld\n",
           (long)DOT4_ITERATIONS, (long)checksum);

    printf("MCYCLE start=%lu end=%lu delta=%lu\n",
           (unsigned long)cycle_start,
           (unsigned long)cycle_end,
           (unsigned long)cycle_delta);

    printf("CYCLES_PER_DOT4=%lu.%03lu\n",
           (unsigned long)cycles_whole,
           (unsigned long)cycles_frac_x1000);

#if DOT4_VERIFY
    uint32_t ref_state = 0x12345678u;
    int32_t ref_checksum = 0;
    for (uint32_t i = 0; i < (uint32_t)DOT4_ITERATIONS; ++i) {
        uint32_t a = next_rand(&ref_state);
        uint32_t b = next_rand(&ref_state);
        ref_checksum += qvdot4_unpacked(a, b);
    }
    printf("%s checksum reference=%ld\n",
           (checksum == ref_checksum) ? "PASS" : "FAIL", (long)ref_checksum);
    return checksum != ref_checksum;
#else
    printf("SKIP checksum verification disabled\n");
    return 0;
#endif
}

int main(void)
{
    printf("dot4 baseline microbenchmark\n");

    int failures = run_directed_tests();
    failures += run_benchmark();

    printf("%s failures=%d\n", failures == 0 ? "RESULT PASS" : "RESULT FAIL",
           failures);
    return failures == 0 ? 0 : 1;
}
