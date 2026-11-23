// Verilator unit test for cv32e40p_qdot4x4 (RI-MAC-inspired) and cv32e40p_qvdot4.
#include "Vtb_dot_units.h"
#include "verilated.h"
#include <cstdint>
#include <cstdio>

static int32_t s4(uint32_t x) { x &= 0xF; return (x & 0x8) ? (int32_t)x - 16 : (int32_t)x; }
static int32_t ref4(uint32_t a, uint32_t b) {
    int32_t s = 0;
    for (int i = 0; i < 4; ++i) s += s4(a >> (4 * i)) * s4(b >> (4 * i));
    return s;
}
static int32_t ref8(uint32_t a, uint32_t b) {
    int32_t s = 0;
    for (int i = 0; i < 8; ++i) s += s4(a >> (4 * i)) * s4(b >> (4 * i));
    return s;
}
static uint32_t rng = 0x12345678u;
static uint32_t rnd() { rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; return rng; }

static Vtb_dot_units* dut;
static long checks = 0, fails = 0;
static void apply(uint32_t a, uint32_t b) {
    dut->a_i = a; dut->b_i = b; dut->eval();
    int32_t lo = (int32_t)dut->qdot4x4_lo_o, hi = (int32_t)dut->qdot4x4_hi_o, q8 = (int32_t)dut->qvdot4_o;
    bool ok = lo == ref4(a, b) && hi == ref4(a >> 16, b >> 16) && q8 == ref8(a, b) && lo + hi == q8;
    ++checks;
    if (!ok && fails++ < 10)
        printf("FAIL a=%08x b=%08x qdot4x4=%d (exp %d) qvdot4=%d (exp %d)\n", a, b, lo, ref4(a, b), q8, ref8(a, b));
}
static void directed(const char* name, uint32_t a, uint32_t b, int32_t expect) {
    apply(a, b);
    int32_t got = (int32_t)dut->qdot4x4_lo_o;
    bool ok = got == expect;
    if (!ok) ++fails;
    printf("%-26s a=0x%08x b=0x%08x expected=%5d qdot4x4=%5d %s\n", name, a, b, expect, got, ok ? "PASS" : "FAIL");
}

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    dut = new Vtb_dot_units;

    directed("RI-MAC Fig.7 example", 0x0000e583u, 0x0000c937u, -30);
    directed("zero", 0x00000000u, 0x00001111u, 0);
    directed("all_ones", 0x00001111u, 0x00001111u, 4);
    directed("all_minus_one", 0x0000FFFFu, 0x00001111u, -4);
    directed("max_positive x 1", 0x00007777u, 0x00001111u, 28);
    directed("min_squared (max result)", 0x00008888u, 0x00008888u, 256);
    directed("max x min (min result)", 0x00007777u, 0x00008888u, -224);
    directed("mixed", 0x0000CDEFu, 0x00004567u, -4*4 + -3*5 + -2*6 + -1*7);
    directed("upper 16 bits ignored", 0xDEAD1111u, 0xBEEF1111u, 4);

    // 1) Exhaustive per lane: all 256 nibble pairs in each of the 4 lanes,
    //    with other lanes zero and with random other lanes and upper halves.
    for (int lane = 0; lane < 4; ++lane)
        for (uint32_t x = 0; x < 16; ++x)
            for (uint32_t y = 0; y < 16; ++y) {
                apply(x << (4 * lane), y << (4 * lane));
                for (int r = 0; r < 16; ++r) {
                    uint32_t m = ~(0xFu << (4 * lane));
                    apply((rnd() & m) | (x << (4 * lane)), (rnd() & m) | (y << (4 * lane)));
                }
            }
    // 2) Exhaustive over a full 16-bit operand A against 256 random 16-bit B values.
    for (int k = 0; k < 256; ++k) {
        uint32_t b = rnd();
        for (uint32_t a = 0; a < 65536; ++a) apply(a | (rnd() & 0xFFFF0000u), b);
    }
    // 3) Random 32-bit operands.
    for (int k = 0; k < 2000000; ++k) apply(rnd(), rnd());

    printf("RANDOM+EXHAUSTIVE checks=%ld failures=%ld\n", checks, fails);
    printf("%s\n", fails == 0 ? "UNIT RESULT PASS" : "UNIT RESULT FAIL");
    delete dut;
    return fails == 0 ? 0 : 1;
}
