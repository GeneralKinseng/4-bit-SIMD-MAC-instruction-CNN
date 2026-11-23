// Solderpad Hardware License, Version 0.51, see LICENSE for details.
//
// Design Name:    RI-MAC-inspired four-lane signed 4-bit dot product (qdot4x4)
// Language:       SystemVerilog
//
// Description:    Independent reimplementation for CV32E40P, inspired by the
//                 C-MUL44 mode of RI-MAC (Longchar et al., AIMLSystems 2025,
//                 doi:10.1109/AIMLSYSTEMS67835.2025.11330892). This is NOT the
//                 authors' RTL.
//
//                 result_o = sum_{i=0..3} sext4(a[4i+3:4i]) * sext4(b[4i+3:4i])
//
//                 Only operand bits [15:0] are used; bits [31:16] are ignored,
//                 matching RI-MAC's 16-bit C-MUL44 waveform example (Fig. 7).
//
//                 Each signed 4x4 product is composed from four 2x2 partial
//                 products, following RI-MAC's recursive decomposition:
//                   A*B = (AH*BH)<<4 + (AH*BL + AL*BH)<<2 + AL*BL
//                 For signed operands the high halves (AH, BH) are signed and
//                 the low halves (AL, BL) are unsigned.
//
//                 Combinational, single-cycle, no internal state.

module cv32e40p_qdot4x4 (
    input  logic [31:0] operand_a_i,
    input  logic [31:0] operand_b_i,
    output logic [31:0] result_o
);

  // 2-bit partial product. Each 2-bit operand is sign-extended when its
  // *_signed flag is set and zero-extended otherwise. Products lie in [-6, 9],
  // so 6 signed bits are sufficient.
  function automatic logic signed [5:0] mul2x2(input logic [1:0] x, input logic x_signed,
                                               input logic [1:0] y, input logic y_signed);
    logic signed [5:0] xs;
    logic signed [5:0] ys;
    // Explicit 6-bit extension so the multiply is evaluated at 6 bits in every tool.
    xs = $signed({{4{x_signed & x[1]}}, x});
    ys = $signed({{4{y_signed & y[1]}}, y});
    return xs * ys;
  endfunction

  logic signed [ 5:0] pp_hh [3:0];  // signed  AH x signed  BH
  logic signed [ 5:0] pp_hl [3:0];  // signed  AH x unsigned BL
  logic signed [ 5:0] pp_lh [3:0];  // unsigned AL x signed  BH
  logic signed [ 5:0] pp_ll [3:0];  // unsigned AL x unsigned BL
  logic signed [ 9:0] prod  [3:0];  // signed 4x4 product, range [-56, 64]
  logic signed [10:0] sum_l1[1:0];
  logic signed [11:0] sum_l2;       // four-lane sum, range [-224, 256]

  always_comb begin
    for (int i = 0; i < 4; i++) begin
      pp_hh[i] = mul2x2(operand_a_i[4*i+2 +: 2], 1'b1, operand_b_i[4*i+2 +: 2], 1'b1);
      pp_hl[i] = mul2x2(operand_a_i[4*i+2 +: 2], 1'b1, operand_b_i[4*i   +: 2], 1'b0);
      pp_lh[i] = mul2x2(operand_a_i[4*i   +: 2], 1'b0, operand_b_i[4*i+2 +: 2], 1'b1);
      pp_ll[i] = mul2x2(operand_a_i[4*i   +: 2], 1'b0, operand_b_i[4*i   +: 2], 1'b0);

      // Shift-and-add composition; all terms explicitly sign-extended to 10 bits.
      prod[i] = (10'(pp_hh[i]) <<< 4)
              + ((10'(pp_hl[i]) + 10'(pp_lh[i])) <<< 2)
              + 10'(pp_ll[i]);
    end

    // Two-level signed adder tree (10b -> 11b -> 12b)
    sum_l1[0] = 11'(prod[0]) + 11'(prod[1]);
    sum_l1[1] = 11'(prod[2]) + 11'(prod[3]);
    sum_l2    = 12'(sum_l1[0]) + 12'(sum_l1[1]);

    // Final sign extension (12b -> 32b)
    result_o  = {{20{sum_l2[11]}}, sum_l2};
  end

endmodule
