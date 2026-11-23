// Copyright 2026 OpenHW Group / Master Research Project
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
//
// Design Name:    Packed Signed 4-bit Dot-Product Functional Unit (qvdot4)
// Language:       SystemVerilog
//
// Description:    Computes the 8-lane signed 4-bit dot product of two
//                 packed 32-bit operands in a single combinational cycle:
//                   result_o = sum_{i=0..7} sext4(a[4i+3:4i]) * sext4(b[4i+3:4i])

module cv32e40p_qvdot4 (
    input  logic [31:0] operand_a_i,
    input  logic [31:0] operand_b_i,
    output logic [31:0] result_o
);

  logic signed [ 7:0] a_sext [7:0];
  logic signed [ 7:0] b_sext [7:0];
  logic signed [ 7:0] prod   [7:0];
  logic signed [ 8:0] sum_l1 [3:0];
  logic signed [ 9:0] sum_l2 [1:0];
  logic signed [10:0] sum_l3;

  always_comb begin
    // 1. Nibble split, sign-extend (4b -> 8b), and 8 parallel signed multiplies
    for (int i = 0; i < 8; i++) begin
      a_sext[i] = $signed({{4{operand_a_i[4*i + 3]}}, operand_a_i[4*i +: 4]});
      b_sext[i] = $signed({{4{operand_b_i[4*i + 3]}}, operand_b_i[4*i +: 4]});
      prod[i]   = a_sext[i] * b_sext[i];
    end

    // 2. Three-level signed adder tree (8b -> 9b -> 10b -> 11b)
    for (int j = 0; j < 4; j++) begin
      sum_l1[j] = $signed({prod[2*j][7], prod[2*j]}) +
                  $signed({prod[2*j + 1][7], prod[2*j + 1]});
    end

    for (int k = 0; k < 2; k++) begin
      sum_l2[k] = $signed({sum_l1[2*k][8], sum_l1[2*k]}) +
                  $signed({sum_l1[2*k + 1][8], sum_l1[2*k + 1]});
    end

    sum_l3   = $signed({sum_l2[0][9], sum_l2[0]}) +
               $signed({sum_l2[1][9], sum_l2[1]});

    // 3. Final sign extension (11b -> 32b)
    result_o = {{21{sum_l3[10]}}, sum_l3};
  end

endmodule
