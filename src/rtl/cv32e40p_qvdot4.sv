module cv32e40p_qvdot4 (
    input  logic [31:0] operand_a_i,
    input  logic [31:0] operand_b_i,
    output logic [31:0] result_o
);

  logic signed [ 3:0] a_lane [8];
  logic signed [ 3:0] b_lane [8];
  logic signed [ 7:0] a_sext [8];
  logic signed [ 7:0] b_sext [8];
  logic signed [ 7:0] prod   [8];

  logic signed [ 8:0] sum_l1 [4];
  logic signed [ 9:0] sum_l2 [2];
  logic signed [10:0] sum_l3;

  always_comb begin
    // 1. Nibble split, sign-extend (4b -> 8b), and 8 parallel signed multipliers
    for (int i = 0; i < 8; i++) begin
      a_lane[i] = signed'(operand_a_i[4*i +: 4]);
      b_lane[i] = signed'(operand_b_i[4*i +: 4]);
      a_sext[i] = 8'(a_lane[i]);
      b_sext[i] = 8'(b_lane[i]);
      prod[i]   = a_sext[i] * b_sext[i];
    end

    // 2. Three-level signed adder tree (8b -> 9b -> 10b -> 11b)
    for (int j = 0; j < 4; j++) begin
      sum_l1[j] = 9'(prod[2*j]) + 9'(prod[2*j + 1]);
    end

    for (int k = 0; k < 2; k++) begin
      sum_l2[k] = 10'(sum_l1[2*k]) + 10'(sum_l1[2*k + 1]);
    end

    sum_l3 = 11'(sum_l2[0]) + 11'(sum_l2[1]);

    // 3. Final sign extension (11b -> 32b)
    result_o = 32'(sum_l3);
  end

endmodule
