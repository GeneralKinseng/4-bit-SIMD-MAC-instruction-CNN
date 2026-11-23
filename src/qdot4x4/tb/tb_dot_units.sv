// Unit-test top: instantiates the existing qvdot4 and the new qdot4x4 side by side.
module tb_dot_units (
    input  logic [31:0] a_i,
    input  logic [31:0] b_i,
    output logic [31:0] qvdot4_o,
    output logic [31:0] qdot4x4_lo_o,
    output logic [31:0] qdot4x4_hi_o
);
  cv32e40p_qvdot4  u_q8   (.operand_a_i(a_i),         .operand_b_i(b_i),         .result_o(qvdot4_o));
  cv32e40p_qdot4x4 u_q4lo (.operand_a_i(a_i),         .operand_b_i(b_i),         .result_o(qdot4x4_lo_o));
  cv32e40p_qdot4x4 u_q4hi (.operand_a_i(a_i >> 16),   .operand_b_i(b_i >> 16),   .result_o(qdot4x4_hi_o));
endmodule
