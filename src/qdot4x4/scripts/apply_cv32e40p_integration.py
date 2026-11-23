#!/usr/bin/env python3
"""Apply qvdot4 + qdot4x4 integration to a pristine CV32E40P checkout (hash fcd5968).

Usage: apply_cv32e40p_integration.py <path/to/core-v-cores/cv32e40p> <path/to/qdot4x4/rtl>
Edits: rtl/include/cv32e40p_pkg.sv, rtl/cv32e40p_decoder.sv, rtl/cv32e40p_ex_stage.sv,
       cv32e40p_manifest.flist; copies cv32e40p_qvdot4.sv and cv32e40p_qdot4x4.sv into rtl/.
Each edit fails loudly if its anchor text is not found exactly once.
"""
import pathlib, shutil, sys

core = pathlib.Path(sys.argv[1]); src = pathlib.Path(sys.argv[2])

def edit(rel, old, new):
    p = core / rel; t = p.read_text()
    if t.count(old) != 1:
        sys.exit(f"anchor not found exactly once in {rel}:\n{old}")
    p.write_text(t.replace(old, new)); print(f"patched {rel}")

# 1. Package: custom-0 opcode name and two new ALU operator codes (unused values).
edit("rtl/include/cv32e40p_pkg.sv",
     "  parameter OPCODE_LOAD_POST = 7'h0b;",
     "  parameter OPCODE_CUSTOM_0 = 7'h0b;  // qvdot4 / qdot4x4 (PULP_XPULP=0 only)\n"
     "  parameter OPCODE_LOAD_POST = 7'h0b;")
edit("rtl/include/cv32e40p_pkg.sv",
     "    ALU_PCKHI = 7'b0111001\n",
     "    ALU_PCKHI = 7'b0111001,\n\n"
     "    // Custom 4-bit dot-product units (write back through the ALU result port)\n"
     "    ALU_QDOT4X4 = 7'b1111110,  // RI-MAC-inspired 4-lane reduction\n"
     "    ALU_QVDOT4  = 7'b1111111   // 8-lane reduction\n")

# 2. Decoder: dedicated custom-0 case. OPCODE_LOAD_POST shares 7'h0b and is only
#    meaningful with PULP_XPULP=1, which core-v-verif's cv32e40p config does not use.
edit("rtl/cv32e40p_decoder.sv",
     "      OPCODE_LOAD,\n      OPCODE_LOAD_POST: begin\n",
     "      OPCODE_CUSTOM_0: begin\n"
     "        if ((instr_rdata_i[14:12] == 3'b000) && (instr_rdata_i[31:25] == 7'b0000000)) begin\n"
     "          // qvdot4 rd, rs1, rs2 : 8-lane signed 4-bit dot product\n"
     "          regfile_alu_we = 1'b1;\n"
     "          rega_used_o    = 1'b1;\n"
     "          regb_used_o    = 1'b1;\n"
     "          alu_operator_o = ALU_QVDOT4;\n"
     "        end else if ((instr_rdata_i[14:12] == 3'b001) && (instr_rdata_i[31:25] == 7'b0000001)) begin\n"
     "          // qdot4x4 rd, rs1, rs2 : 4-lane signed 4-bit dot product on rs1[15:0], rs2[15:0]\n"
     "          // funct3/funct7 follow RI-MAC Table III C-MUL44 fields.\n"
     "          regfile_alu_we = 1'b1;\n"
     "          rega_used_o    = 1'b1;\n"
     "          regb_used_o    = 1'b1;\n"
     "          alu_operator_o = ALU_QDOT4X4;\n"
     "        end else begin\n"
     "          illegal_insn_o = 1'b1;\n"
     "        end\n"
     "      end\n\n"
     "      OPCODE_LOAD: begin\n")

# 3. Execute stage: instantiate both units and select their results on the ALU port.
edit("rtl/cv32e40p_ex_stage.sv",
     "  logic [31:0] alu_result;\n",
     "  logic [31:0] alu_result;\n"
     "  logic [31:0] qvdot4_result;\n"
     "  logic [31:0] qdot4x4_result;\n"
     "  logic        qvdot4_en;\n"
     "  logic        qdot4x4_en;\n")
edit("rtl/cv32e40p_ex_stage.sv",
     "      if (alu_en_i) regfile_alu_wdata_fw_o = alu_result;\n",
     "      if (alu_en_i)\n"
     "        regfile_alu_wdata_fw_o = qvdot4_en  ? qvdot4_result  :\n"
     "                                 qdot4x4_en ? qdot4x4_result : alu_result;\n")
edit("rtl/cv32e40p_ex_stage.sv",
     "  cv32e40p_alu alu_i (\n",
     "  assign qvdot4_en  = alu_en_i && (alu_operator_i == ALU_QVDOT4);\n"
     "  assign qdot4x4_en = alu_en_i && (alu_operator_i == ALU_QDOT4X4);\n\n"
     "  cv32e40p_qvdot4 qvdot4_i (\n"
     "      .operand_a_i(alu_operand_a_i),\n"
     "      .operand_b_i(alu_operand_b_i),\n"
     "      .result_o   (qvdot4_result)\n"
     "  );\n\n"
     "  cv32e40p_qdot4x4 qdot4x4_i (\n"
     "      .operand_a_i(alu_operand_a_i),\n"
     "      .operand_b_i(alu_operand_b_i),\n"
     "      .result_o   (qdot4x4_result)\n"
     "  );\n\n"
     "  cv32e40p_alu alu_i (\n")

# 4. Manifest and sources.
edit("cv32e40p_manifest.flist",
     "${DESIGN_RTL_DIR}/cv32e40p_alu.sv\n",
     "${DESIGN_RTL_DIR}/cv32e40p_alu.sv\n"
     "${DESIGN_RTL_DIR}/cv32e40p_qvdot4.sv\n"
     "${DESIGN_RTL_DIR}/cv32e40p_qdot4x4.sv\n")
for f in ("cv32e40p_qvdot4.sv", "cv32e40p_qdot4x4.sv"):
    shutil.copy(src / f, core / "rtl" / f); print(f"copied rtl/{f}")
