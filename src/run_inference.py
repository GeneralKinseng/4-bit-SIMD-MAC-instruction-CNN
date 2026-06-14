from gem5.components.boards.simple_board import SimpleBoard
from gem5.components.memory.single_channel import SingleChannelDDR3_1600
from gem5.components.processors.simple_processor import SimpleProcessor
from gem5.components.processors.cpu_types import CPUTypes
from gem5.components.cachehierarchies.classic.private_l1_private_l2_cache_hierarchy import PrivateL1PrivateL2CacheHierarchy
from gem5.isas import ISA
from gem5.resources.resource import CustomResource
from gem5.simulate.simulator import Simulator

# 1. Set up a realistic Cache Hierarchy
cache_hierarchy = PrivateL1PrivateL2CacheHierarchy(
    l1d_size="32kB", l1i_size="32kB", l2_size="256kB"
)

# 2. Set up the Memory
memory = SingleChannelDDR3_1600(size="512MB")

# 3. Set up the Processor (Timing CPU for realistic cycle counts)
processor = SimpleProcessor(
    cpu_type=CPUTypes.TIMING,
    isa=ISA.RISCV,
    num_cores=1
)

# THE CRITICAL FIX: Force the RISC-V CPU into 32-bit mode
for core in processor.get_cores():
    core.core.isa[0].riscv_type = "RV32"

# 4. Set up the Board
board = SimpleBoard(
    clk_freq="1GHz",
    processor=processor,
    memory=memory,
    cache_hierarchy=cache_hierarchy
)

# 5. Set the Workload (Your 32-bit executable)
board.set_se_binary_workload(CustomResource("baseline_inference.elf"))

# 6. Run the Simulation
simulator = Simulator(board=board)
print("Starting gem5 RV32 simulation...")
simulator.run()
