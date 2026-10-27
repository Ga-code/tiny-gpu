# Tiny-GPU: Image Brightness Adjustment & Architectural Optimizations[cite: 4]

This repository contains an end-to-end implementation of an 8-bit saturating image brightness adjustment kernel on `tiny-gpu`, along with structural upgrades for **Divergence**, **Instruction Caching (I-Cache)**, and **ALU Bug Fixes**.[cite: 4]

---

## 1. Repository Structure & Setup[cite: 4]

* **`src/`**: SystemVerilog RTL source files (modified `alu.sv`, `core.sv`, `scheduler.sv`, and `fetcher.sv`).[cite: 4]
* **`test/test_brightness.py`**: Cocotb testbench implementing host-to-device memory batching, cycle telemetry, and bit-accurate verification against a NumPy golden model.[cite: 4]
* **`test/parrot.jpg`, `test/rose.jpg` & `test/parrot_brightened_original.jpg`, `test/rose_brightened_original.jpg`**: Input test images and gray-scaled and down-sized into appropriate test examples to compare output against.[cite: 4]
* **`build/gpu.v`**: Compiled Verilog-2005 netlist generated via `sv2v`.[cite: 4]

### Running the Simulation[cite: 4]
```bash
# 1. Compile and run, the file also contains verification
make test_brightness
```
[cite: 4]

---

## 2. Thread Mapping & ISA Kernel Code

### Pixel-to-Thread Mapping
Each 2D grayscale image ($H \times W$) is flattened into a 1D array of unsigned 8-bit pixels. Because `tiny-gpu` has an 8-bit address space ($2^8 = 256$ bytes), the image is then split into batches of **248 pixels** ($31 \text{ blocks} \times 8 \text{ threads/wave}$). 

$$R_0 = (\text{\%blockIdx} \times \text{\%blockDim}) + \text{\%threadIdx}$$

### Saturating Brightness ISA Kernel
To prevent 8-bit overflow ($\text{pixel} + k > 255$) from wrapping around modulo 256, each thread calculates the remaining headroom before saturation ($R_4 = 255 - \text{pixel}$) and compares it against the brightness constant $k$:

```python
program = [
    0b0101000011011110,        # 0:  MUL R0, %blockIdx, %blockDim
    0b0011000000001111,        # 1:  ADD R0, R0, %threadIdx
    0b1001000100000000 | k,    # 2:  CONST R1, k
    0b0111001000000000,        # 3:  LDR R2, R0               ; R2 = pixel[R0]
    0b1001001111111111,        # 4:  CONST R3, #255           ; R3 = 255
    0b0100010000110010,        # 5:  SUB R4, R3, R2           ; R4 = 255 - pixel
    0b0010000001000001,        # 6:  CMP R4, R1               ; compare (255 - pixel) vs k
    0b0001011000001010,        # 7:  BRzp #10                 ; if (255 - pixel) >= k, jump to PC=10
    0b1001001011111111,        # 8:  CONST R2, #255           ; overflow path: clamp R2 = 255
    0b1001000100000000,        # 9:  CONST R1, #0             ; overflow path: R1 = 0
    0b0011010100100001,        # 10: ADD R5, R2, R1           ; reconverge: R5 = R2 + R1
    0b1000000000000101,        # 11: STR R0, R5               ; store R5 back to memory[R0]
    0b1111000000000000         # 12: RET
]
```

---

## 3. Structural RTL Upgrades

1. **SIMT Branch Divergence (`src/scheduler.sv`, `src/core.sv`)**:
   * **Issue**: Baseline `scheduler.sv` updated `current_pc <= next_pc[THREADS_PER_BLOCK-1]`, assuming all threads in a block branch identically. When only a subset of pixels in a block overflowed at `BRzp #10`, non-overflowing threads were forced down the wrong execution path.
   * **Fix**: Added per-thread PC tracking (`thread_pc[i]`) and gated the `.enable` signal on all thread units (`alu`, `lsu`, `registers`, `pc`) with `i < thread_count && thread_pc[i] == current_pc`. On `UPDATE`, the scheduler updates `current_pc` to the **minimum `next_pc`** across active threads, executing the fall-through path while pausing threads waiting at `#10` until both paths automatically reconverge.
2. **ALU Unsigned Comparator Fix (`src/alu.sv`)**:
   * **Issue**: Baseline `alu.sv` evaluated `CMP` flags using `(rs - rt > 0)` and `(rs - rt < 0)`. Because `rs` and `rt` are unsigned 8-bit wires (`[7:0]`), `rs - rt` wraps around modulo 256 and is never `< 0`, leaving the `P` (Positive) flag permanently stuck at `0` and breaking `BRzp`.
   * **Fix**: Replaced subtraction checks with direct unsigned magnitude comparators: `{5'b0, (rs < rt), (rs == rt), (rs > rt)}`.
3. **Instruction Cache (`src/fetcher.sv`)**:
   * **Issue**: Every core re-fetched the same 13 instructions over a 1-channel program memory bus on every wave, stalling in `FETCH` for ~6 cycles per instruction.
   * **Fix**: Added a 256-entry `icache` and `icache_valid` register array inside `fetcher.sv`. After Wave 0 loads the instructions from external memory, all subsequent waves hit in the local cache in **1 cycle**, bypassing both the memory controller and Python VPI overhead.

---

## 4. Timing, Scaling & Verification

### Verification
Output accuracy is verified in `test/test_brightness.py` by comparing every GPU-processed pixel against a clamped 16-bit NumPy reference array (`np.clip(img_array.astype(np.uint16) + k, 0, 255).astype(np.uint8)`). All tested images pass with **100% bit-accurate agreement**.

### Execution Timing & Scaling
| Image Size | Total Pixels | Batches (255 px/batch) | Total Clock Cycles | Wall-Clock Simulation Time |
| :--- | :--- | :--- | :--- | :--- |
| **64 × 64** | 4,096 | 16 | ~26,500 | ~13 s |
| **128 × 128** | 16,384 | 64 | ~106,000 | ~40 s |

* **Linear Hardware Scaling ($O(N)$)**: Both total clock cycles and wall-clock time scale linearly with the pixel count $N$ ($4\times$ increase in pixels from $64 \times 64$ to $128 \times 128$ yields a $\sim 4\times$ increase in cycles and simulation time).
* **Optimization Impact**: Adding the `fetcher.sv` I-Cache and skipping idle `memory.run()` calls in Cocotb reduced $64 \times 64$ simulation time from ~50s down to ~13s.

---

## 5. System Limitations & Trade-Offs

1. **8-Bit Datapath & Memory Addressing Limit**:
   * Because `DATA_MEM_ADDR_BITS = 8` and general-purpose registers (`R0`–`R15`) are 8 bits wide, the hardware cannot natively address more than 256 bytes or index past thread 255 in a single launch.
   * Additionally, the `CONST` instruction (`1001_dddd_iiiiiiii`) only supports an 8-bit immediate field (`#0`–`#255`), tying constants to 8 bits unless the 16-bit instruction format is modified.
2. **Core Scaling vs. Memory Controller Crossbar Overhead**:
   * Scaling `NUM_CORES` and `THREADS_PER_BLOCK` reduces simulated GPU clock cycles per batch, but it requires increasing `DATA_MEM_NUM_CHANNELS` to prevent `LDR`/`STR` bottlenecks, which increases setup time and hence total time taken per batch
3. **Python/C++ VPI Context Switching**:
   * Calling `await RisingEdge(dut.clk)` and evaluating `memory.run()` in Python on every clock tick limits simulation speed since verilog has to talk to python for basically every cycle., making the Python/Verilog VPI handoff the primary wall-clock bottleneck. Ram that is not associated with python is likely to significantly decrease compute time.

---

## 6. Future Architectural Edits

* **Widening Address & Register Datapaths to 16 Bits**: Expanding `DATA_MEM_ADDR_BITS`, `dcr.sv`, `dispatch.sv`, and register/ALU widths (`[7:0]` $\rightarrow$ `[15:0]`) while keeping instructions at 16 bits would allow native single-launch execution of up to $65,536$ threads, which can reduce python batching.
* **Ram inside Verilog**: Replacing the Python-managed `Memory` loop with a behavioral Verilog RAM array (`reg [7:0] data_ram [0:65535]`) would allow the testbench to complete batches in verilog itself rather talking to python every cycle for memory, significantly decreasing compute time.