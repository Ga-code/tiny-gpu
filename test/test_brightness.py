import cocotb
from cocotb.triggers import RisingEdge
from .helpers.setup import setup
from .helpers.memory import Memory
from .helpers.format import format_cycle
from .helpers.logger import logger
from .helpers.setup import Clock
from PIL import Image
import numpy as np
img = Image.open("test/rose.jpg")
img = img.resize((128,128))
img = img.convert("L")
img_array = np.array(img, dtype=np.uint8)
data = img_array.flatten().tolist()
k = 0

program = [
    0b0101000011011110,   # MUL R0, %blockIdx, %blockDim
    0b0011000000001111,   # ADD R0, R0, %threadIdx
    0b1001000100000000 | k,   # CONST R1, k   
    0b0111001000000000,   # LDR R2, R0
    0b1001001111111111,   # CONST R3, #255
    0b0100010000110010,   # SUB R4, R3, R2
    0b0010000001000001,   # CMP R4, R1
    0b0001011000001010,   #BRzp #11
    0b1001001011111111,   # CONST R2, #255
    0b1001000100000000,   # CONST R1, #0
    0b0011010100100001,   # ADD R5, R2, R1      
    0b1000000000000101,   # STR R0, R4
    0b1111000000000000    # RET
]
@cocotb.test()
async def brightness(dut):
    program_memory = Memory(dut, 8, 16, 1, "program")
    data_memory = Memory(dut, 8, 8, 8, "data")

    clock = Clock(dut.clk, 25, units="us")
    cocotb.start_soon(clock.start())
    program_memory.load(program)

    new_img_l = []
    total_cycles = 0
    total_batches = (len(data) + 248) // 249
    logger.info(f"Starting: {len(data)} pixels ({img_array.shape}) across {total_batches} batches")

    for b_idx, i in enumerate(range(0, len(data), 248)):
        batch = data[i : i + 248]
        actual_pixels = len(batch)

        if actual_pixels < 248:
            batch = batch + [0] * (248 - actual_pixels)

        dut.start.value = 0
        dut.reset.value = 1
        await RisingEdge(dut.clk)
        dut.reset.value = 0

        data_memory.load(batch)

        dut.device_control_write_enable.value = 1
        dut.device_control_data.value = actual_pixels
        await RisingEdge(dut.clk)
        dut.device_control_write_enable.value = 0

        dut.start.value = 1
        await RisingEdge(dut.clk)

        cycles = 0
        while dut.done.value != 1:
            if dut.data_mem_read_valid.value or dut.data_mem_write_valid.value or dut.data_mem_read_ready.value or dut.data_mem_write_ready.value:
                data_memory.run()
            if dut.program_mem_read_valid.value or dut.program_mem_read_ready.value:
                program_memory.run()

            #await cocotb.triggers.ReadOnly()
            await RisingEdge(dut.clk)
            cycles += 1

        total_cycles += cycles
        logger.info(f"Batch {b_idx + 1}/{total_batches} finished in {cycles} cycles")

        for idx in range(actual_pixels):
            new_img_l.append(data_memory.memory[idx])

    logger.info(f"Completed all {len(new_img_l)} pixels in {total_cycles} total cycles")

    new_img = np.array(new_img_l, dtype=np.uint8).reshape(img_array.shape)
    actual_img = np.clip(img_array.astype(np.uint16) + k, 0, 255).astype(np.uint8)
    for idx, (gpu_val, exp_val) in enumerate(zip(new_img_l, actual_img.flatten())):
        assert gpu_val == exp_val, f"Pixel {idx} mismatch: GPU got {gpu_val}, expected {exp_val} (orig={data[idx]}, k={k})"
    logger.info("PASS: All pixels match the expected output 100%!")
    nimg = Image.fromarray(new_img, mode="L")
    nimg.save("test/rose_brightened.jpg")
    nimg.show()
    