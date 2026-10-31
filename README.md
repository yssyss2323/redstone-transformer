# Redstone Transformer

A tiny decoder-only transformer, quantized to integers and designed to run on a programmable NPU built from Minecraft redstone.

The goal is not size or speed. It is to make every step of a transformer **visible**: embeddings, Q/K/V, the KV cache filling up, softmax turning scores into probabilities, attention picking a position, and the next token appearing, all as lamps you can walk past.

## The task

Sort four distinct digits (0-7) by generating the answer one token at a time.

```
input:  3 6 1 4 |
output:           1 3 4 6
```

## The model

| | |
|---|---|
| Architecture | Decoder-only, 1 layer, 1 head, no LayerNorm |
| d_model / FFN | 6 / 12 (ReLU) |
| Parameters | 456 |
| Weights | int4, power-of-two scales |
| Activations | int8, requantized with rounding shifts |
| Softmax | 16-entry exp lookup ROM, one division per step |
| Accuracy | 1680 / 1680 (every possible input) |

## The hardware (spec v0.3)

A programmable NPU with a **bit-serial** multiply-accumulate engine. See [`HARDWARE.md`](HARDWARE.md) (Korean) for the full spec; `npu.py` simulates exactly this structure.

**Bit-serial engine.** Instead of parallel multipliers, each lane is a row of AND gates. One operand sits in the lanes (an activation vector, 8 bits per lane). The other operand arrives one bit plane at a time on 12 control lines, one bit per lane. Every round the enabled lanes are summed and accumulated as `acc = 2*acc + S`; the MSB round is negated. In redstone, wiring is the expensive resource, so this trades run time for far less area and almost no wire crossings.

**Two sources drive the control lines.** For matrix-vector products, a bit-plane weight ROM (one row = the same bit of 12 lanes) selected by a ring counter drives the control lines. For attention, the roles swap: the cached keys or values sit in the lanes, and each lane generates its own control bit by picking one bit per round out of its stored query or probability.

**Registers and caches live next to the lanes.** Each lane has 8-bit latch slots on a glass floor above the engine: X, a shared slot for Q then A then H, scores then probabilities, plus eight K slots and six V slots. A select gate and a staircase bring the chosen slot down into the lane. Results are written one element per output column through a broadcast write bus and lock lines.

| Part | Implementation |
|---|---|
| Lane gating | Top-controlled AND gates, one row per lane |
| Lane sum | Chain of 12-bit ripple-carry adders; signed lanes via bit-7 inversion plus one constant |
| Accumulator | 16-bit adder, 16-bit master-slave register (repeater-lock flip-flops), shift-by-one feedback |
| First round | XOR conditional inverter plus carry-in (two's complement negation) |
| Weight ROM | Torch ROM, one row per (column, bit plane), selected by a 24-stage ring counter |
| Round control | 4-stage ring counter generating first-round and feedback-enable signals |
| Lane registers | Repeater latches on an upper glass floor, write strobe = delayed clock, lock line = strobe OR ring row |

The full inference is a 19-instruction program: 73 instructions executed, 1,392 engine rounds per run.

## Files

| File | Purpose |
|---|---|
| `train.py` | Trains the model with quantization-aware training under the hardware's integer rules, exports `weights.json` |
| `int_ref.py` | Pure-integer reference implementation with KV cache. The golden model for everything else |
| `npu.py` | ISA, assembler, inference program, and a simulator of the v0.3 bit-serial hardware (shared lane slots, lane-side KV cache, garbage initial state) |
| `rom_gen.py` | Generates all ROM images into `rom/`; `--run` prints the expected lamp values for every button press |
| `weights.json` | Trained integer weights (d_model 6, FFN 12, seed 1) |
| `rom/` | ROM images; `weight_rom_bitplane.txt` is the one built in Minecraft |
| `docs/measurements.md` | Measured redstone delays and milestone logs |

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate           # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt  # CPU-only torch is enough

python train.py --d 6 --ffn 12 --seed 1 --export   # train + export weights.json
python int_ref.py                # 1680/1680 sorted, 1680/1680 bit-exact vs PyTorch
python int_ref.py --ranges       # exact worst-case range of every value -> bit widths
python npu.py                    # v0.3 simulator matches int_ref at every stage, no bit-width violations
python npu.py --trace 5 2 7 0    # instruction-by-instruction trace
python rom_gen.py                # write ROM images to rom/
python rom_gen.py --run Wq 2 0 3 -3 -7 5   # expected values for every button press
```

## Verification chain

Each layer is checked against the one before it on all 1680 inputs:

```
PyTorch QAT model  ->  int_ref.py (pure integers)  ->  npu.py (hardware simulator)  ->  redstone
```

When a redstone module is built, its lamps are compared against `rom_gen.py --run` and `npu.py --trace`.

## Status

- [x] Model trained and quantized
- [x] Bit widths fixed from exhaustive range analysis
- [x] Hardware spec v0.3 (bit-serial, lane-side KV cache) with a bit-exact simulator
- [x] Redstone: gates, full adder, 8/12/16-bit ripple-carry adders
- [x] Redstone: 8-lane bit-serial MAC engine with 16-bit accumulator
- [x] **M1: first dot product in Minecraft.** Q[0] = 33 from the real embedding and the real Wq column
- [x] **M2: full Q vector, automatic.** Bit-plane weight ROM, ring-counter sequencing, and lane-side registers: one button, 24 presses, Q = [33, -4, 14, 9, -4, 4] stored in redstone latches
- [x] Lane-side slot read path: select gates and staircases bring stored values down into the lanes
- [ ] M3: attention probabilities (K slots, per-lane bit pick for Q, 8-round mode, softmax unit)
- [ ] M4: one full layer (V cache, MIXV, residuals, FFN with 12 lanes)
- [ ] M5: full sort (control unit, instruction ROM, token tape)
- [ ] Exhibition hall with per-stage displays

## Built alongside

A computer architecture course that goes from transistors to a working CPU. This project is the hands-on version: the same building blocks, ending in a transformer instead of a calculator.