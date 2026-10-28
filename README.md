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

## The hardware

A programmable NPU with an 8-lane multiply-accumulate engine, a weight ROM, a KV cache, a softmax unit and a 12-instruction ISA. The whole inference is a 22-instruction program (85 instructions executed, 322 engine operations per run). See [`HARDWARE.md`](HARDWARE.md) (Korean) for the full spec.

The MAC engine is **bit-serial over the weights**. Instead of 8 parallel multipliers, each lane is just a row of AND gates: for every weight bit plane `k` (MSB first), the selected activations are summed and accumulated as `acc = 2*acc + S_k`, with the MSB plane (weight -8) negated. In redstone, wiring is the expensive resource, so this trades about 3x run time for about 5x less area and almost no wire crossings.

| Engine part | Implementation |
|---|---|
| Lane gating | Top-controlled AND gates, 8 lanes x 8 bits |
| Lane sum | Chain of 7 ripple-carry 12-bit adders; signed lanes via MSB inversion plus a single -1024 constant |
| Accumulator | 16-bit adder + 16-bit master-slave register (repeater-lock flip-flops) + shift-by-one feedback |
| First round | XOR conditional inverter on the adder input plus carry-in (two's complement negation) |

## Files

| File | Purpose |
|---|---|
| `train.py` | Trains the model with quantization-aware training under the hardware's integer rules, exports `weights.json` |
| `int_ref.py` | Pure-integer reference implementation with KV cache. The golden model for everything else |
| `npu.py` | ISA, assembler, the inference program, and an instruction-level simulator |
| `rom_gen.py` | Generates the bit layout of all four ROMs into `rom/` |
| `weights.json` | Trained integer weights (d_model 6, FFN 12, seed 1) |
| `rom/` | Weight, constant, exp-LUT and instruction ROM images |

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate           # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt  # CPU-only torch is enough

python train.py --d 6 --ffn 12 --seed 1 --export   # train + export weights.json
python int_ref.py                # 1680/1680 sorted, 1680/1680 bit-exact vs PyTorch
python int_ref.py --ranges       # exact worst-case range of every value -> bit widths
python npu.py                    # simulator matches int_ref at every stage, no bit-width violations
python npu.py --trace 5 2 7 0    # instruction-by-instruction trace
python rom_gen.py                # write ROM images to rom/
```

## Verification chain

Each layer is checked against the one before it on all 1680 inputs:

```
PyTorch QAT model  ->  int_ref.py (pure integers)  ->  npu.py (ISA simulator)  ->  redstone
```

When a redstone module is built, its lamps are compared against `npu.py --trace`.

## Status

- [x] Model trained and quantized
- [x] Bit widths fixed from exhaustive range analysis
- [x] Hardware architecture, ISA and program
- [x] Simulator verified bit-exact
- [x] ROM images
- [x] Redstone: gates, full adder, 8/12/16-bit ripple-carry adders
- [x] Redstone: 8-lane bit-serial MAC engine with 16-bit accumulator
- [x] **M1: first dot product in Minecraft. Q[0] = 33 computed from the real embedding and the real Wq column**
- [ ] M2: weight ROM (bit-plane layout) and automatic round/column sequencing for the full Q vector
- [ ] Remaining redstone modules (KV cache, softmax unit, FFN, control unit)
- [ ] ROM schematic generator
- [ ] Exhibition hall with per-stage displays

## Built alongside

A computer architecture course that goes from transistors to a working CPU. This project is the hands-on version: the same building blocks, ending in a transformer instead of a calculator.