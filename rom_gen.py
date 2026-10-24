"""
레드스톤 NPU - ROM 이미지 생성기

weights.json과 npu.py의 프로그램을 읽어서, 마인크래프트에 넣을 ROM 네 개의 비트 배치를 만듭니다.
  rom/weight_rom.txt   가중치 ROM  (56줄 × 32비트: 레인 8개 × int4)
  rom/const_rom.txt    상수 ROM    (임베딩·편향, 한 줄 = int8 × d_model)
  rom/exp_lut.txt      지수 룩업 ROM (16칸 × 8비트)
  rom/instr_rom.txt    명령어 ROM  (16비트 × 명령어 수)
  rom/rom_images.json  위 네 개의 비트 배열 (스키매틱 생성기의 입력)

비트 순서 규칙 (모든 ROM 공통)
  - 텍스트 파일: 사람이 읽기 쉽게 각 값의 MSB부터 표시
  - JSON 파일 : bits[줄][k], k = 0이 레인(원소) 0의 비트 0(LSB). 마크 배치도 이 순서를 따름

사용법:
  python rom_gen.py
"""
import json
import os

import npu

D = npu.D


def twos(v, n):
    """정수 v를 n비트 2의 보수 비트열(LSB 먼저)로."""
    assert -(1 << (n - 1)) <= v < (1 << (n - 1)) or 0 <= v < (1 << n), f"{v}는 {n}비트에 안 들어감"
    return [(v >> i) & 1 for i in range(n)]


def msb_str(v, n):
    return "".join(str(b) for b in reversed(twos(v, n)))


def image(rows, n):
    """rows: [(값 리스트, 라벨)] -> 텍스트 줄들, 비트 배열"""
    text, bits = [], []
    for addr, (vals, label) in enumerate(rows):
        bits.append([b for v in vals for b in twos(v, n)])
        groups = " ".join(msb_str(v, n) for v in vals)
        nums = " ".join(f"{v:>4}" for v in vals)
        text.append(f"{addr:>3}  {addr:06b}  {groups}  | {nums}  | {label}")
    return text, bits


def main():
    os.makedirs("rom", exist_ok=True)
    out = {"bit_order": "bits[row][k], k=0 is element 0 bit 0 (LSB)"}

    # 가중치 ROM (int4 × 8레인)
    t, b = image(npu.WROM, 4)
    hdr = ["# 가중치 ROM: 한 줄 = 행렬의 한 열, 레인 0..7 × int4 (각 4비트는 MSB부터 표시)",
           "# 주소  주소(2진)  레인0 레인1 ... 레인7  | 10진 값  | 내용"]
    open("rom/weight_rom.txt", "w", encoding="utf-8").write("\n".join(hdr + t) + "\n")
    out["weight_rom"] = dict(width=32, rows=len(b), bits=b, map=npu.WDESC)

    # 상수 ROM (int8 × D)
    t, b = image(npu.CROM, 8)
    hdr = [f"# 상수 ROM: 한 줄 = int8 × {D} (임베딩, 편향)", "# 주소  주소(2진)  원소0 ... | 10진 값 | 내용"]
    open("rom/const_rom.txt", "w", encoding="utf-8").write("\n".join(hdr + t) + "\n")
    out["const_rom"] = dict(width=8 * D, rows=len(b), bits=b, map=npu.CDESC)

    # 지수 룩업 ROM (부호 없는 8비트)
    lut_rows = [([v], f"exp(-{i} × 0.5) × 255") for i, v in enumerate(npu.LUT)]
    t, b = image(lut_rows, 9)            # 부호 없는 255를 담으려고 9비트로 검사 후 하위 8비트만 저장
    b = [row[:8] for row in b]
    t = [f"{i:>3}  {i:04b}  {v:08b}  | {v:>4}  | 최댓값-점수 = {i}" for i, v in enumerate(npu.LUT)]
    open("rom/exp_lut.txt", "w", encoding="utf-8").write("\n".join(["# 지수 룩업 ROM: 주소 = min(최댓값-점수, 15)"] + t) + "\n")
    out["exp_lut"] = dict(width=8, rows=len(b), bits=b)

    # 명령어 ROM (16비트)
    t = [f"{i:>3}  {p['word']:016b}  0x{p['word']:04X}  {p['text']:<18} ; {p['comment']}" for i, p in enumerate(npu.PROG)]
    open("rom/instr_rom.txt", "w", encoding="utf-8").write(
        "\n".join(["# 명령어 ROM: [15:12] opcode | [11:9] a | [8:6] b | [5:0] imm"] + t) + "\n")
    out["instr_rom"] = dict(width=16, rows=len(npu.PROG), bits=[[(p["word"] >> i) & 1 for i in range(16)] for p in npu.PROG])

    json.dump(out, open("rom/rom_images.json", "w"), indent=1)

    total = sum(v["width"] * v["rows"] for k, v in out.items() if isinstance(v, dict))
    print("rom/ 폴더에 저장:")
    for k in ("weight_rom", "const_rom", "exp_lut", "instr_rom"):
        v = out[k]
        print(f"  {k:<11} {v['rows']:>3}줄 × {v['width']:>2}비트 = {v['rows'] * v['width']:>5}비트")
    print(f"  합계 {total}비트")


if __name__ == "__main__":
    main()
