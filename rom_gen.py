"""
레드스톤 NPU - ROM 이미지 생성기

weights.json과 npu.py의 프로그램을 읽어서, 마인크래프트에 넣을 ROM 네 개의 비트 배치를 만듭니다.
  rom/weight_rom.txt   가중치 ROM  (50줄 × 48비트: 레인 12개 × int4, 참고용)
  rom/const_rom.txt    상수 ROM    (임베딩·편향, 한 줄 = int8 × d_model)
  rom/exp_lut.txt      지수 룩업 ROM (16칸 × 8비트)
  rom/instr_rom.txt    명령어 ROM  (16비트 × 명령어 수)
  rom/weight_rom_bitplane.txt  비트 직렬 엔진용 가중치 ROM (한 줄 = 레인 12개의 같은 자리 비트, 열마다 4줄) ← 마크에 짓는 것
  rom/rom_images.json  위 ROM들의 비트 배열 (스키매틱 생성기의 입력)

비트 순서 규칙 (모든 ROM 공통)
  - 텍스트 파일: 사람이 읽기 쉽게 각 값의 MSB부터 표시
  - JSON 파일 : bits[줄][k], k = 0이 레인(원소) 0의 비트 0(LSB). 마크 배치도 이 순서를 따름

사용법:
  python rom_gen.py                 # rom/ 폴더에 전부 생성
  python rom_gen.py --show Wq 0     # 행렬 Wq의 0번 열: 4바퀴 동안 켤 제어 레버(c0~c11) 표
  python rom_gen.py --run Wq 2 0 3 -3 -7 5
                                    # 세팅 직후부터 버튼을 누를 때마다의 기대값 전체 표
                                    # (뒤의 숫자 = 레인 0부터 넣은 X 값, 모자란 레인은 0)
레인 수는 npu.LANES(12). 비트 직렬 ROM 한 줄 = 레인 12개의 같은 자리 비트.
"""
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")   # 윈도우에서 > 파일 저장 시 한글 깨짐 방지
except AttributeError:
    pass

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


def build_bitplane_rom():
    """비트 직렬 엔진용: 예전 ROM 한 줄(레인 12개 × int4)을 자리별 4줄로 나눔.
    바퀴 r = 0, 1, 2, 3 이 자리 −8, 4, 2, 1 (맨 위 비트부터). 주소 = 4 × 예전 줄 번호 + r"""
    rows = []
    for old, (vals, label) in enumerate(npu.WROM):
        for r in range(4):
            k = 3 - r                                    # 이번 바퀴가 보는 비트 위치
            bits = [((v & 0xF) >> k) & 1 for v in vals]  # 레인 0..11
            rows.append((bits, f"{label}, 바퀴 {r + 1} (자리 {['-8', '4', '2', '1'][r]})"))
        # 검증: 4줄로 원래 가중치가 복원되는지
        for lane, v in enumerate(vals):
            b = [rows[-4 + r][0][lane] for r in range(4)]
            assert -8 * b[0] + 4 * b[1] + 2 * b[2] + b[3] == v, (label, lane, v, b)
    return rows


def show(mat, col):
    d = npu.WDESC[mat]
    assert 0 <= col < d["n_out"], f"{mat}의 열은 0 ~ {d['n_out'] - 1}"
    bp = build_bitplane_rom()
    for c in range(d["chunks"]):
        old = d["base"] + col * d["chunks"] + c
        vals, label = npu.WROM[old]
        print(f"{label}: 가중치(레인 0~11) = {vals}")
        for r in range(4):
            bits, _ = bp[4 * old + r]
            on = [f"c{i}" for i, b in enumerate(bits) if b]
            extra = "첫 바퀴 켬, 되먹임 끔" if r == 0 else "첫 바퀴 끔, 되먹임 켬"
            print(f"  바퀴 {r + 1} (자리 {['-8', '4', '2', '1'][r]:>2}): 주소 {4 * old + r:>3}  "
                  f"레인0→11 {''.join(map(str, bits))}  켤 레버: {', '.join(on) or '없음':<24} ({extra})")


def b16(v):
    u = v & 0xFFFF
    return " ".join(f"{u:016b}"[i:i + 4] for i in range(0, 16, 4))


def b4(u, n):
    s = f"{u & ((1 << n) - 1):0{n}b}"
    return " ".join(s[max(0, i - 4):i] for i in range(len(s), 0, -4)[::-1])


def run(mat, xs):
    """세팅 직후부터 버튼을 한 번씩 누를 때의 기대값 (마크 램프와 바로 비교하는 2진수 포함)."""
    d = npu.WDESC[mat]
    xs = (list(xs) + [0] * npu.LANES)[:npu.LANES]
    bp = build_bitplane_rom()
    base = 4 * d["base"]
    n = 4 * d["n_out"]
    print(f"{mat}, X(레인 0~11) = {xs}")
    print("각 버튼마다: [누르기 전, 20~30초 기다린 뒤 보이는 값] -> [누른 후 Q]")
    print("  S  = 맨 위 단 선택 덧셈 출력 (12비트 램프)")
    print("  A  = 누산 덧셈기 A 입력 (되먹임 = 2 x 이전 Q, 첫 바퀴는 0)")
    print("  B  = 누산 덧셈기 B 입력 (반전기 뒤, 비트 12~15는 부호 확장). 첫 바퀴는 S를 뒤집은 값 + Cin 1")
    print()
    acc = 0
    for p in range(n):
        row = base + p
        bits, _ = bp[row]
        r = p % 4
        S = sum(x for x, b in zip(xs, bits) if b)
        A = 0 if r == 0 else 2 * acc
        B = (~S) if r == 0 else S
        cin = 1 if r == 0 else 0
        acc = A + B + cin
        on = ", ".join(f"c{i}" for i, b in enumerate(bits) if b) or "없음"
        mark = f"   <- {mat}[{p // 4}] 완성" if r == 3 else ""
        print(f"버튼 {p + 1:>2} | ROM {row:>3}번 줄, 4링 Q{r}, {'첫 바퀴' if r == 0 else '되먹임'} | 켜진 c: {on}")
        print(f"         S = {b4(S, 12):<14} ({S:>4})")
        print(f"         A = {b4(A, 16):<19} ({A:>5})")
        print(f"         B = {b4(B, 16):<19} ({B:>5}){'  + Cin 1' if cin else ''}")
        print(f"   누른 후 Q = {b4(acc, 16):<19} ({acc:>5}){mark}")
    print(f"버튼 {n + 1:>2} | (다시 {base}번 줄, 4링 Q0 으로 돌아옴)")


def main():
    os.makedirs("rom", exist_ok=True)
    out = {"bit_order": "bits[row][k], k=0 is element 0 bit 0 (LSB)"}

    # 가중치 ROM (int4 × 8레인)
    t, b = image(npu.WROM, 4)
    hdr = ["# 가중치 ROM: 한 줄 = 행렬의 한 열, 레인 0..11 × int4 (각 4비트는 MSB부터 표시)",
           "# 주소  주소(2진)  레인0 레인1 ... 레인7  | 10진 값  | 내용"]
    open("rom/weight_rom.txt", "w", encoding="utf-8").write("\n".join(hdr + t) + "\n")
    out["weight_rom"] = dict(width=32, rows=len(b), bits=b, map=npu.WDESC)

    # 비트 직렬용 가중치 ROM (12비트 × 열마다 4줄)
    bp = build_bitplane_rom()
    t = [f"{i:>3}  {i:08b}  {''.join(map(str, b))}  | 켤 레버: {' '.join(f'c{j}' for j, x in enumerate(b) if x) or '-':<24}| {label}"
         for i, (b, label) in enumerate(bp)]
    hdr = ["# 비트 직렬 가중치 ROM: 한 줄 = 레인 0..11의 같은 자리 비트 (왼쪽이 레인 0)",
           "# 주소 = 4 × (예전 32비트 ROM 줄 번호) + 바퀴(0..3).  바퀴 0 = 자리 −8 (첫 바퀴), 3 = 자리 1"]
    open("rom/weight_rom_bitplane.txt", "w", encoding="utf-8").write("\n".join(hdr + t) + "\n")
    out["weight_rom_bitplane"] = dict(width=npu.LANES, rows=len(bp), bits=[b for b, _ in bp])

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
    for k in ("weight_rom", "weight_rom_bitplane", "const_rom", "exp_lut", "instr_rom"):
        v = out[k]
        print(f"  {k:<11} {v['rows']:>3}줄 × {v['width']:>2}비트 = {v['rows'] * v['width']:>5}비트")
    print(f"  합계 {total}비트")


if __name__ == "__main__":
    if "--run" in sys.argv:
        i = sys.argv.index("--run")
        run(sys.argv[i + 1], [int(v) for v in sys.argv[i + 2:]])
    elif "--show" in sys.argv:
        i = sys.argv.index("--show")
        show(sys.argv[i + 1], int(sys.argv[i + 2]))
    else:
        main()