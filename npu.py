"""
레드스톤 NPU - 명령어 집합(ISA), 어셈블러, 트랜스포머 추론 프로그램, 시뮬레이터

이 파일이 하는 일:
  1. 하드웨어 구조도의 부품(레지스터, ROM, 엔진)을 파이썬 객체로 흉내 냄
  2. 트랜스포머 추론을 NPU 어셈블리 프로그램(PROGRAM)으로 작성
  3. 어셈블러가 프로그램을 16비트 기계어로 바꿈 (시프트 양은 weights.json의 지수로 자동 계산)
  4. 시뮬레이터가 기계어를 한 명령씩 실행하면서 비트 폭 위반을 감시
  5. 결과와 정거장별 중간값을 int_ref.py(정답지)와 비교

사용법:
  python npu.py               # 1680개 전체 검증 + 연산량 요약
  python npu.py --listing     # 어셈블된 프로그램 (주소, 기계어, 시프트 양)
  python npu.py --trace 5 2 7 0   # 명령어 단위 실행 기록
"""
import itertools
import sys

import int_ref as R  # 정답지. 같은 weights.json을 읽음

W, D, FFN, E, LUT, SEP = R.W, R.D, R.FFN, R.E, R.LUT, R.SEP
LANES = 8

# ---------------------------------------------------------------- 하드웨어 규격 (1번 재료: 비트 폭)
WIDTH = {                     # 이름: (최소, 최대)  - 이 범위를 벗어나면 위반으로 기록
    "레지스터(int8)": (-127, 127),   # 포화 회로 없음: 이 범위 밖이면 int_ref와 달라짐
    "누산기(16비트)": (-32768, 32767),
    "점수(8비트)": (-128, 127),
    "합 S(10비트)": (0, 1023),
    "역수 r(9비트)": (0, 511),
    "지수값×r(16비트)": (0, 65535),
    "확률 p(8비트)": (0, 255),
}

# ---------------------------------------------------------------- 명령어 집합
# 16비트 명령어: [15:12] opcode | [11:9] a | [8:6] b | [5:0] imm (부호 있는 6비트)
OPS = {"EMB": 1, "MV": 2, "MVADD": 3, "SCORE": 4, "SOFTMAX": 5, "MIXV": 6,
       "ADDB": 7, "RELU": 8, "ARGMAX": 9, "INCP": 10, "JLT": 11, "HALT": 12}
REGS = {"X": 0, "Q": 1, "A": 2, "H": 3, "KC": 4, "VC": 5}   # KC/VC = KV 캐시의 현재 위치(P) 줄
MATS = ["Wq", "Wk", "Wv", "Wo", "W1", "W2", "U"]
SRC = {"Wq": "X", "Wk": "X", "Wv": "X", "Wo": "A", "W1": "X", "W2": "H", "U": "X"}
BIAS = ["b1", "b2"]
REG_EXP = {"X": "r", "Q": "q", "A": "a", "H": "h", "KC": "k", "VC": "v"}  # 레지스터별 스케일 지수

PROGRAM = """
; ---- 프리필: 입력 숫자 4개(P = 0..3)의 K, V만 계산해서 캐시에 쓰기
prefill:
    EMB                 ; X <- tok_rom[T[P]] + pos_rom[P]
    MV     KC, Wk       ; K캐시[P] <- Wk · X
    MV     VC, Wv       ; V캐시[P] <- Wv · X
    INCP                ; P <- P + 1
    JLT    4, prefill   ; P < 4 이면 반복

; ---- 생성: P = 4..7, 한 바퀴에 토큰 하나
gen:
    EMB
    MV     Q, Wq        ; Q <- Wq · X
    MV     KC, Wk
    MV     VC, Wv
    SCORE               ; s_j <- Q · K_j        (j = 0..P)
    SOFTMAX             ; p_j <- 정수 softmax   (지수 ROM, 역수 1회)
    MIXV                ; A <- Σ p_j · V_j
    MVADD  X, Wo        ; X <- X + Wo · A        (residual 1)
    MV     H, W1        ; H <- W1 · X
    ADDB   H, b1        ; H <- H + b1
    RELU   H            ; H <- max(H, 0)
    MVADD  X, W2        ; X <- X + W2 · H
    ADDB   X, b2        ; X <- X + b2            (residual 2 완성)
    ARGMAX U            ; T[P+1] <- argmax(U · X)
    INCP
    JLT    8, gen
    HALT
"""


# ---------------------------------------------------------------- ROM 이미지
def build_weight_rom():
    """가중치 ROM: 한 줄 = 행렬의 한 열, 레인 8개 × int4. 입력이 8개를 넘으면 열 하나를 여러 줄로."""
    rom, desc, base = [], {}, 0
    for m in MATS:
        M = W[m]
        n_in, n_out = len(M), len(M[0])
        chunks = -(-n_in // LANES)
        desc[m] = dict(base=base, n_in=n_in, n_out=n_out, chunks=chunks)
        for j in range(n_out):
            for c in range(chunks):
                rom.append(([M[i][j] if i < n_in else 0 for i in range(c * LANES, (c + 1) * LANES)],
                            f"{m} 출력 {j}" + (f" (입력 {c * LANES}~{min(n_in, (c + 1) * LANES) - 1})" if chunks > 1 else "")))
        base += n_out * chunks
    return rom, desc


def build_const_rom():
    """상수 ROM: 한 줄 = int8 × D. 토큰 임베딩, 위치 임베딩, 편향 b1(여러 줄), b2."""
    rom, desc = [], {}

    def put(name, vec, label):
        desc.setdefault(name, len(rom))
        rom.append((vec, label))

    for t, v in enumerate(W["tok"]):
        put("tok", v, f"토큰 {'|' if t == SEP else t}")
    for p, v in enumerate(W["pos"]):
        put("pos", v, f"위치 {p}")
    for c in range(0, FFN, D):
        put("b1", (W["b1"][c:c + D] + [0] * D)[:D], f"b1 원소 {c}~{min(FFN, c + D) - 1}")
    put("b2", W["b2"], "b2")
    return rom, desc


WROM, WDESC = build_weight_rom()
CROM, CDESC = build_const_rom()


# ---------------------------------------------------------------- 어셈블러
def auto_shift(op, a, b):
    """재양자화 시프트 양 = 입력 지수 - 출력 지수 (양수: 오른쪽 시프트)."""
    if op in ("MV", "MVADD"):
        mat = MATS[b]
        return E[REG_EXP[SRC[mat]]] + E["w_" + mat] - E[REG_EXP[[k for k, v in REGS.items() if v == a][0]]]
    if op == "SCORE":
        return E["q"] + E["k"] - W["e_score"]
    if op == "MIXV":
        return W["e_prob"] + E["v"] - E["a"]
    return 0


def assemble(src=PROGRAM):
    lines, labels = [], {}
    for raw in src.splitlines():
        line = raw.split(";")[0].strip()
        if not line:
            continue
        if line.endswith(":"):
            labels[line[:-1]] = len(lines)
            continue
        op, *rest = line.split(None, 1)
        args = [x.strip() for x in rest[0].split(",")] if rest else []
        lines.append((op, args, raw.split(";")[1].strip() if ";" in raw else ""))
    prog = []
    for op, args, comment in lines:
        a = b = imm = 0
        if op in ("MV", "MVADD"):
            a, b = REGS[args[0]], MATS.index(args[1])
        elif op == "ADDB":
            a, b = REGS[args[0]], BIAS.index(args[1])
        elif op == "RELU":
            a = REGS[args[0]]
        elif op == "ARGMAX":
            b = MATS.index(args[0])
        elif op == "JLT":
            limit, target = int(args[0]), labels[args[1]]
            a, b, imm = limit >> 3, limit & 7, target          # 한계값 6비트 = a(3) + b(3)
        if op not in ("JLT",):
            imm = auto_shift(op, a, b)
        assert -32 <= imm <= 63, f"{op}: 즉시값 {imm}이 6비트 범위 밖"
        word = (OPS[op] << 12) | (a << 9) | (b << 6) | (imm & 0x3F)
        prog.append(dict(op=op, a=a, b=b, imm=imm, word=word, text=f"{op:<7}{', '.join(args)}", comment=comment))
    return prog


def decode(word):
    op = {v: k for k, v in OPS.items()}[word >> 12]
    a, b, imm = (word >> 9) & 7, (word >> 6) & 7, word & 0x3F
    if op != "JLT" and imm >= 32:
        imm -= 64                                              # 부호 있는 6비트
    return op, a, b, imm


# ---------------------------------------------------------------- 시뮬레이터
def shr(acc, k):  # 반올림 산술 시프트 (int_ref와 같은 회로)
    return (acc + (1 << (k - 1))) >> k if k > 0 else acc << (-k)


class NPU:
    def __init__(self, digits):
        self.T = list(digits) + [SEP] + [0] * 4           # 토큰 테이프 9칸
        self.P = 0
        self.PC = 0
        self.reg = {"X": [0] * D, "Q": [0] * D, "A": [0] * D, "H": [0] * FFN}
        self.K = [[0] * D for _ in range(8)]
        self.V = [[0] * D for _ in range(8)]
        self.s, self.p = [], []
        self.engine_ops = 0                                # 8레인 내적 연산 횟수
        self.instr_count = 0
        self.violations = {}
        self.snap = {}                                     # 정거장별 값 (int_ref 비교용)
        self.snaps = []
        self.trace = []

    def chk(self, kind, vals):
        lo, hi = WIDTH[kind]
        for v in (vals if isinstance(vals, list) else [vals]):
            if not lo <= v <= hi:
                self.violations[kind] = self.violations.get(kind, 0) + 1

    def dot(self, xs, ws):                                 # 엔진 1회: 레인 8개 곱 + 덧셈 트리
        assert len(xs) <= LANES
        self.engine_ops += 1
        return sum(x * w for x, w in zip(xs, ws))

    def matvec(self, mat, x, shift):
        d = WDESC[mat]
        out = []
        for j in range(d["n_out"]):
            acc = 0
            for c in range(d["chunks"]):
                row, _ = WROM[d["base"] + j * d["chunks"] + c]
                acc += self.dot((x + [0] * LANES)[c * LANES:(c + 1) * LANES], row)
                self.chk("누산기(16비트)", acc)
            out.append(acc if shift is None else shr(acc, shift))
        return out

    def write(self, name, vals, add=False):
        if name == "KC":
            self.K[self.P] = vals
        elif name == "VC":
            self.V[self.P] = vals
        else:
            self.reg[name] = [a + b for a, b in zip(self.reg[name], vals)] if add else vals
        self.chk("레지스터(int8)", vals if not add else self.reg[name])

    def step(self):
        op, a, b, imm = decode(PROG_WORDS[self.PC])
        self.instr_count += 1
        nxt = self.PC + 1
        rname = {v: k for k, v in REGS.items()}.get(a, "")
        P, X = self.P, self.reg["X"]
        if op == "EMB":
            h0 = [t + p for t, p in zip(CROM[CDESC["tok"] + self.T[P]][0], CROM[CDESC["pos"] + P][0])]
            self.write("X", h0)
            self.snap = {"h0": h0}
        elif op in ("MV", "MVADD"):
            mat = MATS[b]
            res = self.matvec(mat, self.reg[SRC[mat]], imm)
            self.write(rname, res, add=(op == "MVADD"))
            key = {"Q": "q", "KC": "k", "VC": "v"}.get(rname)
            if key:
                self.snap[key] = res
            if op == "MVADD" and mat == "Wo":
                self.snap["x1"] = list(self.reg["X"])
        elif op == "SCORE":
            self.s = [shr(self.dot(self.reg["Q"], self.K[j]), imm) for j in range(P + 1)]
            self.chk("점수(8비트)", self.s)
        elif op == "SOFTMAX":
            m = max(self.s)
            e = [LUT[min(m - v, len(LUT) - 1)] for v in self.s]
            S = sum(e)
            r = 65536 // S
            self.chk("합 S(10비트)", S)
            self.chk("역수 r(9비트)", r)
            self.chk("지수값×r(16비트)", [x * r for x in e])
            self.p = [(((x * r) >> 8) + 1) >> 1 for x in e]    # = (x*r + 256) >> 9, 16비트 안에서 계산
            self.chk("확률 p(8비트)", self.p)
            self.snap["prob"] = self.p
        elif op == "MIXV":
            out = []
            for i in range(D):                                  # 레인 = 위치 j, 원소 i마다 1회
                acc = self.dot(self.p, [self.V[j][i] for j in range(P + 1)])
                self.chk("누산기(16비트)", acc)
                out.append(shr(acc, imm))
            self.write("A", out)
            self.snap["attn_out"] = out
        elif op == "ADDB":
            base = CDESC[BIAS[b]]
            n = len(self.reg[rname])
            bias = [v for row in range(-(-n // D)) for v in CROM[base + row][0]][:n]
            self.write(rname, bias, add=True)
            if rname == "X":
                self.snap["x2"] = list(self.reg["X"])
        elif op == "RELU":
            self.reg[rname] = [max(0, v) for v in self.reg[rname]]
            self.snap["h"] = list(self.reg[rname])
        elif op == "ARGMAX":
            logits = self.matvec(MATS[b], X, None)
            best = 0
            for j in range(1, len(logits)):                     # 더 클 때만 교체 = 동점이면 작은 번호
                if logits[j] > logits[best]:
                    best = j
            self.T[P + 1] = best
            self.snap.update(logits=logits, pred=best)
            self.snaps.append(self.snap)
        elif op == "INCP":
            self.P += 1
        elif op == "JLT":
            if self.P < ((a << 3) | b):
                nxt = imm
        elif op == "HALT":
            return False
        if TRACE:
            self.trace.append(f"  PC {self.PC:2d}  P={P}  {PROG[self.PC]['text']:<18} "
                              + self.show(op, rname))
        self.PC = nxt
        return True

    def show(self, op, rname):
        if op == "EMB": return f"X = {self.reg['X']}"
        if op in ("MV", "MVADD"):
            v = self.K[self.P] if rname == "KC" else self.V[self.P] if rname == "VC" else self.reg[rname]
            return f"{rname} = {v}"
        if op == "SCORE": return f"s = {self.s}"
        if op == "SOFTMAX": return f"p = {self.p}"
        if op == "MIXV": return f"A = {self.reg['A']}"
        if op in ("ADDB", "RELU"): return f"{rname} = {self.reg[rname]}"
        if op == "ARGMAX": return f"점수 = {self.snap['logits']} -> T[{self.P + 1}] = {self.snap['pred']}"
        return ""

    def run(self):
        while self.step():
            pass
        return self.T[5:9]


PROG = assemble()
PROG_WORDS = [p["word"] for p in PROG]
TRACE = False


# ---------------------------------------------------------------- 검증
KEYS = ["h0", "q", "k", "v", "prob", "attn_out", "x1", "h", "x2", "logits", "pred"]


def verify():
    perms = list(itertools.permutations(range(8), 4))
    ok_out = ok_all = 0
    viol, first_diff = {}, None
    npu = None
    for p in perms:
        npu = NPU(p)
        out = npu.run()
        ref_out, logs = R.sort4(p, return_logs=True)
        ok_out += out == ref_out
        same = all(s.get(k) == l.get(k) for s, l in zip(npu.snaps, logs) for k in KEYS if k in l)
        ok_all += same
        if not same and first_diff is None:
            first_diff = p
        for k, v in npu.violations.items():
            viol[k] = viol.get(k, 0) + v
    print(f"출력 토큰 일치 (int_ref 대비)     : {ok_out}/{len(perms)}")
    print(f"정거장 중간값 전부 일치            : {ok_all}/{len(perms)}")
    if first_diff:
        print(f"  처음 달라진 입력: {first_diff}  ->  python npu.py --trace {' '.join(map(str, first_diff))}")
    print("비트 폭 위반                      : " + ("없음" if not viol else str(viol)))
    print(f"\n한 번 실행당: 명령어 {npu.instr_count}개 실행, 8레인 엔진 연산 {npu.engine_ops}회")
    print(f"프로그램 크기: {len(PROG)}개 명령어 × 16비트 = {len(PROG) * 16}비트")


def listing():
    print(f"{'주소':>4}  {'기계어':<18} {'16진':>6}  {'명령어':<18} {'시프트':>5}  설명")
    for i, p in enumerate(PROG):
        sh = p["imm"] if p["op"] in ("MV", "MVADD", "SCORE", "MIXV") else ""
        print(f"{i:>4}  {p['word']:016b}  0x{p['word']:04X}  {p['text']:<18} {str(sh):>5}  {p['comment']}")


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--listing" in args:
        listing()
    elif "--trace" in args:
        TRACE = True
        nums = [int(a) for a in args if a.isdigit()] or [5, 2, 7, 0]
        npu = NPU(nums)
        out = npu.run()
        print("\n".join(npu.trace))
        print(f"\n입력 {nums} -> 출력 {out}   (명령어 {npu.instr_count}개, 엔진 연산 {npu.engine_ops}회)")
    else:
        verify()
