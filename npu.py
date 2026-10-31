"""
레드스톤 NPU v0.2 (비트 직렬 구조) - 명령어 집합, 어셈블러, 추론 프로그램, 시뮬레이터

이 파일이 하는 일:
  1. 하드웨어 구조도의 부품(레지스터, ROM, 엔진)을 파이썬 객체로 흉내 냄
  2. 트랜스포머 추론을 NPU 어셈블리 프로그램(PROGRAM)으로 작성
  3. 어셈블러가 프로그램을 16비트 기계어로 바꿈 (시프트 양은 weights.json의 지수로 자동 계산)
  4. 시뮬레이터가 마크 회로와 같은 방식으로 실행 (HARDWARE.md v0.2)
     - 엔진: 레인 12개, 비트 직렬 (c 선 비트 평면마다 acc = 2*acc + S, 부호 있는 c 값이면 첫 바퀴만 -S)
     - 레인 옆 레지스터 칸을 이름이 다른 레지스터들이 나눠 씀 (덮어쓰기까지 그대로 흉내)
     - v0.3: K, V 캐시도 레인 옆 칸. SCORE는 레인 = K_j, c 선 = Q의 비트 / MIXV는 레인 = V, c 선 = p의 비트
       (c 선 비트는 각 레인 위 래치에서 바퀴마다 비트 하나를 골라 그 레인 c 선에 넣음)
     - 모든 칸과 캐시를 쓰레기 값으로 채운 채 시작해서, 초기화 없이도 맞는지 확인
     - 쓰기 유닛: 재양자화 시프트 -> 편향 -> ReLU -> (MVADD면) 레인에서 읽은 옛값 더하기
  5. 결과와 정거장별 중간값을 int_ref.py(정답지)와 비교

사용법:
  python npu.py               # 1680개 전체 검증 + 바퀴 수 요약
  python npu.py --listing     # 어셈블된 프로그램 (주소, 기계어, 시프트 양)
  python npu.py --trace 5 2 7 0   # 명령어 단위 실행 기록
"""
import itertools
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")   # 윈도우에서 > 파일 저장 시 한글 깨짐 방지
except AttributeError:
    pass

import int_ref as R  # 정답지. 같은 weights.json을 읽음

W, D, FFN, E, LUT, SEP = R.W, R.D, R.FFN, R.E, R.LUT, R.SEP
LANES = 12          # 레인 수. W2(입력 12개)도 한 번에 들어가게 12개
SLOTS = 3           # 레인마다 레지스터 칸 수

# ---------------------------------------------------------------- 하드웨어 규격 (1번 재료: 비트 폭)
WIDTH = {                     # 이름: (최소, 최대)  - 이 범위를 벗어나면 위반으로 기록
    "레지스터(int8)": (-127, 127),   # 포화 회로 없음: 이 범위 밖이면 int_ref와 달라짐
    "레인 입력(int8)": (-128, 127),  # 보통 모드: 부호 있는 8비트 (비트 7 NOT + 상수 -1536)
    "레인 합 S(12비트)": (-2048, 2047),
    "누산기(16비트)": (-32768, 32767),   # 바퀴 중간값까지 전부 검사
    "쓰기 유닛 중간값(9비트)": (-256, 255),
    "점수(8비트)": (-128, 127),
    "합 S(10비트)": (0, 1023),
    "역수 r(9비트)": (0, 511),
    "지수값×r(16비트)": (0, 65535),
    "확률 p(8비트)": (0, 255),
}

# ---------------------------------------------------------------- 명령어 집합
# 16비트 명령어: [15:12] opcode | [11:9] a | [8:6] b | [5:0] imm (부호 있는 6비트)
OPS = {"EMB": 1, "MV": 2, "MVADD": 3, "SCORE": 4, "SOFTMAX": 5, "MIXV": 6,
       "ARGMAX": 9, "INCP": 10, "JLT": 11, "HALT": 12}      # 7, 8(ADDB, RELU)은 v0.2에서 MV에 합쳐짐
REGS = {"X": 0, "Q": 1, "A": 2, "H": 3, "KC": 4, "VC": 5}   # KC/VC = KV 캐시의 현재 위치(P) 줄
MATS = ["Wq", "Wk", "Wv", "Wo", "W1", "W2", "U"]
SRC = {"Wq": "X", "Wk": "X", "Wv": "X", "Wo": "A", "W1": "X", "W2": "H", "U": "X"}
POST = {"W1": ("b1", True), "W2": ("b2", False)}   # 행렬 설명표: 쓰기 유닛에서 더할 편향, ReLU 여부
# 레인 옆 레지스터 칸 배정: 쓰는 시기가 겹치지 않는 레지스터끼리 칸을 나눠 씀
SLOT_OF = {"X": 0, "Q": 1, "A": 1, "H": 1, "S": 2, "P": 2}
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
    SCORE               ; s_j <- Q · K_j        (j = 0..P): 레인 = K_j 칸, c 선 = Q 비트. 칸 2에 저장
    SOFTMAX             ; p_j <- 정수 softmax   (칸 2를 읽고 같은 칸에 p 쓰기)
    MIXV                ; A <- Σ p_j · V_j      (레인 = 위치 j의 V 칸, c 선 = p 비트, 첫 바퀴 빼기 없음)
    MVADD  X, Wo        ; X <- X + Wo · A        (residual 1)
    MV     H, W1        ; H <- ReLU(W1 · X + b1)  (편향, ReLU는 쓰기 유닛에서)
    MVADD  X, W2        ; X <- X + (W2 · H + b2)  (residual 2 완성)
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


# ---------------------------------------------------------------- 시뮬레이터 (v0.2 비트 직렬)
def shr(acc, k):  # 반올림 산술 시프트 (int_ref와 같은 회로)
    return (acc + (1 << (k - 1))) >> k if k > 0 else acc << (-k)


def bit(v, k, n):  # n비트 2의 보수에서 k번 비트
    return ((v & ((1 << n) - 1)) >> k) & 1


class NPU:
    def __init__(self, digits, seed=0):
        import random
        g = random.Random(seed)
        junk = lambda: g.randint(-127, 127)
        self.T = list(digits) + [SEP] + [0] * 4           # 토큰 테이프 9칸
        self.P = 0
        self.PC = 0
        # 레인 옆 레지스터: [칸][레인]. 전원 켤 때 상태를 모르니 쓰레기 값으로 시작
        self.slot = [[junk() for _ in range(LANES)] for _ in range(SLOTS)]
        # 레인 옆 캐시 칸 (v0.3). 칸이 있는 레인만 쓰레기, 칸이 없는 레인은 레인 입력 0
        self.K = [[junk() if i < D else 0 for i in range(LANES)] for _ in range(8)]   # K 칸[위치 j][레인 i] = K_j[i]
        self.V = [[junk() if j < 8 else 0 for j in range(LANES)] for _ in range(D)]   # V 칸[원소 i][레인 j] = V_j[i]
        self.rounds = {}                                   # 연산별 바퀴 수
        self.writes = 0                                    # 레지스터/캐시 쓰기 횟수
        self.instr_count = 0
        self.violations = {}
        self.ranges = {}
        self.snap = {}
        self.snaps = []
        self.trace = []

    # -- 검사 도구
    def chk(self, kind, vals):
        lo, hi = WIDTH[kind]
        for v in (vals if isinstance(vals, list) else [vals]):
            a, b = self.ranges.get(kind, (v, v))
            self.ranges[kind] = (min(a, v), max(b, v))
            if not lo <= v <= hi:
                self.violations[kind] = self.violations.get(kind, 0) + 1

    # -- 레지스터 칸 읽기/쓰기
    def lanes_of(self, name):
        return self.slot[SLOT_OF[name]]

    def write_lane(self, name, j, v):
        self.chk("레지스터(int8)", v)
        self.slot[SLOT_OF[name]][j] = v
        self.writes += 1

    def read_lane(self, name, j):                          # 읽기 버스: 레인 j의 칸 하나
        return self.slot[SLOT_OF[name]][j]

    # -- 엔진: 비트 직렬 내적
    def serial_dot(self, lanes, gates, nbits, kind, signed_gates=True):
        """lanes: 레인 12개 값 (레인 옆 칸, 부호 있는 8비트), gates: c 선에 나올 값 12개.
        nbits 바퀴, 맨 위 비트부터. 부호 있는 c 값이면 첫 바퀴 acc = -S (무게 -2^(n-1)),
        부호 없는 c 값(p)이면 첫 바퀴도 +S. 나머지 바퀴는 acc = 2*acc + S."""
        used = [x for x, w in zip(lanes, gates) if w != 0]    # 게이트가 0인 레인은 값이 뭐든 안 더해짐
        self.chk("레인 입력(int8)", used)
        acc = 0
        for r in range(nbits):
            k = nbits - 1 - r
            S = sum(x for x, w in zip(lanes, gates) if bit(w, k, nbits))
            self.chk("레인 합 S(12비트)", S)
            acc = (-S if signed_gates else S) if r == 0 else 2 * acc + S
            self.chk("누산기(16비트)", acc)
        self.rounds[kind] = self.rounds.get(kind, 0) + nbits
        return acc

    # -- 쓰기 유닛: 재양자화 -> 편향 -> ReLU -> (MVADD면) 옛값 더하기
    def write_unit(self, acc, k, bias=None, relu=False, old=None):
        v = shr(acc, k)
        self.chk("쓰기 유닛 중간값(9비트)", v)
        if bias is not None:
            v += bias
            self.chk("쓰기 유닛 중간값(9비트)", v)
        if relu:
            v = max(v, 0)
        if old is not None:
            v += old
        return v

    def matvec(self, mat, src, k, dst=None, add=False):
        d = WDESC[mat]
        lanes = list(self.lanes_of(src))                   # 이번 명령 동안 레인에 걸린 값 (쓰레기 레인 포함)
        bias_name, relu = POST.get(mat, (None, False))
        out = []
        for j in range(d["n_out"]):
            row, _ = WROM[d["base"] + j]
            acc = self.serial_dot(lanes, row, 4, mat)
            if dst is None:                                # ARGMAX: 재양자화 없이 비교만
                out.append(acc)
                continue
            bias = W[bias_name][j] if bias_name else None
            old = self.read_lane(dst, j) if add else None
            v = self.write_unit(acc, k, bias, relu, old)
            out.append(v)
            if dst == "KC":
                self.K[self.P][j] = v
                self.writes += 1
            elif dst == "VC":
                self.V[j][self.P] = v
                self.writes += 1
            else:
                self.write_lane(dst, j, v)                 # 열 j가 끝나면 레인 j에 바로 씀
        return out

    def step(self):
        op, a, b, imm = decode(PROG_WORDS[self.PC])
        self.instr_count += 1
        nxt = self.PC + 1
        rname = {v: k for k, v in REGS.items()}.get(a, "")
        P = self.P
        if op == "EMB":
            tok = CROM[CDESC["tok"] + self.T[P]][0]
            pos = CROM[CDESC["pos"] + P][0]
            for j in range(D):
                self.write_lane("X", j, tok[j] + pos[j])
            self.snap = {"h0": self.slot[0][:D]}
        elif op in ("MV", "MVADD"):
            mat = MATS[b]
            res = self.matvec(mat, SRC[mat], imm, dst=rname, add=(op == "MVADD"))
            key = {"Q": "q", "KC": "k", "VC": "v", "H": "h"}.get(rname)
            if key:
                self.snap[key] = res
            if op == "MVADD":
                self.snap["x1" if mat == "Wo" else "x2"] = self.slot[0][:D]
        elif op == "SCORE":
            s = []
            for j in range(P + 1):
                acc = self.serial_dot(self.K[j], self.lanes_of("Q"), 8, "SCORE")   # 레인 = K_j, c = Q 비트
                v = shr(acc, imm)
                self.chk("점수(8비트)", v)
                self.write_lane("S", j, v)
                s.append(v)
            self.s = s
        elif op == "SOFTMAX":                              # 칸 2를 읽기 버스로 세 번 훑음
            s = [self.read_lane("S", j) for j in range(P + 1)]
            m = max(s)                                     # 1회: 최댓값
            e = [LUT[min(m - v, len(LUT) - 1)] for v in s] # 2회: 지수와 합
            S = sum(e)
            r = 65536 // S
            self.chk("합 S(10비트)", S)
            self.chk("역수 r(9비트)", r)
            self.chk("지수값×r(16비트)", [x * r for x in e])
            p = [(((x * r) >> 8) + 1) >> 1 for x in e]
            self.chk("확률 p(8비트)", p)
            for j in range(8):                             # 3회: p 쓰기. P보다 뒤 위치는 0 (캐시 쓰레기 가리기)
                self.write_lane("P", j, p[j] if j <= P else 0)
            self.snap["prob"] = p
        elif op == "MIXV":
            out = []
            for i in range(D):                             # 레인 = 위치 j, c 선 = V 캐시의 원소 i 줄
                acc = self.serial_dot(self.V[i], self.lanes_of("P"), 8, "MIXV", signed_gates=False)  # 레인 = V, c = p 비트
                v = shr(acc, imm)
                self.write_lane("A", i, v)
                out.append(v)
            self.snap["attn_out"] = out
        elif op == "ARGMAX":
            logits = self.matvec(MATS[b], "X", 0, dst=None)
            best = 0
            for j in range(1, len(logits)):                # 더 클 때만 교체 = 동점이면 작은 번호
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
            self.trace.append(f"  PC {self.PC:2d}  P={P}  {PROG[self.PC]['text']:<18} " + self.show(op, rname))
        self.PC = nxt
        return True

    def show(self, op, rname):
        if op == "EMB": return f"X(칸0) = {self.slot[0][:D]}"
        if op in ("MV", "MVADD"):
            if rname == "KC": return f"K 칸(위치 {self.P}) = {self.K[self.P][:D]}"
            if rname == "VC": return f"V 칸들의 레인 {self.P} = {[self.V[i][self.P] for i in range(D)]}"
            n = FFN if rname == "H" else D
            return f"{rname}(칸{SLOT_OF[rname]}) = {self.lanes_of(rname)[:n]}"
        if op == "SCORE": return f"s(칸2) = {self.s}"
        if op == "SOFTMAX": return f"p(칸2) = {self.slot[2][:8]}"
        if op == "MIXV": return f"A(칸1) = {self.slot[1][:D]}"
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
    viol, first_diff, rng = {}, None, {}
    npu = None
    for n, p in enumerate(perms):
        npu = NPU(p, seed=n)
        out = npu.run()
        ref_out, logs = R.sort4(p, return_logs=True)
        ok_out += out == ref_out
        same = all(s.get(k) == l.get(k) for s, l in zip(npu.snaps, logs) for k in KEYS if k in l)
        ok_all += same
        if not same and first_diff is None:
            first_diff = p
        for k, v in npu.violations.items():
            viol[k] = viol.get(k, 0) + v
        for k, (lo, hi) in npu.ranges.items():
            a, b = rng.get(k, (lo, hi))
            rng[k] = (min(a, lo), max(b, hi))
    print(f"출력 토큰 일치 (int_ref 대비)     : {ok_out}/{len(perms)}")
    print(f"정거장 중간값 전부 일치            : {ok_all}/{len(perms)}")
    if first_diff:
        print(f"  처음 달라진 입력: {first_diff}  ->  python npu.py --trace {' '.join(map(str, first_diff))}")
    print("비트 폭 위반                      : " + ("없음" if not viol else str(viol)))
    print("\n값 범위 (1680개 전체, 시작 상태는 매번 다른 쓰레기 값)")
    for k, (lo, hi) in rng.items():
        print(f"  {k:<24} {lo:>7} ~ {hi:<7} (허용 {WIDTH[k][0]} ~ {WIDTH[k][1]})")
    tot = sum(npu.rounds.values())
    print(f"\n한 번 실행당: 명령어 {npu.instr_count}개, 엔진 바퀴 {tot}회, 쓰기 {npu.writes}회")
    print("  바퀴 내역: " + ", ".join(f"{k} {v}" for k, v in npu.rounds.items()))
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
        print(f"\n입력 {nums} -> 출력 {out}   (명령어 {npu.instr_count}개, 엔진 바퀴 {sum(npu.rounds.values())}회)")
    else:
        verify()