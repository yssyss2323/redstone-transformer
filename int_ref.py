"""
레드스톤 트랜스포머 - 정수 레퍼런스 구현 (하드웨어와 비트 단위로 같은 계산)

- 외부 라이브러리 없음. 파이썬 정수와 시프트, 덧셈, 곱셈, 비교, 나눗셈 1회만 사용
- KV 캐시를 쓰는 실제 디코딩 순서 그대로: 프리필(K,V만) -> 생성 패스 4회
- 정거장(①~⑩)별 중간값을 출력해서 마인크래프트 표시판과 한 줄씩 대조하는 용도

사용법:
  python int_ref.py                    # 검증(1680개 전체) + 예시 1개 덤프
  python int_ref.py 3 6 1 4            # 이 입력으로 덤프
  python int_ref.py --no-cache 3 6 1 4 # KV 캐시 없이 (연산량 비교)
  python int_ref.py --ranges           # 1680개 전체에서 모든 값의 범위와 필요한 비트 수
"""
import itertools
import json
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")   # 윈도우에서 > 파일 저장 시 한글 깨짐 방지
except AttributeError:
    pass

W = json.load(open("weights.json"))
D, FFN, SEP, E = W["d"], W["ffn"], W["sep"], W["E"]
LUT = W["exp_lut"]


# ---------------------------------------------------------------- 하드웨어 기본 부품
def sat(x, lo=-127, hi=127):          # 포화 덧셈기의 출력단
    return max(lo, min(hi, x))


def shr(acc, k):                      # 반올림 산술 시프트 = floor(acc / 2^k + 0.5)
    if k > 0:
        return (acc + (1 << (k - 1))) >> k
    return acc << (-k)


def requant(acc, e_in, e_out, lo=-127, hi=127):
    return sat(shr(acc, e_in - e_out), lo, hi)


class Counter:                        # 연산량 집계 (8레인 내적 엔진 기준 사이클 추정용)
    LANES = 8

    def __init__(self):
        self.mac = 0
        self.ops = 0

    def dot(self, n, times=1):
        self.mac += n * times
        self.ops += -(-n // self.LANES) * times


CNT = Counter()


class Ranges:                         # 값별 최소/최대 기록 (하드웨어 비트 폭 결정용)
    def __init__(self):
        self.on = False
        self.r = {}

    def __call__(self, name, vals):
        if not self.on:
            return
        vals = vals if isinstance(vals, list) else [vals]
        lo, hi = self.r.get(name, (min(vals), max(vals)))
        self.r[name] = (min(lo, min(vals)), max(hi, max(vals)))


RNG = Ranges()


def matvec(x, M):                     # x(길이 n) @ M(n x m) -> m개의 내적
    n, m = len(M), len(M[0])
    CNT.dot(n, m)
    return [sum(x[i] * M[i][j] for i in range(n)) for j in range(m)]


# ---------------------------------------------------------------- 트랜스포머 정거장
def embed(tok, pos):                  # ② 임베딩: ROM 두 개 읽고 더하기
    raw = [a + b for a, b in zip(W["tok"][tok], W["pos"][pos])]
    RNG("②  임베딩 합 (포화 전)", raw)
    return [sat(v) for v in raw]


def proj(h0, name, e_out):            # ③ Q/K/V 투영
    acc = matvec(h0, W[name])
    RNG(f"③  {name[1].upper()} 누산기", acc)
    out = [requant(a, E["r"] + E["w_" + name], e_out) for a in acc]
    RNG(f"③  {name[1].upper()}", out)
    return out


def attention(q, K, V, log):
    # ⑤ 점수: q · k_j
    CNT.dot(D, len(K))
    qk = [sum(a * b for a, b in zip(q, k)) for k in K]
    s = [shr(v, E["q"] + E["k"] - W["e_score"]) for v in qk]
    RNG("⑤  Q·K 누산기", qk)
    RNG("⑤  점수", s)
    # ⑤ softmax: 최댓값 빼기 -> 지수 ROM -> 합 -> 역수 1회 -> 곱하기
    m = max(s)
    RNG("⑤  최댓값-점수 (자르기 전)", [m - v for v in s])
    idx = [min(m - v, len(LUT) - 1) for v in s]
    e = [LUT[i] for i in idx]
    S = sum(e)
    r = 65536 // S                    # 패스당 유일한 나눗셈
    p = [(ei * r + 256) >> 9 for ei in e]
    RNG("⑤  합 S", S)
    RNG("⑤  역수 r", r)
    RNG("⑤  지수값 × r", [ei * r for ei in e])
    RNG("⑤  확률 p", p)
    # ⑦ 가중합: Σ p_j v_j  (위치마다 V 벡터를 p_j배 해서 누산 -> 위치 수만큼 8레인 연산)
    CNT.dot(len(V), D)
    acc = [sum(p[j] * V[j][i] for j in range(len(V))) for i in range(D)]
    a = [requant(v, W["e_prob"] + E["v"], E["a"]) for v in acc]
    RNG("⑦  p·V 누산기", acc)
    RNG("⑦  attention 출력", a)
    log.update(score=s, exp_idx=idx, exp=e, S=S, recip=r, prob=p, attn_out=a)
    return a, p


def block(h0, q, K, V, log):
    a, p = attention(q, K, V, log)
    acc = matvec(a, W["Wo"])
    o = [requant(v, E["a"] + E["w_Wo"], E["r"]) for v in acc]
    RNG("⑦  Wo 누산기", acc)
    RNG("⑦  Wo 출력", o)
    RNG("⑦  residual 1 (포화 전)", [x + y for x, y in zip(h0, o)])
    x1 = [sat(x + y) for x, y in zip(h0, o)]                      # ⑦ residual (공용 버스)
    log.update(o=o, x1=x1)
    if FFN:                                                         # ⑧ FFN
        acc1 = matvec(x1, W["W1"])
        pre = [requant(v, E["r"] + E["w_W1"], E["h"]) + b for v, b in zip(acc1, W["b1"])]
        h = [sat(v, 0, 127) for v in pre]
        acc2 = matvec(h, W["W2"])
        fpre = [requant(v, E["h"] + E["w_W2"], E["r"]) + b for v, b in zip(acc2, W["b2"])]
        f = [sat(v) for v in fpre]
        RNG("⑧  W1 누산기", acc1)
        RNG("⑧  은닉 (편향 더한 후, ReLU 전)", pre)
        RNG("⑧  은닉 (ReLU 후)", h)
        RNG("⑧  W2 누산기", acc2)
        RNG("⑧  FFN 출력 (포화 전)", fpre)
        RNG("⑧  residual 2 (포화 전)", [x + y for x, y in zip(x1, f)])
        x2 = [sat(x + y) for x, y in zip(x1, f)]
        log.update(h=h, f=f, x2=x2)
    else:
        x2 = x1
    logits = matvec(x2, W["U"])                                     # ⑨ 출력 점수
    RNG("⑨  출력 점수", logits)
    best = max(range(len(logits)), key=lambda i: (logits[i], -i))  # 동점이면 작은 번호
    log.update(logits=logits, pred=best)
    return best, p


def sort4(digits, use_cache=True, verbose=False, return_logs=False):
    tokens = list(digits) + [SEP]
    K, V = [], []                                                   # ④ KV 캐시
    outs, attn_rows, logs = [], [], []

    # 프리필: 입력 숫자 4개는 K, V만 계산해서 캐시에 쓰기
    for pos in range(4):
        h0 = embed(tokens[pos], pos)
        K.append(proj(h0, "Wk", E["k"]))
        V.append(proj(h0, "Wv", E["v"]))

    # 생성 패스 4회: 위치 4(SEP), 5, 6, 7
    for pos in range(4, 8):
        if not use_cache:   # 캐시가 없으면 앞 위치 K, V를 매번 다시 계산
            K, V = [], []
            for p_ in range(pos):
                hp = embed(tokens[p_], p_)
                K.append(proj(hp, "Wk", E["k"]))
                V.append(proj(hp, "Wv", E["v"]))
        h0 = embed(tokens[pos], pos)
        q = proj(h0, "Wq", E["q"])
        k = proj(h0, "Wk", E["k"])
        v = proj(h0, "Wv", E["v"])
        K, V = K + [k], V + [v]
        log = dict(pos=pos, token=tokens[pos], h0=h0, q=q, k=k, v=v)
        pred, p = block(h0, q, K, V, log)
        outs.append(pred)
        attn_rows.append(p)
        logs.append(log)
        tokens.append(pred)
    if verbose:
        dump(digits, logs, attn_rows, outs)
    if return_logs:
        return outs, logs
    return outs


# ---------------------------------------------------------------- 출력
STATIONS = [("h0", "② 임베딩 합"), ("q", "③ Q"), ("k", "③ K"), ("v", "③ V"),
            ("score", "⑤ 점수"), ("exp_idx", "⑤ 지수ROM 주소"), ("exp", "⑤ 지수값"),
            ("S", "⑤ 합 S"), ("recip", "⑤ 역수 r"), ("prob", "⑤ 확률(/128)"),
            ("attn_out", "⑦ attention 출력"), ("o", "⑦ Wo 출력"), ("x1", "⑦ residual 1"),
            ("h", "⑧ FFN 은닉(ReLU)"), ("f", "⑧ FFN 출력"), ("x2", "⑧ residual 2"),
            ("logits", "⑨ 출력 점수"), ("pred", "⑨ 출력 토큰")]


def fmt(v):
    return " ".join(f"{x:5d}" for x in v) if isinstance(v, list) else str(v)


def dump(digits, logs, attn_rows, outs):
    name = lambda t: "|" if t == SEP else str(t)
    print(f"\n입력 {' '.join(map(str, digits))}  ->  출력 {' '.join(map(str, outs))}")
    for log in logs:
        print(f"\n── 생성 패스: 위치 {log['pos']}, 입력 토큰 '{name(log['token'])}' " + "─" * 30)
        for key, label in STATIONS:
            if key in log:
                print(f"  {label:<16} {fmt(log[key])}")
    # ⑩ attention 히스토리 판
    seq = list(digits) + [SEP] + outs[:3]
    shades = " .:-=+*#%@"
    print("\n⑩ attention 히스토리 (행: 생성 단계, 열: 위치, 진할수록 확률 높음)")
    print("  position  " + " ".join(f"{name(t):>2}" for t in seq))
    for i, row in enumerate(attn_rows):
        cells = [shades[min(9, p * 10 // 129)] * 2 for p in row]
        print(f"  out={outs[i]} ->  " + " ".join(cells))


def verify():
    perms = list(itertools.permutations(range(8), 4))
    ok_sort = ok_qat = 0
    for p in perms:
        out = sort4(p)
        ok_sort += out == sorted(p)
        ok_qat += out == W["qat_outputs"][" ".join(map(str, p))]
    print(f"정렬 정답      : {ok_sort}/{len(perms)}")
    print(f"PyTorch QAT 일치: {ok_qat}/{len(perms)}  (전부 일치해야 하드웨어 규칙이 정확히 재현된 것)")


def cost(digits):
    for cache in (True, False):
        CNT.mac = CNT.ops = 0
        sort4(digits, use_cache=cache)
        print(f"KV 캐시 {'on ' if cache else 'off'}: MAC {CNT.mac:5d}회, 8레인 내적 연산 {CNT.ops:4d}회")


def bits(lo, hi):
    """[lo, hi]를 담는 최소 비트 수. 음수가 있으면 2의 보수 기준."""
    if lo >= 0:
        return max(1, hi.bit_length()), "부호 없음"
    n = 1
    while not (-(1 << (n - 1)) <= lo and hi <= (1 << (n - 1)) - 1):
        n += 1
    return n, "부호 있음"


def ranges():
    RNG.on = True
    for p in itertools.permutations(range(8), 4):
        sort4(p)
    RNG.on = False
    print(f"모델: d_model {D}, FFN {FFN} / 입력 1680개 전체 (KV 캐시 on)")
    print("입력이 이 1680개뿐이라 아래 범위는 추정이 아니라 정확한 최악값이에요.\n")
    print(f"  {'값':<28} {'최소':>8} {'최대':>8}  {'비트':>4}  형식")
    for name, (lo, hi) in RNG.r.items():
        n, kind = bits(lo, hi)
        warn = "  ← int8 포화 발생" if "포화 전" in name and (lo < -127 or hi > 127) else ""
        print(f"  {name:<28} {lo:>8} {hi:>8}  {n:>4}  {kind}{warn}")


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--ranges" in args:
        ranges()
        sys.exit()
    cache = "--no-cache" not in args
    nums = [int(a) for a in args if a.isdigit()]
    if not nums:
        verify()
        nums = [5, 2, 7, 0]
    assert len(nums) == 4 and len(set(nums)) == 4 and all(0 <= n <= 7 for n in nums), "서로 다른 0~7 숫자 4개"
    sort4(nums, use_cache=cache, verbose=True)
    print()
    cost(nums)