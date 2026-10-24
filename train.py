"""
레드스톤 트랜스포머 - 학습 + 정수 양자화 인식 학습(QAT)

과제: 서로 다른 숫자 4개(0~7) 정렬
  입력 토큰열 : 5 2 7 0 | 0 2 5     (위치 0~7)
  예측 대상   :         0 2 5 7     (위치 4~7에서 다음 토큰 예측)

모델: 디코더 온리, 1층, 1헤드, 정규화 없음, 학습형 절대 위치 임베딩, ReLU FFN

양자화 규칙 (= 레드스톤 하드웨어 규칙):
  - 모든 텐서는 "정수 × 2^-e" 형태. e(지수)는 텐서마다 하나, 2의 거듭제곱 스케일
  - 가중치 int4 [-8, 7], 임베딩·활성값 int8 [-127, 127]
  - 재양자화는 곱셈 없이 "반올림 산술 시프트" = floor(x / 2^k + 0.5)
  - softmax: 점수 정수화 -> (최댓값 - 점수)를 주소로 지수 룩업 ROM(16칸, 0~255)
             -> 합 S -> 역수 r = floor(65536 / S) 한 번 -> p = (e * r + 256) >> 9  (0~128)

사용법:
  python train.py --sweep                      # d_model / FFN 조합 비교
  python train.py --d 8 --ffn 16 --export      # 한 조합 학습 후 weights.json 저장
"""
import argparse
import itertools
import json
import math
import random

import torch
import torch.nn.functional as F

torch.set_default_dtype(torch.float64)  # 정수 연산을 float로 정확히 흉내 내기 위해 64비트 사용

SEP = 8          # 구분 토큰
VOCAB = 9        # 0~7 + SEP
N_OUT = 8        # 출력은 숫자 0~7만
SEQ = 8          # 모델 입력 길이 (위치 0~7)
LUT_SIZE = 16    # 지수 룩업 ROM 칸 수
E_SCORE = 1      # 점수 지수: 정수 1 = 0.5 nat  -> 16칸이 0~7.5 nat 범위를 덮음
E_PROB = 7       # 확률 지수: 128 = 1.0
EXP_LUT = [math.floor(255 * math.exp(-i * 2.0 ** -E_SCORE) + 0.5) for i in range(LUT_SIZE)]


# ---------------------------------------------------------------- 데이터
def make_data(seed=0, test_ratio=0.2):
    perms = list(itertools.permutations(range(8), 4))  # 1680개
    random.Random(seed).shuffle(perms)
    n_test = int(len(perms) * test_ratio)

    def to_tensors(ps):
        xs, ys = [], []
        for p in ps:
            s = sorted(p)
            xs.append(list(p) + [SEP] + s[:3])
            ys.append(s)
        return torch.tensor(xs), torch.tensor(ys)

    return to_tensors(perms[n_test:]), to_tensors(perms[:n_test]), perms


# ---------------------------------------------------------------- 양자화 도구
def rnd(x):
    """반올림 floor(x+0.5). 값은 정수, 기울기는 그대로 통과(STE)."""
    return x + (torch.floor(x + 0.5) - x).detach()


def ste(int_val, float_val):
    """값은 정수 경로, 기울기는 실수 경로."""
    return float_val + (int_val - float_val).detach()


def act_exp(maxabs, qmax=127):
    return math.floor(math.log2(qmax / max(maxabs, 1e-8)))


def quant_w(w, qmax=7, qmin=-8):
    """가중치 -> (정수 텐서, 지수). 지수는 현재 최댓값으로 결정."""
    with torch.no_grad():
        e = act_exp(w.abs().max().item(), qmax)
    return torch.clamp(rnd(w * 2.0 ** e), qmin, qmax), e


def requant(acc, e_in, e_out, lo=-127, hi=127):
    """정수 누산값(지수 e_in) -> 정수(지수 e_out). 하드웨어: 반올림 산술 시프트."""
    return torch.clamp(rnd(acc * 2.0 ** (e_out - e_in)), lo, hi)


# ---------------------------------------------------------------- 모델
class TinyGPT(torch.nn.Module):
    def __init__(self, d, ffn):
        super().__init__()
        self.d, self.ffn = d, ffn
        P = lambda *s: torch.nn.Parameter(torch.randn(*s) * (1 / math.sqrt(s[0]) if len(s) > 1 else 1))
        self.tok = torch.nn.Parameter(torch.randn(VOCAB, d))
        self.pos = torch.nn.Parameter(torch.randn(SEQ, d))
        self.Wq, self.Wk, self.Wv, self.Wo = P(d, d), P(d, d), P(d, d), P(d, d)
        if ffn:
            self.W1, self.b1 = P(d, ffn), torch.nn.Parameter(torch.zeros(ffn))
            self.W2, self.b2 = P(ffn, d), torch.nn.Parameter(torch.zeros(d))
        self.U = P(d, N_OUT)
        self.qat = False
        self.pct = 1.0  # 보정 기준: 1.0이면 최댓값, 0.999면 상위 0.1%는 잘라냄(포화)
        self.E = {}  # 활성값 지수 (보정 후 고정)

    # ---- 실수 모델 (1단계 학습 + 보정용)
    def forward_float(self, x, collect=None):
        T = x.shape[1]
        h0 = self.tok[x] + self.pos[:T]
        q = h0 @ self.Wq / math.sqrt(self.d)  # 1/sqrt(d)를 Wq에 흡수한 형태로 기록
        k, v = h0 @ self.Wk, h0 @ self.Wv
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool), 1)
        att = (q @ k.transpose(1, 2)).masked_fill(mask, -1e9).softmax(-1)
        a = att @ v
        o = a @ self.Wo
        x1 = h0 + o
        if self.ffn:
            hh = F.relu(x1 @ self.W1 + self.b1)
            f = hh @ self.W2 + self.b2
            x2 = x1 + f
        else:
            hh = f = None
            x2 = x1
        if collect is not None:
            for name, t in dict(h0=h0, q=q, k=k, v=v, a=a, o=o, x1=x1, h=hh, f=f, x2=x2).items():
                if t is not None:
                    v = t.abs().flatten()
                    v = v.max().item() if self.pct >= 1.0 else torch.quantile(v, self.pct).item()
                    collect[name] = max(collect.get(name, 0), v)
        return x2 @ self.U, att

    # ---- 정수 모델 (QAT): 값은 레드스톤 하드웨어와 비트 단위로 동일
    def forward_int(self, x):
        E, d = self.E, self.d
        T = x.shape[1]
        er = E["r"]
        tok_i = torch.clamp(rnd(self.tok * 2.0 ** er), -127, 127)
        pos_i = torch.clamp(rnd(self.pos * 2.0 ** er), -127, 127)
        h0 = torch.clamp(tok_i[x] + pos_i[:T], -127, 127)              # 포화 덧셈

        Wq, eq_w = quant_w(self.Wq / math.sqrt(d))
        Wk, ek_w = quant_w(self.Wk)
        Wv, ev_w = quant_w(self.Wv)
        q = requant(h0 @ Wq, er + eq_w, E["q"])
        k = requant(h0 @ Wk, er + ek_w, E["k"])
        v = requant(h0 @ Wv, er + ev_w, E["v"])

        # --- 점수 + 정수 softmax
        qk = q @ k.transpose(1, 2)                                       # 지수 E[q]+E[k]
        s_real = qk * 2.0 ** -(E["q"] + E["k"])                          # 실수 점수 (기울기용)
        s = torch.floor(s_real * 2.0 ** E_SCORE + 0.5)                   # 정수 점수
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool), 1)
        s = s.masked_fill(mask, -1e9)
        m = s.max(-1, keepdim=True).values
        idx = torch.clamp(m - s, 0, LUT_SIZE - 1).long()
        lut = torch.tensor(EXP_LUT, dtype=torch.float64)
        e = lut[idx].masked_fill(mask, 0)
        S = e.sum(-1, keepdim=True)
        r = torch.floor(65536 / S)
        p_int = torch.floor((e * r + 256) / 512)                         # 0~128
        p_float = s_real.masked_fill(mask, -1e9).softmax(-1) * 128
        p = ste(p_int, p_float)

        a = requant(p @ v, E_PROB + E["v"], E["a"])
        Wo, eo_w = quant_w(self.Wo)
        o = requant(a @ Wo, E["a"] + eo_w, er)
        x1 = torch.clamp(h0 + o, -127, 127)

        if self.ffn:
            W1, e1 = quant_w(self.W1)
            W2, e2 = quant_w(self.W2)
            b1 = torch.clamp(rnd(self.b1 * 2.0 ** E["h"]), -127, 127)
            b2 = torch.clamp(rnd(self.b2 * 2.0 ** er), -127, 127)
            hh = torch.clamp(requant(x1 @ W1, er + e1, E["h"]) + b1, 0, 127)   # ReLU = 0에서 자르기
            f = torch.clamp(requant(hh @ W2, E["h"] + e2, er) + b2, -127, 127)
            x2 = torch.clamp(x1 + f, -127, 127)
        else:
            x2 = x1

        U, eu = quant_w(self.U)
        logits_int = x2 @ U                                              # argmax만 하므로 재양자화 없음
        return logits_int * 2.0 ** -(er + eu), p_int / 128

    def forward(self, x):
        return self.forward_int(x) if self.qat else self.forward_float(x)

    # ---- 보정: 실수 모델의 활성값 범위로 지수 결정
    def calibrate(self, x):
        c = {}
        with torch.no_grad():
            self.forward_float(x, collect=c)
        r_max = max(c[n] for n in ("h0", "o", "x1", "f", "x2") if n in c)
        self.E = {"r": act_exp(r_max), "q": act_exp(c["q"]), "k": act_exp(c["k"]),
                  "v": act_exp(c["v"]), "a": act_exp(c["a"])}
        if self.ffn:
            self.E["h"] = act_exp(c["h"])
        self.qat = True


# ---------------------------------------------------------------- 학습/평가
def loss_fn(model, x, y):
    logits, _ = model(x)
    return F.cross_entropy(logits[:, 4:].reshape(-1, N_OUT), y.reshape(-1))


@torch.no_grad()
def generate(model, prompts):
    """자기회귀 그리디 생성. prompts: (B,5) -> (B,4)"""
    seq = prompts.clone()
    outs = []
    for _ in range(4):
        logits, _ = model(seq)
        nxt = logits[:, -1].argmax(-1)  # 동점이면 작은 번호 (하드웨어 비교기와 동일하게 맞출 것)
        outs.append(nxt)
        seq = torch.cat([seq, nxt[:, None]], 1)
    return torch.stack(outs, 1)


def exact_acc(model, x, y):
    return (generate(model, x[:, :5]) == y).all(1).double().mean().item()


def train(model, x, y, steps, lr, log=False, wd=0.0):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for i in range(steps):
        loss = loss_fn(model, x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if log and (i % 500 == 0 or i == steps - 1):
            print(f"  step {i:5d}  loss {loss.item():.4f}")
    return loss.item()


def run(d, ffn, seed, float_steps=1500, qat_steps=1500, log=False,
        lr=1e-2, qat_lr=2e-3, wd=0.0, pct=1.0):
    torch.manual_seed(seed)
    (xtr, ytr), (xte, yte), _ = make_data()
    model = TinyGPT(d, ffn)
    model.pct = pct
    if log: print("[1] 실수 학습")
    train(model, xtr, ytr, float_steps, lr, log, wd)
    f_tr, f_te = exact_acc(model, xtr, ytr), exact_acc(model, xte, yte)
    model.calibrate(xtr)
    q0_te = exact_acc(model, xte, yte)
    if log: print(f"[2] 정수 QAT (보정 직후 테스트 정확도 {q0_te:.3f})")
    train(model, xtr, ytr, qat_steps, qat_lr, log)
    q_tr, q_te = exact_acc(model, xtr, ytr), exact_acc(model, xte, yte)
    return model, dict(float_train=f_tr, float_test=f_te, int_test_before_qat=q0_te,
                       int_train=q_tr, int_test=q_te)


def n_params(d, ffn):
    n = VOCAB * d + SEQ * d + 4 * d * d + d * N_OUT
    return n + (2 * d * ffn + ffn + d if ffn else 0)


# ---------------------------------------------------------------- 내보내기
def export(model, path="weights.json"):
    E, d = model.E, model.d
    with torch.no_grad():
        ints = lambda t: [[int(v) for v in row] for row in t.tolist()]
        out = {"d": d, "ffn": model.ffn, "vocab": VOCAB, "sep": SEP, "seq": SEQ,
               "exp_lut": EXP_LUT, "e_score": E_SCORE, "e_prob": E_PROB, "E": dict(E)}
        out["tok"] = ints(torch.clamp(torch.floor(model.tok * 2.0 ** E["r"] + 0.5), -127, 127))
        out["pos"] = ints(torch.clamp(torch.floor(model.pos * 2.0 ** E["r"] + 0.5), -127, 127))
        W = {"Wq": model.Wq / math.sqrt(d), "Wk": model.Wk, "Wv": model.Wv, "Wo": model.Wo, "U": model.U}
        if model.ffn:
            W.update(W1=model.W1, W2=model.W2)
            out["b1"] = [int(v) for v in torch.clamp(torch.floor(model.b1 * 2.0 ** E["h"] + 0.5), -127, 127).tolist()]
            out["b2"] = [int(v) for v in torch.clamp(torch.floor(model.b2 * 2.0 ** E["r"] + 0.5), -127, 127).tolist()]
        for name, w in W.items():
            wi, ew = quant_w(w)
            out[name] = ints(wi)
            out["E"]["w_" + name] = ew
        # 전체 1680개 입력에 대한 QAT 모델의 출력 (정수 레퍼런스 검증용)
        _, _, perms = make_data()
        prompts = torch.tensor([list(p) + [SEP] for p in perms])
        out["qat_outputs"] = {" ".join(map(str, p)): o for p, o in zip(perms, generate(model, prompts).tolist())}
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"저장: {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--d", type=int, default=8)
    ap.add_argument("--ffn", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--export", action="store_true")
    ap.add_argument("--seeds", type=int, default=1, help="시드 여러 개를 돌려 가장 좋은 모델 선택")
    ap.add_argument("--float-steps", type=int, default=1500)
    ap.add_argument("--qat-steps", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--qat-lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=0.0, help="실수 학습 weight decay")
    ap.add_argument("--pct", type=float, default=1.0, help="보정 퍼센타일 (예: 0.999)")
    args = ap.parse_args()
    hp = dict(float_steps=args.float_steps, qat_steps=args.qat_steps, lr=args.lr,
              qat_lr=args.qat_lr, wd=args.wd, pct=args.pct)

    if args.sweep:
        print(f"{'d':>2} {'ffn':>3} {'seed':>4} {'params':>6} | {'float_te':>8} {'int_noQAT':>9} {'int_tr':>6} {'int_te':>6}")
        for d in (4, 6, 8):
            for ffn in (0, 2 * d):
                for seed in (0, 1):
                    _, r = run(d, ffn, seed, **hp)
                    print(f"{d:>2} {ffn:>3} {seed:>4} {n_params(d, ffn):>6} | {r['float_test']:8.3f} "
                          f"{r['int_test_before_qat']:9.3f} {r['int_train']:6.3f} {r['int_test']:6.3f}", flush=True)
    elif args.seeds > 1:
        print(f"d={args.d} ffn={args.ffn} params={n_params(args.d, args.ffn)}  {hp}")
        print(f"{'seed':>4} | {'float_te':>8} {'int_noQAT':>9} {'int_tr':>6} {'int_te':>6}")
        best = None
        for seed in range(args.seed, args.seed + args.seeds):
            model, r = run(args.d, args.ffn, seed, **hp)
            print(f"{seed:>4} | {r['float_test']:8.3f} {r['int_test_before_qat']:9.3f} "
                  f"{r['int_train']:6.3f} {r['int_test']:6.3f}", flush=True)
            key = (r["int_test"], r["int_train"])
            if best is None or key > best[0]:
                best = (key, seed, model)
        print(f"최고: seed {best[1]}  (int_test, int_train) = {best[0]}")
        if args.export:
            export(best[2])
    else:
        model, r = run(args.d, args.ffn, args.seed, log=True, **hp)
        print({k: round(v, 4) for k, v in r.items()})
        print("지수:", model.E)
        if args.export:
            export(model)
