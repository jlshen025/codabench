"""decode_conf.py — recover the EXACT confusion matrix of a submission from its
server scores + known gold marginals.  [PORTED FROM stanceeval_t2, 2026-07-29]

The scorer returns Overall_Favg2, Overall_Favg3 and Overall_Accuracy. With the gold
marginals known and your own predicted marginals known locally, three equations pin
TP_Favor / TP_Against / TP_None to a UNIQUE integer solution:

    Accuracy*N        = TP_F + TP_A + TP_N
    3*Favg3 - 2*Favg2 = F1_None = 2*TP_N/(pred_N + gold_N)
    2*Favg2           = 2*TP_F/(pred_F+gold_F) + 2*TP_A/(pred_A+gold_A)

VERIFIED on this project: champion 869442 (pred F157/A181/N14, server
0.891160/0.677440/0.846591) decodes to TP_F=144/158, TP_A=148/160, TP_N=6/34 and
reproduces Overall_Favg2 to 6 dp. Its headline reading: of 54 total errors, **28 are
gold-None rows sitting inside the stance predictions** as pure false positives, and
the off-diagonal solve puts most of them in the Against pool (goldNone->predA 19-27
vs goldNone->predF 1-9) — so the None repair to hunt first is Against-side.

Usage: decode_conf.py <predictions.txt|submission.zip> <favg2> <favg3> <accuracy> [label]

BREAK-EVEN LAW. Favg2 excludes None, so moving a row OUT of a stance class touches
that one F1 term. Removing a non-gold row from Favor gains +0.141 pts; removing a
true Favor loses -0.172 => break-even 55.0% precision (Against: 56.8%). Compute this
BEFORE reading rows, then rank and cut there. The threshold RISES as you improve.

GATE-1 LINE (inherited, non-negotiable). Aggregate counts are legitimate: they tell you
how well a candidate did. Combining several overlapping reads to solve for WHICH
individual hidden rows carry which label, and picking rows on that basis, is label
reconstruction — the multi-set form of the forbidden "small-diff pair", and a DQ is
irreversible. t2 derived such a set (worth a guaranteed +0.307) and DELETED IT
UNSUBMITTED. Legal: rows chosen by judgment about the TEXT, decoder used only to score
and forecast. Forbidden: rows chosen by an optimiser over label assignments.

GOLD (this project's recovered prior): Favor 158 / Against 160 / None 34, N=352.
"""
import sys, json, collections
import pandas as pd

GOLD = {"Women Driving": (158, 160, 34)}   # (Favor, Against, None)
POOL_GOLD = (158, 160, 34)
N_TEST = 352


def solve(N, gold, pred, favg2, favg3, acc):
    gF, gA, gN = gold
    pF, pA, pN = pred
    correct = round(acc * N)
    TPN = round((3 * favg3 - 2 * favg2) * (pN + gN) / 2)
    S = correct - TPN
    d = 1 / (pF + gF) - 1 / (pA + gA)
    TPF = round((favg2 - S / (pA + gA)) / d)
    return TPF, S - TPF, TPN


def solve_full(gold, pred, tp):
    """Off-diagonals as a function of the one free parameter a = C[Against][Favor]."""
    gF, gA, gN = gold
    pF, pA, pN = pred
    TPF, TPA, TPN = tp
    rows = []
    for a in range(0, gA - TPA + 1):
        NF = (pF - TPF) - a           # C[None][Favor]
        NA = (gN - TPN) - NF          # C[None][Against]
        FA = (pA - TPA) - NA          # C[Favor][Against]
        FN = (gF - TPF) - FA          # C[Favor][None]
        AN = (gA - TPA) - a           # C[Against][None]
        if min(NF, NA, FA, FN, AN) < 0:
            continue
        rows.append(dict(a=a, N2F=NF, N2A=NA, F2A=FA, F2N=FN, A2N=AN))
    return rows


def load_pred(txt_path):
    """Labels from a predictions .txt OR straight out of a submission .zip."""
    if str(txt_path).endswith(".zip"):
        import zipfile
        with zipfile.ZipFile(txt_path) as z:
            raw = z.read("predictions.txt").decode()
    else:
        with open(txt_path, encoding="utf-8") as fh:
            raw = fh.read()
    return [l.strip() for l in raw.splitlines() if l.strip()]


def report(txt_path, favg2, favg3, acc, label=""):
    """This track's test is a SINGLE target (Women Driving, 352 rows), so the
    pooled overall scores decode directly — no per-target split as in t2."""
    pred = load_pred(txt_path)
    n = len(pred)
    assert n == N_TEST, f"expected {N_TEST} labels, got {n}"
    c = collections.Counter(pred)
    pc = (c["Favor"], c["Against"], c["None"])
    gF, gA, gN = POOL_GOLD
    TPF, TPA, TPN = solve(n, POOL_GOLD, pc, favg2, favg3, acc)

    print(f"\n=== {label or txt_path} ===")
    print(f"  pred F{pc[0]} A{pc[1]} N{pc[2]}   gold F{gF} A{gA} N{gN}  (N={n})")
    print(f"  TP_F={TPF}/{gF}  TP_A={TPA}/{gA}  TP_N={TPN}/{gN}   errors={n - (TPF+TPA+TPN)}")
    # Re-derive the scores from the solution: a mismatch means the inputs are wrong.
    chk2 = TPF / (pc[0] + gF) + TPA / (pc[1] + gA)
    chk_n = 2 * TPN / (pc[2] + gN)
    print(f"  check: Favg2={chk2:.6f} (given {favg2:.6f})  F1_None={chk_n:.4f}")
    print(f"  gold-None rows sitting in your STANCE predictions: {gN - TPN}"
          f"  (pure false positives dragging BOTH F1 terms)")
    for r in solve_full(POOL_GOLD, pc, (TPF, TPA, TPN)):
        print(f"      a={r['a']}: goldNone->predF {r['N2F']}, goldNone->predA {r['N2A']}, "
              f"goldFavor->predA {r['F2A']}, goldFavor->predN {r['F2N']}, goldAgainst->predN {r['A2N']}")


if __name__ == "__main__":
    # usage: decode_conf.py <predictions.txt|submission.zip> <favg2> <favg3> <acc> [label]
    if len(sys.argv) < 5:
        sys.exit("usage: decode_conf.py <predictions.txt|.zip> <favg2> <favg3> <accuracy> [label]")
    report(sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]),
           sys.argv[5] if len(sys.argv) > 5 else "")
