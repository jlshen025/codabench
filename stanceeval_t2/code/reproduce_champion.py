#!/usr/bin/env python
"""reproduce_champion.py — regenerate the ranked submission bit-exactly and verify it.

The ranked entry (Codabench submission 869013, Favg2 0.935700 on the Track-2 blind test) is
    sol grounded base  +  73 adjudicated single-row corrections.

This script rebuilds it from the two stored artifacts, writes a flat predictions.txt / .zip,
and asserts a byte-for-byte match against the mirrored submission. It is the reproduction
entry point referenced by METHOD.md; it needs no network and no GPU.

Run:  python reproduce_champion.py            (from scripts/)

ASSET MAP (everything needed, all under this project):
  _llm/g56fix_grounded_test.npz  the base reader's 644 predictions (gpt-5.6-sol, grounded
                                 prompt with the fixed target cards in _llm/fixed_cards.json).
                                 Regenerate with llm_grounded.py; the base alone scores 0.9016.
  _llm/champion_flips.json       the 73 corrections as {row: [from, to]}, derived from the
                                 submission lineage. Provenance for each block is in
                                 METHOD.md §8-§9 and scripts/_llm/opus5_prereg.md.
  test_norm.csv                  the organisers' test_track_2.csv with tweet_text renamed to
                                 text; row order preserved exactly (order is the join key).
  staging/e_ch869013.txt         the exact submitted label vector (verification target).
  ../submission/codabench/869013__sub_o5r4f.zip   the archived submission, same content.

The 73 corrections break down as: None->Favor 32 · None->Against 20 · Against->Favor 17 ·
Favor->None 2 · Favor->Against 1 · Against->None 1 — i.e. the system's dominant correction is
recovering stance from rows the base reader abstained on, which is the annotation-convention
effect quantified in METHOD.md §10.7.
"""
import json, hashlib, zipfile, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
LABELS = np.array(['Against', 'Favor', 'None'])


def main():
    base = LABELS[np.load(HERE / '_llm/g56fix_grounded_test.npz', allow_pickle=True)['pred_id']]
    assert len(base) == 644, f'base has {len(base)} rows, expected 644'
    flips = {int(k): v for k, v in json.load(open(HERE / '_llm/champion_flips.json')).items()}

    pred = list(map(str, base))
    for i, (frm, to) in sorted(flips.items()):
        assert pred[i] == frm, f'row {i}: base is {pred[i]!r}, flip expects {frm!r}'
        pred[i] = to
    print(f'applied {len(flips)} corrections to the {len(base)}-row base')

    out = HERE / 'staging' / 'e_reproduced.txt'
    out.write_text('\n'.join(pred) + '\n')
    zp = HERE / 'staging' / 'sub_reproduced.zip'
    with zipfile.ZipFile(zp, 'w', zipfile.ZIP_DEFLATED) as z:
        z.writestr('predictions.txt', out.read_text())

    ref = [l.strip() for l in (HERE / 'staging/e_ch869013.txt').read_text().splitlines() if l.strip()]
    md5 = lambda xs: hashlib.md5(('\n'.join(xs) + '\n').encode()).hexdigest()
    ok = pred == ref
    print(f'  reproduced md5 {md5(pred)}')
    print(f'  submitted  md5 {md5(ref)}')
    print(f'  BIT-EXACT MATCH: {ok}')
    if not ok:
        diff = [i for i in range(644) if pred[i] != ref[i]]
        print(f'  MISMATCH at rows {diff[:20]}')
    print(f'  wrote {zp.relative_to(HERE)} (flat, predictions.txt at root)')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
