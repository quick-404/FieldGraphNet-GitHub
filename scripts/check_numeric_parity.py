"""Numeric parity check between the English ICASSP paper and its Chinese mirror.

Task 10 of the ICASSP 2027 plan: extract every result-like decimal from both sources
and require that the two sets are identical, not merely overlapping.

Two traps this script exists to avoid, both of which produced false alarms when the
check was first run by hand:

1.  A naive comment strip of ``%[^\\n]*`` also eats everything after an escaped
    ``\\%``, so ``21.1\\% to 49.7\\%`` loses 49.7. Comments are therefore matched with
    ``(?<!\\\\)%`` so that escaped percent signs survive.
2.  LaTeX layout parameters (``2.6cm``, ``0.95\\textwidth``) are decimals but are not
    results. They are removed before comparison, otherwise the two files disagree on
    page-fitting constants that are expected to differ.

Run from the repository root:

    python scripts/check_numeric_parity.py

Exit status is 0 when parity holds and 1 when it does not, so this can gate a build.
"""
import os
import re
import sys
from collections import Counter

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPER = os.path.join(REPO, 'paper', 'icassp')
EN = os.path.join(PAPER, 'icassp2027_fieldgraphnet.tex')
ZH = os.path.join(PAPER, 'icassp2027_fieldgraphnet_zh.tex')


def result_numbers(path):
    """Decimals that represent results, with comments and layout constants removed."""
    src = open(path, encoding='utf-8').read()
    src = re.sub(r'(?<!\\)%[^\n]*', '', src)                 # comments, keeping \%
    src = re.sub(r'\\(label|cite|ref|includegraphics)\{[^}]*\}', ' ', src)
    # layout constants: \parbox[c][2.6cm][c]{...} and N.NN\textwidth / \linewidth
    src = re.sub(r'\\parbox\[c\]\[[^\]]*\]\[c\]\{[\d.]+\\?[a-z]*\}', ' ', src)
    src = re.sub(r'\d+\.\d+\\?(textwidth|linewidth|columnwidth)', ' ', src)
    return re.findall(r'\d+\.\d+', src)


def main():
    for p in (EN, ZH):
        if not os.path.exists(p):
            print(f'MISSING: {p}')
            return 1

    en, zh = result_numbers(EN), result_numbers(ZH)
    ce, cz = Counter(en), Counter(zh)

    print(f'English : {len(en)} decimals, {len(ce)} distinct   ({os.path.basename(EN)})')
    print(f'Chinese : {len(zh)} decimals, {len(cz)} distinct   ({os.path.basename(ZH)})')
    print()

    only_en = sorted(set(ce) - set(cz), key=float)
    only_zh = sorted(set(cz) - set(ce), key=float)
    differing = [(t, ce[t], cz[t]) for t in sorted(set(ce) & set(cz), key=float)
                 if ce[t] != cz[t]]

    if only_en:
        print('present in English but absent from Chinese:')
        for t in only_en:
            print(f'    {t}   (EN {ce[t]}, ZH 0)')
    if only_zh:
        print('present in Chinese but absent from English:')
        for t in only_zh:
            print(f'    {t}   (ZH {cz[t]}, EN 0)')
    if differing:
        print('present in both with different multiplicities:')
        for t, a, b in differing:
            print(f'    {t}   (EN {a}, ZH {b})')

    ok = not (only_en or only_zh or differing)
    print('VERDICT:', 'NUMERIC PARITY OK' if ok else 'MISMATCH FOUND')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
