# -*- coding: utf-8 -*-
"""Figure: per-protocol F1 of the label-free fusion vs baselines at ~20% FPR.

Reproducible replacement for the previously hand-made fig_ood_f1.pdf, which had no
generator script and still carried pre-alignment (falsified) numbers.

Data source: eval_results/fusion_comparison/multisignal_fusion_rules.json
  (5-seed mean F1, each method at its own realised ~20% benign FPR).

The label-free LOF arm is plotted deliberately: after the flow-alignment fix LOF is
stronger than the fusion on Mirai and Recon-PortScan, so a figure showing only
SVM / RF / Fusion would misleadingly imply the fusion is the strongest method.

Run from the repository root:
    .venv\\Scripts\\python.exe experiments/generate_ood_f1_figure.py
"""
import json
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "pdf.fonttype": 42,
    "font.size": 12,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 1.0,
    "legend.frameon": False,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
})

JSON = 'eval_results/fusion_comparison/multisignal_fusion_rules.json'
OUT_PDF = 'paper/figures/fig_ood_f1.pdf'
OUT_PNG = 'paper/figures/fig_ood_f1.png'

with open(JSON, encoding='utf-8') as f:
    agg = json.load(f)['aggregate_full_test']

PROTOCOLS = ['DNS_Spoofing.pcap', 'Mirai-udpplain.pcap', 'SqlInjection.pcap',
             'DictionaryBruteForce.pcap', 'Recon-PortScan.pcap']
SHORT = ['DNS\nSpoofing', 'Mirai', 'SQL\nInjection', 'Brute\nForce', 'Recon']

# (key in JSON, legend label, colour) — supervised first, then the label-free pair
SERIES = [
    ('SVM', 'SVM (supervised)', '#BDC3C7'),
    ('RF', 'Random Forest\n(supervised)', '#808090'),
    ('F3_fisher', 'Fusion\n(label-free)', '#00568D'),
    ('flow_lof', 'LOF\n(label-free)', '#8FB4D9'),
]

fig, ax = plt.subplots(figsize=(7.2, 3.4))
n = len(SERIES)
w = 0.8 / n
x = np.arange(len(PROTOCOLS))

for i, (key, label, colour) in enumerate(SERIES):
    vals = [agg[p][key]['f1_mean'] for p in PROTOCOLS]
    offs = (i - (n - 1) / 2.0) * w
    ax.bar(x + offs, vals, width=w * 0.92, color=colour, label=label,
           edgecolor='black', linewidth=0.4, zorder=3)

ax.set_ylabel('F1 @ ~20% FPR', fontsize=11)
ax.set_ylim(0.0, 1.0)
ax.set_xticks(x)
ax.set_xticklabels(SHORT, fontsize=9)
ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.tick_params(axis='y', labelsize=9)
ax.yaxis.grid(True, linestyle='--', alpha=0.35, linewidth=0.4, zorder=0)
ax.set_axisbelow(True)
ax.legend(fontsize=7.5, ncol=4, loc='upper center', bbox_to_anchor=(0.5, 1.22),
          handlelength=1.1, columnspacing=1.0, handletextpad=0.4)

fig.tight_layout()
os.makedirs(os.path.dirname(OUT_PDF), exist_ok=True)
fig.savefig(OUT_PDF, bbox_inches='tight')
fig.savefig(OUT_PNG, dpi=600, bbox_inches='tight')
print('saved', OUT_PDF, '+', OUT_PNG)

# --- print the plotted numbers so the figure can be audited against the JSON ---
for p, s in zip(PROTOCOLS, SHORT):
    row = '  '.join(f'{k}={agg[p][k]["f1_mean"]:.3f}' for k, _, _ in SERIES)
    print(f'{s.replace(chr(10), " "):<16} {row}')
