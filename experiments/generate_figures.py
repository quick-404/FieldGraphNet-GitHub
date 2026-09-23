"""
Nature-style figure: line chart (top) + bar chart (bottom).
Following nature-figure SKILL.md contract:

Core conclusion: Fusion system achieves highest or second-highest recall
on all 5 OOD protocols and ranks first in mean recall.
Archetype: quantitative grid (2-panel stacked)
Backend: Python (matplotlib)
Export: PDF, column-width (8 cm), editable text
"""
import json, numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# --- Nature-style rcParams ---
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

# --- Data ---
with open('eval_results/fusion_comparison/per_protocol_baselines.json') as f:
    data = json.load(f)

protocols = ['DNS_Spoofing', 'Mirai-udpplain', 'SqlInjection', 'DictionaryBruteForce', 'Recon-PortScan']
short_names = ['DNS\nSpoofing', 'Mirai', 'SQL\nInjection', 'Brute\nForce', 'Recon']

sklearn_methods = ['Random Forest', 'KNN', 'SVM', 'Naive Bayes', 'C4.5 (Tree)', 'AdaBoost']
# Nature NMI pastel palette for sklearn baselines
sklearn_colors = ['#7884B4', '#B4C0E4', '#E4CCD8', '#F0C0CC', '#E0E0F0', '#E0F0F0']

seq_methods = ['LSTM', 'GRU', 'Transformer']
seq_colors = ['#3E8E5A', '#7BB089', '#B7D4BE']

fusion_color = '#0F4D92'   # blue_main
gnn4id_color = '#606060'   # neutral_dark

x = np.arange(5)
gnn_recalls = [data[p]['GNN4ID (raw)']['recall'] for p in protocols]
fusion_recalls = [data[p]['FieldGraph-Net (Fusion)']['recall'] for p in protocols]

mean_recalls = {m: np.mean([data[p][m]['recall'] for p in protocols]) for m in sklearn_methods}
mean_recalls.update({m: np.mean([data[p][m]['recall'] for p in protocols]) for m in seq_methods})
mean_recalls['GNN4ID (raw)'] = np.mean(gnn_recalls)
mean_recalls['FieldGraph-Net (Fusion)'] = np.mean(fusion_recalls)

# --- Figure: single-column width, 3 panels (line / composition / mean recall) ---
fig, (ax_top, ax_mid, ax_bot) = plt.subplots(3, 1, figsize=(6.7, 7.6), dpi=300,
                                             gridspec_kw={'height_ratios': [3.0, 1.8, 2.2],
                                                          'hspace': 0.45})
fig.subplots_adjust(left=0.09, right=0.97, top=0.95, bottom=0.05)

# ===================== TOP: LINE CHART =====================
ax = ax_top
ax.set_xlim(-0.45, 4.45)
ax.set_ylim(-0.02, 1.08)
ax.set_ylabel('Recall', fontsize=14, labelpad=6)
ax.set_xticks(x)
ax.set_xticklabels(short_names, fontsize=12)
ax.tick_params(axis='both', labelsize=12)
ax.yaxis.grid(True, linestyle='--', alpha=0.3, linewidth=0.3)
ax.set_axisbelow(True)
# Panel label


for i, method in enumerate(sklearn_methods):
    recalls = [data[p][method]['recall'] for p in protocols]
    ax.plot(x, recalls, '-', color=sklearn_colors[i], linewidth=0.6,
            alpha=0.7, marker='o', markersize=3, label=method)

for i, method in enumerate(seq_methods):
    recalls = [data[p][method]['recall'] for p in protocols]
    ax.plot(x, recalls, '-', color=seq_colors[i], linewidth=0.7,
            alpha=0.85, marker='^', markersize=3, label=method)

ax.plot(x, gnn_recalls, '--', color=gnn4id_color, linewidth=0.8,
        marker='s', markersize=3.5, label='GNN4ID (raw)', zorder=4)
ax.plot(x, fusion_recalls, '-', color=fusion_color, linewidth=1.8,
        marker='D', markersize=5, label='FieldGraph-Net (Fusion)', zorder=5)

# Improvement annotations
for i in range(5):
    proto = protocols[i]
    fus_v = fusion_recalls[i]
    best_base = max(data[proto][m]['recall'] for m in sklearn_methods + seq_methods)
    imp = fus_v - best_base
    if imp > 0.01:
        ax.annotate(f'+{imp:.0%}', xy=(i, fus_v + 0.06),
                    fontsize=10, ha='center', color=fusion_color, fontweight='bold')
    elif fus_v > gnn_recalls[i] + 0.01:
        ax.annotate(f'+{(fus_v-gnn_recalls[i]):.0%}', xy=(i, fus_v + 0.06),
                    fontsize=10, ha='center', color=gnn4id_color)

# Legend: 3-column, compact, placed above the axes to avoid overlap
ax.legend(frameon=False, fontsize=8, loc='lower left', ncol=3,
           handletextpad=0.6, labelspacing=0.25, columnspacing=0.8,
           bbox_to_anchor=(0, 1.02), borderaxespad=0)

# ===================== MIDDLE: PROTOCOL COMPOSITION =====================
ax = ax_mid
comp = [('Benign', 500), ('DDoS', 500), ('DNS', 500), ('SQLi', 500),
        ('Brute', 500), ('Recon', 500), ('Mirai', 233)]
comp_in_domain = {'Benign', 'DDoS'}
comp_values = [c[1] for c in comp]
comp_colors = ['#0F4D92' if c[0] in comp_in_domain else '#B0BEC5' for c in comp]
bars = ax.bar([c[0] for c in comp], comp_values, color=comp_colors, width=0.6)
ax.set_ylabel('Flows', fontsize=10)
ax.set_ylim(0, 560)
ax.yaxis.grid(True, linestyle='--', alpha=0.3, linewidth=0.3)
ax.set_axisbelow(True)
for b, v in zip(bars, comp_values):
    ax.text(b.get_x() + b.get_width() / 2, v + 6, f'{v}', ha='center', fontsize=7)
ax.tick_params(axis='x', labelsize=8)
# compact legend placed above the panel to avoid overlapping the bars
from matplotlib.patches import Patch
ax.legend(handles=[
    Patch(color='#0F4D92', label='In-domain'),
    Patch(color='#B0BEC5', label='OOD protocol'),
], fontsize=7, frameon=False, ncol=2, loc='upper left',
    bbox_to_anchor=(0.0, 1.15), handletextpad=0.4, columnspacing=0.8,
    borderaxespad=0)

# ===================== BOTTOM: BAR CHART =====================
ax = ax_bot
ax.tick_params(axis='both', labelsize=12)


order = sorted(mean_recalls, key=mean_recalls.get)
values = [mean_recalls[m] for m in order]
colors_bar = [fusion_color if m == 'FieldGraph-Net (Fusion)' else
              gnn4id_color if m == 'GNN4ID (raw)' else '#D5D8DC'
              for m in order]

ax.barh(range(len(order)), values, color=colors_bar, height=0.55, linewidth=0)
ax.set_yticks(range(len(order)))
labels = [m.replace(' (raw)', '').replace(' (Tree)', '') for m in order]
ax.set_yticklabels(labels, fontsize=11)
ax.invert_yaxis()
ax.set_xlim(0, 0.72)
ax.set_xlabel('Mean Recall', fontsize=14, labelpad=6)
ax.xaxis.grid(True, linestyle='--', alpha=0.3, linewidth=0.3)
ax.set_axisbelow(True)

for i, v in enumerate(values):
    ax.text(v + 0.01, i, f'{v:.3f}', fontsize=10, va='center',
            color=fusion_color if v == max(values) else '#555555',
            fontweight='bold' if v == max(values) else 'normal')

# --- Export ---
plt.savefig('paper/figures/fig_ood_comparison.pdf', bbox_inches='tight')
plt.savefig('paper/figures/fig_ood_comparison.png', dpi=600, bbox_inches='tight')
print('Saved: paper/figures/fig_ood_comparison.pdf + .png (600 dpi)')


# ===================== SCALABILITY FIGURE =====================
def fig_scalability():
    with open('eval_results/fusion_comparison/scalability.json') as f:
        d = json.load(f)
    pts = d['points']
    flows = [p['flows'] for p in pts]
    times = [p['elapsed_seconds'] for p in pts]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.7, 2.6), dpi=300,
                                   layout='constrained')

    # Left: total time vs flows with linear reference
    ax = ax1
    ax.plot(flows, times, '-o', color=fusion_color, linewidth=1.6, markersize=4)
    # linear reference from the 50-flow point
    ref = [times[0] * f / flows[0] for f in flows]
    ax.plot(flows, ref, '--', color='#888888', linewidth=1.0,
            label='linear reference')
    ax.set_xlabel('Flows processed', fontsize=12)
    ax.set_ylabel('End-to-end time (s)', fontsize=12)
    ax.set_xlim(0, 550)
    ax.yaxis.grid(True, linestyle='--', alpha=0.3, linewidth=0.3)
    ax.set_axisbelow(True)
    ax.legend(fontsize=9, frameon=False)

    # Right: per-flow time vs flows
    ax = ax2
    per_flow = [t * 1000 / f for t, f in zip(times, flows)]
    ax.plot(flows, per_flow, '-s', color=gnn4id_color, linewidth=1.4, markersize=4)
    ax.set_xlabel('Flows processed', fontsize=12)
    ax.set_ylabel('Per-flow time (ms)', fontsize=12)
    ax.set_xlim(0, 550)
    ax.yaxis.grid(True, linestyle='--', alpha=0.3, linewidth=0.3)
    ax.set_axisbelow(True)

    plt.savefig('paper/figures/fig_scalability.pdf', bbox_inches='tight')
    plt.savefig('paper/figures/fig_scalability.png', dpi=600, bbox_inches='tight')
    print('Saved: paper/figures/fig_scalability.pdf + .png (600 dpi)')


# ===================== PROTOCOL COMPOSITION FIGURE =====================
def fig_protocol_composition():
    """Bar chart of CIC IoT 2023 flow distribution by source protocol."""
    comp = [
        ('Benign', 'BenignTraffic', 500),
        ('DDoS', 'DDoS-HTTP_Flood', 500),
        ('DNS', 'DNS_Spoofing', 500),
        ('SQLi', 'SqlInjection', 500),
        ('Brute', 'DictBruteForce', 500),
        ('Recon', 'Recon-PortScan', 500),
        ('Mirai', 'Mirai-UDP', 233),
    ]
    in_domain = {'Benign', 'DDoS'}
    labels = [c[0] for c in comp]
    values = [c[2] for c in comp]
    colors = ['#0F4D92' if c[0] in in_domain else '#B0BEC5' for c in comp]

    fig, ax = plt.subplots(figsize=(6.7, 2.8), dpi=300, layout='constrained')
    bars = ax.bar(labels, values, color=colors, width=0.6)
    ax.set_ylabel('Flows (CIC IoT 2023)', fontsize=12)
    ax.set_ylim(0, 560)
    ax.yaxis.grid(True, linestyle='--', alpha=0.3, linewidth=0.3)
    ax.set_axisbelow(True)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + 10, f'{v}',
                ha='center', fontsize=10)
    # legend: in-domain vs OOD
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color='#0F4D92', label='In-domain'),
        Patch(color='#B0BEC5', label='OOD protocol'),
    ], fontsize=9, frameon=False, loc='upper right')
    ax.annotate(f'OOD = {2233} flows (69.1\\%)',
                xy=(5.0, 470), fontsize=11, ha='center', color='#555555')

    plt.savefig('paper/figures/fig_protocol_composition.pdf', bbox_inches='tight')
    plt.savefig('paper/figures/fig_protocol_composition.png', dpi=600, bbox_inches='tight')
    print('Saved: paper/figures/fig_protocol_composition.pdf + .png (600 dpi)')


if __name__ == '__main__':
    fig_scalability()
    fig_protocol_composition()
