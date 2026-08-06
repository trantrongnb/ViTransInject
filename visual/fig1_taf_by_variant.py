"""
fig1_taf_by_variant.py — TAF grouped bar chart (log Y), paper-quality figure.
Usage: python visual/fig1_taf_by_variant.py
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import os

OUT_PNG = os.path.join(os.path.dirname(__file__), 'fig1_taf_by_variant.png')
OUT_PDF = os.path.join(os.path.dirname(__file__), 'fig1_taf_by_variant.pdf')

# ---------- Data ----------
variants   = ['mt_en', 'bilingual', 'vi_nodiacritic', 'mixed', 'teencode']
models     = ['Llama-1B', 'Qwen-1.5B', 'Gemma-2B', 'Qwen-3B', 'Sailor2-1B']
colors     = ['#2563EB', '#EF4444', '#10B981', '#F59E0B', '#8B5CF6']
hatches    = ['', '', '', '', '']

# taf[model_idx, variant_idx]
taf = np.array([
    [1.58, 2.33, 1.25, 0.92, 0.67],  # Llama-1B
    [1.62, 1.08, 1.15, 0.92, 0.92],  # Qwen-1.5B
    [1.23, 1.23, 1.15, 1.15, 0.85],  # Gemma-2B
    [1.00, 0.60, 0.95, 0.90, 1.00],  # Qwen-3B
    [1.23, 1.54, 1.00, 1.31, 0.69],  # Sailor2-1B
])

# ---------- Style ----------
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'Nimbus Roman', 'DejaVu Serif'],
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.8,
    'xtick.labelsize': 9,
    'ytick.labelsize': 8.5,
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'figure.dpi': 100,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.facecolor': 'white',
})

fig, ax = plt.subplots(figsize=(10.4, 5.0), facecolor='white')
ax.set_facecolor('white')

n_v = len(variants)
n_m = len(models)
x   = np.arange(n_v)

bar_w     = 0.135
group_w   = n_m * bar_w + (n_m - 1) * 0.015
offsets   = np.linspace(-(group_w - bar_w) / 2, (group_w - bar_w) / 2, n_m)

for mi, (model, color, hatch, off) in enumerate(zip(models, colors, hatches, offsets)):
    vals = taf[mi]
    bars = ax.bar(x + off, vals, bar_w, color=color, alpha=0.90,
                  label=model, zorder=3, edgecolor='#111827',
                  linewidth=0.45, hatch=hatch)
    # Value labels on top of each bar
    for bar, val in zip(bars, vals):
        xc = bar.get_x() + bar.get_width() / 2
        if val >= 1.0:
            ax.text(xc, val * 1.04, f'{val:.2f}',
                    ha='center', va='bottom', fontsize=7.5,
                    color='#111827', fontweight='bold', zorder=5)
        else:
            ax.text(xc, val * 0.96, f'{val:.2f}',
                    ha='center', va='top', fontsize=7.5,
                    color='#111827', fontweight='bold', zorder=5)

# Reference line at TAF = 1.0
ax.axhline(1.0, color='#6B7280', linewidth=1.0, linestyle=(0, (5, 4)), zorder=2, alpha=0.8)
ax.text(n_v - 0.42, 1.035, 'TAF = 1', fontsize=8, color='#9CA3AF', ha='right', va='bottom')


# Y axis — log scale
ax.set_yscale('log', base=10)
ax.set_ylim(0.52, 2.7)
y_ticks = [0.6, 0.7, 0.8, 1.0, 1.25, 1.5, 2.0, 2.5]
ax.yaxis.set_major_locator(mticker.FixedLocator(y_ticks))
ax.yaxis.set_major_formatter(mticker.FixedFormatter(['0.6', '0.7', '0.8', '1.0', '1.25', '1.5', '2.0', '2.5']))
ax.yaxis.set_minor_locator(mticker.NullLocator())
ax.set_ylabel('TAF  (log scale)', fontsize=10, labelpad=6, color='#374151')

# X axis
ax.set_xticks(x)
ax.set_xticklabels(variants, fontsize=9.5, fontfamily='monospace')
ax.set_xlim(-0.55, n_v - 0.45)

# Keep the plotting area plain white for paper reproduction.
ax.grid(False)
ax.set_axisbelow(True)

# Legend
legend = ax.legend(fontsize=8.6, loc='upper right', framealpha=0.95,
                   edgecolor='#E5E7EB', ncol=3, handlelength=1.2,
                   handletextpad=0.5, columnspacing=1.0)



plt.tight_layout(pad=1.2)
assert ax.get_yscale() == 'log'
plt.savefig(OUT_PDF, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.savefig(OUT_PNG, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
print(f"Saved: {OUT_PDF}")
print(f"Saved: {OUT_PNG}")
