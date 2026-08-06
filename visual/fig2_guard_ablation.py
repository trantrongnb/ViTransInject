"""
fig2_guard_ablation.py — Guard ablation dual-axis chart, paper-quality figure.
Usage: python visual/fig2_guard_ablation.py
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import os

OUT_PNG = os.path.join(os.path.dirname(__file__), 'fig2_guard_ablation.png')
OUT_PDF = os.path.join(os.path.dirname(__file__), 'fig2_guard_ablation.pdf')

# ---------- Data ----------
x = np.array([0, 1, 2, 3])
config_labels = ['none', 'v1\n(vi)', 'v2\n(+nodiac)', 'v3\n(+en)']

llama_asr = np.array([8.08, 8.08, 7.08, 4.42])
qwen_asr  = np.array([7.33, 7.17, 4.67, 2.67])
llama_dr  = np.array([0.0,  0.0,  9.2,  24.4])
qwen_dr   = np.array([0.0,  0.0,  12.3, 21.6])

C_LLAMA = '#2563EB'
C_QWEN  = '#EF4444'

# ---------- Style ----------
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'Nimbus Roman', 'DejaVu Serif'],
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'axes.spines.top': False,
    'axes.linewidth': 1.5,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'xtick.major.size': 4,
    'ytick.major.size': 4,
    'xtick.major.width': 1.5,
    'ytick.major.width': 1.5,
    'figure.dpi': 100,
})

fig, ax1 = plt.subplots(figsize=(8.5, 4.8))
ax2 = ax1.twinx()

# Shade the "+EN" step
ax1.axvspan(2.45, 3.55, alpha=0.06, color='#F59E0B', zorder=0)

# ---- ASR lines (left axis, solid) ----
ms = 8
ax1.plot(x, llama_asr, '-o', color=C_LLAMA, markersize=ms, linewidth=2.2,
         zorder=4, label='Llama-1B · ASR', clip_on=False)
ax1.plot(x, qwen_asr,  '-o', color=C_QWEN,  markersize=ms, linewidth=2.2,
         zorder=4, label='Qwen-1.5B · ASR', clip_on=False)

# ---- DR lines (right axis, dashed + open square) ----
kw_dr = dict(markersize=ms, linewidth=2.2, zorder=4,
             markerfacecolor='white', markeredgewidth=2.0, clip_on=False)
ax2.plot(x, llama_dr, '--s', color=C_LLAMA, **kw_dr, label='Llama-1B · DR')
ax2.plot(x, qwen_dr,  '--s', color=C_QWEN,  **kw_dr, label='Qwen-1.5B · DR')

# ---- Left Y (ASR) ----
ax1.set_ylim(0, 9.8)
ax1.set_yticks([0, 2, 4, 6, 8])
ax1.set_yticklabels(['0%', '2%', '4%', '6%', '8%'])
ax1.set_ylabel('ASR  (↓ better)', fontsize=10, color='#374151', labelpad=6)
ax1.yaxis.grid(True, linestyle=':', linewidth=0.6, color='#E5E7EB', zorder=0)
ax1.set_axisbelow(True)

# ---- Right Y (DR): linear percentage scale ----
ax2.set_ylim(0, 32)
ax2.set_yticks([0, 10, 20, 30])
ax2.set_yticklabels(['0%', '10%', '20%', '30%'])
ax2.set_ylabel('DR  (↑ better)', fontsize=10, color='#374151', rotation=270, labelpad=14)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_linewidth(1.5)
ax2.spines['right'].set_color('#374151')
ax2.tick_params(axis='y', colors='#111827')
# ---- X axis ----
ax1.set_xticks(x)
ax1.set_xticklabels(config_labels, fontsize=9.5, fontfamily='monospace')
ax1.set_xlim(-0.35, 3.45)
ax1.spines['bottom'].set_color('#374151')
ax1.spines['left'].set_color('#374151')


# ---- Legend ----
handles = [
    mlines.Line2D([], [], color=C_LLAMA, marker='o', linestyle='-',
                  markersize=7, linewidth=2, label='Llama-1B  ASR'),
    mlines.Line2D([], [], color=C_QWEN,  marker='o', linestyle='-',
                  markersize=7, linewidth=2, label='Qwen-1.5B  ASR'),
    mlines.Line2D([], [], color=C_LLAMA, marker='s', linestyle='--',
                  markersize=7, linewidth=2, markerfacecolor='white',
                  markeredgewidth=1.8, label='Llama-1B  DR'),
    mlines.Line2D([], [], color=C_QWEN,  marker='s', linestyle='--',
                  markersize=7, linewidth=2, markerfacecolor='white',
                  markeredgewidth=1.8, label='Qwen-1.5B  DR'),
]
ax1.legend(handles=handles, fontsize=8.5, loc='upper left', framealpha=0.95,
           edgecolor='#E5E7EB', ncol=2, handlelength=1.8)

plt.tight_layout(pad=1.2)
plt.savefig(OUT_PDF, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.savefig(OUT_PNG, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
print(f"Saved: {OUT_PDF}")
print(f"Saved: {OUT_PNG}")
