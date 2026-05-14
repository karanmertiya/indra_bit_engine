"""
Generate all performance visualizations for the Indra-Bit paper/README.
Run from the indra_bit_engine directory.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Color palette ─────────────────────────────────────────────
BG     = "#0d1117"
CARD   = "#161b22"
GREEN  = "#39d353"
BLUE   = "#58a6ff"
ORANGE = "#f0883e"
GRAY   = "#8b949e"
WHITE  = "#e6edf3"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": CARD,
    "axes.edgecolor": GRAY, "axes.labelcolor": WHITE,
    "xtick.color": WHITE, "ytick.color": WHITE,
    "text.color": WHITE, "grid.color": "#30363d",
    "font.family": "DejaVu Sans", "font.size": 11,
})

# ═══════════════════════════════════════════════════════════════
# FIGURE 1 — Scalability: Speedup vs Matrix Size
# ═══════════════════════════════════════════════════════════════
sizes   = [512, 1024, 2048, 4096]
fp32_ms = [5.82, 24.30, 100.30, 412.06]
apot_ms = [1.49,  5.63,  26.99, 124.23]
speedup = [f/a for f,a in zip(fp32_ms, apot_ms)]

fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=BG)
fig.suptitle("Indra-Bit APoT vs FP32 — AMD Ryzen 5 5600H · Single Thread · -O3 -mavx2",
             color=WHITE, fontsize=13, y=1.02)

# Left: tok/s comparison
ax = axes[0]
x   = np.arange(len(sizes))
w   = 0.35
fp32_tps = [1000/m for m in fp32_ms]
apot_tps = [1000/m for m in apot_ms]
bars1 = ax.bar(x - w/2, fp32_tps, w, label="FP32 Baseline",  color=BLUE,   alpha=0.85)
bars2 = ax.bar(x + w/2, apot_tps, w, label="Indra-Bit APoT", color=GREEN,  alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels([f"{s}x{s}" for s in sizes])
ax.set_xlabel("Matrix Size (square)")
ax.set_ylabel("Tokens / Second")
ax.set_title("Throughput Comparison", color=WHITE)
ax.legend(facecolor=CARD, edgecolor=GRAY, labelcolor=WHITE)
ax.grid(axis='y', alpha=0.3)
for bar in bars2:
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
            f"{bar.get_height():.0f}", ha='center', va='bottom',
            color=GREEN, fontsize=9, fontweight='bold')

# Right: Speedup line
ax = axes[1]
ax.plot(sizes, speedup, color=ORANGE, linewidth=2.5, marker='o', markersize=8)
ax.fill_between(sizes, speedup, alpha=0.15, color=ORANGE)
ax.axhline(1.0, color=GRAY, linestyle='--', alpha=0.5, label="Baseline (1x)")
for x_, y_ in zip(sizes, speedup):
    ax.annotate(f"{y_:.2f}x", (x_, y_), textcoords="offset points",
                xytext=(0, 10), ha='center', color=ORANGE, fontweight='bold')
ax.set_xlabel("Matrix Size (N×N)")
ax.set_ylabel("Speedup over FP32")
ax.set_title("Consistent 3.3x–4.3x Speedup Across All Scales", color=WHITE)
ax.set_xscale('log', base=2)
ax.legend(facecolor=CARD, edgecolor=GRAY, labelcolor=WHITE)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("plots/fig1_scalability.png", dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print("[SAVED] plots/fig1_scalability.png")

# ═══════════════════════════════════════════════════════════════
# FIGURE 2 — MACs Eliminated
# ═══════════════════════════════════════════════════════════════
macs = [s*s*32 for s in sizes]  # D*D*N_LAYERS
labels = [f"{s}x{s}" for s in sizes]

fig, ax = plt.subplots(figsize=(8, 5), facecolor=BG)
colors = [BLUE, ORANGE, GREEN, "#b392f0"]
bars = ax.bar(labels, [m/1e6 for m in macs], color=colors, alpha=0.85, edgecolor=GRAY)
for bar, m in zip(bars, macs):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
            f"{m/1e6:.0f}M", ha='center', va='bottom', color=WHITE, fontweight='bold')
ax.set_xlabel("Matrix Size")
ax.set_ylabel("Float Multiplications Eliminated (Millions)")
ax.set_title("Floating-Point MACs Eliminated per Token (all -> 0 in Indra-Bit)", color=WHITE)
ax.grid(axis='y', alpha=0.3)
fig.text(0.5, -0.04, "Indra-Bit replaces ALL of these with integer bit-shifts. Zero FP MACs.",
         ha='center', color=ORANGE, fontsize=10, style='italic')
plt.tight_layout()
plt.savefig("plots/fig2_macs_eliminated.png", dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print("[SAVED] plots/fig2_macs_eliminated.png")

# ═══════════════════════════════════════════════════════════════
# FIGURE 3 — APoT Architecture Diagram
# ═══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor=BG)

# Left — Standard FP32 path
ax = axes[0]
ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis('off')
ax.set_title("Standard FP32 Forward Pass", color=BLUE, fontsize=13)
boxes = [
    (5, 8.5, "Input x  (float32)", BLUE),
    (5, 6.5, "Weight W (float32)", BLUE),
    (5, 4.5, "x * W  ← FLOAT MULTIPLY\n(cuBLAS / BLAS)", "#f85149"),
    (5, 2.5, "+ Bias  →  Output", BLUE),
]
for (bx, by, txt, col) in boxes:
    ax.add_patch(mpatches.FancyBboxPatch((bx-3, by-0.7), 6, 1.2,
        boxstyle="round,pad=0.1", facecolor=CARD, edgecolor=col, linewidth=2))
    ax.text(bx, by, txt, ha='center', va='center', color=col, fontsize=9, fontweight='bold')
ax.annotate("", xy=(5, 5.2), xytext=(5, 5.8), arrowprops=dict(arrowstyle="->", color=GRAY))
ax.annotate("", xy=(5, 3.2), xytext=(5, 3.8), arrowprops=dict(arrowstyle="->", color=GRAY))
ax.text(5, 1.2, "COST: 536M float multiplications per token", ha='center', color="#f85149",
        fontsize=9, style='italic')

# Right — Indra-Bit APoT path
ax = axes[1]
ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis('off')
ax.set_title("Indra-Bit APoT Forward Pass", color=GREEN, fontsize=13)
boxes2 = [
    (5, 9.0, "Input x  (int8, quantized)", GREEN),
    (5, 7.2, "Weights stored as exponents: k1 k2 k3 k4", GREEN),
    (5, 5.4, "x>>k1 + x>>k2 + x>>k3 + x>>k4", GREEN),
    (5, 3.6, "Apply sign ×  (+1 or -1)", ORANGE),
    (5, 1.8, "+ Bias  →  Output  (dequantize)", GREEN),
]
for (bx, by, txt, col) in boxes2:
    ax.add_patch(mpatches.FancyBboxPatch((bx-3.8, by-0.65), 7.6, 1.2,
        boxstyle="round,pad=0.1", facecolor=CARD, edgecolor=col, linewidth=2))
    ax.text(bx, by, txt, ha='center', va='center', color=col, fontsize=9, fontweight='bold')
for y in [6.3, 4.7, 3.0]:
    ax.annotate("", xy=(5, y), xytext=(5, y+0.5), arrowprops=dict(arrowstyle="->", color=GRAY))
ax.text(5, 0.5, "COST: 0 float multiplications. Pure bit-shifts.", ha='center',
        color=GREEN, fontsize=9, style='italic', fontweight='bold')

plt.tight_layout()
plt.savefig("plots/fig3_architecture.png", dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print("[SAVED] plots/fig3_architecture.png")

# ═══════════════════════════════════════════════════════════════
# FIGURE 4 — Summary Dashboard
# ═══════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(14, 7), facecolor=BG)
fig.suptitle("Indra-Bit APoT — Results Summary", color=WHITE, fontsize=15, fontweight='bold')

# Stats grid
stats = [
    ("3.3x – 4.3x",   "Consistent CPU Speedup",       GREEN),
    ("0",              "Float Multiplications",         ORANGE),
    ("536M → 0",       "MACs per Token (4096 scale)",  BLUE),
    ("0.009",          "LLM Loss Degradation",          GREEN),
    ("4-bit equiv.",   "Weight Compression",            ORANGE),
    ("Ryzen 5 5600H",  "No GPU Required",               BLUE),
]
for i, (val, label, col) in enumerate(stats):
    ax = fig.add_subplot(2, 3, i+1)
    ax.set_facecolor(CARD)
    ax.axis('off')
    ax.text(0.5, 0.6, val, ha='center', va='center', transform=ax.transAxes,
            color=col, fontsize=22, fontweight='bold')
    ax.text(0.5, 0.2, label, ha='center', va='center', transform=ax.transAxes,
            color=GRAY, fontsize=10)
    for spine in ax.spines.values():
        spine.set_edgecolor(col)
        spine.set_linewidth(2)
        spine.set_visible(True)

plt.tight_layout()
plt.savefig("plots/fig4_summary.png", dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print("[SAVED] plots/fig4_summary.png")
print("\nAll 4 figures generated successfully.")
