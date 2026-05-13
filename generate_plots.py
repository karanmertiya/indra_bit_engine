import matplotlib.pyplot as plt
import numpy as np
import os

# Data
models = ['FP32 ResNet-110\n(Standard)', 'Indra-Bit 1.7M\n(Multiplier-Free)']
accuracy = [93.5, 91.10]
memory = [6.8, 0.85] # MB
inference_time = [1200, 54.6] # ms on CPU (estimated vs actual WASM)

# Set up the style
plt.style.use('dark_background')
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
fig.patch.set_facecolor('#0a0a0f')

colors = ['#ff3366', '#00ffcc']

# Accuracy Bar
bars1 = ax1.bar(models, accuracy, color=colors)
ax1.set_title('CIFAR-10 Accuracy (%)', color='white', pad=20, fontsize=14)
ax1.set_ylim(0, 100)
for bar in bars1:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, yval + 1, f"{yval}%", ha='center', va='bottom', color='white', fontweight='bold')

# Memory Bar
bars2 = ax2.bar(models, memory, color=colors)
ax2.set_title('Physical RAM Usage (MB)', color='white', pad=20, fontsize=14)
for bar in bars2:
    yval = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2, yval + 0.2, f"{yval} MB", ha='center', va='bottom', color='white', fontweight='bold')

# Inference Speed Bar
bars3 = ax3.bar(models, inference_time, color=colors)
ax3.set_title('CPU Browser Inference (ms)', color='white', pad=20, fontsize=14)
for bar in bars3:
    yval = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2, yval + 20, f"{yval} ms", ha='center', va='bottom', color='white', fontweight='bold')

# Cleanup aesthetics
for ax in [ax1, ax2, ax3]:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#333')
    ax.spines['bottom'].set_color('#333')
    ax.tick_params(colors='white')
    ax.set_facecolor('#0a0a0f')

plt.tight_layout()
plt.savefig('C:/Users/ksmer/PlacementPrep/vibe-coded certificate/indra_bit_engine/performance_comparison.png', dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
print("Saved visualization!")
