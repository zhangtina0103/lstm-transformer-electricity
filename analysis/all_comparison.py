import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Manually entered data
dlinear_epochs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
dlinear_train = [0.3716, 0.2308, 0.2152, 0.2103, 0.2081, 0.2071, 0.2067, 0.2064, 0.2063, 0.2063, 0.2062, 0.2062, 0.2062, 0.2062, 0.2062]
dlinear_val = [0.2247, 0.1972, 0.1901, 0.1874, 0.1860, 0.1854, 0.1852, 0.1850, 0.1850, 0.1849, 0.1849, 0.1848, 0.1849, 0.1848, 0.1849]
dlinear_final_mse = 0.2065
dlinear_final_mae = 0.2948

hybrid_epochs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
hybrid_train = [0.4362, 0.3160, 0.2540, 0.2276, 0.2159, 0.2103, 0.2076, 0.2062, 0.2054, 0.2051, 0.2049, 0.2048, 0.2046, 0.2047, 0.2047]
hybrid_val = [0.3470, 0.2812, 0.2527, 0.2421, 0.2387, 0.2354, 0.2347, 0.2342, 0.2337, 0.2337, 0.2336, 0.2336, 0.2336, 0.2336, 0.2335]
hybrid_final_mse = 0.2881
hybrid_final_mae = 0.3694

# Training curves
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Training loss
axes[0].plot(dlinear_epochs, dlinear_train, 'o-', color='#2E86AB',
            label='DLinear', linewidth=2.5, markersize=7, alpha=0.8)
axes[0].plot(hybrid_epochs, hybrid_train, 's-', color='#A23B72',
            label='FreqHybrid', linewidth=2.5, markersize=7, alpha=0.8)
axes[0].set_xlabel('Epoch', fontsize=13, fontweight='bold')
axes[0].set_ylabel('Training Loss (MSE)', fontsize=13, fontweight='bold')
axes[0].set_title('Training Loss', fontsize=15, fontweight='bold', pad=15)
axes[0].legend(fontsize=12, framealpha=0.9)
axes[0].grid(True, alpha=0.4, linestyle='--')
axes[0].set_facecolor('#F8F9FA')

# Validation loss
axes[1].plot(dlinear_epochs, dlinear_val, 'o-', color='#2E86AB',
            label='DLinear', linewidth=2.5, markersize=7, alpha=0.8)
axes[1].plot(hybrid_epochs, hybrid_val, 's-', color='#A23B72',
            label='FreqHybrid', linewidth=2.5, markersize=7, alpha=0.8)
axes[1].set_xlabel('Epoch', fontsize=13, fontweight='bold')
axes[1].set_ylabel('Validation Loss (MSE)', fontsize=13, fontweight='bold')
axes[1].set_title('Validation Loss', fontsize=15, fontweight='bold', pad=15)
axes[1].legend(fontsize=12, framealpha=0.9)
axes[1].grid(True, alpha=0.4, linestyle='--')
axes[1].set_facecolor('#F8F9FA')

# Add final values
dl_final = min(dlinear_val)
hy_final = min(hybrid_val)

axes[1].axhline(y=dl_final, color='#2E86AB', linestyle='--', alpha=0.6, linewidth=1.5)
axes[1].axhline(y=hy_final, color='#A23B72', linestyle='--', alpha=0.6, linewidth=1.5)

# Add text annotations
axes[1].text(0.98, 0.98,
            f'DLinear: {dl_final:.4f}\nFreqHybrid: {hy_final:.4f}\nΔ: {hy_final-dl_final:+.4f}',
            transform=axes[1].transAxes, fontsize=11,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))

plt.tight_layout()
os.makedirs('analysis', exist_ok=True)
plt.savefig('analysis/training_curves.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

# Results comparison
models = ['DLinear', 'FreqHybrid']
mse_values = [dlinear_final_mse, hybrid_final_mse]
mae_values = [dlinear_final_mae, hybrid_final_mae]

x = np.arange(len(models))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))

bars1 = ax.bar(x - width/2, mse_values, width, label='MSE',
              color='#2E86AB', alpha=0.8, edgecolor='black', linewidth=1.5)
bars2 = ax.bar(x + width/2, mae_values, width, label='MAE',
              color='#A23B72', alpha=0.8, edgecolor='black', linewidth=1.5)

ax.set_xlabel('Model', fontsize=14, fontweight='bold')
ax.set_ylabel('Error', fontsize=14, fontweight='bold')
ax.set_title('Model Performance Comparison (Lower is Better)',
            fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=13)
ax.legend(fontsize=12, framealpha=0.9)
ax.grid(True, alpha=0.3, axis='y', linestyle='--')
ax.set_facecolor('#F8F9FA')

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.4f}',
               ha='center', va='bottom', fontsize=11, fontweight='bold')

# Add improvement annotation
improvement = ((hybrid_final_mse - dlinear_final_mse) / dlinear_final_mse) * 100
ax.text(0.5, max(mse_values + mae_values) * 0.95,
       f'FreqHybrid: {improvement:+.1f}% vs DLinear',
       ha='center', fontsize=12, fontweight='bold',
       bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

plt.tight_layout()
plt.savefig('analysis/results_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

# Convergence comparison
fig, ax = plt.subplots(figsize=(12, 6))

# Plot both validation curves on same plot
ax.plot(dlinear_epochs, dlinear_val, 'o-', color='#2E86AB',
        label='DLinear', linewidth=3, markersize=8, alpha=0.8)
ax.plot(hybrid_epochs, hybrid_val, 's-', color='#A23B72',
        label='FreqHybrid', linewidth=3, markersize=8, alpha=0.8)

ax.set_xlabel('Epoch', fontsize=14, fontweight='bold')
ax.set_ylabel('Validation Loss (MSE)', fontsize=14, fontweight='bold')
ax.set_title('Convergence Comparison: Validation Loss Over Time',
            fontsize=16, fontweight='bold', pad=20)
ax.legend(fontsize=13, framealpha=0.9, loc='upper right')
ax.grid(True, alpha=0.4, linestyle='--')
ax.set_facecolor('#F8F9FA')

# Annotate key points
ax.annotate('DLinear converges\nfaster and lower',
            xy=(12, dlinear_val[11]), xytext=(10, 0.25),
            fontsize=11, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#2E86AB', lw=2),
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

ax.annotate('FreqHybrid\nplateau higher',
            xy=(15, hybrid_val[14]), xytext=(13, 0.27),
            fontsize=11, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#A23B72', lw=2),
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

plt.tight_layout()
plt.savefig('analysis/convergence_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
