import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import re
import glob
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

# Style settings
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 6)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14

# Extract results from logs
def extract_from_log(log_file):
    """Extract final MSE/MAE from log"""
    with open(log_file, 'r') as f:
        content = f.read()
        matches = re.findall(r'mse:([\d.]+),?\s*mae:([\d.]+)', content)
        if matches:
            mse, mae = matches[-1]
            return float(mse), float(mae)
    return None, None

# Find all logs
all_logs = glob.glob('logs/*.log')
results = []

for log in all_logs:
    name = os.path.basename(log).replace('.log', '')
    mse, mae = extract_from_log(log)

    if mse is not None:
        if 'dlinear' in name.lower():
            model = 'DLinear'
        elif 'hybrid' in name.lower():
            model = 'FreqHybrid'
        else:
            continue

        if 'weather' in name.lower():
            dataset = 'Weather'
        elif 'ecl' in name.lower() or 'electricity' in name.lower():
            dataset = 'Electricity'
        else:
            continue

        if '96' in name and '336' not in name and '720' not in name:
            horizon = 96
        elif '336' in name:
            horizon = 336
        elif '720' in name:
            horizon = 720
        else:
            continue

        results.append({
            'Model': model,
            'Dataset': dataset,
            'Horizon': horizon,
            'MSE': mse,
            'MAE': mae
        })

df = pd.DataFrame(results)
os.makedirs('analysis', exist_ok=True)
os.makedirs('results', exist_ok=True)

# Improvement heatmap
improvements = []
for dataset in df['Dataset'].unique():
    for horizon in sorted(df['Horizon'].unique()):
        dl_row = df[(df['Dataset'] == dataset) &
                   (df['Horizon'] == horizon) &
                   (df['Model'] == 'DLinear')]
        hy_row = df[(df['Dataset'] == dataset) &
                   (df['Horizon'] == horizon) &
                   (df['Model'] == 'FreqHybrid')]

        if len(dl_row) > 0 and len(hy_row) > 0:
            dl_mse = dl_row['MSE'].values[0]
            hy_mse = hy_row['MSE'].values[0]
            improvement = ((dl_mse - hy_mse) / dl_mse * 100)

            improvements.append({
                'Dataset': dataset,
                'Horizon': f'{horizon}h',
                'Improvement': improvement
            })

if improvements:
    imp_df = pd.DataFrame(improvements)
    pivot_imp = imp_df.pivot(index='Dataset', columns='Horizon', values='Improvement')

    fig, ax = plt.subplots(figsize=(12, 6))

    # Custom colormap: red (negative) to green (positive)
    sns.heatmap(pivot_imp, annot=True, fmt='.1f',
               cmap='RdYlGn', center=0, vmin=-100, vmax=20,
               cbar_kws={'label': 'Improvement over DLinear (%)', 'shrink': 0.8},
               ax=ax, linewidths=2, linecolor='white',
               annot_kws={'fontsize': 14, 'fontweight': 'bold'})

    ax.set_title('FreqHybrid Performance vs DLinear\n(Positive = FreqHybrid Better)',
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Forecast Horizon', fontsize=13, fontweight='bold')
    ax.set_ylabel('Dataset', fontsize=13, fontweight='bold')

    # Add interpretation text
    fig.text(0.5, -0.05,
            'Red = DLinear wins  |  Green = FreqHybrid wins',
            ha='center', fontsize=11, style='italic')

    plt.tight_layout()
    plt.savefig('analysis/improvement_heatmap.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

# Side-by-side bar graph comparison
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

datasets = df['Dataset'].unique()
horizons = sorted(df['Horizon'].unique())

for idx, dataset in enumerate(datasets):
    ax = axes[idx]

    dataset_df = df[df['Dataset'] == dataset]

    x = np.arange(len(horizons))
    width = 0.35

    dlinear_mses = []
    hybrid_mses = []

    for h in horizons:
        dl = dataset_df[(dataset_df['Horizon'] == h) & (dataset_df['Model'] == 'DLinear')]
        hy = dataset_df[(dataset_df['Horizon'] == h) & (dataset_df['Model'] == 'FreqHybrid')]

        dlinear_mses.append(dl['MSE'].values[0] if len(dl) > 0 else 0)
        hybrid_mses.append(hy['MSE'].values[0] if len(hy) > 0 else 0)

    bars1 = ax.bar(x - width/2, dlinear_mses, width, label='DLinear',
                  color='#2E86AB', alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, hybrid_mses, width, label='FreqHybrid',
                  color='#A23B72', alpha=0.8, edgecolor='black', linewidth=1.5)

    ax.set_xlabel('Forecast Horizon', fontsize=13, fontweight='bold')
    ax.set_ylabel('MSE (Lower is Better)', fontsize=13, fontweight='bold')
    ax.set_title(f'{dataset} Dataset', fontsize=15, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{h}h' for h in horizons])
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_facecolor('#F8F9FA')

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('analysis/bar_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

# Line plot showing MSE vs. Horizon
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for idx, dataset in enumerate(df['Dataset'].unique()):
    ax = axes[idx]

    dataset_df = df[df['Dataset'] == dataset].sort_values('Horizon')

    for model in ['DLinear', 'FreqHybrid']:
        model_df = dataset_df[dataset_df['Model'] == model]

        if len(model_df) > 0:
            color = '#2E86AB' if model == 'DLinear' else '#A23B72'
            marker = 'o' if model == 'DLinear' else 's'

            ax.plot(model_df['Horizon'], model_df['MSE'],
                   marker=marker, linestyle='-', linewidth=3, markersize=12,
                   color=color, label=model, alpha=0.8)

    ax.set_xlabel('Forecast Horizon (hours)', fontsize=13, fontweight='bold')
    ax.set_ylabel('MSE', fontsize=13, fontweight='bold')
    ax.set_title(f'{dataset} - Performance vs Horizon', fontsize=15, fontweight='bold')
    ax.legend(fontsize=12, loc='best')
    ax.grid(True, alpha=0.4, linestyle='--')
    ax.set_facecolor('#F8F9FA')

    # Highlight winner at each point
    for h in dataset_df['Horizon'].unique():
        dl_mse = dataset_df[(dataset_df['Horizon'] == h) & (dataset_df['Model'] == 'DLinear')]['MSE']
        hy_mse = dataset_df[(dataset_df['Horizon'] == h) & (dataset_df['Model'] == 'FreqHybrid')]['MSE']

        if len(dl_mse) > 0 and len(hy_mse) > 0:
            if dl_mse.values[0] < hy_mse.values[0]:
                # DLinear wins - mark with star
                ax.scatter([h], [dl_mse.values[0]], s=200, marker='*',
                          color='gold', edgecolors='black', linewidths=2, zorder=10)

plt.tight_layout()
plt.savefig('analysis/line_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
