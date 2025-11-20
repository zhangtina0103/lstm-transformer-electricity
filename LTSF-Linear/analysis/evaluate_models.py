"""
Deep Analysis Script for FreqHybrid vs Baselines
Implements all 6 analysis methods
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from scipy import signal
import re
import os
import glob

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 11

class ModelAnalyzer:
    def __init__(self, results_dir='./results', logs_dir='./logs'):
        self.results_dir = results_dir
        self.logs_dir = logs_dir

    # =========================================================================
    # 1. STANDARD METRICS (for tables)
    # =========================================================================
    def extract_results_table(self):
        """Parse logs and create comparison table"""
        print("\n" + "="*70)
        print("QUANTITATIVE RESULTS (Electricity Dataset, Horizon=96)")
        print("="*70)

        results = {}

        # Find all log files
        log_files = glob.glob(f'{self.logs_dir}/*.log')

        for log_file in log_files:
            model_name = os.path.basename(log_file).replace('.log', '')

            with open(log_file, 'r') as f:
                content = f.read()

                # Extract MSE and MAE
                mse_match = re.search(r'mse:([\d.]+)', content)
                mae_match = re.search(r'mae:([\d.]+)', content)

                if mse_match and mae_match:
                    results[model_name] = {
                        'MSE': float(mse_match.group(1)),
                        'MAE': float(mae_match.group(1))
                    }

        # Create DataFrame
        df = pd.DataFrame(results).T
        df = df.sort_values('MSE')

        print(df.to_string())
        print("="*70)

        # Calculate improvements
        if 'dlinear_ecl_96' in results and 'freqhybrid_ecl_96' in results:
            baseline_mse = results['dlinear_ecl_96']['MSE']
            hybrid_mse = results['freqhybrid_ecl_96']['MSE']
            improvement = ((baseline_mse - hybrid_mse) / baseline_mse) * 100

            print(f"\n📊 FreqHybrid vs DLinear:")
            print(f"   MSE Improvement: {improvement:+.2f}%")

            if improvement > 0:
                print(f"   ✅ FreqHybrid BEATS DLinear by {improvement:.2f}%")
            else:
                print(f"   ❌ FreqHybrid is {abs(improvement):.2f}% worse than DLinear")

        # Save table
        df.to_csv('results/comparison_table.csv')
        print(f"\n✅ Saved results/comparison_table.csv")

        return df

    # =========================================================================
    # 2. FREQUENCY DOMAIN ANALYSIS
    # =========================================================================
    def plot_frequency_comparison(self, save_path='analysis/frequency_comparison.png'):
        """Compare predictions in frequency domain"""
        print("\n🔬 Analyzing frequency domain...")

        # Load predictions
        dlinear_pred = self._load_predictions('DLinear')
        hybrid_pred = self._load_predictions('FreqHybrid')

        if dlinear_pred is None or hybrid_pred is None:
            print("⚠️  Prediction files not found. Run inference first.")
            return

        # Load ground truth (from test set)
        # Assuming predictions are saved with matching true values

        # Take first 5 samples for visualization
        n_samples = 5

        fig, axes = plt.subplots(n_samples, 2, figsize=(14, 3*n_samples))

        for i in range(n_samples):
            # Time domain
            axes[i, 0].plot(dlinear_pred[i, :, 0], label='DLinear', alpha=0.7)
            axes[i, 0].plot(hybrid_pred[i, :, 0], label='FreqHybrid', alpha=0.7)
            axes[i, 0].set_title(f'Sample {i+1}: Time Domain')
            axes[i, 0].legend()
            axes[i, 0].grid(True, alpha=0.3)

            # Frequency domain
            d_fft = np.abs(np.fft.rfft(dlinear_pred[i, :, 0]))
            h_fft = np.abs(np.fft.rfft(hybrid_pred[i, :, 0]))

            axes[i, 1].plot(d_fft, label='DLinear', alpha=0.7)
            axes[i, 1].plot(h_fft, label='FreqHybrid', alpha=0.7)
            axes[i, 1].set_title(f'Sample {i+1}: Frequency Domain')
            axes[i, 1].set_xlabel('Frequency Bin')
            axes[i, 1].set_ylabel('Magnitude')
            axes[i, 1].legend()
            axes[i, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        os.makedirs('analysis', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Saved {save_path}")
        plt.close()

    # =========================================================================
    # 3. COMPONENT-WISE ERROR ANALYSIS
    # =========================================================================
    def component_wise_error(self, save_path='analysis/component_errors.png'):
        """Compute MSE for trend/seasonal/residual components"""
        print("\n🔬 Computing component-wise errors...")

        # This requires access to the decomposition
        # You'll need to modify your model to save decomposed predictions

        # Placeholder for now
        print("⚠️  Implement this after saving decomposed predictions")

        # Structure:
        # 1. Load predictions
        # 2. Decompose into trend/seasonal/residual (using moving average)
        # 3. Compute MSE for each component
        # 4. Plot bar chart comparing models

    # =========================================================================
    # 4. GATE BEHAVIOR ANALYSIS
    # =========================================================================
    def analyze_gate_behavior(self, model_path, save_path='analysis/gate_behavior.png'):
        """Analyze adaptive gating weights"""
        print("\n🔬 Analyzing gate behavior...")

        # You need to modify FreqHybrid to return gate weights
        # For now, this is a placeholder

        print("⚠️  To enable this:")
        print("   1. Modify FreqHybrid.forward() to return alphas")
        print("   2. Save alphas during testing")
        print("   3. Run this analysis")

        # Structure:
        # 1. Load saved gate weights
        # 2. Plot distribution (box plot)
        # 3. Plot over time
        # 4. Analyze by time-of-day or forecast step

    # =========================================================================
    # 5. ERROR VS HORIZON CURVE
    # =========================================================================
    def plot_error_accumulation(self, save_path='analysis/error_vs_horizon.png'):
        """Plot MSE as function of forecast step"""
        print("\n🔬 Analyzing error accumulation...")

        dlinear_pred = self._load_predictions('DLinear')
        hybrid_pred = self._load_predictions('FreqHybrid')

        if dlinear_pred is None or hybrid_pred is None:
            print("⚠️  Prediction files not found")
            return

        # Compute MSE at each time step
        # Note: You need ground truth for this

        print("⚠️  Need ground truth to compute step-wise errors")
        print("   Save ground truth during testing")

    # =========================================================================
    # 6. STATISTICAL STABILITY (multiple seeds)
    # =========================================================================
    def statistical_stability_analysis(self):
        """If you ran multiple seeds, compute mean ± std"""
        print("\n🔬 Statistical stability analysis...")
        print("⚠️  Run experiments with multiple seeds (--itr 3)")
        print("   Then aggregate results here")

    # =========================================================================
    # HELPER FUNCTIONS
    # =========================================================================
    def _load_predictions(self, model_name):
        """Load prediction numpy files"""
        pattern = f'{self.results_dir}/*{model_name}*/pred.npy'
        files = glob.glob(pattern)

        if len(files) == 0:
            return None

        return np.load(files[0])

    def run_all_analysis(self):
        """Run all available analyses"""
        print("\n" + "="*70)
        print("RUNNING COMPREHENSIVE ANALYSIS")
        print("="*70)

        # 1. Standard metrics
        results_df = self.extract_results_table()

        # 2. Frequency analysis
        self.plot_frequency_comparison()

        # 3. Component-wise (placeholder)
        # self.component_wise_error()

        # 4. Gate behavior (placeholder)
        # self.analyze_gate_behavior()

        # 5. Error accumulation (placeholder)
        # self.plot_error_accumulation()

        print("\n" + "="*70)
        print("ANALYSIS COMPLETE")
        print("="*70)
        print("Check analysis/ folder for plots")

if __name__ == '__main__':
    analyzer = ModelAnalyzer(
        results_dir='./results',
        logs_dir='./logs'
    )

    analyzer.run_all_analysis()
