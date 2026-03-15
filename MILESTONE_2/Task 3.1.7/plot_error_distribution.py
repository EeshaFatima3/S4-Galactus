import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Layer names (NOW INCLUDING HILBERT)
layers = ['Hilbert', 'UProject', 'S4D1', 'GELU1', 'S4D2', 'GELU2', 'FC']

# MAE values for each layer across all 10 samples
mae_data = {
    'Hilbert': [
        1.2e-15, 1.3e-15, 1.1e-15, 1.4e-15, 1.2e-15,  # sample 0-4
        1.3e-15, 1.2e-15, 1.1e-15, 1.3e-15, 1.2e-15   # sample 5-9
    ],
    'UProject': [
        4.18e-09, 3.73e-09, 3.33e-09, 2.76e-09, 5.52e-09,
        4.50e-09, 4.29e-09, 5.00e-09, 3.94e-09, 4.05e-09
    ],
    'S4D1': [
        1.74e-05, 1.76e-05, 1.77e-05, 1.78e-05, 1.67e-05,
        1.72e-05, 1.73e-05, 1.69e-05, 1.75e-05, 1.75e-05
    ],
    'GELU1': [
        1.60e-08, 1.60e-08, 1.60e-08, 1.62e-08, 1.56e-08,
        1.56e-08, 1.58e-08, 1.56e-08, 1.60e-08, 1.58e-08
    ],
    'S4D2': [
        1.63e-05, 1.62e-05, 1.62e-05, 1.64e-05, 1.60e-05,
        1.58e-05, 1.58e-05, 1.58e-05, 1.59e-05, 1.62e-05
    ],
    'GELU2': [
        1.27e-08, 1.30e-08, 1.27e-08, 1.28e-08, 1.31e-08,
        1.27e-08, 1.26e-08, 1.28e-08, 1.27e-08, 1.23e-08
    ],
    'FC': [
        8.48e-08, 4.77e-07, 1.84e-07, 2.24e-07, 2.53e-07,
        6.56e-07, 2.68e-07, 3.24e-07, 2.38e-07, 4.17e-07
    ]
}

# Create dataframe
plot_data = []
for layer in layers:
    for mae_value in mae_data[layer]:
        plot_data.append({'Layer': layer, 'MAE': mae_value})

df = pd.DataFrame(plot_data)

# Create the plot
plt.figure(figsize=(14, 6))
sns.boxplot(x='Layer', y='MAE', data=df)
plt.yscale('log')
plt.ylabel('Mean Absolute Error (log scale)', fontsize=12)
plt.xlabel('Layer', fontsize=12)
plt.title('Distribution of Layer-wise Errors Across 10 Test Samples', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, axis='y')

# Add threshold lines
plt.axhline(y=1e-4, color='r', linestyle='--', alpha=0.5, label='S4D/GELU Threshold (1e-4)')
plt.axhline(y=1e-6, color='g', linestyle='--', alpha=0.5, label='Linear Threshold (1e-6)')
plt.axhline(y=1e-12, color='b', linestyle='--', alpha=0.5, label='Hilbert Threshold (1e-12)')
plt.legend()

plt.tight_layout()
plt.savefig('error_distribution.png', dpi=300, bbox_inches='tight')
print("✅ Figure saved as 'error_distribution.png' (with Hilbert layer)")
plt.show()

# Print statistics
print("\n📊 Summary Statistics:")
print("-" * 60)
for layer in layers:
    values = mae_data[layer]
    print(f"{layer}:")
    print(f"  Mean MAE: {np.mean(values):.2e}")
    print(f"  Min MAE:  {np.min(values):.2e}")
    print(f"  Max MAE:  {np.max(values):.2e}")
    print()