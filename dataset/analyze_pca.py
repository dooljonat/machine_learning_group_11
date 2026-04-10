"""
Quick script to analyze PCA explained variance and determine optimal n_components.
Run this before tuning to validate your choice of PCA components.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from dataset.loader import DataConfig, get_numpy

print("Loading training data...")
# Use a reasonable sample size for analysis (all 500 per class)
config = DataConfig(max_samples_per_class=500)
X_train, y_train, X_val, y_val = get_numpy(config)
print(f"Loaded {X_train.shape[0]} training samples, {X_train.shape[1]} features")

print("\nFitting PCA with all components (this may take a minute)...")
# Fit PCA with maximum possible components (min of n_samples, n_features)
max_components = min(X_train.shape[0], X_train.shape[1])
pca = PCA(n_components=max_components)
pca.fit(X_train)

# Calculate cumulative explained variance
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

# Find components needed for different variance thresholds
thresholds = [0.80, 0.85, 0.90, 0.95, 0.99]
print("\n" + "="*60)
print("EXPLAINED VARIANCE ANALYSIS")
print("="*60)
for threshold in thresholds:
    n_comp = np.argmax(cumulative_variance >= threshold) + 1
    print(f"  {threshold*100:.0f}% variance: {n_comp:4d} components")

# Check specific values of interest
print("\n" + "="*60)
print("VARIANCE CAPTURED BY SPECIFIC COMPONENT COUNTS")
print("="*60)
for n_comp in [200, 500, 1000, 2000]:
    if n_comp <= len(cumulative_variance):
        variance = cumulative_variance[n_comp - 1]
        print(f"  {n_comp:4d} components: {variance*100:.2f}% variance")
    else:
        print(f"  {n_comp:4d} components: (exceeds max {max_components})")

# Create visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Scree plot (first 100 components)
ax1.plot(range(1, min(101, len(pca.explained_variance_ratio_) + 1)),
         pca.explained_variance_ratio_[:100], 'b-', linewidth=2)
ax1.set_xlabel('Component Number')
ax1.set_ylabel('Explained Variance Ratio')
ax1.set_title('Scree Plot (First 100 Components)')
ax1.grid(True, alpha=0.3)

# Cumulative variance plot
ax2.plot(range(1, len(cumulative_variance) + 1),
         cumulative_variance * 100, 'g-', linewidth=2)
ax2.axhline(y=90, color='r', linestyle='--', label='90% variance', alpha=0.7)
ax2.axhline(y=95, color='orange', linestyle='--', label='95% variance', alpha=0.7)
ax2.axvline(x=500, color='purple', linestyle='--', label='500 components', alpha=0.7)
ax2.set_xlabel('Number of Components')
ax2.set_ylabel('Cumulative Explained Variance (%)')
ax2.set_title('Cumulative Explained Variance')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, min(3000, len(cumulative_variance)))

plt.tight_layout()
plot_path = "dataset/pca_variance_analysis.png"
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f"\nPlot saved to {plot_path}")
plt.close()

# Recommendation
print("\n" + "="*60)
print("RECOMMENDATION")
print("="*60)
n_95 = np.argmax(cumulative_variance >= 0.95) + 1
n_90 = np.argmax(cumulative_variance >= 0.90) + 1

if 500 >= n_95:
    print(f"✓ 500 components captures {cumulative_variance[499]*100:.2f}% variance")
    print(f"  This is GOOD - exceeds the 95% threshold ({n_95} components needed)")
elif 500 >= n_90:
    print(f"✓ 500 components captures {cumulative_variance[499]*100:.2f}% variance")
    print(f"  This is REASONABLE - between 90% and 95% thresholds")
    print(f"  Consider testing {n_95} components for 95% variance if feasible")
else:
    print(f"⚠ 500 components only captures {cumulative_variance[499]*100:.2f}% variance")
    print(f"  This is LOW - consider increasing to {n_90} for 90% or {n_95} for 95%")

print("="*60)
