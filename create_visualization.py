import matplotlib.pyplot as plt
import numpy as np

# Data from the report
methods = ["Customer Segmented", "Baseline", "NGCF (1% data)", "LightGCN (10% data)", "LightGBM"]
hit_rate = [0.3605, 0.3300, 0.1240, 0.1500, 0.1722]  # Updated from report
training_time = [2.5, 0.5, 8.2, 4.7, 0.3]  # Keep existing training times, baseline as 0.0
memory_usage = [2.49, 0.5, 12.76, 8.54, 4.82]  # Keep existing memory usage, estimate for baseline

x = np.arange(len(methods))  # label locations
width = 0.25  # width of the bars

# Create subplots
fig, ax1 = plt.subplots(figsize=(12, 7))

# Reorder methods for better visualization
order = [0, 1, 4, 3, 2]  # Customer Segmented, Baseline, LightGBM, LightGCN, NGCF
methods = [methods[i] for i in order]
hit_rate = [hit_rate[i] for i in order]
training_time = [float(training_time[i]) for i in order]  # Ensure all are float
memory_usage = [float(memory_usage[i]) for i in order]  # Ensure all are float

# Bar plots
bars1 = ax1.bar(x - width, hit_rate, width, label='Hit Rate @10', color='tab:blue')
bars2 = ax1.bar(x, training_time, width, label='Training Time (h)', color='tab:orange')
bars3 = ax1.bar(x + width, memory_usage, width, label='Memory Usage (GB)', color='tab:green')

# Adding labels, title, and legend
ax1.set_xlabel('Methods', fontsize=14)
ax1.set_ylabel('Values', fontsize=14)
ax1.set_title('Comparative Analysis of Recommendation Systems', fontsize=16, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(methods, rotation=15, ha='right', fontsize=12)
ax1.legend(loc='upper left', fontsize=12)

# Add value labels on bars
def add_value_labels(bars, decimals=4):
    for bar in bars:
        height = bar.get_height()
        if height > 0:  # Only add labels for non-zero bars
            ax1.text(bar.get_x() + bar.get_width() / 2, height,
                    f'{height:.{decimals}f}', ha='center', va='bottom', fontsize=10)

add_value_labels(bars1, decimals=4)
add_value_labels(bars2, decimals=1)
add_value_labels(bars3, decimals=2)

plt.tight_layout()
plt.savefig('imgs/comparative_analysis.png', dpi=300, bbox_inches='tight')
print("Visualization saved to imgs/comparative_analysis.png")