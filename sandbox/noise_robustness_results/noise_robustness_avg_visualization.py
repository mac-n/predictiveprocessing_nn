import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patches as patches

# Results from your experiment
noise_types = ['gaussian', 'structured', 'missing']
noise_levels = {
    'gaussian': np.linspace(0, 5.0, 10),
    'structured': np.linspace(0, 5.0, 10),
    'missing': np.linspace(0, 0.5, 10)
}

# Pattern network results
pattern_results = {
    'gaussian': [0.009588, 0.213940, 0.817316, 1.824504, 3.281041, 4.949613, 7.169182, 9.477708, 12.063607, 16.137249],
    'structured': [0.009588, 0.297156, 1.175469, 2.611777, 4.650095, 6.961847, 10.126974, 13.306460, 17.606556, 21.718816],
    'missing': [0.009588, 2.710601, 5.257819, 8.254156, 11.402543, 14.055655, 17.467655, 20.417320, 24.490345, 28.223699]
}

# Standard network results
standard_results = {
    'gaussian': [0.014690, 0.236424, 0.852874, 1.913178, 3.348873, 4.927573, 7.047575, 9.068693, 11.624240, 15.055751],
    'structured': [0.014690, 0.454636, 1.773241, 3.921768, 6.781353, 9.994598, 14.172055, 18.342524, 23.387042, 28.087598],
    'missing': [0.014690, 2.763661, 5.104599, 8.102411, 10.867169, 13.392071, 16.335049, 18.872622, 22.934232, 26.428974]
}

# Calculate relative improvement percentages
improvements = {}
for noise_type in noise_types:
    pattern_loss = np.array(pattern_results[noise_type])
    standard_loss = np.array(standard_results[noise_type])
    rel_improvement = (standard_loss - pattern_loss) / standard_loss * 100
    improvements[noise_type] = rel_improvement

# Create the figure
plt.figure(figsize=(12, 8))
gs = GridSpec(2, 5, height_ratios=[2, 1])

# Title and explanation
plt.suptitle("Pattern Networks Show Superior Robustness to Structured Noise", fontsize=16, y=0.98)
summary_text = (
    "This experiment compares pattern networks with standard neural networks on the Lorenz system prediction task "
    "under different types of noise. The results show pattern networks excel particularly on structured noise, "
    "demonstrating their ability to capture fundamental data structure even in noisy conditions."
)

# Main plots (top row) - Performance curves
colors = {'pattern': '#1f77b4', 'standard': '#ff7f0e'}
line_styles = {'gaussian': '-', 'structured': '-', 'missing': '-'}
markers = {'gaussian': 'o', 'structured': 's', 'missing': '^'}

# Create the three main plots
for i, noise_type in enumerate(noise_types):
    ax = plt.subplot(gs[0, i])
    
    # Plot both network performances
    ax.plot(noise_levels[noise_type], pattern_results[noise_type], 
            line_styles[noise_type], color=colors['pattern'], marker=markers[noise_type], 
            markersize=4, label='Pattern Network')
    
    ax.plot(noise_levels[noise_type], standard_results[noise_type], 
            line_styles[noise_type], color=colors['standard'], marker=markers[noise_type], 
            markersize=4, label='Standard Network')
    
    # Shade the region between curves for structured noise to highlight the difference
    if noise_type == 'structured':
        ax.fill_between(noise_levels[noise_type], 
                         pattern_results[noise_type], 
                         standard_results[noise_type], 
                         color='lightblue', alpha=0.5)
    
    # Set titles and labels
    noise_titles = {'gaussian': 'Gaussian Noise', 'structured': 'Structured Noise', 'missing': 'Missing Data'}
    ax.set_title(noise_titles[noise_type])
    ax.set_xlabel('Noise Level')
    ax.set_ylabel('Mean Squared Error')
    ax.grid(alpha=0.3)
    
    # Add improvement annotation for structured noise
    if noise_type == 'structured':
        max_improvement = np.max(improvements[noise_type])
        max_idx = np.argmax(improvements[noise_type])
        level = noise_levels[noise_type][max_idx]
        ax.annotate(f'Max: {max_improvement:.1f}% better', 
                    xy=(level, pattern_results[noise_type][max_idx]),
                    xytext=(level-1, pattern_results[noise_type][max_idx] + 5),
                    arrowprops=dict(arrowstyle='->'))

# Improvement comparison (right side) - AVERAGE Relative improvement across all noise levels
ax_improvement = plt.subplot(gs[0, 3:])

# Calculate average improvement across all noise levels
avg_improvements = []
noise_labels = []

for noise_type in noise_types:
    pattern_loss = np.array(pattern_results[noise_type])
    standard_loss = np.array(standard_results[noise_type])
    rel_improvement = (standard_loss - pattern_loss) / standard_loss * 100
    avg_improvements.append(np.mean(rel_improvement))
    noise_labels.append(noise_type.capitalize())

bars = ax_improvement.bar(noise_labels, avg_improvements, color=['#1f77b4', '#2ca02c', '#9467bd'])

# Add labels and values
ax_improvement.set_title('Average Performance Advantage Across All Noise Levels')
ax_improvement.set_ylabel('Error Reduction (%)')
ax_improvement.set_ylim([-10, 30])  # Adjusted for average values
ax_improvement.axhline(y=0, color='r', linestyle='--', alpha=0.3)
ax_improvement.grid(axis='y', alpha=0.3)

for i, bar in enumerate(bars):
    height = avg_improvements[i]
    ax_improvement.annotate(f'{height:.1f}%',
                          xy=(bar.get_x() + bar.get_width()/2, height),
                          xytext=(0, 3),  # 3 points vertical offset
                          textcoords="offset points",
                          ha='center', va='bottom')

# Highlight structured noise
bars[1].set_color('#2ca02c')
bars[1].set_edgecolor('black')
bars[1].set_linewidth(1.5)

# Add common legend at the top of the figure
handles = [
    plt.Line2D([0], [0], color=colors['pattern'], marker='o', linestyle='-', markersize=6),
    plt.Line2D([0], [0], color=colors['standard'], marker='o', linestyle='-', markersize=6)
]
labels = ['Pattern Network', 'Standard Network']

fig = plt.gcf()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.96), ncol=2, frameon=True, fontsize=10)

# Add summary text at the bottom
plt.figtext(0.5, 0.04, summary_text, ha='center', fontsize=10, 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

# Pattern advantage explanation - Updated for average
avg_structured_improvement = avg_improvements[1]  # Index 1 is structured noise
pattern_explanation = (
    f"Pattern networks excel with structured noise (avg. {avg_structured_improvement:.1f}% better) because they can model the underlying structure separately from the noise pattern. "
    "This suggests patterns capture fundamental system properties that enable robustness to systematic perturbations."
)
plt.figtext(0.5, 0.11, pattern_explanation, ha='center', fontsize=10, 
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.4))

# Tighten layout and save
plt.tight_layout(rect=[0, 0.15, 1, 0.93])
plt.savefig('pattern_robustness_avg_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

print("Visualization saved as pattern_robustness_avg_comparison.png") 