import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import MinMaxScaler

def visualize_pattern_projections(model, n_steps=1000):
    """Visualize how each pattern projects onto the Lorenz attractor with normalized patterns"""
    plt.rcParams.update({'font.size': 30,
                        'axes.labelsize': 30,
                        'axes.titlesize': 30,
                        'xtick.labelsize': 24,
                        'ytick.labelsize': 24})

    # Generate Lorenz trajectory
    def lorenz(x, y, z, s=10, r=28, b=2.667):
        dx = s*(y - x)
        dy = r*x - y - x*z
        dz = x*y - b*z
        return dx, dy, dz

    # Generate trajectory
    dt = 0.01
    xs, ys, zs = [], [], []
    x, y, z = 1, 1, 1
    
    for _ in range(n_steps):
        dx, dy, dz = lorenz(x, y, z)
        x += dx * dt
        y += dy * dt
        z += dz * dt
        xs.append(x)
        ys.append(y)
        zs.append(z)
    
    trajectory = np.array([xs, ys, zs]).T
    
    # Get pattern activations for both layers
    X = torch.FloatTensor([xs[i:i+20] for i in range(len(xs)-20)])  # Sequence length 20
    layer_pattern_activations = []
    
    model.eval()
    with torch.no_grad():
        for x in X:
            model(x.unsqueeze(0))
            stats = model.get_layer_stats()
            # Get patterns for both layers
            layer_pattern_activations.append([
                stats[0].pattern_usage.cpu().numpy(),
                stats[1].pattern_usage.cpu().numpy()
            ])
    
    layer_pattern_activations = np.array(layer_pattern_activations)
    n_patterns = layer_pattern_activations.shape[2]  # number of patterns per layer
    
    # Create a figure with 4 rows (2 per layer) and 4 columns
    fig = plt.figure(figsize=(30, 32))
    gs = fig.add_gridspec(4, 4, hspace=0.1, wspace=0.05)
    
    scatter_plots = []
    
    for layer in range(2):  # For each layer
        for i in range(n_patterns):  # For each pattern
            # Calculate position in the grid
            row = (layer * 2) + (i // 4)  # First two rows for layer 0, next two for layer 1
            col = i % 4  # Four patterns per row
            
            ax = fig.add_subplot(gs[row, col], projection='3d')
            
            # Normalize this pattern's activations
            scaler = MinMaxScaler()
            pattern_i_norm = scaler.fit_transform(
                layer_pattern_activations[:, layer, i].reshape(-1, 1)
            ).ravel()
            
            # Plot trajectory with color based on normalized pattern activation
            scatter = ax.scatter(xs[:-20], ys[:-20], zs[:-20], 
                               c=pattern_i_norm,
                               cmap='viridis',
                               vmin=0,
                               vmax=1,
                               alpha=0.6,
                               s=50)
            scatter_plots.append(scatter)
            
            ax.set_title(f'Layer {layer}, Pattern {i}', pad=20, fontsize=36)
            
            # Set common viewing angle
            ax.view_init(elev=20, azim=45)
            ax.dist = 11
            
            # Remove axis labels except for leftmost plots
            if col != 0:
                ax.set_yticklabels([])
            if col != 0 or row != (layer * 2 + 1):  # Show x labels only on bottom row of each layer
                ax.set_xticklabels([])
            if col != 0:  # Show z labels only on leftmost plots
                ax.set_zticklabels([])
    
    # Add a single colorbar on the right side of the figure
    plt.subplots_adjust(right=0.95)
    cbar_ax = fig.add_axes([0.96, 0.15, 0.01, 0.7])
    cbar = fig.colorbar(scatter_plots[0], cax=cbar_ax, 
                       label='Normalized Pattern Activation')
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(['Low', 'Medium', 'High'])
    cbar.ax.tick_params(labelsize=24)
    cbar.ax.set_ylabel('Normalized Pattern Activation', fontsize=30, labelpad=20)
    
    # Add overall title
    fig.suptitle('Pattern Projections onto Lorenz Attractor', 
                y=0.95, fontsize=42)
    
    return fig

def run_pattern_projection_analysis(model):
    """Run pattern projection analysis for both layers in a single figure"""
    fig = visualize_pattern_projections(model)
    plt.savefig('combined_pattern_projections.pdf', 
                bbox_inches='tight', 
                format='pdf', 
                dpi=300)
    plt.close()