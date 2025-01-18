import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_pattern_projections(model, layer_idx=0, n_steps=1000):
    """Visualize how each pattern projects onto the Lorenz attractor"""
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
    
    # Get pattern activations for each point
    X = torch.FloatTensor([xs[i:i+20] for i in range(len(xs)-20)])  # Sequence length 20
    pattern_activations = []
    
    model.eval()
    with torch.no_grad():
        for x in X:
            model(x.unsqueeze(0))
            stats = model.get_layer_stats()
            pattern_activations.append(stats[layer_idx].pattern_usage.cpu().numpy())
    
    pattern_activations = np.array(pattern_activations)
    
    # Create subplot for each pattern
    n_patterns = pattern_activations.shape[1]
    fig = plt.figure(figsize=(20, 5*((n_patterns+3)//4)))
    
    for i in range(n_patterns):
        ax = fig.add_subplot(((n_patterns+3)//4), 4, i+1, projection='3d')
        
        # Color points based on pattern activation
        colors = plt.cm.viridis(pattern_activations[:, i])
        
        # Plot trajectory with color based on pattern activation
        scatter = ax.scatter(xs[:-20], ys[:-20], zs[:-20], 
                           c=pattern_activations[:, i],
                           cmap='viridis',
                           alpha=0.6)
        
        ax.set_title(f'Pattern {i} Activation')
        plt.colorbar(scatter, ax=ax)
        
        # Set common viewing angle
        ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    return fig

def run_pattern_projection_analysis(model):
    """Run pattern projection analysis for both layers"""
    for layer in [0, 1]:  # Skip layer 2 as discussed
        fig = visualize_pattern_projections(model, layer_idx=layer)
        fig.suptitle(f'Layer {layer} Pattern Projections onto Lorenz Attractor', y=1.02, size=16)
        plt.savefig(f'layer_{layer}_pattern_projections.png', bbox_inches='tight', dpi=300)
        plt.close()