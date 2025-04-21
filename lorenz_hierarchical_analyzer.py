import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from typing import List, Dict
import os
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler

class LorenzHierarchicalAnalyzer:
    def __init__(self, model):
        self.model = model
        
    def generate_lorenz_trajectory(self, n_steps=1000, dt=0.01):
        """Generate Lorenz trajectory for analysis"""
        def lorenz(x, y, z, s=10, r=28, b=2.667):
            dx = s*(y - x)
            dy = r*x - y - x*z
            dz = x*y - b*z
            return dx, dy, dz
        
        trajectory = np.zeros((n_steps, 3))
        x, y, z = 1, 1, 1
        
        for i in range(n_steps):
            dx, dy, dz = lorenz(x, y, z)
            x += dx * dt
            y += dy * dt
            z += dz * dt
            trajectory[i] = [x, y, z]
            
        return trajectory
    
    def get_pattern_activations(self, n_steps=1000, window_size=20):
        """Analyze hierarchical pattern activations along Lorenz trajectory"""
        trajectory = self.generate_lorenz_trajectory(n_steps)
        pattern_data = []
        
        # Create sliding windows of trajectory
        X = torch.FloatTensor([trajectory[i:i+window_size, 0] 
                             for i in range(len(trajectory)-window_size)])
        
        self.model.eval()
        with torch.no_grad():
            for i, x in enumerate(X):
                # Forward pass
                self.model(x.unsqueeze(0))
                
                # Get pattern activations for each layer and level
                stats = self.model.get_layer_stats()
                
                for layer_idx, stat in enumerate(stats):
                    for level_idx, level_patterns in enumerate(stat.pattern_usage_per_level):
                        pattern_data.append({
                            'layer': layer_idx,
                            'level': level_idx,
                            'position': i + window_size - 1,
                            'point': trajectory[i + window_size - 1],
                            'pattern_activations': level_patterns.cpu().numpy(),
                            'trajectory_idx': i
                        })
        
        return pattern_data, trajectory
    
    def visualize_pattern_regions(self, data: List[Dict], trajectory: np.ndarray, 
                                layer: int = 0, window_size: int = 20):
        """Visualize how different hierarchical levels segment the Lorenz attractor"""
        levels = sorted(set(d['level'] for d in data if d['layer'] == layer))
        
        # Reduced figure size and tightened spacing
        fig = plt.figure(figsize=(16, 4*len(levels)))
        gs = fig.add_gridspec(len(levels), 4, width_ratios=[1]*4, 
                            hspace=0.1, wspace=0.1)  # Reduced spacing
        
        plt.rcParams.update({'font.size': 20})  # Base font size
        scatter_plots = []
        
        for level_idx, level in enumerate(levels):
            level_data = [d for d in data if d['layer'] == layer and d['level'] == level]
            pattern_activations = np.array([d['pattern_activations'] for d in level_data])
            n_patterns = pattern_activations.shape[1]
            
            for pattern_idx in range(n_patterns):
                ax = fig.add_subplot(gs[level_idx, pattern_idx], projection='3d')
                
                scaler = MinMaxScaler()
                pattern_norm = scaler.fit_transform(pattern_activations[:, pattern_idx].reshape(-1, 1)).ravel()
                
                scatter = ax.scatter(trajectory[:-window_size, 0],
                                   trajectory[:-window_size, 1],
                                   trajectory[:-window_size, 2],
                                   c=pattern_norm,
                                   cmap='viridis',
                                   vmin=0,
                                   vmax=1,
                                   alpha=0.6)
                scatter_plots.append(scatter)
                
                ax.set_title(f'Level {level} Pattern {pattern_idx}', fontsize=24, pad=20)
                ax.view_init(elev=20, azim=45)
                
                # Axis labels
                if level_idx == len(levels)-1:
                    ax.tick_params(axis='both', which='major', labelsize=20)
                else:
                    ax.set_xticklabels([])
                    ax.set_yticklabels([])
                    ax.set_zticklabels([])
        
        # Colorbar
        plt.subplots_adjust(right=0.92)
        cbar_ax = fig.add_axes([0.93, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(scatter_plots[0], cax=cbar_ax)
        cbar.set_label('Normalized Pattern Activation', fontsize=24)
        cbar.ax.tick_params(labelsize=20)
        cbar.set_ticks([0, 0.5, 1])
        cbar.set_ticklabels(['Low', 'Medium', 'High'])
        
        plt.tight_layout()
        return fig
    
    def analyze_compression_ratios(self, data: List[Dict]):
        """Analyze how effectively each level compresses information"""
        levels = sorted(set(d['level'] for d in data))
        layers = sorted(set(d['layer'] for d in data))
        
        plt.rcParams.update({'font.size': 30})  # Increased font size globally
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for layer_idx in layers:
            layer_data = [d for d in data if d['layer'] == layer_idx]
            
            compression_ratios = []
            level_labels = []
            
            for level_idx in levels:
                level_data = [d for d in layer_data if d['level'] == level_idx]
                pattern_activations = np.array([d['pattern_activations'] for d in level_data])
                
                # Calculate entropy/sparsity
                pattern_usage = pattern_activations.mean(axis=0)
                entropy = -np.sum(pattern_usage * np.log2(pattern_usage + 1e-10))
                compression_ratios.append(entropy)
                level_labels.append(f'L{level_idx}')
            
            ax.plot(compression_ratios, marker='o', label=f'Layer {layer_idx}', linewidth=3, markersize=12)
        
        ax.set_xlabel('Hierarchical Level', fontsize=30, labelpad=20)
        ax.set_ylabel('Pattern Entropy (bits)', fontsize=30, labelpad=20)
        ax.set_xticks(range(len(levels)))
        ax.set_xticklabels(level_labels, fontsize=24)
        ax.tick_params(axis='both', which='major', labelsize=24)
        ax.legend(fontsize=24, loc='best')
        ax.set_title('Information Compression Across Hierarchical Levels', fontsize=30, pad=20)
        
        plt.tight_layout()
        return fig
    
    def analyze_pattern_correlations(self, data: List[Dict], layer: int = 0):
        """Analyze correlations between patterns at different levels"""
        levels = sorted(set(d['level'] for d in data if d['layer'] == layer))
        
        plt.rcParams.update({'font.size': 30})  # Increased font size globally
        fig, axs = plt.subplots(1, len(levels), figsize=(6*len(levels), 5))
        if len(levels) == 1:
            axs = [axs]
        
        for level_idx, level in enumerate(levels):
            level_data = [d for d in data if d['layer'] == layer and d['level'] == level]
            pattern_activations = np.array([d['pattern_activations'] for d in level_data])
            
            # Calculate correlation matrix
            corr_matrix = np.corrcoef(pattern_activations.T)
            
            # Plot correlation matrix
            sns.heatmap(corr_matrix, 
                       ax=axs[level_idx],
                       cmap='coolwarm',
                       center=0,
                       vmin=-1,
                       vmax=1,
                       annot=True,  # Show correlation values
                       fmt='.2f',   # Format to 2 decimal places
                       annot_kws={'size': 20})  # Larger correlation numbers
            
            axs[level_idx].set_title(f'Layer {layer} Level {level}\nPattern Correlations', 
                                   fontsize=30, pad=20)
            axs[level_idx].tick_params(axis='both', which='major', labelsize=24)
        
        plt.tight_layout()
        return fig

def analyze_lorenz_hierarchical_patterns(model):
    """Run complete hierarchical pattern analysis on Lorenz system"""
    # Create timestamped directory for visualizations with full path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    viz_dir = os.path.join(os.getcwd(), f"lorenz_hierarchical_viz_{timestamp}")
    print(f"\nCreating visualization directory at: {viz_dir}")
    os.makedirs(viz_dir, exist_ok=True)
    
    # Initialize analyzer
    analyzer = LorenzHierarchicalAnalyzer(model)
    
    # Run analysis
    print("\nAnalyzing hierarchical patterns...")
    data, trajectory = analyzer.get_pattern_activations()
    
    # Generate visualizations for each layer
    print("\nGenerating visualizations...")
    for layer in range(len(model.layers)):
        # Pattern regions
        plt.figure(figsize=(16, 12))
        fig = analyzer.visualize_pattern_regions(data, trajectory, layer=layer)
        plt.savefig(os.path.join(viz_dir, f'layer_{layer}_pattern_regions.pdf'), 
                   format='pdf', bbox_inches='tight', dpi=300)
        plt.close()
        
        # Pattern correlations
        plt.figure(figsize=(15, 5))
        fig = analyzer.analyze_pattern_correlations(data, layer=layer)
        plt.savefig(os.path.join(viz_dir, f'layer_{layer}_pattern_correlations.pdf'),
                   format='pdf', bbox_inches='tight', dpi=300)
        plt.close()
    
    # Overall compression analysis
    plt.figure(figsize=(10, 6))
    fig = analyzer.analyze_compression_ratios(data)
    plt.savefig(os.path.join(viz_dir, 'compression_analysis.pdf'),
                format='pdf', bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"\nVisualizations saved in directory: {viz_dir}")
    return analyzer, data