import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Tuple, List

class LorenzPatternMapper:
    def __init__(self, model):
        self.model = model
        # Set global font sizes
        plt.rcParams.update({
            'font.size': 30,
            'axes.labelsize': 30,
            'axes.titlesize': 30,
            'xtick.labelsize': 24,
            'ytick.labelsize': 24,
            'legend.fontsize': 24
        })
    
    def generate_lorenz_trajectory(self, n_steps=1000, dt=0.01) -> Tuple[np.ndarray, np.ndarray]:
        """Generate full Lorenz trajectory with state variables"""
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
        
        wing = np.sign(trajectory[:, 0])
        transitions = np.where(np.diff(wing) != 0)[0]
        
        return trajectory, transitions
    
    def get_pattern_activations(self, input_sequence: torch.Tensor) -> List[np.ndarray]:
        pattern_activations = []
        self.model.eval()
        
        with torch.no_grad():
            for x in input_sequence:
                self.model(x.unsqueeze(0))
                stats = self.model.get_layer_stats()
                layer_patterns = [stat.pattern_usage.cpu().numpy() for stat in stats]
                pattern_activations.append(layer_patterns)
        
        return pattern_activations
    
    def analyze_wing_patterns(self, sequence_length=20):
        trajectory, transitions = self.generate_lorenz_trajectory()
        X = torch.FloatTensor([trajectory[i:i+sequence_length, 0] 
                             for i in range(len(trajectory)-sequence_length)])
        
        pattern_activations = self.get_pattern_activations(X)
        
        left_wing_patterns = []
        right_wing_patterns = []
        transition_patterns = []
        
        for i in range(len(pattern_activations)):
            is_transition = any(abs(i - t) < sequence_length for t in transitions)
            wing = np.sign(trajectory[i+sequence_length-1, 0])
            
            if is_transition:
                transition_patterns.append(pattern_activations[i])
            elif wing < 0:
                left_wing_patterns.append(pattern_activations[i])
            else:
                right_wing_patterns.append(pattern_activations[i])
        
        left_avg = np.mean(left_wing_patterns, axis=0)
        right_avg = np.mean(right_wing_patterns, axis=0)
        trans_avg = np.mean(transition_patterns, axis=0)
        
        return left_avg, right_avg, trans_avg
    
    def visualize_pattern_mapping(self):
        left_patterns, right_patterns, trans_patterns = self.analyze_wing_patterns()
        
        # Create single figure with two subplots stacked vertically
        fig, axes = plt.subplots(2, 1, figsize=(20, 16), height_ratios=[1, 1])
        
        # Plot patterns for both layers vertically
        for layer in range(2):  # Assuming 2 layers
            ax = axes[layer]
            
            x = np.arange(len(left_patterns[layer]))
            width = 0.25
            
            ax.bar(x - width, left_patterns[layer], width, label='Left Wing', linewidth=2)
            ax.bar(x, trans_patterns[layer], width, label='Transitions', linewidth=2)
            ax.bar(x + width, right_patterns[layer], width, label='Right Wing', linewidth=2)
            
            ax.set_xlabel('Pattern Index', fontsize=30, labelpad=20)
            ax.set_ylabel('Average Activation', fontsize=30, labelpad=20)
            ax.set_title(f'Layer {layer} Pattern Usage', 
                        fontsize=36, pad=20)
            
            ax.tick_params(axis='both', which='major', labelsize=24)
            ax.legend(fontsize=24, loc='upper right')
        
        plt.suptitle('Pattern Usage by Lorenz State', fontsize=42, y=0.95)
        plt.tight_layout()
        
        # Plot 3D trajectory 
        fig3d = plt.figure(figsize=(20, 20))
        ax3d = fig3d.add_subplot(111, projection='3d')
        
        trajectory, transitions = self.generate_lorenz_trajectory()
        colors = ['blue', 'red']
        wing = np.sign(trajectory[:, 0])
        
        for i in range(len(trajectory)-1):
            color = colors[int(wing[i] > 0)]
            if i in transitions:
                color = 'green'
            
            ax3d.plot(trajectory[i:i+2, 0], 
                     trajectory[i:i+2, 1], 
                     trajectory[i:i+2, 2], 
                     color=color, alpha=0.6, linewidth=2)
        
        ax3d.set_title('Lorenz Attractor with Pattern Regions', 
                      fontsize=36, pad=20)
        
        ax3d.tick_params(axis='both', which='major', labelsize=24)
        ax3d.set_xlabel('X', fontsize=30, labelpad=20)
        ax3d.set_ylabel('Y', fontsize=30, labelpad=20)
        ax3d.set_zlabel('Z', fontsize=30, labelpad=20)
        
        ax3d.view_init(elev=20, azim=45)
        ax3d.dist = 11
        
        return fig, fig3d

def map_lorenz_patterns(model):
    mapper = LorenzPatternMapper(model)
    fig_bar, fig_3d = mapper.visualize_pattern_mapping()
    return fig_3d, fig_bar