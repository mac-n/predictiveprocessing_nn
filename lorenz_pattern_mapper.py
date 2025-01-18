import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Tuple, List

class LorenzPatternMapper:
    def __init__(self, model):
        self.model = model
    
    def generate_lorenz_trajectory(self, n_steps=1000, dt=0.01) -> Tuple[np.ndarray, np.ndarray]:
        """Generate full Lorenz trajectory with state variables"""
        def lorenz(x, y, z, s=10, r=28, b=2.667):
            dx = s*(y - x)
            dy = r*x - y - x*z
            dz = x*y - b*z
            return dx, dy, dz

        # Initialize arrays to store trajectory
        trajectory = np.zeros((n_steps, 3))
        x, y, z = 1, 1, 1
        
        # Generate trajectory
        for i in range(n_steps):
            dx, dy, dz = lorenz(x, y, z)
            x += dx * dt
            y += dy * dt
            z += dz * dt
            trajectory[i] = [x, y, z]
        
        # Calculate wing identification and transitions
        wing = np.sign(trajectory[:, 0])  # Using x-coordinate to identify wings
        transitions = np.where(np.diff(wing) != 0)[0]
        
        return trajectory, transitions
    
    def get_pattern_activations(self, input_sequence: torch.Tensor) -> List[np.ndarray]:
        """Get pattern activations for a sequence of inputs"""
        pattern_activations = []
        self.model.eval()
        
        with torch.no_grad():
            for x in input_sequence:
                # Forward pass
                self.model(x.unsqueeze(0))
                # Get layer stats
                stats = self.model.get_layer_stats()
                # Record pattern usage for each layer
                layer_patterns = [stat.pattern_usage.cpu().numpy() for stat in stats]
                pattern_activations.append(layer_patterns)
        
        return pattern_activations
    
    def analyze_wing_patterns(self, sequence_length=20):
        """Analyze which patterns are active in different wings of the attractor"""
        # Generate trajectory and prepare input sequence
        trajectory, transitions = self.generate_lorenz_trajectory()
        X = torch.FloatTensor([trajectory[i:i+sequence_length, 0] 
                             for i in range(len(trajectory)-sequence_length)])
        
        # Get pattern activations
        pattern_activations = self.get_pattern_activations(X)
        
        # Separate patterns by wing
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
        
        # Average patterns for each state
        left_avg = np.mean(left_wing_patterns, axis=0)
        right_avg = np.mean(right_wing_patterns, axis=0)
        trans_avg = np.mean(transition_patterns, axis=0)
        
        return left_avg, right_avg, trans_avg
    
    def visualize_pattern_mapping(self):
        """Visualize how patterns map to Lorenz system states"""
        left_patterns, right_patterns, trans_patterns = self.analyze_wing_patterns()
        
        # Plot pattern usage for each layer
        n_layers = len(left_patterns)
        fig, axes = plt.subplots(n_layers, 1, figsize=(12, 4*n_layers))
        
        for layer in range(n_layers):
            ax = axes[layer] if n_layers > 1 else axes
            
            # Plot patterns
            x = np.arange(len(left_patterns[layer]))
            width = 0.25
            
            ax.bar(x - width, left_patterns[layer], width, label='Left Wing')
            ax.bar(x, trans_patterns[layer], width, label='Transitions')
            ax.bar(x + width, right_patterns[layer], width, label='Right Wing')
            
            ax.set_xlabel('Pattern Index')
            ax.set_ylabel('Average Activation')
            ax.set_title(f'Layer {layer} Pattern Usage by Lorenz State')
            ax.legend()
        
        plt.tight_layout()
        
        # Plot 3D trajectory with pattern highlights
        trajectory, transitions = self.generate_lorenz_trajectory()
        
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Color points based on dominant pattern
        colors = ['blue', 'red']  # for left/right wing
        wing = np.sign(trajectory[:, 0])
        
        for i in range(len(trajectory)-1):
            color = colors[int(wing[i] > 0)]
            if i in transitions:
                color = 'green'
            
            ax.plot(trajectory[i:i+2, 0], 
                   trajectory[i:i+2, 1], 
                   trajectory[i:i+2, 2], 
                   color=color, alpha=0.6)
        
        ax.set_title('Lorenz Attractor with Pattern Regions')
        
        return fig, axes

def map_lorenz_patterns(model):
    mapper = LorenzPatternMapper(model)
    return mapper.visualize_pattern_mapping()