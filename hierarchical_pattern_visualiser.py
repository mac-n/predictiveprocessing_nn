import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple
import os
from datetime import datetime

class HierarchicalPatternAnalyzer:
    def __init__(self, model):
        self.model = model
        self.idx_to_char = {0: ' '}  # Space character
        for i, c in enumerate('abcdefghijklmnopqrstuvwxyz'):
            self.idx_to_char[i + 1] = c
        self.char_to_idx = {v: k for k, v in self.idx_to_char.items()}
    
    def analyze_pattern_activations(self, sequences: List[str], window_size: int = 20) -> Dict:
        """Analyze pattern activations across hierarchical levels"""
        pattern_data = []
        
        self.model.eval()
        with torch.no_grad():
            for seq_idx, seq in enumerate(sequences):
                # Convert sequence to indices
                char_indices = [self.char_to_idx[c] for c in seq]
                
                # Create sliding windows
                for i in range(len(char_indices) - window_size + 1):
                    window = char_indices[i:i+window_size]
                    input_tensor = torch.FloatTensor(window).unsqueeze(0)
                    
                    # Forward pass
                    self.model(input_tensor)
                    
                    # Get pattern activations
                    stats = self.model.get_layer_stats()
                    
                    # Store data for each layer and level
                    for layer_idx, stat in enumerate(stats):
                        for level_idx, level_patterns in enumerate(stat.pattern_usage_per_level):
                            pattern_data.append({
                                'layer': layer_idx,
                                'level': level_idx,
                                'position': i + window_size - 1,
                                'char': seq[i + window_size - 1],
                                'pattern_activations': level_patterns.cpu().numpy(),
                                'sequence': seq,
                                'seq_idx': seq_idx
                            })
        
        return pattern_data
    
    def visualize_pattern_character_associations(self, data: List[Dict], layer: int = 0):
        """Create heatmaps of pattern activations for different characters at each level"""
        # Get number of levels
        levels = sorted(set(d['level'] for d in data if d['layer'] == layer))
        
        fig, axs = plt.subplots(len(levels), 1, figsize=(12, 6*len(levels)))
        if len(levels) == 1:
            axs = [axs]
            
        for level_idx, level in enumerate(levels):
            # Filter data for specified layer and level
            level_data = [d for d in data if d['layer'] == layer and d['level'] == level]
            
            # Create character-pattern activation matrix
            chars = sorted(set(d['char'] for d in level_data))
            n_patterns = len(level_data[0]['pattern_activations'])
            
            activation_matrix = np.zeros((len(chars), n_patterns))
            counts = np.zeros(len(chars))
            
            for d in level_data:
                char_idx = chars.index(d['char'])
                activation_matrix[char_idx] += d['pattern_activations']
                counts[char_idx] += 1
            
            # Average activations
            activation_matrix = activation_matrix / counts[:, np.newaxis]
            
            # Create heatmap
            sns.heatmap(activation_matrix, 
                       xticklabels=[f'P{i}' for i in range(n_patterns)],
                       yticklabels=chars,
                       cmap='viridis',
                       ax=axs[level_idx])
            axs[level_idx].set_title(f'Layer {layer} Level {level} Pattern-Character Associations')
            axs[level_idx].set_xlabel('Pattern')
            axs[level_idx].set_ylabel('Character')
        
        plt.tight_layout()
    
    def visualize_sequence_patterns(self, data: List[Dict], seq_idx: int = 0, layer: int = 0):
        """Visualize pattern activations across a sequence for each level"""
        levels = sorted(set(d['level'] for d in data if d['layer'] == layer))
        
        fig, axs = plt.subplots(len(levels), 1, figsize=(15, 6*len(levels)))
        if len(levels) == 1:
            axs = [axs]
            
        for level_idx, level in enumerate(levels):
            # Filter data for specific sequence, layer, and level
            level_data = [d for d in data 
                         if d['layer'] == layer and 
                         d['level'] == level and 
                         d['seq_idx'] == seq_idx]
            seq = level_data[0]['sequence']
            
            # Create pattern activation matrix
            n_patterns = len(level_data[0]['pattern_activations'])
            seq_length = len(seq)
            pattern_matrix = np.zeros((n_patterns, seq_length))
            
            for d in level_data:
                pos = d['position']
                pattern_matrix[:, pos] = d['pattern_activations']
            
            # Plot
            im = axs[level_idx].imshow(pattern_matrix, aspect='auto', cmap='viridis')
            plt.colorbar(im, ax=axs[level_idx], label='Pattern Activation')
            axs[level_idx].set_title(f'Layer {layer} Level {level} Pattern Activations')
            axs[level_idx].set_xlabel('Position in Sequence')
            axs[level_idx].set_ylabel('Pattern Index')
            
            # Add sequence text
            axs[level_idx].set_xticks(range(len(seq)))
            axs[level_idx].set_xticklabels(list(seq), rotation=45)
        
        plt.tight_layout()
    
    def analyze_word_boundary_patterns(self, data: List[Dict], layer: int = 0):
        """Analyze pattern activations around word boundaries for each level"""
        levels = sorted(set(d['level'] for d in data if d['layer'] == layer))
        
        fig, axs = plt.subplots(len(levels), 1, figsize=(10, 6*len(levels)))
        if len(levels) == 1:
            axs = [axs]
            
        for level_idx, level in enumerate(levels):
            # Filter data for specified layer and level
            level_data = [d for d in data if d['layer'] == layer and d['level'] == level]
            
            # Find space characters (word boundaries)
            boundary_activations = []
            non_boundary_activations = []
            
            for d in level_data:
                if d['char'] == ' ':
                    boundary_activations.append(d['pattern_activations'])
                else:
                    non_boundary_activations.append(d['pattern_activations'])
            
            # Convert to arrays
            boundary_activations = np.array(boundary_activations)
            non_boundary_activations = np.array(non_boundary_activations)
            
            # Plot comparison
            n_patterns = boundary_activations.shape[1]
            
            axs[level_idx].bar(np.arange(n_patterns) - 0.2, 
                             np.mean(boundary_activations, axis=0), 
                             width=0.4, 
                             label='Word Boundary')
            axs[level_idx].bar(np.arange(n_patterns) + 0.2, 
                             np.mean(non_boundary_activations, axis=0), 
                             width=0.4, 
                             label='Non-Boundary')
            
            axs[level_idx].set_title(f'Layer {layer} Level {level} Pattern Activation at Word Boundaries')
            axs[level_idx].set_xlabel('Pattern Index')
            axs[level_idx].set_ylabel('Average Activation')
            axs[level_idx].legend()
            axs[level_idx].set_xticks(range(n_patterns))
        
        plt.tight_layout()

def analyze_hierarchical_patterns(model, n_samples=100):
    """Run complete pattern analysis on hierarchical network"""
    from data_generators import generate_language_data
    
    # Create timestamped directory for visualizations
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    viz_dir = f"hierarchical_viz_{timestamp}"
    os.makedirs(viz_dir, exist_ok=True)
    
    # Generate sample sequences
    X, _ = generate_language_data(n_samples=n_samples)
    
    # Convert tensor data back to sequences
    analyzer = HierarchicalPatternAnalyzer(model)
    sequences = []
    for x in X:
        seq = ''.join([analyzer.idx_to_char[int(idx.item())] for idx in x])
        sequences.append(seq)
    
    # Run analysis
    data = analyzer.analyze_pattern_activations(sequences)
    
    # Generate visualizations
    for layer in range(len(model.layers)):
        plt.figure()
        analyzer.visualize_pattern_character_associations(data, layer=layer)
        plt.savefig(os.path.join(viz_dir, f'layer_{layer}_char_patterns.png'))
        plt.close()
        
        plt.figure()
        analyzer.visualize_sequence_patterns(data, layer=layer)
        plt.savefig(os.path.join(viz_dir, f'layer_{layer}_sequence_patterns.png'))
        plt.close()
        
        plt.figure()
        analyzer.analyze_word_boundary_patterns(data, layer=layer)
        plt.savefig(os.path.join(viz_dir, f'layer_{layer}_word_boundaries.png'))
        plt.close()
    
    print(f"Visualizations saved in directory: {viz_dir}")
    
    return analyzer, data