import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple

class LanguagePatternAnalyzer:
    def __init__(self, model):
        self.model = model
        self.idx_to_char = {0: ' '} # Space character
        for i, c in enumerate('abcdefghijklmnopqrstuvwxyz'):
            self.idx_to_char[i + 1] = c
        self.char_to_idx = {v: k for k, v in self.idx_to_char.items()}
    
    def analyze_pattern_activations(self, sequences: List[str], window_size: int = 20) -> Dict:
        """Analyze pattern activations for character sequences"""
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
                    
                    # Store data
                    for layer_idx, stat in enumerate(stats):
                        pattern_data.append({
                            'layer': layer_idx,
                            'position': i + window_size - 1,
                            'char': seq[i + window_size - 1],
                            'pattern_activations': stat.pattern_usage.cpu().numpy(),
                            'sequence': seq,
                            'seq_idx': seq_idx
                        })
        
        return pattern_data
    
    def visualize_pattern_character_associations(self, data: List[Dict], layer: int = 0):
        """Create heatmap of pattern activations for different characters"""
        # Filter data for specified layer
        layer_data = [d for d in data if d['layer'] == layer]
        
        # Create character-pattern activation matrix
        chars = sorted(set(d['char'] for d in layer_data))
        n_patterns = len(layer_data[0]['pattern_activations'])
        
        activation_matrix = np.zeros((len(chars), n_patterns))
        counts = np.zeros(len(chars))
        
        for d in layer_data:
            char_idx = chars.index(d['char'])
            activation_matrix[char_idx] += d['pattern_activations']
            counts[char_idx] += 1
        
        # Average activations
        activation_matrix = activation_matrix / counts[:, np.newaxis]
        
        # Create heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(activation_matrix, 
                   xticklabels=[f'P{i}' for i in range(n_patterns)],
                   yticklabels=chars,
                   cmap='viridis')
        plt.title(f'Layer {layer} Pattern-Character Associations')
        plt.xlabel('Pattern')
        plt.ylabel('Character')
    
    def visualize_sequence_patterns(self, data: List[Dict], seq_idx: int = 0, layer: int = 0):
        """Visualize pattern activations across a sequence"""
        # Filter data for specific sequence and layer
        seq_data = [d for d in data 
                   if d['layer'] == layer and d['seq_idx'] == seq_idx]
        seq = seq_data[0]['sequence']
        
        # Create pattern activation matrix
        n_patterns = len(seq_data[0]['pattern_activations'])
        seq_length = len(seq)
        pattern_matrix = np.zeros((n_patterns, seq_length))
        
        for d in seq_data:
            pos = d['position']
            pattern_matrix[:, pos] = d['pattern_activations']
        
        # Plot
        plt.figure(figsize=(15, 6))
        plt.imshow(pattern_matrix, aspect='auto', cmap='viridis')
        plt.colorbar(label='Pattern Activation')
        plt.title(f'Layer {layer} Pattern Activations Across Sequence')
        plt.xlabel('Position in Sequence')
        plt.ylabel('Pattern Index')
        
        # Add sequence text
        plt.xticks(range(len(seq)), list(seq), rotation=45)
    
    def analyze_word_boundary_patterns(self, data: List[Dict], layer: int = 0):
        """Analyze pattern activations around word boundaries"""
        # Filter data for specified layer
        layer_data = [d for d in data if d['layer'] == layer]
        
        # Find space characters (word boundaries)
        boundary_activations = []
        non_boundary_activations = []
        
        for d in layer_data:
            if d['char'] == ' ':
                boundary_activations.append(d['pattern_activations'])
            else:
                non_boundary_activations.append(d['pattern_activations'])
        
        # Convert to arrays
        boundary_activations = np.array(boundary_activations)
        non_boundary_activations = np.array(non_boundary_activations)
        
        # Plot comparison
        plt.figure(figsize=(10, 6))
        n_patterns = boundary_activations.shape[1]
        
        plt.bar(np.arange(n_patterns) - 0.2, 
               np.mean(boundary_activations, axis=0), 
               width=0.4, 
               label='Word Boundary')
        plt.bar(np.arange(n_patterns) + 0.2, 
               np.mean(non_boundary_activations, axis=0), 
               width=0.4, 
               label='Non-Boundary')
        
        plt.title(f'Layer {layer} Pattern Activation at Word Boundaries')
        plt.xlabel('Pattern Index')
        plt.ylabel('Average Activation')
        plt.legend()
        plt.xticks(range(n_patterns))

def analyze_language_patterns(model, n_samples=100):
    """Run complete pattern analysis on language data"""
    from data_generators import generate_language_data
    
    # Generate sample sequences
    X, _ = generate_language_data(n_samples=n_samples)
    
    # Convert tensor data back to sequences
    analyzer = LanguagePatternAnalyzer(model)
    sequences = []
    for x in X:
        seq = ''.join([analyzer.idx_to_char[int(idx.item())] for idx in x])
        sequences.append(seq)
    
    # Run analysis
    data = analyzer.analyze_pattern_activations(sequences)
    
    # Generate visualizations
    analyzer.visualize_pattern_character_associations(data, layer=0)
    plt.savefig('char_pattern_associations.png')
    
    analyzer.visualize_sequence_patterns(data, layer=0)
    plt.savefig('sequence_patterns.png')
    
    analyzer.analyze_word_boundary_patterns(data, layer=0)
    plt.savefig('word_boundary_patterns.png')
    
    return analyzer, data