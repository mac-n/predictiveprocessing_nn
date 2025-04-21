import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
from sklearn.cluster import KMeans

class PatternAnalyzer:
    def __init__(self, model):
        self.model = model
        self.pattern_history = []  # Track pattern usage over time
        self.confidence_history = []  # Track confidence values
        self.flow_history = []  # Track information flow
        
    def record_step(self, layer_stats: List):
        """Record a single step of pattern usage and routing"""
        pattern_usage = {}
        confidence_vals = {}
        flows = {}
        
        for stat in layer_stats:
            layer_idx = stat.layer_idx
            pattern_usage[layer_idx] = stat.pattern_usage.cpu().numpy()
            confidence_vals[layer_idx] = stat.confidence_values.mean().cpu().item()
            flows[layer_idx] = {
                'penult': stat.penultimate_magnitude.cpu().item(),
                'continue': stat.continue_magnitude.cpu().item()
            }
        
        self.pattern_history.append(pattern_usage)
        self.confidence_history.append(confidence_vals)
        self.flow_history.append(flows)
    
    def plot_pattern_evolution(self, layer_idx: int):
        """Plot how pattern usage evolves over time for a specific layer"""
        patterns = np.array([h[layer_idx] for h in self.pattern_history])
        
        plt.figure(figsize=(12, 6))
        plt.subplot(2, 1, 1)
        
        # Pattern usage heatmap
        plt.imshow(patterns.T, aspect='auto', cmap='viridis')
        plt.colorbar(label='Pattern Activation')
        plt.ylabel('Pattern Index')
        plt.title(f'Pattern Usage Evolution - Layer {layer_idx}')
        
        # Confidence and flow subplot
        plt.subplot(2, 1, 2)
        confidences = [h[layer_idx] for h in self.confidence_history]
        penult_flows = [h[layer_idx]['penult'] for h in self.flow_history]
        cont_flows = [h[layer_idx]['continue'] for h in self.flow_history]
        
        plt.plot(confidences, label='Confidence', color='blue')
        plt.plot(penult_flows, label='To Penultimate', color='green')
        plt.plot(cont_flows, label='Continue Up', color='red')
        plt.xlabel('Time Step')
        plt.ylabel('Magnitude')
        plt.legend()
        plt.tight_layout()
    
    def analyze_pattern_correlations(self):
        """Analyze how patterns relate to each other within and across layers"""
        correlations = self.model.get_pattern_correlations()
        
        n_layers = len(correlations)
        plt.figure(figsize=(4*n_layers, 4))
        
        for i, corr in enumerate(correlations):
            plt.subplot(1, n_layers, i+1)
            plt.imshow(corr.detach().cpu().numpy(), cmap='coolwarm', vmin=-1, vmax=1)
            plt.colorbar()
            plt.title(f'Layer {i} Pattern\nCorrelations')
        plt.tight_layout()
    
    def pattern_phase_analysis(self, inputs: torch.Tensor, n_clusters: int = 3):
        """Analyze which patterns are active in different phases of the input"""
        # Get pattern activations for all inputs
        pattern_activations = []
        self.model.eval()
        with torch.no_grad():
            for x in inputs:
                # Forward pass
                self.model(x.unsqueeze(0))
                # Get layer stats
                stats = self.model.get_layer_stats()
                # Record pattern usage
                layer_patterns = {stat.layer_idx: stat.pattern_usage.cpu().numpy() 
                                for stat in stats}
                pattern_activations.append(layer_patterns)
        
        # Cluster inputs based on pattern usage
        for layer_idx in range(len(self.model.layers)):
            layer_patterns = np.array([p[layer_idx] for p in pattern_activations])
            
            # Cluster pattern activations
            kmeans = KMeans(n_clusters=n_clusters)
            clusters = kmeans.fit_predict(layer_patterns)
            
            # Plot results
            plt.figure(figsize=(12, 4))
            
            # Pattern usage by cluster
            for i in range(n_clusters):
                cluster_patterns = layer_patterns[clusters == i].mean(axis=0)
                plt.bar(np.arange(len(cluster_patterns)) + i*0.25, 
                       cluster_patterns, 
                       width=0.25,
                       label=f'Cluster {i}')
            
            plt.xlabel('Pattern Index')
            plt.ylabel('Average Activation')
            plt.title(f'Layer {layer_idx} Pattern Usage by Cluster')
            plt.legend()
            plt.tight_layout()


def visualize_lorenz_patterns(data_generator, model, n_steps=1000):
    """Visualize pattern usage on Lorenz attractor"""
    # Generate Lorenz data
    X, y = data_generator(n_samples=n_steps)
    
    # Create analyzer
    analyzer = PatternAnalyzer(model)
    
    # Process sequence
    model.eval()
    with torch.no_grad():
        for i in range(len(X)):
            _, _ = model(X[i:i+1])
            analyzer.record_step(model.get_layer_stats())
    
    # Plot pattern evolution for each layer
    for layer_idx in range(len(model.layers)):
        analyzer.plot_pattern_evolution(layer_idx)
    
    # Plot pattern correlations
    analyzer.analyze_pattern_correlations()
    
    # Analyze patterns in different phases
    analyzer.pattern_phase_analysis(X, n_clusters=3)
    
    return analyzer