import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class HierarchicalLayerStats:
    """Extended stats to track hierarchy"""
    prediction_errors: torch.Tensor
    confidence_values: torch.Tensor
    penultimate_magnitude: torch.Tensor
    continue_magnitude: torch.Tensor
    layer_idx: int
    pattern_usage_per_level: List[torch.Tensor]  # Track patterns at each level
    compression_ratios: List[float]  # Track how much each level compresses


class PatternDictionary(nn.Module):
    """A module to hold a single level's patterns"""
    def __init__(self, n_patterns, dim):
        super().__init__()
        self.patterns = nn.Parameter(torch.randn(n_patterns, dim) / dim**0.5)
        
    def forward(self, x=None):
        return self.patterns

class HierarchicalPatternLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, next_dim, penultimate_dim, 
                 patterns_per_level=4, n_levels=2, compression_factor=2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.next_dim = next_dim
        self.n_levels = n_levels
        self.compression_factor = compression_factor
        
        # Main processing pathway
        self.process = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Create hierarchical pattern dictionaries
        self.level_dims = [hidden_dim // (compression_factor ** i) for i in range(n_levels)]
        
        # Pattern dictionaries for each level
        self.pattern_dicts = nn.ModuleList([
            PatternDictionary(patterns_per_level, dim)
            for dim in self.level_dims
        ])
        
        self.next_pattern_dicts = nn.ModuleList([
            PatternDictionary(patterns_per_level, next_dim)
            for _ in range(n_levels)
        ])
        
        # Attention mechanisms for each level
        self.pattern_attentions = nn.ModuleList([
            nn.Linear(dim, patterns_per_level)
            for dim in self.level_dims
        ])
        
        # Dimension reduction between levels
        self.level_reducers = nn.ModuleList([
            nn.Linear(self.level_dims[i], self.level_dims[i+1])
            for i in range(n_levels-1)
        ])
        
        # Path to penultimate layer
        self.to_penultimate = nn.Linear(sum(self.level_dims), penultimate_dim)
        
        self.last_stats: Optional[HierarchicalLayerStats] = None
        
    def compress_activity(self, x: torch.Tensor, 
                         is_next_layer: bool = False) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Hierarchical pattern compression"""
        if x.dim() == 1:
            x = x.unsqueeze(0)
            
        all_compressed = []
        all_pattern_weights = []
        current = x
        
        pattern_dicts = self.next_pattern_dicts if is_next_layer else self.pattern_dicts
        
        for level in range(self.n_levels):
            # Get patterns for this level
            patterns = pattern_dicts[level]()
            
            # Get pattern attention weights
            attn = self.pattern_attentions[level](current)
            pattern_weights = F.gumbel_softmax(attn, tau=0.1, hard=True, dim=-1)
            
            # Compress using patterns
            compressed = pattern_weights @ patterns
            
            all_compressed.append(compressed)
            all_pattern_weights.append(pattern_weights)
            
            # Prepare input for next level if needed
            if level < self.n_levels - 1:
                # Ensure the compressed tensor is the right shape for the next reducer
                next_level_dim = self.level_dims[level+1]
                if compressed.size(1) != next_level_dim:
                    # Use a projection to match the required dimension
                    proj = nn.Linear(compressed.size(1), next_level_dim).to(compressed.device)
                    current = proj(compressed)
                else:
                    current = compressed
            
        # Concatenate all levels of compression
        final_compressed = torch.cat(all_compressed, dim=1)
        return final_compressed, all_pattern_weights

    def forward(self, x: torch.Tensor, 
                next_layer: Optional['HierarchicalPatternLayer'], 
                layer_idx: int) -> Tuple[Optional[torch.Tensor], torch.Tensor, Optional[torch.Tensor]]:
        if x.dim() == 1:
            x = x.unsqueeze(0)
            
        # Process input
        hidden = self.process(x)
        
        if next_layer is not None:
            # Compress current layer's activity and predict next layer
            my_compressed, my_patterns = self.compress_activity(hidden, is_next_layer=False)
            predicted_next = my_compressed
            
            # Get actual next layer transformation
            with torch.no_grad():
                actual_next = next_layer.process(hidden)
                compressed_next, _ = next_layer.compress_activity(actual_next, is_next_layer=True)
            
            # Match dimensions
            min_dim = min(predicted_next.size(1), compressed_next.size(1))
            predicted_next = predicted_next[:, :min_dim]
            compressed_next = compressed_next[:, :min_dim]
            
            # Prediction error and routing
            pred_error = torch.mean((compressed_next - predicted_next)**2, dim=1, keepdim=True)
            pred_certainty = torch.abs(pred_error - torch.mean(pred_error))
            temperature = torch.sigmoid(pred_certainty)
            
            scaled_error = -pred_error * temperature
            confidence = 0.5 * (torch.tanh(scaled_error) + 1)
            
            # Add routing cost
            routing_balance = confidence * (1 - confidence)
            routing_cost = 0.1 * torch.mean(routing_balance)
            pred_error = pred_error + routing_cost
            
            # Route information based on prediction accuracy
            penultimate_features = self.to_penultimate(my_compressed)
            penultimate_contribution = penultimate_features * confidence
            continue_up = hidden * (1 - confidence)
            
            # Calculate compression ratios
            compression_ratios = [
                input_dim / output_dim 
                for input_dim, output_dim in zip(self.level_dims[:-1], self.level_dims[1:])
            ]
            
            # Track statistics
            self.last_stats = HierarchicalLayerStats(
                prediction_errors=pred_error.detach(),
                confidence_values=confidence.detach(),
                penultimate_magnitude=torch.mean(torch.norm(penultimate_contribution.detach(), dim=1)),
                continue_magnitude=torch.mean(torch.norm(continue_up.detach(), dim=1)),
                layer_idx=layer_idx,
                pattern_usage_per_level=[p.detach().mean(0) for p in my_patterns],
                compression_ratios=compression_ratios
            )
            
            return continue_up, penultimate_contribution, pred_error
            
        else:
            # Last layer
            my_compressed, my_patterns = self.compress_activity(hidden, is_next_layer=False)
            penultimate_contribution = self.to_penultimate(my_compressed)
            
            compression_ratios = [
                input_dim / output_dim 
                for input_dim, output_dim in zip(self.level_dims[:-1], self.level_dims[1:])
            ]
            
            self.last_stats = HierarchicalLayerStats(
                prediction_errors=torch.zeros(1, 1, device=x.device),
                confidence_values=torch.ones(1, 1, device=x.device),
                penultimate_magnitude=torch.mean(torch.norm(penultimate_contribution.detach(), dim=1)),
                continue_magnitude=torch.tensor(0.0, device=x.device),
                layer_idx=layer_idx,
                pattern_usage_per_level=[p.detach().mean(0) for p in my_patterns],
                compression_ratios=compression_ratios
            )
            
            return None, penultimate_contribution, None

class HierarchicalPatternPredictiveNet(nn.Module):
    def __init__(self, input_dim, hidden_dims, penultimate_dim, output_dim,
                 patterns_per_level=4, n_levels=2, compression_factor=2):
        super().__init__()
        
        self.layers = nn.ModuleList()
        current_dim = input_dim
        
        # Create layers
        for i, hidden_dim in enumerate(hidden_dims):
            # Determine next dimension
            next_dim = hidden_dims[i + 1] if i < len(hidden_dims) - 1 else penultimate_dim
            
            layer = HierarchicalPatternLayer(
                input_dim=current_dim,
                hidden_dim=hidden_dim,
                next_dim=next_dim,
                penultimate_dim=penultimate_dim,
                patterns_per_level=patterns_per_level,
                n_levels=n_levels,
                compression_factor=compression_factor
            )
            self.layers.append(layer)
            current_dim = hidden_dim
        
        self.final = nn.Linear(penultimate_dim, output_dim)
    
    def get_layer_stats(self) -> List[HierarchicalLayerStats]:
        """Return hierarchical statistics from all layers"""
        return [layer.last_stats for layer in self.layers if layer.last_stats is not None]
    
    def get_pattern_correlations(self) -> List[List[torch.Tensor]]:
        """Get correlation matrices between patterns at each level for each layer"""
        pattern_correlations = []
        for layer in self.layers:
            layer_correlations = []
            for pattern_dict in layer.pattern_dicts:
                # Get the patterns from the PatternDictionary module
                patterns = pattern_dict.patterns
                # Normalize patterns
                normalized = F.normalize(patterns, dim=1)
                # Get correlation matrix
                corr = normalized @ normalized.T
                layer_correlations.append(corr)
            pattern_correlations.append(layer_correlations)
        return pattern_correlations
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        penultimate_contributions = []
        current = x
        all_errors = []
        
        # Process through layers
        for i, layer in enumerate(self.layers):
            next_layer = self.layers[i+1] if i < len(self.layers)-1 else None
            current, penultimate, error = layer(current, next_layer, i)
            
            if error is not None:
                all_errors.append(error)
            penultimate_contributions.append(penultimate)
        
        # Combine penultimate contributions
        penultimate = torch.sum(torch.stack(penultimate_contributions), dim=0)
        
        # Final output
        output = self.final(penultimate)
        
        return output, torch.cat(all_errors, dim=1) if all_errors else None