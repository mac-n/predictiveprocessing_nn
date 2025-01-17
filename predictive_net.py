import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class LayerStats:
    """Track statistics for a single layer during forward pass"""
    prediction_errors: torch.Tensor
    confidence_values: torch.Tensor
    penultimate_magnitude: torch.Tensor
    continue_magnitude: torch.Tensor
    layer_idx: int
    pattern_usage: torch.Tensor  # Track which patterns are being used
    
class PatternPredictiveLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, next_dim, penultimate_dim, n_patterns=12):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.next_dim = next_dim
        self.n_patterns = n_patterns
        
        # Main processing pathway
        self.process = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Pattern-based compression for current layer's hidden state
        self.pattern_dict = nn.Parameter(torch.randn(n_patterns, hidden_dim) / hidden_dim**0.5)
        self.pattern_attention = nn.Linear(hidden_dim, n_patterns)
        
        # Separate pattern-based compression for predicting next layer's hidden state
        self.next_pattern_dict = nn.Parameter(torch.randn(n_patterns, next_dim) / next_dim**0.5)
        self.next_pattern_attention = nn.Linear(hidden_dim, n_patterns)
        
        # Path to penultimate layer
        self.to_penultimate = nn.Linear(hidden_dim, penultimate_dim)
        
        # For tracking statistics
        self.last_stats: Optional[LayerStats] = None
    
    def compress_activity(self, x: torch.Tensor, is_next_layer: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compress activity using pattern dictionary"""
        # Ensure x has the right dimensionality for attention
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        # Choose appropriate attention and patterns based on whether we're compressing next layer
        if is_next_layer:
            attention = self.next_pattern_attention
            patterns = self.next_pattern_dict
        else:
            attention = self.pattern_attention
            patterns = self.pattern_dict
        
        # Compute attention weights
        attn = attention(x)
        pattern_weights = F.softmax(attn, dim=-1)
        
        # Compress using weighted combination of patterns
        compressed = pattern_weights @ patterns
        return compressed, pattern_weights
    
    def forward(self, x: torch.Tensor, next_layer: Optional['PatternPredictiveLayer'], layer_idx: int) -> Tuple[Optional[torch.Tensor], torch.Tensor, Optional[torch.Tensor]]:
        # Ensure x is 2D tensor
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        # Process input
        hidden = self.process(x)
        
        if next_layer is not None:
            # Compress current layer's hidden state and predict next layer
            my_compressed, my_patterns = self.compress_activity(hidden, is_next_layer=False)
            predicted_next = my_compressed  # Using compressed representation as prediction
            
            # Get actual next layer transformation
            with torch.no_grad():
                actual_next = next_layer.process(hidden)
                compressed_next, _ = next_layer.compress_activity(actual_next, is_next_layer=True)
            
            # Ensure consistent dimensions for prediction error
            # If dimensions differ, pad or truncate
            min_dim = min(predicted_next.size(1), compressed_next.size(1))
            predicted_next = predicted_next[:, :min_dim]
            compressed_next = compressed_next[:, :min_dim]
            
            # Prediction error (per sample)
            pred_error = torch.mean((compressed_next - predicted_next)**2, dim=1, keepdim=True)
            
            # Route based on prediction accuracy
            # Dynamic temperature based on prediction certainty
            pred_certainty = torch.abs(pred_error - torch.mean(pred_error))
            temperature = torch.sigmoid(pred_certainty)
            
            # Route based on prediction accuracy using tanh scaling
            scaled_error = -pred_error * temperature
            confidence = 0.5 * (torch.tanh(scaled_error) + 1)
            
            # Add routing cost - penalize using both paths
            routing_balance = confidence * (1 - confidence)
            routing_cost = 0.1 * torch.mean(routing_balance)
            pred_error = pred_error + routing_cost
            
            # Route information based on prediction accuracy
            # Higher confidence (better prediction) -> more to penultimate
            # Lower confidence (worse prediction) -> more continuing up
            penultimate_features = self.to_penultimate(hidden)
            penultimate_contribution = penultimate_features * confidence
            continue_up = hidden * (1 - confidence)
            
            # Track statistics
            self.last_stats = LayerStats(
                prediction_errors=pred_error.detach(),
                confidence_values=confidence.detach(),
                penultimate_magnitude=torch.mean(torch.norm(penultimate_contribution.detach(), dim=1)),
                continue_magnitude=torch.mean(torch.norm(continue_up.detach(), dim=1)),
                layer_idx=layer_idx,
                pattern_usage=my_patterns.detach().mean(0)  # Average pattern usage across batch
            )
            
            return continue_up, penultimate_contribution, pred_error
        else:
            # Last layer just contributes to penultimate
            penultimate_contribution = self.to_penultimate(hidden)
            
            # Get pattern usage for last layer
            _, my_patterns = self.compress_activity(hidden, is_next_layer=False)
            
            # Track statistics
            self.last_stats = LayerStats(
                prediction_errors=torch.zeros(1, 1, device=x.device),
                confidence_values=torch.ones(1, 1, device=x.device),
                penultimate_magnitude=torch.mean(torch.norm(penultimate_contribution.detach(), dim=1)),
                continue_magnitude=torch.tensor(0.0, device=x.device),
                layer_idx=layer_idx,
                pattern_usage=my_patterns.detach().mean(0)
            )
            
            return None, penultimate_contribution, None

class PatternPredictiveNet(nn.Module):
    def __init__(self, input_dim, hidden_dims, penultimate_dim, output_dim, n_patterns=8):
        super().__init__()
        
        self.layers = nn.ModuleList()
        current_dim = input_dim
        
        # Create layers
        for i, hidden_dim in enumerate(hidden_dims):
            # Determine next dimension (for prediction)
            next_dim = hidden_dims[i + 1] if i < len(hidden_dims) - 1 else penultimate_dim
            
            layer = PatternPredictiveLayer(
                input_dim=current_dim, 
                hidden_dim=hidden_dim, 
                next_dim=next_dim, 
                penultimate_dim=penultimate_dim,
                n_patterns=n_patterns
            )
            self.layers.append(layer)
            current_dim = hidden_dim
        
        # Final output layer
        self.final = nn.Linear(penultimate_dim, output_dim)
    
    def get_layer_stats(self) -> List[LayerStats]:
        """Return statistics from the last forward pass for all layers"""
        return [layer.last_stats for layer in self.layers if layer.last_stats is not None]
    
    def get_pattern_correlations(self) -> List[torch.Tensor]:
        """Get correlation matrices between patterns in each layer"""
        pattern_correlations = []
        for layer in self.layers:
            patterns = layer.pattern_dict
            # Normalize patterns
            normalized = F.normalize(patterns, dim=1)
            # Get correlation matrix
            corr = normalized @ normalized.T
            pattern_correlations.append(corr)
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
        
        # Combine all contributions in penultimate layer
        penultimate = torch.sum(torch.stack(penultimate_contributions), dim=0)
        
        # Final output
        output = self.final(penultimate)
        
        return output, torch.cat(all_errors, dim=1) if all_errors else None