import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class LayerStats:
    """Track statistics for a single layer during forward pass"""
    prediction_errors: torch.Tensor  # [batch_size, 1]
    confidence_values: torch.Tensor  # [batch_size, 1]
    penultimate_magnitude: torch.Tensor  # Average magnitude of penultimate contribution
    continue_magnitude: torch.Tensor    # Average magnitude of continue_up values
    layer_idx: int

class PredictiveLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, next_dim, penultimate_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.compressed_dim = max(next_dim // 4, 8)  # Ensure minimum size
        
        # Main processing pathway
        self.process = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Lightweight prediction pathway
        self.predict_next = nn.Linear(hidden_dim, self.compressed_dim)
        self.compress_next = nn.Linear(next_dim, self.compressed_dim)
        
        # Path to penultimate layer
        self.to_penultimate = nn.Linear(hidden_dim, penultimate_dim)
        
        # For tracking statistics
        self.last_stats: Optional[LayerStats] = None
        
    def forward(self, x: torch.Tensor, next_layer: Optional['PredictiveLayer'], layer_idx: int) -> Tuple[Optional[torch.Tensor], torch.Tensor, Optional[torch.Tensor]]:
        # Process input
        hidden = self.process(x)
        
        if next_layer is not None:
            # Predict next layer's compressed transformation
            predicted_next = self.predict_next(hidden)
            
            # Get actual next layer transformation (compressed)
            with torch.no_grad():
                actual_next = next_layer.process(hidden)
                compressed_next = self.compress_next(actual_next)
            
            # Prediction error (per sample)
            pred_error = torch.mean((compressed_next - predicted_next)**2, dim=1, keepdim=True)
            
            # Route based on prediction accuracy
            # Dynamic temperature based on prediction certainty
            pred_certainty = torch.abs(pred_error - torch.mean(pred_error))
            temperature = torch.tanh(pred_certainty)
            
            # Route based on prediction accuracy using logit scaling
            scaled_error = -pred_error * temperature
            confidence = 0.5 * (torch.tanh(scaled_error) + 1)  # logit-like but bounded 0-1
            
            # Add routing cost - penalize using both paths
            routing_balance = confidence * (1 - confidence)  # High when routing is ambiguous
            routing_cost = 0.1 * torch.mean(routing_balance)
            pred_error = pred_error + routing_cost
            
            # Route information based on prediction accuracy
            penultimate_features = self.to_penultimate(hidden)
            penultimate_contribution = penultimate_features * confidence
            continue_up = hidden * (1 - confidence)
            
            # Track statistics
            self.last_stats = LayerStats(
                prediction_errors=pred_error.detach(),
                confidence_values=confidence.detach(),
                penultimate_magnitude=torch.mean(torch.norm(penultimate_contribution.detach(), dim=1)),
                continue_magnitude=torch.mean(torch.norm(continue_up.detach(), dim=1)),
                layer_idx=layer_idx
            )
            
            return continue_up, penultimate_contribution, pred_error
        else:
            # Last layer just contributes to penultimate
            penultimate_contribution = self.to_penultimate(hidden)
            
            # Track statistics (no prediction for last layer)
            self.last_stats = LayerStats(
                prediction_errors=torch.zeros(1, 1, device=x.device),
                confidence_values=torch.ones(1, 1, device=x.device),
                penultimate_magnitude=torch.mean(torch.norm(penultimate_contribution.detach(), dim=1)),
                continue_magnitude=torch.tensor(0.0, device=x.device),
                layer_idx=layer_idx
            )
            
            return None, penultimate_contribution, None

class PredictiveNet(nn.Module):
    def __init__(self, input_dim, hidden_dims, penultimate_dim, output_dim):
        super().__init__()
        
        self.layers = nn.ModuleList()
        current_dim = input_dim
        
        # Create layers
        for i, hidden_dim in enumerate(hidden_dims):
            next_dim = hidden_dims[i + 1] if i < len(hidden_dims) - 1 else penultimate_dim
            layer = PredictiveLayer(current_dim, hidden_dim, next_dim, penultimate_dim)
            self.layers.append(layer)
            current_dim = hidden_dim
        
        # Final output layer
        self.final = nn.Linear(penultimate_dim, output_dim)
    
    def get_layer_stats(self) -> List[LayerStats]:
        """Return statistics from the last forward pass for all layers"""
        return [layer.last_stats for layer in self.layers if layer.last_stats is not None]
    
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