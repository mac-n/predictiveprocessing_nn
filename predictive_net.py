import torch
import torch.nn as nn
import torch.nn.functional as F

class PredictiveLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, next_dim, penultimate_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.next_dim = next_dim
        self.penultimate_dim = penultimate_dim
        
        # Main processing pathway
        self.process = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Predict next layer's transformation
        self.predict_next = nn.Linear(hidden_dim, next_dim)  # Now predicts correct dimension
        
        # Path to penultimate layer
        self.to_penultimate = nn.Linear(hidden_dim, penultimate_dim)
        
    def forward(self, x, next_layer=None):
        # Process input
        hidden = self.process(x)
        
        if next_layer is not None:
            # Predict next layer's transformation
            predicted_next = self.predict_next(hidden)
            
            # Get actual next layer transformation
            with torch.no_grad():
                actual_next = next_layer.process(hidden)
            
            # Prediction error (per sample)
            pred_error = torch.mean((actual_next - predicted_next)**2, dim=1, keepdim=True)
            
            # Route based on prediction accuracy
            confidence = torch.sigmoid(-pred_error)  # High when error is low
            
            # Well-predicted information goes to penultimate
            penultimate_features = self.to_penultimate(hidden)
            penultimate_contribution = penultimate_features * confidence
            
            # Poorly-predicted information continues upward
            continue_up = hidden * (1 - confidence)
            
            return continue_up, penultimate_contribution, pred_error
        else:
            # Last layer just contributes to penultimate
            return None, self.to_penultimate(hidden), None

class PredictiveNet(nn.Module):
    def __init__(self, input_dim, hidden_dims, penultimate_dim, output_dim):
        super().__init__()
        
        self.layers = nn.ModuleList()
        current_dim = input_dim
        
        # Create layers with correct next dimensions
        for i, hidden_dim in enumerate(hidden_dims):
            next_dim = hidden_dims[i + 1] if i < len(hidden_dims) - 1 else penultimate_dim
            layer = PredictiveLayer(current_dim, hidden_dim, next_dim, penultimate_dim)
            self.layers.append(layer)
            current_dim = hidden_dim
        
        # Final output layer
        self.final = nn.Linear(penultimate_dim, output_dim)
    
    def forward(self, x):
        penultimate_contributions = []
        current = x
        all_errors = []
        
        # Process through layers
        for i, layer in enumerate(self.layers):
            next_layer = self.layers[i+1] if i < len(self.layers)-1 else None
            current, penultimate, error = layer(current, next_layer)
            
            if error is not None:
                all_errors.append(error)
            penultimate_contributions.append(penultimate)
        
        # Combine all contributions in penultimate layer
        penultimate = torch.sum(torch.stack(penultimate_contributions), dim=0)
        
        # Final output
        output = self.final(penultimate)
        
        # Return both outputs and mean prediction error
        return output, torch.cat(all_errors, dim=1) if all_errors else None