import torch
import torch.nn as nn
import torch.nn.functional as F

class PredictiveLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, next_dim, penultimate_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Main processing pathway
        self.process = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Predict next layer's transformation
        self.predict_next = nn.Linear(hidden_dim, next_dim)
        
        # Path to penultimate - now with more structure
        self.to_penultimate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, penultimate_dim)
        )
        
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
            
            # Create penultimate features with confidence information
            penultimate_features = self.to_penultimate(hidden)
            penultimate_contribution = penultimate_features * confidence
            
            # Poorly-predicted information continues upward
            continue_up = hidden * (1 - confidence)
            
            return continue_up, penultimate_contribution, pred_error
        else:
            # Last layer's contribution to penultimate
            return None, self.to_penultimate(hidden), None

class PenultimateLayer(nn.Module):
    def __init__(self, penultimate_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        
        # Weighting mechanism for different layers' contributions
        self.contribution_weights = nn.Parameter(torch.ones(num_layers)/num_layers)
        
        # Processing for combined features
        self.process = nn.Sequential(
            nn.Linear(penultimate_dim, penultimate_dim*2),
            nn.ReLU(),
            nn.Linear(penultimate_dim*2, penultimate_dim)
        )
    
    def forward(self, contributions):
        # Weight and combine contributions from different layers
        weights = F.softmax(self.contribution_weights, dim=0)
        weighted_sum = sum(w * c for w, c in zip(weights, contributions))
        
        # Process the combined features
        return self.process(weighted_sum)

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
        
        # Dedicated penultimate layer
        self.penultimate = PenultimateLayer(penultimate_dim, len(self.layers))
        
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
        
        # Process in dedicated penultimate layer
        penultimate = self.penultimate(penultimate_contributions)
        
        # Final output
        output = self.final(penultimate)
        
        return output, torch.cat(all_errors, dim=1) if all_errors else None