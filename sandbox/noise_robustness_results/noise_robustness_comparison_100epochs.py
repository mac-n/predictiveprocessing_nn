import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from torch.utils.data import DataLoader, TensorDataset
import os

from base_predictive_net import DiscretePatternPredictiveNet
from data_generators import generate_lorenz_data
from experiment_harness import create_base_ppn

class StandardNN(nn.Module):
    """Standard neural network with similar architecture to PPN but without patterns"""
    def __init__(self, input_size=20, hidden_sizes=[64, 32, 16, 32], output_size=1):
        super().__init__()
        
        # Create a network with same number of parameters as PPN
        layers = []
        sizes = [input_size] + hidden_sizes
        
        for i in range(len(sizes)-1):
            layers.append(nn.Linear(sizes[i], sizes[i+1]))
            layers.append(nn.ReLU())
        
        layers.append(nn.Linear(sizes[-1], output_size))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x), None  # Return None for consistency with PPN

def generate_noisy_data(base_data, noise_type, noise_level):
    """Add noise to existing Lorenz data"""
    X, y = base_data
    X_noisy = X.clone()
    y_noisy = y.clone()
    
    if noise_type == 'gaussian':
        # Add Gaussian noise
        X_noisy += torch.randn_like(X) * noise_level
    
    elif noise_type == 'structured':
        # Add structured sine wave noise
        batch_size = X.shape[0]
        steps = torch.arange(0, X.shape[1]).float()
        
        for i in range(batch_size):
            phase = torch.rand(1) * 2 * np.pi  # Random phase
            X_noisy[i] += noise_level * torch.sin(0.5 * steps + phase)
    
    elif noise_type == 'missing':
        # Randomly zero out values
        mask = torch.rand_like(X) < noise_level
        X_noisy[mask] = 0
    
    return X_noisy, y_noisy

def evaluate_model(model, dataloader, device):
    """Evaluate model on dataset and return metrics"""
    model.eval()
    total_loss = 0
    criterion = nn.MSELoss()
    
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred, _ = model(X)
            loss = criterion(pred, y)
            total_loss += loss.item() * len(X)
    
    avg_loss = total_loss / len(dataloader.dataset)
    return avg_loss

def run_noise_robustness_experiment():
    """Compare robustness of PPN vs standard NN on noisy data"""
    print("Starting Noise Robustness Experiment...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create experiment directory
    experiment_dir = 'noise_robustness_results'
    os.makedirs(experiment_dir, exist_ok=True)
    
    # 1. Generate clean Lorenz data
    print("Generating clean Lorenz data...")
    X, y = generate_lorenz_data(n_samples=5000)
    clean_dataset = TensorDataset(
        torch.FloatTensor(X),
        torch.FloatTensor(y).reshape(-1, 1)
    )
    train_size = int(0.8 * len(clean_dataset))
    val_size = len(clean_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(clean_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)
    
    # 2. Train both models on clean data (now with 100 epochs)
    print("\nTraining pattern network...")
    pattern_net = create_base_ppn().to(device)
    train_model(pattern_net, train_loader, val_loader, name="Pattern Network", n_epochs=100)
    
    print("\nTraining standard network...")
    standard_net = StandardNN().to(device)
    train_model(standard_net, train_loader, val_loader, name="Standard Network", n_epochs=100)
    
    # 3. Evaluate on increasingly noisy data
    print("\nEvaluating on noisy data...")
    noise_types = ['gaussian', 'structured', 'missing']
    
    results = {
        'gaussian': {'pattern_net': [], 'standard_net': [], 'levels': []},
        'structured': {'pattern_net': [], 'standard_net': [], 'levels': []},
        'missing': {'pattern_net': [], 'standard_net': [], 'levels': []}
    }
    
    # Testing noise levels (including more fine-grained low noise levels)
    noise_ranges = {
        'gaussian': np.concatenate([np.linspace(0, 0.5, 5), np.linspace(0.5, 5.0, 6)[1:]]),
        'structured': np.concatenate([np.linspace(0, 0.5, 5), np.linspace(0.5, 5.0, 6)[1:]]),
        'missing': np.concatenate([np.linspace(0, 0.05, 5), np.linspace(0.05, 0.5, 6)[1:]])
    }
    
    for noise_type in noise_types:
        print(f"\nTesting {noise_type} noise robustness:")
        for noise_level in noise_ranges[noise_type]:
            # Create test dataset with this noise level
            X_noisy, y_noisy = generate_noisy_data((torch.FloatTensor(X), torch.FloatTensor(y).reshape(-1, 1)), 
                                                  noise_type, noise_level)
            
            noisy_dataset = TensorDataset(X_noisy, y_noisy)
            noisy_loader = DataLoader(noisy_dataset, batch_size=64)
            
            # Evaluate both models
            pattern_loss = evaluate_model(pattern_net, noisy_loader, device)
            standard_loss = evaluate_model(standard_net, noisy_loader, device)
            
            # Store results
            results[noise_type]['pattern_net'].append(pattern_loss)
            results[noise_type]['standard_net'].append(standard_loss)
            results[noise_type]['levels'].append(noise_level)
            
            print(f"  Noise level {noise_level:.2f}: Pattern Net Loss: {pattern_loss:.6f}, Standard Net Loss: {standard_loss:.6f}")
    
    # 4. Visualize results
    print("\nCreating visualizations...")
    create_robustness_visualizations(results, experiment_dir)
    
    print(f"\nExperiment completed! Results saved to: {experiment_dir}")
    return results

def train_model(model, train_loader, val_loader, name="Model", n_epochs=100, lr=0.001):
    """Train a model and print progress"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    for epoch in range(n_epochs):
        model.train()
        train_loss = 0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            output, pred_errors = model(X_batch)
            
            loss = criterion(output, y_batch)
            if pred_errors is not None:
                pred_loss = 0.1 * torch.mean(pred_errors)
                loss = loss + pred_loss
            
            loss.backward()
            optimizer.step()
            
            if hasattr(model, 'update_temperatures'):
                model.update_temperatures()
            
            train_loss += loss.item() * len(X_batch)
        
        train_loss = train_loss / len(train_loader.dataset)
        
        # Validation
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                output, _ = model(X_batch)
                loss = criterion(output, y_batch)
                val_loss += loss.item() * len(X_batch)
        
        val_loss = val_loss / len(val_loader.dataset)
        
        # Print progress at intervals
        if epoch % 10 == 0 or epoch == n_epochs - 1:
            print(f"  {name}: Epoch {epoch+1}/{n_epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

if __name__ == "__main__":
    run_noise_robustness_experiment() 