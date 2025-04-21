import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from torch.utils.data import DataLoader, TensorDataset

from base_predictive_net import DiscretePatternPredictiveNet
from data_generators import generate_lorenz_data
from experiment_harness import create_base_ppn

def generate_lorenz_trajectory(n_steps=1000, dt=0.01):
    """Generate Lorenz trajectory for visualization"""
    print("Generating Lorenz trajectory...")
    def lorenz(x, y, z, s=10, r=28, b=2.667):
        dx = s*(y - x)
        dy = r*x - y - x*z
        dz = x*y - b*z
        return dx, dy, dz

    # Generate trajectory
    trajectory = np.zeros((n_steps, 3))
    x, y, z = 1, 1, 1
    
    for i in range(n_steps):
        dx, dy, dz = lorenz(x, y, z)
        x += dx * dt
        y += dy * dt
        z += dz * dt
        trajectory[i] = [x, y, z]
    
    print(f"Trajectory shape: {trajectory.shape}")
    return trajectory

def train_with_pattern_evolution():
    """Train a model while collecting pattern activations for visualization"""
    print("Starting pattern evolution training...")
    
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Generate Lorenz data
    print("Generating Lorenz data...")
    X, y = generate_lorenz_data(n_samples=5000)
    print(f"Generated data shapes: X={X.shape}, y={y.shape}")
    
    # Generate full trajectory for visualization
    trajectory = generate_lorenz_trajectory(n_steps=1500)
    
    # Split into train and test
    train_size = int(0.8 * len(X))
    X_train, y_train = X[:train_size], y[:train_size]
    X_test, y_test = X[train_size:], y[train_size:]
    
    # Create data loader
    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=32,
        shuffle=True
    )
    
    # Initialize model
    try:
        model = create_base_ppn().to(device)
        print("Model created successfully")
    except:
        print("Falling back to direct model creation")
        model = DiscretePatternPredictiveNet(
            input_dim=20,
            hidden_dims=[64, 32, 16],
            penultimate_dim=32,
            output_dim=1,
            n_patterns=8
        ).to(device)
    
    # Set up optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    # Training parameters
    num_epochs = 100
    visualization_interval = 5
    
    # Create sequence windows from trajectory for visualization
    window_size = 20
    x_data = np.array([trajectory[i:i+window_size, 0] for i in range(len(trajectory)-window_size)])
    X_windows = torch.FloatTensor(x_data)
    
    # Storage for all epoch data
    all_epoch_data = []
    
    # Initial pattern activations (untrained model)
    print("Getting initial pattern activations...")
    model.eval()
    initial_activations = []
    with torch.no_grad():
        for x in X_windows:
            model(x.unsqueeze(0))
            stats = model.get_layer_stats()[0]  # Layer 0
            initial_activations.append(stats.pattern_usage.cpu().numpy())
    
    all_epoch_data.append({
        'epoch': 0,
        'activations': np.array(initial_activations)
    })
    
    # Training loop with periodic pattern collection
    print(f"Training for {num_epochs} epochs with visualizations every {visualization_interval} epochs...")
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Training
        model.train()
        total_loss = 0
        
        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            if batch_idx % 20 == 0:
                print(f"  Batch {batch_idx+1}/{len(train_loader)}")
            
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            output, pred_errors = model(X_batch)
            
            # Calculate loss
            loss = criterion(output, y_batch)
            if pred_errors is not None:
                pred_loss = 0.1 * torch.mean(pred_errors)
                loss = loss + pred_loss
                
            loss.backward()
            optimizer.step()
            model.update_temperatures()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
        
        # Save pattern activations at specified intervals
        if (epoch + 1) % visualization_interval == 0 or epoch == num_epochs - 1:
            print(f"Collecting pattern activations for epoch {epoch+1}...")
            model.eval()
            activations = []
            with torch.no_grad():
                for x in X_windows:
                    model(x.unsqueeze(0))
                    stats = model.get_layer_stats()[0]  # Layer 0
                    activations.append(stats.pattern_usage.cpu().numpy())
            
            all_epoch_data.append({
                'epoch': epoch + 1,
                'activations': np.array(activations)
            })
    
    # Create animation with collected data
    create_pattern_animation(trajectory, all_epoch_data)
    
    return model, all_epoch_data

def create_pattern_animation(trajectory, epoch_data):
    """Create a direct Plotly animation without HTML parsing"""
    print("\nCreating pattern animation...")
    n_patterns = 8
    
    # Create 2×4 grid for 8 patterns, with reduced spacing
    fig = make_subplots(
        rows=2, cols=4,
        specs=[[{'type': 'scene'} for _ in range(4)] for _ in range(2)],
        subplot_titles=[f'Pattern {i}' for i in range(n_patterns)],
        horizontal_spacing=0.01,  # Reduced from default
        vertical_spacing=0.02,    # Reduced from default
    )
    
    # Create frames for each epoch
    frames = []
    for data in epoch_data:
        epoch = data['epoch']
        activations = data['activations']
        
        frame_data = []
        for i in range(n_patterns):
            # Get pattern activations for pattern i
            pattern_activations = activations[:, i]
            
            # Min-max scaling for consistency
            min_val = pattern_activations.min()
            max_val = pattern_activations.max()
            if max_val > min_val:
                pattern_norm = (pattern_activations - min_val) / (max_val - min_val)
            else:
                pattern_norm = np.zeros_like(pattern_activations)
            
            # Determine row and column for this pattern
            row = (i // 4) + 1
            col = (i % 4) + 1
            
            # Create scatter trace
            scatter = go.Scatter3d(
                x=trajectory[:len(pattern_norm), 0],
                y=trajectory[:len(pattern_norm), 1],
                z=trajectory[:len(pattern_norm), 2],
                mode='markers',
                marker=dict(
                    size=2.5,  # Slightly smaller points for compactness
                    color=pattern_norm,
                    colorscale='Viridis',
                    opacity=0.7,
                    cmin=0,
                    cmax=1,
                    showscale=False
                ),
                name=f'Pattern {i}'
            )
            
            frame_data.append(scatter)
        
        # Create frame for this epoch
        frames.append(go.Frame(
            data=frame_data,
            name=f"Epoch {epoch}",
            traces=list(range(n_patterns))
        ))
    
    # Add initial data (first epoch)
    first_epoch_data = frames[0].data
    for i in range(n_patterns):
        row = (i // 4) + 1
        col = (i % 4) + 1
        fig.add_trace(first_epoch_data[i], row=row, col=col)
    
    # Set consistent camera angle for all subplots - slightly zoomed out to see whole pattern
    camera = dict(
        eye=dict(x=1.6, y=1.6, z=1.6),  # Slightly adjusted for better view
        up=dict(x=0, y=0, z=1)
    )
    
    for i in range(1, 3):
        for j in range(1, 5):
            fig.update_scenes(
                row=i, col=j,
                aspectmode='cube',
                camera=camera,
                xaxis_visible=False,
                yaxis_visible=False,
                zaxis_visible=False
            )
    
    # Add animation controls
    fig.update_layout(
        title="Pattern Evolution Through 100 Epochs",
        height=700,  # Reduced for tighter display
        width=1200,
        margin=dict(l=10, r=10, t=50, b=10),  # Tighter margins
        updatemenus=[{
            'type': 'buttons',
            'showactive': False,
            'buttons': [{
                'label': 'Play',
                'method': 'animate',
                'args': [None, {
                    'frame': {'duration': 500, 'redraw': True},  # Faster playback
                    'fromcurrent': True,
                    'transition': {'duration': 200, 'easing': 'quadratic-in-out'}
                }]
            }, {
                'label': 'Pause',
                'method': 'animate',
                'args': [[None], {
                    'frame': {'duration': 0, 'redraw': True},
                    'mode': 'immediate',
                    'transition': {'duration': 0}
                }]
            }]
        }],
        sliders=[{
            'active': 0,
            'yanchor': 'top',
            'xanchor': 'left',
            'currentvalue': {
                'font': {'size': 16},
                'prefix': 'Epoch: ',
                'visible': True,
                'xanchor': 'right'
            },
            'steps': [
                {
                    'method': 'animate',
                    'label': f"{data['epoch']}",
                    'args': [[f"Epoch {data['epoch']}"], {
                        'frame': {'duration': 300, 'redraw': True},
                        'mode': 'immediate',
                        'transition': {'duration': 300}
                    }]
                }
                for data in epoch_data
            ]
        }]
    )
    
    # Add frames to figure
    fig.frames = frames
    
    # Save interactive HTML
    os.makedirs('pattern_evolution', exist_ok=True)
    html_path = 'pattern_evolution/lorenz_pattern_evolution_100epochs.html'
    fig.write_html(html_path)
    print(f"Saved animation to {html_path}")
    
    return fig

if __name__ == "__main__":
    model, data = train_with_pattern_evolution()
    print("Process completed!") 