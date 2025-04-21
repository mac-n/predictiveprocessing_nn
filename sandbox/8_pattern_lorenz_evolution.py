import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import time
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler

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

def get_pattern_activations(model, trajectory, window_size=20):
    """Get pattern activations for trajectory windows - optimized version"""
    print("Getting pattern activations...")
    # Create all windows at once using numpy operations (fixes warning)
    n_windows = len(trajectory) - window_size
    x_data = np.array([trajectory[i:i+window_size, 0] for i in range(n_windows)])
    X_windows = torch.FloatTensor(x_data)
    
    # Get pattern activations for each window
    model.eval()
    layer_pattern_activations = []
    
    with torch.no_grad():
        batch_size = 32
        for i in range(0, len(X_windows), batch_size):
            batch = X_windows[i:i+batch_size]
            for x in batch:
                model(x.unsqueeze(0))
                stat = model.get_layer_stats()[0]  # Layer 0
                layer_pattern_activations.append(stat.pattern_usage.cpu().numpy())
    
    return np.array(layer_pattern_activations)

def create_8_pattern_visualization(model, trajectory, epoch, save_dir, n_patterns=8):
    """Create 8 separate 3D spaces, one for each pattern"""
    try:
        print(f"\nCreating 8-pattern visualization for epoch {epoch}...")
        
        # Get pattern activations
        layer_pattern_activations = get_pattern_activations(model, trajectory)
        
        # Create a 3×3 grid (8 patterns + title in center)
        fig = make_subplots(
            rows=3, cols=3,
            specs=[
                [{'type': 'scatter3d'}, {'type': 'scatter3d'}, {'type': 'scatter3d'}],
                [{'type': 'scatter3d'}, {'type': 'scatter3d'}, {'type': 'scatter3d'}],
                [{'type': 'scatter3d'}, {'type': 'scatter3d'}, {'type': 'scatter3d'}]
            ],
            subplot_titles=["Pattern 0", "Pattern 1", "Pattern 2", 
                           "Pattern 3", "Lorenz Patterns", "Pattern 4",
                           "Pattern 5", "Pattern 6", "Pattern 7"],
            horizontal_spacing=0.01,
            vertical_spacing=0.02
        )
        
        # We'll use this mapping to place patterns in the grid
        # Grid positions are: (row, col)
        # The center position (2,2) is reserved for a title
        grid_positions = [
            (1, 1), (1, 2), (1, 3),  # Top row
            (2, 1), (2, 3),          # Middle row (skip center)
            (3, 1), (3, 2), (3, 3)   # Bottom row
        ]
        
        # Add each pattern's Lorenz attractor in its own subplot
        for i in range(n_patterns):
            row, col = grid_positions[i]
            
            # Scale pattern activations to [0,1]
            scaler = MinMaxScaler()
            pattern_norm = scaler.fit_transform(
                layer_pattern_activations[:, i].reshape(-1, 1)
            ).ravel()
            
            # FIX: Use a single opacity value instead of an array
            # Plotly 3D doesn't support per-point opacity arrays
            
            # Add this pattern's trace to the figure
            fig.add_trace(
                go.Scatter3d(
                    x=trajectory[:len(pattern_norm), 0],
                    y=trajectory[:len(pattern_norm), 1],
                    z=trajectory[:len(pattern_norm), 2],
                    mode='markers',
                    marker=dict(
                        size=2.5,
                        color=pattern_norm,  # Color still varies by point
                        colorscale='Viridis',
                        opacity=0.7,  # Single opacity for all points
                        showscale=(i == 0),  # Only show colorbar for first pattern
                        colorbar=dict(
                            title='Activation',
                            thickness=15,
                            len=0.6,
                            x=1.0,
                            y=0.5
                        ),
                        cmin=0,  # Fix color scale range
                        cmax=1
                    ),
                    name=f'Pattern {i}'
                ),
                row=row, col=col
            )
        
        # Add title in center cell
        fig.add_annotation(
            x=0.5, y=0.5,
            text=f"Layer 0 Pattern Evolution<br>Epoch {epoch}",
            showarrow=False,
            font=dict(size=20),
            xref="paper", yref="paper"
        )
        
        # Set consistent camera positions for all subplots - adjusted to see both wings
        camera = dict(
            eye=dict(x=1.2, y=1.8, z=1.0),  # Adjusted to see both wings clearly
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0)
        )
        
        for i in range(1, 4):
            for j in range(1, 4):
                if i == 2 and j == 2:  # Skip center cell
                    continue
                fig.update_scenes(
                    row=i, col=j,
                    aspectmode='cube',
                    camera=camera,
                    xaxis_visible=False,
                    yaxis_visible=False,
                    zaxis_visible=False
                )
        
        # Update layout
        fig.update_layout(
            title=f'Layer 0 Pattern Evolution - Epoch {epoch}',
            height=900,
            width=1100,
            showlegend=False,
            margin=dict(l=10, r=10, t=50, b=10)  # Tighter margins
        )
        
        # Save as interactive HTML and static image
        # Use zero-padded epoch numbers for proper sorting (001, 002, ... 099, 100)
        html_file = os.path.join(save_dir, f'8pattern_epoch_{epoch:03d}.html')
        fig.write_html(html_file)
        
        png_file = os.path.join(save_dir, f'8pattern_epoch_{epoch:03d}.png')
        fig.write_image(png_file, scale=2)
        
        print(f"Saved visualization to {html_file} and {png_file}")
        return fig
        
    except Exception as e:
        print(f"Error creating visualization: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def create_animated_visualization(frames_dir):
    """Create a single seamless HTML animation from all pattern visualizations"""
    print("\nCreating seamless animated visualization...")
    
    # Get all plotly figure HTML files - only those with epoch numbers
    html_files = sorted([f for f in os.listdir(frames_dir) 
                        if f.endswith('.html') and 'pattern_epoch_' in f])
    
    if not html_files:
        print("No pattern epoch HTML files found for animation")
        return
    
    # Extract epoch numbers from filenames - with error handling
    epochs = []
    sorted_html_files = []
    
    for f in html_files:
        try:
            # Extract just the number part
            epoch_part = f.split('epoch_')[-1].split('.')[0]
            epoch_num = int(epoch_part)
            epochs.append(epoch_num)
            sorted_html_files.append((epoch_num, f))
        except (ValueError, IndexError):
            print(f"Skipping file with invalid format: {f}")
    
    if not sorted_html_files:
        print("No valid HTML files with epoch numbers found")
        return
    
    # Sort by epoch number
    sorted_html_files.sort(key=lambda x: x[0])
    epochs.sort()
    html_files = [f for _, f in sorted_html_files]
        
    print(f"Found {len(html_files)} valid epoch files")
    
    # Create a new combined HTML file with all frames preloaded
    animation_file = os.path.join(frames_dir, 'seamless_animation.html')
    
    # Extract Plotly JSON data from each HTML file to embed directly
    plot_data = []
    for html_file in html_files:
        try:
            with open(os.path.join(frames_dir, html_file), 'r') as f:
                html_content = f.read()
                
                # Find the plotly data in the HTML
                start_marker = 'Plotly.newPlot('
                start_idx = html_content.find(start_marker)
                if start_idx > 0:
                    data_start = html_content.find('[', start_idx)
                    data_end = html_content.find(');', data_start)
                    if data_start > 0 and data_end > 0:
                        # Extract just enough data to recreate the plot
                        plot_json = html_content[data_start:data_end]
                        plot_data.append(plot_json)
        except Exception as e:
            print(f"Error processing {html_file}: {e}")
    
    if not plot_data:
        print("Could not extract Plotly data from any HTML files")
        return
        
    print(f"Successfully extracted Plotly data from {len(plot_data)} files")
    
    with open(animation_file, 'w') as f:
        f.write(f'''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Seamless Lorenz Pattern Evolution</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                    text-align: center;
                }}
                #animation-container {{
                    width: 100%;
                    height: 900px;
                    border: 1px solid #ddd;
                    margin: 20px 0;
                }}
                .controls {{
                    margin: 20px;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    flex-wrap: wrap;
                }}
                button {{
                    padding: 10px 20px;
                    margin: 0 10px;
                    font-size: 16px;
                    cursor: pointer;
                    background-color: #4CAF50;
                    color: white;
                    border: none;
                    border-radius: 4px;
                }}
                button:hover {{
                    background-color: #45a049;
                }}
                .slider-container {{
                    width: 80%;
                    margin: 20px auto;
                    display: flex;
                    align-items: center;
                }}
                #epochSlider {{
                    flex-grow: 1;
                    margin: 0 10px;
                }}
                .speed-control {{
                    display: flex;
                    align-items: center;
                    margin: 0 20px;
                }}
                label {{
                    margin-right: 10px;
                }}
            </style>
        </head>
        <body>
            <h1>Lorenz Pattern Evolution Through Training</h1>
            
            <div id="animation-container"></div>
            
            <div class="controls">
                <button id="playButton">Play</button>
                <button id="pauseButton">Pause</button>
                <div class="speed-control">
                    <label for="speedControl">Speed:</label>
                    <input type="range" id="speedControl" min="0.5" max="5" step="0.5" value="1">
                    <span id="speedValue">1 fps</span>
                </div>
            </div>
            
            <div class="slider-container">
                <span>Epoch:</span>
                <input type="range" id="epochSlider" min="0" max="{len(epochs)-1}" value="0">
                <span id="epochDisplay">{epochs[0]}</span>
            </div>
            
            <script>
                // Store all the plot data for each frame
                const plotData = {plot_data};
                const epochs = {epochs};
                const container = document.getElementById('animation-container');
                const epochSlider = document.getElementById('epochSlider');
                const epochDisplay = document.getElementById('epochDisplay');
                const playButton = document.getElementById('playButton');
                const pauseButton = document.getElementById('pauseButton');
                const speedControl = document.getElementById('speedControl');
                const speedValue = document.getElementById('speedValue');
                
                let currentFrame = 0;
                let isPlaying = false;
                let animationInterval;
                let speed = 1; // frames per second
                
                // Use a try-catch for better error handling in the browser
                try {{
                    // Initialize Plotly with first frame
                    Plotly.newPlot(container, ...JSON.parse(plotData[0]));
                    
                    // Pre-process all frames for faster animation
                    const frames = plotData.map(data => {{
                        try {{
                            return JSON.parse(data);
                        }} catch (e) {{
                            console.error('Error parsing frame data:', e);
                            // Return a simple empty plot as fallback
                            return [[{{x: [], y: [], z: [], type: 'scatter3d'}}], {{}}];
                        }}
                    }});
                    
                    // Update display with current frame
                    function updateDisplay() {{
                        try {{
                            // Use Plotly.react for smoother transitions (no flashing)
                            Plotly.react(container, ...frames[currentFrame]);
                            epochDisplay.textContent = epochs[currentFrame];
                            epochSlider.value = currentFrame;
                        }} catch (e) {{
                            console.error('Error updating display:', e);
                        }}
                    }}
                    
                    // Event listeners
                    epochSlider.addEventListener('input', function() {{
                        currentFrame = parseInt(this.value);
                        updateDisplay();
                    }});
                    
                    playButton.addEventListener('click', playAnimation);
                    pauseButton.addEventListener('click', pauseAnimation);
                    
                    function playAnimation() {{
                        if (isPlaying) return;
                        
                        isPlaying = true;
                        animationInterval = setInterval(function() {{
                            currentFrame = (currentFrame + 1) % frames.length;
                            updateDisplay();
                        }}, 1000 / speed);
                    }}
                    
                    function pauseAnimation() {{
                        if (!isPlaying) return;
                        
                        isPlaying = false;
                        clearInterval(animationInterval);
                    }}
                    
                    speedControl.addEventListener('input', function() {{
                        speed = parseFloat(this.value);
                        speedValue.textContent = speed + ' fps';
                        if (isPlaying) {{
                            pauseAnimation();
                            playAnimation();
                        }}
                    }});
                    
                    // Initial display
                    updateDisplay();
                }} catch (e) {{
                    console.error('Error initializing animation:', e);
                    container.innerHTML = '<p style="color:red">Error loading animation. Check console for details.</p>';
                }}
            </script>
        </body>
        </html>
        ''')
    
    print(f"Created seamless animation: {animation_file}")

def train_lorenz_pattern_model():
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directories
    frames_dir = '8_pattern_frames'
    os.makedirs(frames_dir, exist_ok=True)
    print(f"Created output directory: {frames_dir}")
    
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
    
    # Training loop with visualization
    num_epochs = 100  # Increased to 100 epochs
    visualization_interval = 2  # Every 2 epochs
    
    print(f"Training model for {num_epochs} epochs with visualizations every {visualization_interval} epochs...")
    
    # Create initial visualization
    create_8_pattern_visualization(model, trajectory, 0, frames_dir)
    
    try:
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            # Train one epoch
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
            
            # Create visualization at specified intervals
            if (epoch + 1) % visualization_interval == 0 or epoch == num_epochs - 1:
                create_8_pattern_visualization(model, trajectory, epoch + 1, frames_dir)
    
    except Exception as e:
        print(f"Error during training: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # Create animation of all visualizations
    create_animated_visualization(frames_dir)
    
    # Create movie using ffmpeg
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.png')])
    if frame_files:
        print("Creating movie from frames...")
        ffmpeg_cmd = f"ffmpeg -framerate 2 -pattern_type glob -i '{frames_dir}/8pattern_epoch_*.png' -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p 8_pattern_lorenz_evolution.mp4"
        print(f"Running command: {ffmpeg_cmd}")
        
        ret_code = os.system(ffmpeg_cmd)
        if ret_code == 0:
            print("Movie created: 8_pattern_lorenz_evolution.mp4")
        else:
            print("Error creating movie. Check if ffmpeg is installed.")

if __name__ == "__main__":
    print("Starting 8-pattern Lorenz evolution visualization...")
    start_time = time.time()
    train_lorenz_pattern_model()
    elapsed_time = time.time() - start_time
    print(f"Process completed in {elapsed_time:.2f} seconds") 