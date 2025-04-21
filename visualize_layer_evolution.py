import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from base_predictive_net import DiscretePatternPredictiveNet
from data_generators import generate_lorenz_data

def create_pattern_predictive_net():
    return DiscretePatternPredictiveNet(
        input_dim=20,
        hidden_dims=[64, 32, 16],
        penultimate_dim=32,
        output_dim=1,
        n_patterns=8
    )

def collect_layer_activations(model, data_loader, layer_idx=0):
    """Collect pattern activations and flow data for a specific layer"""
    pattern_activations = []
    confidences = []
    penult_flows = []
    continue_flows = []
    
    model.eval()
    with torch.no_grad():
        for X, _ in data_loader:
            # Forward pass
            model(X)
            # Get layer stats
            stats = model.get_layer_stats()
            layer_stat = stats[layer_idx]
            
            # Collect data
            pattern_activations.append(layer_stat.pattern_usage.cpu().numpy())
            confidences.append(layer_stat.confidence_values.mean().cpu().numpy())
            penult_flows.append(layer_stat.penultimate_magnitude.cpu().numpy())
            continue_flows.append(layer_stat.continue_magnitude.cpu().numpy())
    
    return (np.array(pattern_activations), np.array(confidences), 
            np.array(penult_flows), np.array(continue_flows))

def plot_layer_evolution(pattern_data, confidence_data, penult_flow, continue_flow, save_path='layer_evolution.pdf'):
    """Plot layer evolution with pattern usage and dual-axis flow dynamics"""
    plt.rcParams.update({
        'font.size': 14,
        'axes.labelsize': 16,
        'axes.titlesize': 18,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14
    })

    fig = plt.figure(figsize=(12, 10))
    # Reduce space between subplots
    gs = GridSpec(2, 1, height_ratios=[1, 1], hspace=0.2)

    # Pattern evolution plot
    ax0 = fig.add_subplot(gs[0])
    im = ax0.imshow(pattern_data.T, aspect='auto', cmap='viridis', 
                    interpolation='nearest')
    ax0.set_ylabel('Pattern Index')
    ax0.set_title('Pattern Usage Evolution - Layer 0')
    
    # Add colorbar with specific position and size
    cax = fig.add_axes([0.92, 0.525, 0.02, 0.35])  # [left, bottom, width, height]
    plt.colorbar(im, cax=cax, label='Pattern Activation')

    # Dual-axis flow dynamics plot
    ax1 = fig.add_subplot(gs[1])
    
    # Normalize flow values
    continue_flow = continue_flow / np.max(continue_flow)
    penult_flow = penult_flow / np.max(penult_flow)
    
    # First axis for flows
    lns1 = ax1.plot(continue_flow, 'r-', label='Continue Up', linewidth=2)
    lns2 = ax1.plot(penult_flow, 'g-', label='To Penultimate', linewidth=2)
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Flow Magnitude')
    ax1.grid(True, alpha=0.3)
    
    # Second axis for confidence, aligned with main plot
    ax2 = ax1.twinx()
    lns3 = ax2.plot(confidence_data, 'b-', label='Inter Layer Confidence', linewidth=2)
    ax2.set_ylabel('Confidence', color='b')
    
    # Adjust the position of the right y-axis to align with colorbar
    ax2.spines['right'].set_position(('outward', 60))
    
    # Combine legends with updated position
    lns = lns1 + lns2 + lns3
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc='lower right', bbox_to_anchor=(0.98, 0.02))

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Load model state from your latest run
    model = create_pattern_predictive_net()
    state_dict = torch.load('./model_state.pt')
    model.load_state_dict(state_dict)
    
    # Generate fresh Lorenz data for visualization
    X, y = generate_lorenz_data()
    data_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X, y),
        batch_size=32,
        shuffle=False
    )
    
    # Collect activations for layer 0
    pattern_data, confidence_data, penult_flow, continue_flow = collect_layer_activations(
        model, data_loader, layer_idx=0
    )
    
    # Create visualization
    plot_layer_evolution(pattern_data, confidence_data, penult_flow, continue_flow)
    print("Visualization saved to layer_evolution.pdf")

if __name__ == "__main__":
    main()