import torch
from lorenz_pattern_mapper import map_lorenz_patterns
import matplotlib.pyplot as plt
from predictive_net import PatternPredictiveNet  # Import your model class

def main():
    # Create a new model instance with the same architecture
    model = PatternPredictiveNet(
        input_dim=20,  # Your sequence length
        hidden_dims=[64, 32, 16],  # Your hidden dimensions
        penultimate_dim=32,  # Your penultimate dimension
        output_dim=1,  # Output dimension
        n_patterns=8  # Number of patterns
    )
    
    # Load the state dict
    state_dict = torch.load('trained_pattern_model.pt')
    if isinstance(state_dict, dict):
        if 'state_dict' in state_dict:  # If it's a checkpoint dictionary
            model.load_state_dict(state_dict['state_dict'])
        else:  # If it's just the state dict
            model.load_state_dict(state_dict)
    
    model.eval()
    
    # Run the analysis
    print("Analyzing Lorenz patterns...")
    fig_3d, pattern_axes = map_lorenz_patterns(model)
    
    # Save the figures
    fig_3d.savefig('lorenz_trajectory_patterns.png')
    plt.figure(pattern_axes[0].figure)
    plt.savefig('pattern_state_mapping.png')
    
    print("Analysis complete! Check lorenz_trajectory_patterns.png and pattern_state_mapping.png")
    plt.close('all')

if __name__ == "__main__":
    main()