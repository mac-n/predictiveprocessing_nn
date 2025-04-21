import torch
import os
from datetime import datetime
from experiment_harness import ExperimentHarness
from base_predictive_net import DiscretePatternPredictiveNet
from data_generators import generate_lorenz_data
from lorenz_pattern_mapper import map_lorenz_patterns
from lorenz_projections import run_pattern_projection_analysis, visualize_pattern_projections  # Added import
import matplotlib.pyplot as plt
import numpy as np

def create_pattern_predictive_net():
    return DiscretePatternPredictiveNet(
        input_dim=20,
        hidden_dims=[64, 32, 16],
        penultimate_dim=32,
        output_dim=1,
        n_patterns=8
    )

def convert_state_dict(list_state_dict):
    """Convert a state dict containing lists back to tensors"""
    tensor_state_dict = {}
    for key, value in list_state_dict.items():
        if isinstance(value, list):
            tensor_state_dict[key] = torch.tensor(value)
        else:
            tensor_state_dict[key] = value
    return tensor_state_dict

def save_model_and_visualizations(model, output_dir):
    """Generate and save all visualizations in the specified directory"""
    # Save model state
    torch.save(model.state_dict(), os.path.join(output_dir, 'model_state.pt'))
    
    # Generate and save pattern mappings
    print("\nGenerating pattern analysis visualizations...")
    fig_3d, fig_bar = map_lorenz_patterns(model)
    
    print(f"\nSaving plots to {output_dir}...")
    fig_3d.savefig(os.path.join(output_dir, 'lorenz_trajectory_patterns.pdf'), 
                   bbox_inches='tight', format='pdf', dpi=300)
    fig_bar.savefig(os.path.join(output_dir, 'pattern_state_mapping.pdf'), 
                    bbox_inches='tight', format='pdf', dpi=300)
    
    # Generate and save pattern projections
    print("\nGenerating pattern projections...")
    run_pattern_projection_analysis(model)
    
    # Save experiment metadata
    metadata = {
        'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
        'model_config': {
            'input_dim': 20,
            'hidden_dims': [64, 32, 16],
            'penultimate_dim': 32,
            'output_dim': 1,
            'n_patterns': 8
        }
    }
    
    with open(os.path.join(output_dir, 'experiment_metadata.txt'), 'w') as f:
        for key, value in metadata.items():
            f.write(f'{key}: {value}\n')
    
    plt.close('all')
    
    # Save experiment metadata
    metadata = {
        'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
        'model_config': {
            'input_dim': 20,
            'hidden_dims': [64, 32, 16],
            'penultimate_dim': 32,
            'output_dim': 1,
            'n_patterns': 8
        }
    }
    
    with open(os.path.join(output_dir, 'experiment_metadata.txt'), 'w') as f:
        for key, value in metadata.items():
            f.write(f'{key}: {value}\n')

def main():
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join('pattern_analysis_outputs', f'analysis_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    
    # Train model
    print("\nTraining model...")
    harness = ExperimentHarness(
        data_generator=generate_lorenz_data,
        model_factory=create_pattern_predictive_net,
        n_seeds=1,
        epochs=100
    )
    results = harness.run_experiment()
    
    # Get trained model
    model = create_pattern_predictive_net()
    list_state_dict = next(iter(results.values())).model_state_dict
    tensor_state_dict = convert_state_dict(list_state_dict)
    model.load_state_dict(tensor_state_dict)
    
    # Generate all visualizations and save results
    save_model_and_visualizations(model, output_dir)
    
    print(f"\nAnalysis complete! All outputs saved to {output_dir}")
    plt.close('all')

if __name__ == "__main__":
    main()