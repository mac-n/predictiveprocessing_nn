from lorenz_pattern_mapper import map_lorenz_patterns
from data_generators import generate_lorenz_data
from experiment_harness import ExperimentHarness
from predictive_net import PatternPredictiveNet
import matplotlib.pyplot as plt
import torch
import os
from datetime import datetime

def create_pattern_predictive_net():
    return PatternPredictiveNet(
        input_dim=20,
        hidden_dims=[64, 32, 16],
        penultimate_dim=32,
        output_dim=1,
        n_patterns=8
    )

def main():
    # Create a timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join('pattern_analysis_outputs', f'analysis_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    
    # Train model
    print("\nTraining model...")
    harness = ExperimentHarness(
        data_generator=generate_lorenz_data,
        model_factory=create_pattern_predictive_net,
        n_seeds=1,  # Just train one model for now
        epochs=100
    )
    results = harness.run_experiment()
    
    # Get the trained model from the first (only) seed
    model = harness.model_factory()
    model.load_state_dict(next(iter(results.values())).model_state_dict)
    
    # Save the trained model
    torch.save(model.state_dict(), os.path.join(output_dir, 'pattern_model.pt'))
    
    print("\nGenerating pattern analysis visualizations...")
    fig_3d, pattern_axes = map_lorenz_patterns(model)
    
    # Save plots
    print(f"\nSaving plots to {output_dir}...")
    fig_3d.savefig(os.path.join(output_dir, 'lorenz_trajectory_patterns.png'))
    plt.figure(pattern_axes[0].figure)
    plt.savefig(os.path.join(output_dir, 'pattern_state_mapping.png'))
    
    # Close all plots to prevent blocking
    plt.close('all')
    
    print(f"\nAnalysis complete! Check the plots in {output_dir}")

if __name__ == "__main__":
    main()