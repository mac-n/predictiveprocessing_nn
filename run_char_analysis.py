from language_pattern_analyzer import analyze_language_patterns
from base_predictive_net import DiscretePatternPredictiveNet
from experiment_harness import ExperimentHarness
from data_generators import generate_language_data
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import os  # Add this import

def create_pattern_predictive_net():
    return DiscretePatternPredictiveNet(
        input_dim=20,  # sequence length
        hidden_dims=[64, 32, 16],
        penultimate_dim=32,
        output_dim=1,
        n_patterns=8
    )

def main():
    # Create figures directory if it doesn't exist
    os.makedirs('figures', exist_ok=True)
    
    print("Training model on language data...")
    # Train model
    harness = ExperimentHarness(
        data_generator=generate_language_data,
        model_factory=create_pattern_predictive_net,
        n_seeds=1,
        epochs=100,
        batch_size=32,
        test_size=0.2
    )
    results = harness.run_experiment()
    
    # Get the trained model from the first (only) seed
    model = create_pattern_predictive_net()
    
    # Convert the serialized state dict back to tensor format
    state_dict = next(iter(results.values())).model_state_dict
    tensor_state_dict = {}
    for k, v in state_dict.items():
        if isinstance(v, list):
            tensor_state_dict[k] = torch.tensor(v)
        else:
            tensor_state_dict[k] = v
            
    model.load_state_dict(tensor_state_dict)
    
    # Run analysis
    print("\nAnalyzing language patterns...")
    analyzer, data = analyze_language_patterns(model, n_samples=100)
    
    # Set larger fonts before the visualization
    plt.rcParams.update({'font.size': 16,
                        'axes.labelsize': 20,
                        'axes.titlesize': 24,
                        'xtick.labelsize': 16,
                        'ytick.labelsize': 16})
    
    # Generate visualization with the data
    print("\nGenerating visualization...")
    viz = analyzer.visualize_pattern_character_associations(data)
    
    # Save with high quality settings
    plt.gcf().set_size_inches(10, 8)
    plt.tight_layout()
    plt.savefig('figures/layer0_patterns.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\nAnalysis complete!")
    print(f"\nFinal test loss: {next(iter(results.values())).final_test_loss:.4f}")
    print(f"Visualization saved to figures/layer0_patterns.pdf")

if __name__ == "__main__":
    main()