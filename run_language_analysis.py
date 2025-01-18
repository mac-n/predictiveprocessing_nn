from language_pattern_analyzer import analyze_language_patterns
from predictive_net import PatternPredictiveNet
from experiment_harness import ExperimentHarness
from data_generators import generate_language_data
import torch
import matplotlib.pyplot as plt

def create_pattern_predictive_net():
    return PatternPredictiveNet(
        input_dim=20,  # sequence length
        hidden_dims=[64, 32, 16],
        penultimate_dim=32,
        output_dim=1,
        n_patterns=8
    )

def main():
    print("Training model on language data...")
    # Train model
    harness = ExperimentHarness(
        data_generator=generate_language_data,
        model_factory=create_pattern_predictive_net,
        n_seeds=1,  # Just train one model
        epochs=100,
        batch_size=32,
        test_size=0.2
    )
    results = harness.run_experiment()
    
    # Get the trained model from the first (only) seed
    model = create_pattern_predictive_net()
    model.load_state_dict(next(iter(results.values())).model_state_dict)
    
    # Run analysis
    print("\nAnalyzing language patterns...")
    analyzer, data = analyze_language_patterns(model, n_samples=100)
    
    print("\nAnalysis complete! Check the generated PNG files.")
    plt.close('all')
    
    # Print final test loss
    print(f"\nFinal test loss: {next(iter(results.values())).final_test_loss:.4f}")

if __name__ == "__main__":
    main()