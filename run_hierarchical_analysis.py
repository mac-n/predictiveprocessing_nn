from hierarchical_pattern_analyzer import analyze_hierarchical_patterns
from hierarchical_net import HierarchicalPatternPredictiveNet
from experiment_harness import ExperimentHarness
from data_generators import generate_language_data
import torch

def create_pattern_predictive_net():
    return HierarchicalPatternPredictiveNet(
        input_dim=20,  # sequence length
        hidden_dims=[64, 32, 16],
        penultimate_dim=32,
        output_dim=1,
        patterns_per_level=4,  # number of patterns at each level
        n_levels=2,  # letter-level and word-level
        compression_factor=2 
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
    analyzer, data = analyze_hierarchical_patterns(model, n_samples=100)
    
    print("\nAnalysis complete!")
    
    # Print final test loss
    print(f"\nFinal test loss: {next(iter(results.values())).final_test_loss:.4f}")

if __name__ == "__main__":
    main()