import torch
import matplotlib.pyplot as plt
from lorenz_hierarchical_analyzer import LorenzHierarchicalAnalyzer, analyze_lorenz_hierarchical_patterns
from hierarchical_net import HierarchicalPatternPredictiveNet
from experiment_harness import ExperimentHarness
from data_generators import generate_lorenz_data

def create_hierarchical_net():
    return HierarchicalPatternPredictiveNet(
        input_dim=20,
        hidden_dims=[64, 32, 16],
        penultimate_dim=32,
        output_dim=1,
        patterns_per_level=4,
        n_levels=2,
        compression_factor=2
    )

def main():
    print("Training model on Lorenz data...")
    harness = ExperimentHarness(
        data_generator=generate_lorenz_data,
        model_factory=create_hierarchical_net,
        n_seeds=1,
        epochs=100
    )
    results = harness.run_experiment()
    
    def convert_state_dict(list_state_dict):
        """Convert a state dict containing lists back to tensors"""
        tensor_state_dict = {}
        for key, value in list_state_dict.items():
            if isinstance(value, list):
                tensor_state_dict[key] = torch.tensor(value)
            else:
                tensor_state_dict[key] = value
        return tensor_state_dict

    # Get trained model
    model = create_hierarchical_net()
    list_state_dict = next(iter(results.values())).model_state_dict
    tensor_state_dict = convert_state_dict(list_state_dict)
    model.load_state_dict(tensor_state_dict)
    
    # Use the wrapper function to generate all visualizations
    analyzer, data = analyze_lorenz_hierarchical_patterns(model)
    
    print(f"\nFinal test loss: {next(iter(results.values())).final_test_loss:.4f}")

if __name__ == "__main__":
    main()