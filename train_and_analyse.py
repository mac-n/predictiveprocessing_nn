from pattern_analysis import visualize_lorenz_patterns
from data_generators import generate_lorenz_data
from experiment_harness import ExperimentHarness, create_predictive_net
import matplotlib.pyplot as plt
import torch
import numpy as np
import os

def train_model(seed=3):  # Using seed 3 since it had good results in your tests
    # Set random seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Create and train model
    harness = ExperimentHarness(
        data_generator=generate_lorenz_data,
        model_factory=lambda: create_predictive_net(),
        n_seeds=1,  # We're just training one model
        epochs=100,
        batch_size=32,
        test_size=0.2,
        eval_frequency=5
    )
    
    print("Training model...")
    result = harness.run_trial(seed)
    model = harness.model_factory()
    
    # Save the trained model
    save_path = 'trained_pattern_model.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'test_loss': result.final_test_loss,
        'train_losses': result.train_losses,
        'test_losses': result.test_losses
    }, save_path)
    print(f"Model saved to {save_path}")
    
    return model, result

def load_model():
    save_path = 'trained_pattern_model.pt'
    if os.path.exists(save_path):
        print(f"Loading saved model from {save_path}")
        checkpoint = torch.load(save_path)
        model = create_predictive_net()
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model with test loss: {checkpoint['test_loss']:.6f}")
        return model, checkpoint
    else:
        print("No saved model found, training new model...")
        return train_model()

def main():
    # Load or train the model
    model, result = load_model()
    
    # If we got a checkpoint, result is the checkpoint dict
    if isinstance(result, dict):
        print(f"\nTest loss from loaded model: {result['test_loss']:.6f}")
    else:
        print(f"\nTest loss from new model: {result.final_test_loss:.6f}")
    
    # Create visualizations
    print("\nGenerating pattern analysis visualizations...")
    analyzer = visualize_lorenz_patterns(generate_lorenz_data, model)
    
    # Save plots to files
    print("\nSaving plots...")
    for i in plt.get_fignums():
        plt.figure(i)
        plt.savefig(f'pattern_analysis_plot_{i}.png')
        
    print("\nAnalysis complete! Check the saved plot files.")
    
    # Optionally show plots (comment out if running on a server without display)
    plt.show()

if __name__ == "__main__":
    main()