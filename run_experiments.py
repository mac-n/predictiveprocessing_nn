import json
import numpy as np
from experiment_harness import ExperimentHarness, create_standard_net, create_predictive_net, run_comparison
from data_generators import generate_switching_sine_data, generate_lorenz_data, generate_mixed_frequency_data, generate_memory_data, generate_language_data
from scipy import stats

def run_all_experiments(output_file="new_experiment_results.json", n_patterns=8):
    data_generators = {
        "lorenz": generate_lorenz_data,
        "memory": generate_memory_data,
        "language": generate_language_data
    }
    
    all_results = {}
    for data_name, data_generator in data_generators.items():
        print(f"\nRunning experiments for {data_name} data...")
        try:
            results = run_comparison(data_generator, n_seeds=5, n_patterns=n_patterns)
            all_results[data_name] = {
                "standard_losses": results["standard_losses"],
                "predictive_losses": results["predictive_losses"],
                "t_stat": float(results["t_stat"]),  # Convert from numpy type
                "p_value": float(results["p_value"]),  # Convert from numpy type
                "summary": {
                    "standard_mean": float(np.mean(results["standard_losses"])),
                    "standard_std": float(np.std(results["standard_losses"])),
                    "predictive_mean": float(np.mean(results["predictive_losses"])),
                    "predictive_std": float(np.std(results["predictive_losses"]))
                }
            }
            print(f"Completed {data_name} experiments successfully")
            print(f"Standard: {all_results[data_name]['summary']['standard_mean']:.4f} ± {all_results[data_name]['summary']['standard_std']:.4f}")
            print(f"Predictive: {all_results[data_name]['summary']['predictive_mean']:.4f} ± {all_results[data_name]['summary']['predictive_std']:.4f}")
            print(f"p-value: {all_results[data_name]['p_value']:.4f}")
            
        except Exception as e:
            print(f"Error running experiments for {data_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            all_results[data_name] = {"error": str(e)}
    
    # Save results to file
    try:
        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=4)
        print(f"\nResults saved to {output_file}")
    except Exception as e:
        print(f"Error saving results to file: {str(e)}")
        
    return all_results


if __name__ == "__main__":
    results = run_all_experiments(n_patterns=8) # Specify the desired number of patterns here.