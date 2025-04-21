from experiment_harness import run_comparison, create_standard_net, create_base_ppn, create_hierarchical_net
from data_generators import generate_lorenz_data, generate_memory_data, generate_language_data
import os

# Define data generators
data_generators = {
    "lorenz": generate_lorenz_data,
    "memory": generate_memory_data,
    "language": generate_language_data
}

# Define model factories
model_factories = {
    "standard": create_standard_net,
    "base_ppn": create_base_ppn,
    "hierarchical_ppn": create_hierarchical_net
}

# Create output directory if it doesn't exist
save_path = "experiment_results"
os.makedirs(save_path, exist_ok=True)


# Run the comparison
run_comparison(
    data_generators=data_generators,
    model_factories=model_factories,
    n_seeds=5,
    save_path=save_path
)