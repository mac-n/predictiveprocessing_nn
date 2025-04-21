import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Callable
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from hierarchical_net import HierarchicalPatternPredictiveNet, HierarchicalLayerStats
import torch.nn.functional as F
import json
from scipy import stats
from base_predictive_net import DiscretePatternPredictiveNet
from legacy_predictive_net import PatternPredictiveNet


@dataclass
class EpochStats:
    """Statistics for a single epoch"""
    layer_confidences: Dict[int, float]  # Average confidence per layer
    layer_pred_errors: Dict[int, float]  # Average prediction error per layer
    penultimate_flows: Dict[int, float]  # Average flow to penultimate per layer
    continue_flows: Dict[int, float]     # Average flow upward per layer
    train_loss: float
    prediction_loss: Optional[float] = None
    pattern_entropy: Optional[Dict[int, float]] = None # Average pattern entropy per layer


@dataclass
class ExperimentResult:
    """Store results from a single experimental run"""
    train_losses: List[float]
    test_losses: List[float]
    final_test_loss: float
    prediction_errors: Optional[List[float]] = None
    epoch_stats: Optional[List[EpochStats]] = None  # Track stats across epochs
    model_state_dict: Optional[Dict] = None


class ExperimentHarness:
    def __init__(
        self,
        data_generator,
        model_factory,
        n_seeds: int = 5,
        epochs: int = 100,
        batch_size: int = 32,
        test_size: float = 0.2,
        eval_frequency: int = 5
    ):
        self.data_generator = data_generator
        self.model_factory = model_factory
        self.n_seeds = n_seeds
        self.epochs = epochs
        self.batch_size = batch_size
        self.test_size = test_size
        self.eval_frequency = eval_frequency
    
    def collect_epoch_stats(self, model, epoch_loss: float, pred_loss: Optional[float] = None) -> Optional[EpochStats]:
      """Collect statistics from all layers for the epoch if relevant"""
      if isinstance(model, (HierarchicalPatternPredictiveNet, DiscretePatternPredictiveNet)):
            layer_stats = model.get_layer_stats()
      else:
           return None
            
      # Initialize aggregates
      confidences = {}
      pred_errors = {}
      penult_flows = {}
      cont_flows = {}
      entropies = {}
        
      # Aggregate stats across layers
      for stats in layer_stats:
            idx = stats.layer_idx
            confidences[idx] = float(torch.mean(stats.confidence_values).item())
            pred_errors[idx] = float(torch.mean(stats.prediction_errors).item())
            penult_flows[idx] = float(stats.penultimate_magnitude.item())
            cont_flows[idx] = float(stats.continue_magnitude.item())
            
            # Calculate entropy from pattern usage
            if hasattr(stats, 'pattern_usage') and stats.pattern_usage is not None:
                  probs = F.softmax(stats.pattern_usage, dim=-1)
                  entropy = -torch.sum(probs * torch.log2(probs + 1e-8))
                  entropies[idx] = float(entropy.item())
            elif hasattr(stats, 'pattern_usage_per_level') and stats.pattern_usage_per_level:
                pattern_entropy = 0
                for usage in stats.pattern_usage_per_level:
                    probs = F.softmax(usage, dim=-1)
                    entropy = -torch.sum(probs * torch.log2(probs + 1e-8))
                    pattern_entropy += entropy / len(stats.pattern_usage_per_level)  # Average entropy across levels
                entropies[idx] = float(pattern_entropy.item())
            else:
                entropies[idx] = None
            
      return EpochStats(
            layer_confidences=confidences,
            layer_pred_errors=pred_errors,
            penultimate_flows=penult_flows,
            continue_flows=cont_flows,
            train_loss=epoch_loss,
            prediction_loss=pred_loss,
            pattern_entropy=entropies
        )

    def analyze_epoch_stats(self, epoch_stats: List[EpochStats]) -> pd.DataFrame:
        """Analyze layer behavior across epochs"""
        records = []
        
        for epoch, stats in enumerate(epoch_stats):
            for layer_idx in stats.layer_confidences.keys():
                records.append({
                    'epoch': epoch,
                    'layer': layer_idx,
                    'confidence': stats.layer_confidences[layer_idx],
                    'pred_error': stats.layer_pred_errors[layer_idx],
                    'penult_flow': stats.penultimate_flows[layer_idx],
                    'continue_flow': stats.continue_flows[layer_idx],
                    'train_loss': stats.train_loss,
                    'pred_loss': stats.prediction_loss,
                    'pattern_entropy': stats.pattern_entropy[layer_idx] if stats.pattern_entropy is not None else None
                })
        
        return pd.DataFrame(records)

    def run_trial(self, seed: int) -> ExperimentResult:
        """Run a single trial with given seed"""
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        train_loader, test_loader, metadata = self.prepare_data()
        model = self.model_factory()
        optimizer = torch.optim.Adam(model.parameters())
        criterion = nn.MSELoss()
        
        train_losses = []
        test_losses = []
        prediction_errors = []
        epoch_stats_list = []
        
        for epoch in range(self.epochs):
            model.train()
            epoch_loss = 0
            epoch_pred_errors = []
            n_batches = 0
            
            for X, y in train_loader:
                optimizer.zero_grad()
                output = model(X)
                
                # Handle both single output and (output, pred_errors) tuple
                if isinstance(output, tuple):
                    outputs, pred_errors = output
                else:
                    outputs = output
                    pred_errors = None
                
                # Main task loss
                task_loss = criterion(outputs, y)
                
                # Add weighted prediction error if available
                if pred_errors is not None:
                    pred_loss = 0.1 * torch.mean(pred_errors)
                    total_loss = task_loss + pred_loss
                    epoch_pred_errors.append(pred_loss.item())
                else:
                    total_loss = task_loss
                
                total_loss.backward()
                optimizer.step()
                
                epoch_loss += task_loss.item()
                n_batches += 1
            
            avg_train_loss = epoch_loss / n_batches
            train_losses.append(avg_train_loss)
            
            avg_pred_loss = np.mean(epoch_pred_errors) if epoch_pred_errors else None
            prediction_errors.append(avg_pred_loss)
            
             # Update temperatures for discrete model
            if hasattr(model, 'update_temperatures') and callable(getattr(model, 'update_temperatures')):
               model.update_temperatures()
                
            # Collect statistics only for predictive network
            if isinstance(model, (HierarchicalPatternPredictiveNet, DiscretePatternPredictiveNet)):
                epoch_stats = self.collect_epoch_stats(model, avg_train_loss, avg_pred_loss)
                if epoch_stats:
                    epoch_stats_list.append(epoch_stats)
            
            if epoch % self.eval_frequency == 0:
                test_loss = self.evaluate_model(model, test_loader)
                test_losses.append(test_loss)
                print(f"Seed {seed}, Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, "
                      f"Test Loss = {test_loss:.4f}")
                
                # Print layer statistics for predictive network
                if isinstance(model, (HierarchicalPatternPredictiveNet, DiscretePatternPredictiveNet)) and epoch_stats:
                    print("\nLayer Statistics:")
                    for layer_idx, conf in epoch_stats.layer_confidences.items():
                        print(f"Layer {layer_idx}:")
                        print(f"  Confidence: {conf:.3f}")
                        print(f"  Pred Error: {epoch_stats.layer_pred_errors[layer_idx]:.3f}")
                        print(f"  Penult Flow: {epoch_stats.penultimate_flows[layer_idx]:.3f}")
                        print(f"  Continue Flow: {epoch_stats.continue_flows[layer_idx]:.3f}")
                        if epoch_stats.pattern_entropy:
                          print(f"  Pattern Entropy: {epoch_stats.pattern_entropy[layer_idx]:.3f}")
        
        final_test_loss = self.evaluate_model(model, test_loader)
        
        # Convert model state dict to list of lists for JSON serialization
        model_state_dict_serializable = {}
        for key, value in model.state_dict().items():
            if isinstance(value, torch.Tensor):
                model_state_dict_serializable[key] = value.cpu().numpy().tolist()
            else:
                model_state_dict_serializable[key] = value

        return ExperimentResult(
            train_losses=train_losses,
            test_losses=test_losses,
            final_test_loss=final_test_loss,
            prediction_errors=prediction_errors if prediction_errors else None,
            epoch_stats=epoch_stats_list,
            model_state_dict=model_state_dict_serializable
            )
        
    def analyze_layer_behavior(self, results: Dict[int, ExperimentResult]) -> pd.DataFrame:
        """Analyze layer behavior across all trials"""
        all_stats = []
        
        for seed, result in results.items():
            if result.epoch_stats:
                stats_df = self.analyze_epoch_stats(result.epoch_stats)
                stats_df['seed'] = seed
                all_stats.append(stats_df)
        
        if all_stats:
            combined_stats = pd.concat(all_stats)
            return combined_stats
        
        return pd.DataFrame()

    def run_experiment(self) -> Dict[int, ExperimentResult]:
        """Run full experiment with multiple seeds"""
        results = {}
        for seed in range(self.n_seeds):
            print(f"\nRunning trial with seed {seed}")
            results[seed] = self.run_trial(seed)
        return results


    def prepare_data(self):
        """Prepare train and test dataloaders"""
        generator_output = self.data_generator()
        if len(generator_output) == 2:
            X, y = generator_output
            metadata = None
        else:
            X, y, metadata = generator_output
        
        n_samples = len(X)
        n_test = int(n_samples * self.test_size)
        indices = torch.randperm(n_samples)
        
        train_indices = indices[:-n_test]
        test_indices = indices[-n_test:]
        
        train_dataset = TensorDataset(X[train_indices], y[train_indices])
        test_dataset = TensorDataset(X[test_indices], y[test_indices])
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size)
        
        return train_loader, test_loader, metadata

    def evaluate_model(self, model, dataloader) -> float:
        """Evaluate model on given dataloader"""
        model.eval()
        total_loss = 0
        n_batches = 0
        criterion = nn.MSELoss()
        
        with torch.no_grad():
            for X, y in dataloader:
                output = model(X)
                # Handle both single output and (output, pred_errors) tuple
                outputs = output[0] if isinstance(output, tuple) else output
                loss = criterion(outputs, y)
                total_loss += loss.item()
                n_batches += 1
                
        return total_loss / n_batches


def create_standard_net(sequence_length=20, hidden_dims=[64, 32, 16]):
    """Create baseline network with same architecture"""
    return nn.Sequential(
        nn.Linear(sequence_length, hidden_dims[0]),
        nn.ReLU(),
        nn.Linear(hidden_dims[0], hidden_dims[1]),
        nn.ReLU(),
        nn.Linear(hidden_dims[1], hidden_dims[2]),
        nn.ReLU(),
        nn.Linear(hidden_dims[2], 32),  # Added penultimate layer matching PPN
        nn.ReLU(),
        nn.Linear(32, 1)
    )


def create_base_ppn(sequence_length=20, hidden_dims=[64, 32, 16], n_patterns=8):
    """Create base predictive network with new architecture"""
    return DiscretePatternPredictiveNet(
        input_dim=sequence_length,
        hidden_dims=hidden_dims,
        penultimate_dim=32,
        output_dim=1,
        n_patterns=n_patterns
    )


def create_hierarchical_net(sequence_length=20, hidden_dims=[64, 32, 16], n_patterns=4, n_levels=2, compression_factor=2):
    """Create hierarchical predictive network with new architecture"""
    return HierarchicalPatternPredictiveNet(
        input_dim=sequence_length,
        hidden_dims=hidden_dims,
        penultimate_dim=32,  # Size of integration layer
        output_dim=1,
        patterns_per_level=n_patterns,
        n_levels=n_levels,
        compression_factor=compression_factor
    )

def run_comparison(
        data_generators: Dict[str, Callable],
        model_factories: Dict[str, Callable],
        n_seeds: int = 5,
        save_path: str = "experiment_results"
    ):
    """
    Run a comparison between different models across various datasets.

    Args:
        data_generators (Dict[str, Callable]): A dictionary mapping dataset names to data generation functions.
        model_factories (Dict[str, Callable]): A dictionary mapping model names to model creation functions.
        n_seeds (int): Number of random seeds to use for each experiment.
        save_path (str): Path to save output JSON files

    Returns:
        None (saves results to JSON files)
    """
    all_results = {
        "experiment_results": {},
        "summary": {}
    }

    for data_name, data_generator in data_generators.items():
            print(f"\nRunning experiments for {data_name} data...")
            all_results["experiment_results"][data_name] = {}
            all_results["summary"][data_name] = {}
            
            losses = {}
            
            for model_name, model_factory in model_factories.items():
                print(f"\n  Running {model_name} model...")
                harness = ExperimentHarness(
                    data_generator=data_generator,
                    model_factory=model_factory,
                    n_seeds=n_seeds
                )
                results = harness.run_experiment()

                losses[model_name] = [r.final_test_loss for r in results.values()]
                mean_loss = np.mean(losses[model_name])
                std_loss = np.std(losses[model_name])
                
                layer_stats = []
                for seed, result in results.items():
                    if result.epoch_stats:
                        stats_df = harness.analyze_epoch_stats(result.epoch_stats)
                        layer_stats += stats_df.to_dict('records')

                # Convert model state dict to list of lists
                model_state_dicts = [result.model_state_dict for result in results.values()]
                
                all_results["experiment_results"][data_name][model_name] = {
                   "losses": losses[model_name],
                   "mean_loss": float(mean_loss),
                   "std_loss": float(std_loss),
                    "layer_stats": layer_stats,
                    "model_state_dict":  model_state_dicts
                 }

            # Perform t-tests
            model_names = list(model_factories.keys())
            for i in range(len(model_names)):
                for j in range(i + 1, len(model_names)):
                    model1 = model_names[i]
                    model2 = model_names[j]

                    t_stat, p_value = stats.ttest_ind(losses[model1], losses[model2])
                    all_results["summary"][data_name][f"{model1}_vs_{model2}"] = {
                        "t_stat": float(t_stat),
                        "p_value": float(p_value)
                    }

    # Save results to file
    try:
        with open(f"{save_path}/detailed_experiment_results.json", 'w') as f:
            json.dump(all_results, f, indent=4)
            
        # Create summary dictionary for short output
        summary_results = {"summary": {}}
        for data_name in all_results["summary"].keys():
          summary_results["summary"][data_name] = all_results["summary"][data_name]
          for model_name in all_results["experiment_results"][data_name].keys():
             summary_results["summary"][data_name][model_name] = {
                    "mean_loss": all_results["experiment_results"][data_name][model_name]["mean_loss"],
                    "std_loss": all_results["experiment_results"][data_name][model_name]["std_loss"]
                }

        with open(f"{save_path}/summary_experiment_results.json", 'w') as f:
            json.dump(summary_results, f, indent=4)

        print(f"\nResults saved to {save_path}")
    except Exception as e:
        print(f"Error saving results to file: {str(e)}")
        import traceback
        traceback.print_exc()