import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd

@dataclass
class EpochStats:
    """Statistics for a single epoch"""
    layer_confidences: Dict[int, float]  # Average confidence per layer
    layer_pred_errors: Dict[int, float]  # Average prediction error per layer
    penultimate_flows: Dict[int, float]  # Average flow to penultimate per layer
    continue_flows: Dict[int, float]     # Average flow upward per layer
    train_loss: float
    prediction_loss: Optional[float] = None

@dataclass
class ExperimentResult:
    """Store results from a single experimental run"""
    train_losses: List[float]
    test_losses: List[float]
    final_test_loss: float
    prediction_errors: Optional[List[float]] = None
    epoch_stats: Optional[List[EpochStats]] = None  # Track stats across epochs

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
    
    def collect_epoch_stats(self, model, epoch_loss: float, pred_loss: Optional[float] = None) -> EpochStats:
        """Collect statistics from all layers for the epoch"""
        layer_stats = model.get_layer_stats()
        
        # Initialize aggregates
        confidences = {}
        pred_errors = {}
        penult_flows = {}
        cont_flows = {}
        
        # Aggregate stats across layers
        for stats in layer_stats:
            idx = stats.layer_idx
            confidences[idx] = float(torch.mean(stats.confidence_values).item())
            pred_errors[idx] = float(torch.mean(stats.prediction_errors).item())
            penult_flows[idx] = float(stats.penultimate_magnitude.item())
            cont_flows[idx] = float(stats.continue_magnitude.item())
        
        return EpochStats(
            layer_confidences=confidences,
            layer_pred_errors=pred_errors,
            penultimate_flows=penult_flows,
            continue_flows=cont_flows,
            train_loss=epoch_loss,
            prediction_loss=pred_loss
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
                    'pred_loss': stats.prediction_loss
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
            
            # Collect statistics
            epoch_stats = self.collect_epoch_stats(model, avg_train_loss, avg_pred_loss)
            epoch_stats_list.append(epoch_stats)
            
            if epoch % self.eval_frequency == 0:
                test_loss = self.evaluate_model(model, test_loader)
                test_losses.append(test_loss)
                print(f"Seed {seed}, Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, "
                      f"Test Loss = {test_loss:.4f}")
                
                # Print layer statistics
                print("\nLayer Statistics:")
                for layer_idx, conf in epoch_stats.layer_confidences.items():
                    print(f"Layer {layer_idx}:")
                    print(f"  Confidence: {conf:.3f}")
                    print(f"  Pred Error: {epoch_stats.layer_pred_errors[layer_idx]:.3f}")
                    print(f"  Penult Flow: {epoch_stats.penultimate_flows[layer_idx]:.3f}")
                    print(f"  Continue Flow: {epoch_stats.continue_flows[layer_idx]:.3f}")
        
        final_test_loss = self.evaluate_model(model, test_loader)
        
        return ExperimentResult(
            train_losses=train_losses,
            test_losses=test_losses,
            final_test_loss=final_test_loss,
            prediction_errors=prediction_errors if prediction_errors else None,
            epoch_stats=epoch_stats_list
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
        nn.Linear(hidden_dims[2], 1)
    )

def create_predictive_net(sequence_length=20, hidden_dims=[64, 32, 16]):
    """Create predictive network with new architecture"""
    from predictive_net import PredictiveNet
    return PredictiveNet(
        input_dim=sequence_length,
        hidden_dims=hidden_dims,
        penultimate_dim=32,  # Size of integration layer
        output_dim=1
    )

def run_comparison(data_generator, n_seeds=5):
    """Run comparison between standard and predictive networks"""
    # Standard network experiment
    standard_harness = ExperimentHarness(
        data_generator=data_generator,
        model_factory=lambda: create_standard_net(),
        n_seeds=n_seeds
    )
    standard_results = standard_harness.run_experiment()
    
    # Predictive network experiment
    predictive_harness = ExperimentHarness(
    data_generator=data_generator,
    model_factory=lambda: create_predictive_net(),
    n_seeds=n_seeds
    )
    predictive_results = predictive_harness.run_experiment()

    # Statistical comparison
    from scipy import stats
    standard_losses = [r.final_test_loss for r in standard_results.values()]
    predictive_losses = [r.final_test_loss for r in predictive_results.values()]
    
    t_stat, p_value = stats.ttest_ind(standard_losses, predictive_losses)
    
    return {
        'standard': standard_results,
        'predictive': predictive_results,
        't_stat': t_stat,
        'p_value': p_value,
        'standard_losses': standard_losses,
        'predictive_losses': predictive_losses
    }           