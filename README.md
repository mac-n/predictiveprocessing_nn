# Training-Transparent Neural Networks

Neural network architecture that learns interpretable features during training while improving performance on both chaotic systems and pattern recognition tasks. [Read more about the architectural details and performance results](https://mac-n.github.io/).

## Key Results
- Lorenz attractor prediction: p<0.0002 improvement over baseline
- Pattern memory tasks: p<0.0001 improvement
- Directly visualizable learned features 

## Quick Start
```bash
python run_experiments.py  # Main experiment, outputs to newexperimentresults.json
python train_fresh.py      # Generates Lorenz pattern visualizations
```

## Architecture Variants
- Main branch: Current implementation (p=0.0005 on Lorenz)
- `hierarchy_patterns` branch: Hierarchical variation with improved pattern memory performance

## Note on File Names
The best-performing Lorenz implementation is currently in `discrete_predictive_net.py` (yes, I know that name is ironic - refactoring in progress!)

## Status
Active development - visualizations and documentation are being consolidated. The core architecture and results are stable.

