# Pattern Predictive Networks (PPN)

This repository contains the implementation of Pattern Predictive Networks (PPNs), a neural architecture designed for inherent transparency. PPNs learn interpretable internal representations through pattern dictionaries and use prediction-based routing to make information flow observable during both training and inference.

**For a detailed explanation of the architecture, key results (including demonstrations on chaotic systems), and future directions towards transparent LLMs, please see the main project writeup:** [**https://mac-n.github.io/**](https://mac-n.github.io/)

## Core Innovation

The PPN architecture introduces two key components:

1. **Pattern Dictionaries:** Each layer learns to compress information into interpretable 'patterns' via attention and predicts the patterns the *next* layer will use
2. **Prediction-Error Based Routing:** Information flow is dynamically routed based on how accurately a layer anticipates how the next layer will compress its output using its patterns

This combination allows for:
- Direct observation of internal representations during training
- Explicit tracking of information flow through the network
- Transparency in both success and failure modes

## Installation

This code is implemented in Python and relies on PyTorch for model training. To set up the environment:

```bash
# Clone the repository
git clone https://github.com/mac-n/predictiveprocessing_nn.git
cd predictiveprocessing_nn

# Install dependencies
pip install -r requirements.txt
```

## Running Experiments

To train and evaluate the models, run:

```bash
python run_experiments.py
```

This script will execute experiments primarily focused on chaotic sequence prediction (Lorenz attractor) and simple language modeling, as discussed in the main project writeup. Code for additional experiments (e.g., pattern memory) is also included. Results and logs are saved for analysis.

## Visualizations

To generate visualizations discussed in the project writeup:

```bash
# Visualize base PPN patterns/behavior on Lorenz data
python run_lorenz_viz_basemodel.py  

# Track layer-wise pattern evolution (e.g., on Lorenz data)
python visualize_layer_evolution.py 

# Visualize pattern specialization (or lack thereof) on the language task
python run_char_analysis.py         
```

Additional scripts for other experiments (e.g., hierarchical model variant, an adaptation of the architecture for more structured data such as pattern memory) are also included:

```bash
# Visualize hierarchical model on Lorenz data (demonstrates transparent failure)
python run_lorenz_hierarchical.py   
```

## Current Results

The base PPN architecture achieves:
- 78% reduction in prediction error on chaotic Lorenz attractor data vs. standard neural networks
- Comparable performance to standard networks on language prediction
- Transparent internal representations that reveal how information is processed

## Future Directions

We are currently working on:
1. Integration with transformer mechanisms for improved language task performance
2. Pattern-to-pattern attention for capturing higher-order relationships
3. Scaling to larger models and more complex datasets

## Citation

If you use this code in your research, please cite:

```
@misc{ppn2024,
  author = {Mccombe,Niamh, Wong-Lin, KongFatt},
  title = {Pattern Predictive Networks: Neural Architectures with Built-in Transparency},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/mac-n/predictiveprocessing_nn}
}
```
