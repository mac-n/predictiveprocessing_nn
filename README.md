# Training Transparent Neural Networks with Learned Interpretable Features

This repository contains the code for running experiments and visualizations related to the Pattern Predictive Network (PPN), introduced in our paper *Training Transparent Neural Networks with Learned Interpretable Features*. The PPN architecture is designed to maintain interpretability throughout training while achieving competitive performance on sequential prediction tasks.

## Installation

This code is implemented in Python and relies on PyTorch for model training. To set up the environment, install dependencies using:

```bash
pip install -r requirements.txt
```

## Running Experiments

To train and evaluate the models, run:

```bash
python run_experiments.py
```

This script will execute experiments on chaotic sequence prediction (Lorenz attractor), pattern memory tasks, and language modeling, logging the results for analysis.

## Visualizations

To generate visualizations of network behavior and information flow, use the following scripts:

```bash
python run_lorenz_viz_basemodel.py  # Visualize base model on Lorenz data
python run_lorenz_hierarchical.py   # Visualize hierarchical model on Lorenz data
python visualize_layer_evolution.py # Track layer-wise pattern evolution
python run_char_analysis.py         # Character-level analysis for language task
```

## SHAP Analysis

SHAP (SHapley Additive exPlanations) visualizations provide a comparison between our inherently interpretable model and traditional post-hoc interpretability methods. Due to the computational cost, we recommend running this in Google Colab:

- `SHAP_visualisation.ipynb`

## Paper

If you use this code or find it useful, please cite our work:

> **Training Transparent Neural Networks with Learned Interpretable Features**\
> Anonymous Authors 

## Contact

For any questions, please reach out to the authors.




