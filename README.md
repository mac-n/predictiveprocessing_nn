This code implements the transparent neural network architecture described at https://mac-n.github.io

This is still a work in progress. Tidying up and pruning of branches is ongoing. 

You can run the main experiment with python run_experiments.py. Results will be output to newexperimentresults.json.

The branch with the hierarchical variation is found at hierarchy_patterns with the appropriate variation of experiment_harness.py

The code that works best with Lorenz data is called discrete_predictive_net.py. This is a very confusing accident of the evolution of this architecture, so i apologise if you see this. In the main branch at the moment you are looking at a very slightly outdated version of the predictive_net architecture which runs at p=0.0005 rather than p=0.0002 improvement for Lorenz. It has a softmax instead of a Gumbel-softmax layer and doesn't work quite as well. But it has the advantage of not being called discrete_predictive_net while working best on Lorenz data. 

The code for different visualisations is currently scattered throughout branches. Sensible refactoring of visualisation code for different data types/ architecture combinations is ongoing and will be fully documented when complete.  However, if you want to generate those cool projections of the learned patterns onto the lorenz system,and why wouldn't you, then python train_fresh.py -  they will be created in the same directory where the script runs, as layer_0_pattern_projections.png and layer_1_pattern_projections.png. Other lorenz pattern graphs are generated in a timestamped subfolder at the same time. Yes, I know. I'm on it. 


