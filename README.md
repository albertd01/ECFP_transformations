### ECFP vs. NGF similarity analysis
### Setup
Step 1: **Clone the repository**
```
git clone https://github.com/albertd01/BSC_thesis.git
```
Step 2: **Navigate to root of the repository**
Step 3: **Install requirements** (create venv if you want to isolate dependencies)
```
pip install requirements.txt
```
#### run_experiment.py
runs all experiments in config directory. Saves results to src/logs
##### Experiment config files
An experiment can be configured using this yaml file structure:
```yaml
experiment:
  dataset: # Options: ESOL, BACE, lipo
  ngf:
    "implementation": # options: pytorch_geometric, from_scratch
    "hidden_dim": # int: hidden dimension of ngf 
    "fingerprint_dim": # int: embedding/fingerprint dimension
    "num_layers": # int thesis experiments used 2
    "weight_scale": # float: scales ngf weights
    "sum_fn":   # ngf node aggregation - options: default, bond_weighted
    "smooth_fn": # ngf activation - options: tanh, identity, relu, sin
    "sparsify_fn": # ngf sparsification - options: gumbel, sparsemax, softmax
  ecfp:
    implementation: # implementation options: rdkit_binary, rdkit_count, algorithm1
    radius: # int: ECFP radius
    fingerprint_dim: # int: fingerprint dimension    

  evaluation:
    num_pairs: # int: Number of random pairs for correlation plot
    downstream_task: # task type - options: regression, classification
    train_end_to_end: # flag for enabling end-to-end training (true, false)    
    corrupt_labels: # flag for corrupting labels (true, false)         
```
#### weight_analysis.ipynb
creates some plots showcasing the effect of using large random weights on a test molecule. Serves as a preliminary experiment.
#### compare ecfp_variants.py
compares pairwise distance correlation of RDKit binary ECFP, RDKit count ECFP, Algorithm 1 from the NGF paper