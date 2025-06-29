import yaml
import torch
from pathlib import Path
from utils.dataset_utils import DuvenaudDataset
from utils.ecfp_utils import compute_ecfp_bit_vectors, compute_algorithm1_fps, compute_ecfp_count_vectors
from models.ngf import NeuralGraphFingerprint, NGFWithHead
from utils.evaluation import run_pairwise_analysis
from utils.sparsity_analysis import compute_sparsity
from utils.frozen_downstream import run_frozen_downstream_task
import numpy as np
from torch_geometric.loader import DataLoader
from utils.logging_utils import create_experiment_dir, save_results, save_distances_csv, save_distance_plot
from torch_geometric.nn.models import NeuralFingerprint
from models.ngf_adapter import NGFAdapter
from utils.end_to_end_downstream import run_end_to_end_training

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def run_experiment(config, experiment):
    np.random.seed(1312)
    torch.manual_seed(1312)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    
    # Location of results
    log_dir = create_experiment_dir(experiment)
    
    # Load dataset + custom PyG graphs
    dataset = DuvenaudDataset(
        name=config['experiment']['dataset']
    )
    dataset.process()

    # Compute ECFP fingerprints
    ecfp_impl = config['experiment']['ecfp']['implementation']
    radius = config['experiment']['ecfp']['radius']
    nBits = config['experiment']['ecfp']['fingerprint_dim']

    if ecfp_impl == 'rdkit_binary':
        fps_ecfp = compute_ecfp_bit_vectors(dataset.smiles_list, radius=radius, nBits=nBits)

    elif ecfp_impl == 'rdkit_count':
        fps_ecfp = compute_ecfp_count_vectors(dataset.smiles_list, radius=radius, nBits=nBits)

    elif ecfp_impl == 'algorithm1':
        fps_ecfp = compute_algorithm1_fps(dataset.smiles_list, radius=radius, nBits=nBits)

    else:
        raise ValueError("Invalid ECFP implementation.")

    # Construct NGF model
    example_input = dataset[0]
    in_channels = example_input.x.shape[1]
    if config['experiment']['ngf']['implementation'] == 'from_scratch':

        model_core = NeuralGraphFingerprint(
            in_channels=in_channels,
            hidden_dim=config['experiment']['ngf']['hidden_dim'],
            fingerprint_dim=config['experiment']['ngf']['fingerprint_dim'],
            num_layers=config['experiment']['ngf']['num_layers'],
            weight_scale=config['experiment']['ngf']['weight_scale'],
            sum_fn=config['experiment']['ngf']['sum_fn'],
            smooth_fn=config['experiment']['ngf']['smooth_fn'],
            sparsify_fn=config['experiment']['ngf']['sparsify_fn'],
        )
        ngf = NGFAdapter(model_core,mode="custom")
        
    elif config['experiment']['ngf']['implementation'] == 'pytorch_geometric':

        model_core = NeuralFingerprint(
            in_channels=in_channels,
            hidden_channels=config['experiment']['ngf']['hidden_dim'],
            out_channels=config['experiment']['ngf']['fingerprint_dim'],
            num_layers=config['experiment']['ngf']['num_layers']
        )
        ngf = NGFAdapter(model_core, mode="pytorch_geometric")
    else:
        raise ValueError("Invalid NGF implementation.")
    
    
    for p in ngf.model.parameters():
        p.requires_grad = False

    # Generate NGF embeddings
    ngf.model.eval()
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    emb_list = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to('cpu')
            emb = ngf(batch)
            emb_list.append(emb.cpu())
    emb_mat = torch.cat(emb_list, dim=0).numpy()
    print("Number of ECFP fingerprints:", len(fps_ecfp))
    print("NGF embeddings shape:", emb_mat.shape)
    
    # Compute bit sparsity
    sparsity_analysis_ngf = compute_sparsity(emb_mat)
    print("BIT SPARSITY ANAYLSIS - NGF")
    print("Mean active bits:", sparsity_analysis_ngf["mean_active_bits"])
    print("Mean sparsity:", sparsity_analysis_ngf["mean_sparsity"])
    print("Bit usage:", sparsity_analysis_ngf["bit_usage"])
    
    sparsity_analysis_ecfp = compute_sparsity(fps_ecfp)
    print("BIT SPARSITY ANAYLSIS - ECFP")
    print("Mean active bits:", sparsity_analysis_ecfp["mean_active_bits"])
    print("Mean sparsity:", sparsity_analysis_ecfp["mean_sparsity"])
    print("Bit usage:", sparsity_analysis_ecfp["bit_usage"])
    
    # Run distance analysis
    ecfp_all, ngf_all, ecfp_sample, ngf_sample, r = run_pairwise_analysis(
        emb_mat, fps_ecfp,
        sample_size=config['experiment']['evaluation']['num_pairs']
    )
    

    # Save the plot
    save_distance_plot(log_dir, ecfp_sample, ngf_sample, r, title=config['experiment']['dataset']+ " frozen NGF embeddings vs. ECFP")
    
    # Downstream Evaluation
    labels_np = np.array([data.y.item() for data in dataset])

    task_type = config['experiment']['evaluation']['downstream_task']
    dataset_name = config['experiment']['dataset']

    results_frozen = run_frozen_downstream_task(
        ecfp_array=fps_ecfp,
        ngf_array=emb_mat,
        labels=np.array(labels_np),
        task_type=task_type
    )

    print(f"\n[Frozen Downstream Evaluation] {task_type} on {dataset_name} dataset")
    print(f"ECFP {results_frozen['ecfp']:.4f}")
    print(f"NGF {results_frozen['ngf']:.4f}")
    
    if config['experiment']['evaluation'].get("train_end_to_end", False):    
        full_model = NGFWithHead(
            ngf_base=NeuralGraphFingerprint(
                in_channels=in_channels,
                hidden_dim=config['experiment']['ngf']['hidden_dim'],
                fingerprint_dim=config['experiment']['ngf']['fingerprint_dim'],
                num_layers=config['experiment']['ngf']['num_layers'],
                weight_scale=config['experiment']['ngf']['weight_scale'],
                sum_fn=config['experiment']['ngf']['sum_fn'],
                smooth_fn=config['experiment']['ngf']['smooth_fn'],
                sparsify_fn=config['experiment']['ngf']['sparsify_fn'],
            ),
            task_type=config['experiment']['evaluation']['downstream_task'],
            hidden_dim=config['experiment']['ngf']['hidden_dim'],  
            mode=config['experiment']['ngf']['implementation']
        )
        if config['experiment']['evaluation'].get("corrupt_labels", False):
            print("corrupting labels before training...")
            dataset.corrupt_labels(task_type)
        print("Running end-to-end training with NGF...")
        results_trained, trained_ngf_with_head = run_end_to_end_training(full_model, dataset, task_type)
        
        print(f"\n[End to end trained Downstream Evaluation] {task_type} on {dataset_name} dataset")
        print(f"NGF {results_trained:.4f}")
        
        trained_ngf = trained_ngf_with_head.ngf
        for p in trained_ngf.parameters():
            p.requires_grad = False

        trained_ngf.eval()
        loader = DataLoader(dataset, batch_size=64, shuffle=False)
        emb_list = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to('cpu')
                emb = trained_ngf(batch)
                emb_list.append(emb.cpu())
        trained_ngf_embeddings = torch.cat(emb_list, dim=0).numpy()
        
        ecfp_dists, ngf_dists,ecfp_sample, ngf_sample, r_trained = run_pairwise_analysis(
            ngf_embeddings=trained_ngf_embeddings,
            ecfp_fps=fps_ecfp,
            sample_size=config['experiment']['evaluation']['num_pairs']
        )
        
        save_distance_plot(log_dir, ecfp_sample, ngf_sample, r_trained, title=config['experiment']['dataset']+" trained NGF embeddings vs. ECFP")
        
    
    
    # Save results 
    
    
    results_to_log = {
        "dataset": config['experiment']['dataset'],
        "frozen downstream task": {
            "pearson_r": r,
            "task": config['experiment']['evaluation']['downstream_task'],
            "ecfp_result": float(results_frozen['ecfp']),
            "ngf_result": float(results_frozen['ngf']),
        },
        "sparsity analysis" : {
            "NGF Mean active bits:": float(sparsity_analysis_ngf["mean_active_bits"]),
            "NGF Mean sparsity:": float(sparsity_analysis_ngf["mean_sparsity"]),
            "ECFP Mean active bits:": float(sparsity_analysis_ecfp["mean_active_bits"]),
            "ECFP Mean sparsity:": float(sparsity_analysis_ecfp["mean_sparsity"])
        },
        "config": config['experiment']  
    }

    if config['experiment']['evaluation'].get("train_end_to_end", False):
        results_to_log["end to end trained downstream task"]= {
            "pearson_r": r_trained,
            "task": config['experiment']['evaluation']['downstream_task'],
            "result": float(results_trained),
        },
    save_results(log_dir, results_to_log)

if __name__ == "__main__":
    config_dir = Path("config")
    config_files = sorted(config_dir.glob("*.yaml"))  # All .yaml files in config/

    for config_path in config_files:
        print(f"\n=== Running experiment: {config_path.name} ===")
        config = load_config(config_path)
        run_experiment(config, config_path.name)
