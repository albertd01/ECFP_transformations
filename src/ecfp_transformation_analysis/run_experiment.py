from dataset_utils import ECFPDataset
from downstream import run_downstream_task
from pairwise_distances import analyze_distance_preservation
from transforms import BlockRadiusLinearMixing


def run_experiment(
    experiment_name,
    dataset_name,
    use_count,
    transform_obj=None,
    n_bits=2048,
    radius=2,
    epochs=100,
    analyze_distances=True,
    sample_size=300
):
    """Run single experiment and store results."""
    task_type = 'classification' if dataset_name == 'bace' else 'regression'
    metric_name = 'ROC-AUC' if task_type == 'classification' else 'RMSE'
    
    print(f'\n[{experiment_name}] {dataset_name.upper()} - {"Count" if use_count else "Binary"} ECFP')
    
    # Load dataset
    dataset = ECFPDataset(
        name=dataset_name,
        split_type='random',
        target_index=0,
        n_bits=n_bits,
        radius=radius,
        use_count=use_count
    )
    
    # Apply transformation if provided
    if transform_obj is not None:
        original_dataset = ECFPDataset(
            name=dataset_name,
            split_type='random',
            target_index=0,
            n_bits=n_bits,
            radius=radius,
            use_count=use_count
        )
        dataset.apply_transform(transform_obj)
    else:
        original_dataset = None
    
    # Downstream task
    result = run_downstream_task(
        dataset,
        task_type=task_type,
        hidden_dim=128,
        epochs=epochs,
        lr=1e-3,
        device='cpu'
    )
    
    print(f'  {metric_name}: Val={result["val"]:.4f}, Test={result["test"]:.4f}')
    
    # Distance preservation
    distances = None
    if analyze_distances and transform_obj is not None:
        distances = analyze_distance_preservation(
            original_dataset,
            dataset,
            correlation_method='spearman',
            sample_size=sample_size
        )
        print(f'  Distance preservation: J={distances["tanimoto"]["correlation"]:.3f}, '
              f'E={distances["euclidean"]["correlation"]:.3f}, '
              f'C={distances["cosine"]["correlation"]:.3f}')
    
    return result, distances

def run_multiradius_experiment(
    experiment_name,
    dataset_name,
    use_count,
    nonlinearity='relu',
    normalize = True,
    n_bits=6144,
    n_bits_per_radius=2048,
    radius=2,
    epochs=100,
    sample_size=300
):
    """Run multi-radius experiment with block mixing for Binary or Count fingerprints."""
    task_type = 'classification' if dataset_name == 'bace' else 'regression'
    metric_name = 'ROC-AUC' if task_type == 'classification' else 'RMSE'
    ecfp_type = 'Count' if use_count else 'Binary'
    
    print(f'\n[{experiment_name}] {dataset_name.upper()} - Multi-Radius {ecfp_type}')
    
    # Load multi-radius dataset
    original_dataset = ECFPDataset(
        name=dataset_name,
        split_type='random',
        target_index=0,
        radius=radius,
        n_bits=n_bits,
        use_count=use_count,
        multi_radius=False,
        #n_bits_per_radius=n_bits_per_radius
    )
    
    # Create transformation if nonlinearity is provided
    if nonlinearity is not None:
        transform = BlockRadiusLinearMixing(
            radius_blocks=original_dataset.radius_schema.blocks,
            nonlinearity=nonlinearity,
            seed=42,
            normalize=normalize
        )
        
        dataset = ECFPDataset(
            name=dataset_name,
            split_type='random',
            target_index=0,
            radius=radius,
            n_bits=n_bits,
            use_count=use_count,
            multi_radius=True,
            n_bits_per_radius=n_bits_per_radius
        )
        dataset.apply_transform(transform)
    else:
        dataset = original_dataset
    
    # Downstream task
    result = run_downstream_task(
        dataset,
        task_type=task_type,
        hidden_dim=128,
        epochs=epochs,
        lr=1e-3,
        device='cpu'
    )
    
    print(f'  {metric_name}: Val={result["val"]:.4f}, Test={result["test"]:.4f}')
    
    # Distance preservation (compare transformed to original multi-radius)
    distances = None
    if nonlinearity is not None:
        distances = analyze_distance_preservation(
            original_dataset,
            dataset,
            correlation_method='spearman',
            sample_size=sample_size
        )
        print(f'  Distance preservation: T={distances["tanimoto"]["correlation"]:.3f}, '
              f'E={distances["euclidean"]["correlation"]:.3f}, '
              f'C={distances["cosine"]["correlation"]:.3f}')
    
    
    return result, distances
