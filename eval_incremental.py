import torch
import numpy as np
import copy
from util import evaluate

def evaluate_incremental(model, datasets, args, device='cuda'):
    """
    Evaluate model on multiple time slices
    
    Args:
        model: Model to evaluate
        datasets: List of datasets for each time slice
        args: Command line arguments
        device: Device to run evaluation on
        
    Returns:
        Dictionary of results for each slice
    """
    model.eval()
    results = {}
    
    # Evaluate on each time slice
    for idx, dataset in enumerate(datasets):
        ndcg, hr = evaluate(model, dataset, args, device)
        results[f'slice_{idx}'] = {
            'ndcg': ndcg, 
            'hr': hr
        }
    
    model.train()
    return results

def evaluate_forgetting(base_model, incremental_model, datasets, args, device='cuda'):
    """
    Measure knowledge retention and forgetting
    
    Args:
        base_model: Model trained on T1
        incremental_model: Model incrementally trained on T2+
        datasets: Time-sliced datasets
        args: Command line arguments
        device: Device to run evaluation on
        
    Returns:
        Dictionary of retention metrics
    """
    base_model.eval()
    incremental_model.eval()
    
    metrics = {}
    
    # Evaluate each model on each slice
    for slice_idx, dataset in enumerate(datasets):
        # Base model performance
        base_ndcg, base_hr = evaluate(base_model, dataset, args, device)
        
        # Incremental model performance
        incr_ndcg, incr_hr = evaluate(incremental_model, dataset, args, device)
        
        # Calculate retention rate
        if base_ndcg > 0:
            ndcg_retention = incr_ndcg / base_ndcg
        else:
            ndcg_retention = 0
            
        if base_hr > 0:
            hr_retention = incr_hr / base_hr
        else:
            hr_retention = 0
        
        metrics[f'slice_{slice_idx}'] = {
            'base_ndcg': base_ndcg,
            'base_hr': base_hr,
            'incr_ndcg': incr_ndcg,
            'incr_hr': incr_hr,
            'ndcg_retention': ndcg_retention,
            'hr_retention': hr_retention
        }
    
    # Calculate average retention
    avg_ndcg_retention = np.mean([metrics[f'slice_{idx}']['ndcg_retention'] 
                                 for idx in range(len(datasets))])
    avg_hr_retention = np.mean([metrics[f'slice_{idx}']['hr_retention'] 
                               for idx in range(len(datasets))])
    
    metrics['average'] = {
        'ndcg_retention': avg_ndcg_retention,
        'hr_retention': avg_hr_retention
    }
    
    return metrics