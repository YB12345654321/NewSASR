import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
import argparse

def plot_incremental_performance(results, output_dir):
    """
    Plot performance across slices for different models
    """
    # Extract data
    base_results = results['base_model']
    incremental_results = results['incremental_model']
    joint_results = results['joint_model']
    
    # Determine number of slices
    num_slices = len([k for k in base_results.keys() if k.startswith('slice_')])
    
    # Extract NDCG for base model
    base_ndcg = []
    for i in range(num_slices):
        base_ndcg.append(base_results[f'slice_{i}']['ndcg'])
    
    # Find latest incremental update
    max_slice = max([int(k.split('_')[2]) for k in incremental_results.keys() 
                   if k.startswith('after_slice_')])
    
    # Extract NDCG for final incremental model
    inc_ndcg = []
    for i in range(num_slices):
        inc_ndcg.append(incremental_results[f'after_slice_{max_slice}_eval_on_{i}']['ndcg'])
    
    # Extract NDCG for final joint model with adaptive weighting
    joint_ndcg = []
    for i in range(num_slices):
        joint_ndcg.append(joint_results[f'after_slice_{max_slice}_adaptive_eval_on_{i}']['ndcg'])
    
    # Create bar chart
    plt.figure(figsize=(12, 6))
    
    x = np.arange(num_slices)
    width = 0.25
    
    plt.bar(x - width, base_ndcg, width, label='Base Model (T1)')
    plt.bar(x, inc_ndcg, width, label='Incremental Model')
    plt.bar(x + width, joint_ndcg, width, label='Joint Model (Adaptive)')
    
    plt.xlabel('Time Slice')
    plt.ylabel('NDCG@10')
    plt.title('Model Performance Across Time Slices')
    plt.xticks(x, [f'T{i+1}' for i in range(num_slices)])
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_comparison.png'))
    plt.close()
    
    # Track performance over incremental updates
    if max_slice > 1:
        plt.figure(figsize=(12, 6))
        
        # Performance on T1 (knowledge retention)
        t1_performance = [base_results['slice_0']['ndcg']]
        
        for i in range(1, max_slice + 1):
            t1_performance.append(incremental_results[f'after_slice_{i}_eval_on_0']['ndcg'])
        
        # Performance on latest slice (new knowledge acquisition)
        new_performance = [base_results[f'slice_{max_slice}']['ndcg']]
        
        for i in range(1, max_slice + 1):
            new_performance.append(incremental_results[f'after_slice_{i}_eval_on_{max_slice}']['ndcg'])
        
        x = np.arange(max_slice + 1)
        plt.plot(x, t1_performance, 'b-o', label='T1 Performance (Knowledge Retention)')
        plt.plot(x, new_performance, 'r-s', label=f'T{max_slice+1} Performance (New Knowledge)')
        
        plt.xlabel('Training Stage')
        plt.ylabel('NDCG@10')
        plt.title('Knowledge Retention and Acquisition')
        plt.xticks(x, ['Base'] + [f'After T{i+1}' for i in range(1, max_slice + 1)])
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'knowledge_retention.png'))
        plt.close()
    
    # Compare joint model alpha values
    plt.figure(figsize=(10, 6))
    
    alphas = [0.3, 0.5, 0.7]
    alpha_perf = []
    
    for alpha in alphas:
        avg_ndcg = 0
        for i in range(num_slices):
            avg_ndcg += joint_results[f'after_slice_{max_slice}_alpha_{alpha}_eval_on_{i}']['ndcg']
        avg_ndcg /= num_slices
        alpha_perf.append(avg_ndcg)
    
    # Get adaptive performance
    adaptive_perf = 0
    for i in range(num_slices):
        adaptive_perf += joint_results[f'after_slice_{max_slice}_adaptive_eval_on_{i}']['ndcg']
    adaptive_perf /= num_slices
    
    plt.bar(alphas, alpha_perf, width=0.1, color='blue', label='Static Alpha')
    plt.axhline(y=adaptive_perf, color='r', linestyle='--', label='Adaptive Weighting')
    
    plt.xlabel('Alpha Value (Weight for Base Model)')
    plt.ylabel('Average NDCG@10')
    plt.title('Joint Model Performance with Different Alpha Values')
    plt.legend()
    
    plt.savefig(os.path.join(output_dir, 'alpha_comparison.png'))
    plt.close()
    
    return True

def plot_forgetting_curve(results, output_dir):
    """
    Plot forgetting curves over time
    """
    # Extract data for forgetting analysis
    base_results = results['base_model']
    incremental_results = results['incremental_model']
    
    # Determine number of slices
    num_slices = len([k for k in base_results.keys() if k.startswith('slice_')])
    
    # Find maximum slice
    max_slice = max([int(k.split('_')[2]) for k in incremental_results.keys() 
                   if k.startswith('after_slice_')])
    
    if max_slice < 2:
        return  # Not enough incremental updates for meaningful forgetting curve
    
    plt.figure(figsize=(12, 6))
    
    # Calculate relative performance (% of original) for each slice
    for slice_idx in range(num_slices):
        base_perf = base_results[f'slice_{slice_idx}']['ndcg']
        
        relative_perf = [1.0]  # Start at 100% (base model)
        
        for update_idx in range(1, max_slice + 1):
            # After each incremental update, what's the performance on this slice?
            current_perf = incremental_results[f'after_slice_{update_idx}_eval_on_{slice_idx}']['ndcg']
            relative_perf.append(current_perf / base_perf)
        
        # Plot relative performance curve
        plt.plot(range(max_slice + 1), relative_perf, 
                marker='o' if slice_idx == 0 else ('s' if slice_idx == num_slices-1 else '^'),
                label=f'Slice T{slice_idx+1}')
    
    plt.xlabel('Incremental Updates')
    plt.ylabel('Relative Performance (% of Original)')
    plt.title('Knowledge Retention Over Incremental Updates')
    plt.xticks(range(max_slice + 1), ['Base'] + [f'After T{i+1}' for i in range(1, max_slice + 1)])
    plt.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'forgetting_curve.png'))
    plt.close()
    
    return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', required=True, help='Directory containing results.pkl')
    args = parser.parse_args()
    
    results_path = os.path.join(args.results_dir, 'results.pkl')
    
    if not os.path.exists(results_path):
        print(f"Results file not found: {results_path}")
        return
    
    with open(results_path, 'rb') as f:
        results = pickle.load(f)
    
    plot_incremental_performance(results, args.results_dir)
    plot_forgetting_curve(results, args.results_dir)
    
    print(f"Visualizations saved to {args.results_dir}")

if __name__ == "__main__":
    main()