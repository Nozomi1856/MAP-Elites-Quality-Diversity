"""
Training script for MAP-Elites Creative Architecture Search
"""
import torch
import argparse
import os
from datetime import datetime

from map_elites import MAPElites, BehaviorSpace, MutationOperator, QualityDiversityArchive
from architecture import ArchitectureState
from evaluation import train_architecture
from utils import save_all_results
from visualize import create_results_report


def main():
    parser = argparse.ArgumentParser(description='MAP-Elites Creative Architecture Search')
    parser.add_argument('--iterations', type=int, default=1000,
                       help='Number of MAP-Elites iterations')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--output_dir', type=str, default='results_mapelites',
                       help='Output directory')
    parser.add_argument('--eval_top_k', type=int, default=20,
                       help='Number of top architectures to fully evaluate')
    parser.add_argument('--dataset', type=str, default='cifar10',
                       choices=['cifar10', 'mnist', 'fashion'],
                       help='Dataset to use')
    parser.add_argument('--depth_bins', type=int, default=5,
                       help='Number of bins for depth dimension')
    parser.add_argument('--width_bins', type=int, default=5,
                       help='Number of bins for width dimension')
    parser.add_argument('--skip_bins', type=int, default=4,
                       help='Number of bins for skip connection dimension')
    parser.add_argument('--eval_epochs', type=int, default=3,
                       help='Epochs for quick evaluation during search')
    parser.add_argument('--final_epochs', type=int, default=20,
                       help='Epochs for final evaluation')
    
    args = parser.parse_args()
    
    # Set seeds
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(args.output_dir, f'{args.dataset}_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    
    # Initialize MAP-Elites components
    print("\n" + "="*60)
    print("Initializing MAP-Elites")
    print("="*60)
    
    behavior_space = BehaviorSpace(
        depth_bins=args.depth_bins,
        width_bins=args.width_bins,
        skip_bins=args.skip_bins
    )
    
    mutation_operator = MutationOperator()
    
    map_elites = MAPElites(
        behavior_space=behavior_space,
        mutation_operator=mutation_operator,
        operation_strategy='diverse'
    )
    
    print(f"Behavior space: {behavior_space.get_total_cells()} cells")
    print(f"  Depth bins: {args.depth_bins}")
    print(f"  Width bins: {args.width_bins}")
    print(f"  Skip bins: {args.skip_bins}")
    
    # Define evaluation function
    def evaluate_architecture(arch: ArchitectureState) -> float:
        """Quick evaluation during search."""
        try:
            accuracy = train_architecture(
                arch,
                epochs=args.eval_epochs,
                device=args.device,
                subset_size=10000
            )
            return accuracy
        except Exception as e:
            print(f"Error evaluating architecture: {e}")
            return 0.1
    
    # Run MAP-Elites
    print(f"\n" + "="*60)
    print(f"Running MAP-Elites for {args.iterations} iterations")
    print("="*60 + "\n")
    
    archive = map_elites.run(
        num_iterations=args.iterations,
        evaluate_fn=evaluate_architecture,
        verbose=True
    )
    
    # Print final statistics
    stats = archive.get_stats()
    print(f"\n" + "="*60)
    print("MAP-Elites Complete")
    print("="*60)
    print(f"Coverage: {stats['coverage']:.2%}")
    print(f"Cells filled: {stats['num_filled']}/{behavior_space.get_total_cells()}")
    print(f"Best performance: {stats['best_performance']:.4f}")
    print(f"Mean performance: {stats['mean_performance']:.4f}")
    print(f"Total evaluated: {stats['total_evaluated']}")
    
    # Get all architectures sorted by performance
    all_archs = archive.get_all_architectures()
    all_archs.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n" + "="*60)
    print(f"Final Evaluation: Top {args.eval_top_k} Architectures")
    print("="*60 + "\n")
    
    # Prepare for final evaluation
    final_results = []
    
    for i, (arch, search_perf, behavior) in enumerate(all_archs[:args.eval_top_k]):
        print(f"\nArchitecture {i+1}/{args.eval_top_k}")
        print(f"  Search performance: {search_perf:.4f}")
        print(f"  Behavior: depth={behavior[0]}, width={behavior[1]}, skip={behavior[2]}")
        print(f"  Nodes: {len(arch.nodes)}, Edges: {len(arch.edges)}")
        print(f"  Depth: {arch.depth}, Avg Width: {arch.avg_width:.1f}")
        
        print("  Training (full evaluation)...")
        final_acc = train_architecture(
            arch,
            epochs=args.final_epochs,
            device=args.device,
            subset_size=None
        )
        
        print(f"  Final Accuracy: {final_acc:.4f}")
        
        final_results.append({
            'architecture': arch,
            'search_reward': search_perf,  # Use search performance as "reward"
            'final_accuracy': final_acc,
            'trajectory': [],  # MAP-Elites doesn't have trajectories
            'behavior': behavior
        })
    
    # Convert archive history to "best_archs" format for compatibility
    best_archs = []
    for item in archive.history:
        best_archs.append({
            'architecture': item['architecture'],
            'reward': item['performance'],
            'trajectory': [],
            'episode': 0
        })
    
    # Create dummy stats for compatibility with save_all_results
    dummy_stats = {
        'episode_rewards': [item['performance'] for item in archive.history],
        'losses': [],
        'epsilons': []
    }
    
    # Save all results
    print(f"\n" + "="*60)
    print("Saving Results")
    print("="*60 + "\n")
    
    save_all_results(output_dir, best_archs, final_results, dummy_stats, args.dataset)
    
    # Create visualization report
    create_results_report(output_dir)
    
    # Save MAP-Elites specific data
    import json
    mapelites_data = {
        'algorithm': 'MAP-Elites',
        'behavior_space': {
            'depth_bins': args.depth_bins,
            'width_bins': args.width_bins,
            'skip_bins': args.skip_bins,
            'total_cells': behavior_space.get_total_cells()
        },
        'final_stats': stats,
        'iterations': args.iterations
    }
    
    with open(os.path.join(output_dir, 'mapelites_info.json'), 'w') as f:
        json.dump(mapelites_data, f, indent=2)
    
    # Print summary
    print(f"\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Algorithm: MAP-Elites")
    print(f"Total iterations: {args.iterations}")
    print(f"Architectures explored: {stats['total_evaluated']}")
    print(f"Archive coverage: {stats['coverage']:.2%}")
    print(f"Top architectures evaluated: {len(final_results)}")
    
    if final_results:
        best_result = max(final_results, key=lambda x: x['final_accuracy'])
        print(f"\nBest Architecture:")
        print(f"  Final Accuracy: {best_result['final_accuracy']:.4f}")
        print(f"  Search Performance: {best_result['search_reward']:.4f}")
        print(f"  Behavior: {best_result['behavior']}")
    
    print(f"\nView results: {output_dir}/report.html")
    print("="*60)


if __name__ == '__main__':
    main()
