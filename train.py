"""
Main training script (updated with model saving)
"""
import torch
import argparse
import os
from datetime import datetime

from agent import CreativityDQN
from evaluation import train_architecture
from utils import save_all_results, plot_training_stats
from visualize import create_results_report


def evaluate_top_architectures(best_archs, top_k=20, device='cuda'):
    """Fully evaluate top architectures"""
    print(f"\n{'='*60}")
    print(f"Evaluating Top {top_k} Architectures (Full Training)")
    print(f"{'='*60}\n")
    
    results = []
    
    for i, arch_data in enumerate(best_archs[:top_k]):
        arch = arch_data['architecture']
        search_reward = arch_data['reward']
        
        print(f"\nArchitecture {i+1}/{top_k}")
        print(f"  Search Reward: {search_reward:.4f}")
        print(f"  Nodes: {len(arch.nodes)}, Edges: {len(arch.edges)}")
        print(f"  Depth: {arch.depth}, Avg Width: {arch.avg_width:.1f}")
        
        print("  Training (20 epochs)...")
        final_acc = train_architecture(
            arch,
            epochs=20,
            device=device,
            subset_size=None
        )
        
        print(f"  Final Accuracy: {final_acc:.4f}")
        
        results.append({
            'architecture': arch,
            'search_reward': search_reward,
            'final_accuracy': final_acc,
            'trajectory': arch_data.get('trajectory', [])
        })
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Creative Architecture Search')
    parser.add_argument('--episodes', type=int, default=1000,
                       help='Number of training episodes')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Output directory')
    parser.add_argument('--eval_top_k', type=int, default=20,
                       help='Number of top architectures to fully evaluate')
    parser.add_argument('--dataset', type=str, default='cifar10',
                       choices=['cifar10', 'mnist', 'fashion'],
                       help='Dataset to use')
    
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
    
    # Initialize agent
    print("\nInitializing agent...")
    agent = CreativityDQN(device=args.device)
    
    # Training
    print(f"\nStarting training for {args.episodes} episodes...")
    best_archs, stats = agent.train(num_episodes=args.episodes)
    
    # Save agent
    agent_path = os.path.join(output_dir, 'agent.pt')
    agent.save(agent_path)
    print(f"\nSaved agent to {agent_path}")
    
    # Plot training stats
    plot_path = os.path.join(output_dir, 'training_stats.png')
    plot_training_stats(stats, plot_path)
    
    # Evaluate top architectures
    final_results = evaluate_top_architectures(
        best_archs, 
        top_k=args.eval_top_k,
        device=args.device
    )
    
    # Save all results
    save_all_results(output_dir, best_archs, final_results, stats, args.dataset)
    
    # Create visualization report
    create_results_report(output_dir)
    
    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total episodes: {args.episodes}")
    print(f"Architectures explored: {len(best_archs)}")
    print(f"Top architectures evaluated: {len(final_results)}")
    
    if final_results:
        best_result = max(final_results, key=lambda x: x['final_accuracy'])
        print(f"\nBest Architecture:")
        print(f"  Final Accuracy: {best_result['final_accuracy']:.4f}")
        print(f"  Search Reward: {best_result['search_reward']:.4f}")
    
    print(f"\nView results: {output_dir}/report.html")


if __name__ == '__main__':
    main()
