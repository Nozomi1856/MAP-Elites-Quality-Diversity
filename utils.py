"""
Helper utilities for saving/loading models and architectures
"""
import json
import os
import torch
from datetime import datetime
from typing import Dict, List
import matplotlib.pyplot as plt


def save_architecture_json(arch, filepath: str, metadata: Dict = None):
    """
    Save architecture as JSON
    
    Args:
        arch: ArchitectureState object
        filepath: Path to save JSON
        metadata: Optional metadata (rewards, accuracy, etc.)
    """
    arch_dict = {
        'nodes': arch.nodes,
        'edges': arch.edges,
        'operations': arch.operations,
        'channels': arch.channels,
        'positions': arch.positions,
        'input_node': arch.input_node,
        'output_node': arch.output_node,
        'metadata': metadata or {}
    }
    
    with open(filepath, 'w') as f:
        json.dump(arch_dict, f, indent=2)
    
    return filepath


def load_architecture_json(filepath: str):
    """
    Load architecture from JSON
    
    Returns:
        ArchitectureState object
    """
    from architecture import ArchitectureState
    
    with open(filepath, 'r') as f:
        arch_dict = json.load(f)
    
    arch = ArchitectureState()
    arch.nodes = arch_dict['nodes']
    arch.edges = [(tuple(e) if isinstance(e, list) else e) 
                  for e in arch_dict['edges']]
    arch.operations = {int(k): v for k, v in arch_dict['operations'].items()}
    arch.channels = {int(k): v for k, v in arch_dict['channels'].items()}
    arch.positions = {int(k): v for k, v in arch_dict['positions'].items()}
    arch.input_node = arch_dict['input_node']
    arch.output_node = arch_dict['output_node']
    arch.node_counter = max(arch.nodes) + 1 if arch.nodes else 0
    
    return arch


def save_model_pth(model, filepath: str, metadata: Dict = None):
    """
    Save PyTorch model with metadata
    
    Args:
        model: PyTorch model (ConvNet)
        filepath: Path to save .pth file
        metadata: Optional metadata
    """
    save_dict = {
        'model_state_dict': model.state_dict(),
        'architecture': {
            'nodes': model.arch.nodes,
            'edges': model.arch.edges,
            'operations': model.arch.operations,
            'channels': model.arch.channels,
            'positions': model.arch.positions,
            'input_node': model.arch.input_node,
            'output_node': model.arch.output_node,
        },
        'metadata': metadata or {},
        'timestamp': datetime.now().isoformat()
    }
    
    torch.save(save_dict, filepath)
    return filepath


def save_all_results(output_dir: str, best_archs: List, final_results: List, 
                     stats: Dict, dataset_name: str):
    """
    Save all results in multiple formats
    
    Creates:
    - models/ folder with .pth files
    - architectures/ folder with .json files
    - results.jsonl with all architecture metadata
    - summary.json with overview
    """
    os.makedirs(output_dir, exist_ok=True)
    
    models_dir = os.path.join(output_dir, 'models')
    arch_dir = os.path.join(output_dir, 'architectures')
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(arch_dir, exist_ok=True)
    
    # Save each result
    results_jsonl = []
    
    for i, result in enumerate(final_results):
        arch = result['architecture']
        
        # Create unique ID
        arch_id = f"arch_{i:03d}"
        
        # Save model as .pth
        from evaluation import ConvNet
        model = ConvNet(arch, num_classes=10)
        model_path = os.path.join(models_dir, f"{arch_id}.pth")
        
        metadata = {
            'arch_id': arch_id,
            'search_reward': result['search_reward'],
            'final_accuracy': result['final_accuracy'],
            'dataset': dataset_name,
            'depth': arch.depth,
            'avg_width': arch.avg_width,
            'total_params': arch.total_params,
            'num_skip_connections': arch.num_skip_connections,
        }
        
        if 'trajectory' in result and len(result['trajectory']) > 0:
            last_step = result['trajectory'][-1]
            if 'components' in last_step:
                components = last_step['components']
                metadata.update({
                    'topological_novelty': components.get('topological_novelty', 0),
                    'scale_novelty': components.get('scale_novelty', 0),
                })
        
        save_model_pth(model, model_path, metadata)
        
        # Save architecture as JSON
        json_path = os.path.join(arch_dir, f"{arch_id}.json")
        save_architecture_json(arch, json_path, metadata)
        
        # Add to JSONL
        results_jsonl.append(metadata)
        
        print(f"Saved {arch_id}: accuracy={metadata['final_accuracy']:.4f}")
    
    # Save results.jsonl
    jsonl_path = os.path.join(output_dir, 'results.jsonl')
    with open(jsonl_path, 'w') as f:
        for result in results_jsonl:
            f.write(json.dumps(result) + '\n')
    
    # Save summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'dataset': dataset_name,
        'total_episodes': len(stats['episode_rewards']),
        'architectures_explored': len(best_archs),
        'architectures_evaluated': len(final_results),
        'best_accuracy': max(r['final_accuracy'] for r in final_results) if final_results else 0,
        'avg_accuracy': sum(r['final_accuracy'] for r in final_results) / len(final_results) if final_results else 0,
        'training_stats': {
            'final_epsilon': stats['epsilons'][-1] if stats['epsilons'] else 0,
            'avg_reward': sum(stats['episode_rewards']) / len(stats['episode_rewards']) if stats['episode_rewards'] else 0,
        }
    }
    
    summary_path = os.path.join(output_dir, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"All results saved to: {output_dir}")
    print(f"  - Models: {models_dir}/ ({len(final_results)} .pth files)")
    print(f"  - Architectures: {arch_dir}/ ({len(final_results)} .json files)")
    print(f"  - Results: {jsonl_path}")
    print(f"  - Summary: {summary_path}")
    print(f"{'='*60}")


def plot_training_stats(stats, save_path='training_stats.png'):
    """Plot training statistics"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Episode rewards
    axes[0, 0].plot(stats['episode_rewards'])
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].grid(True)
    
    # Training loss
    if stats['losses']:
        axes[0, 1].plot(stats['losses'])
        axes[0, 1].set_title('Training Loss')
        axes[0, 1].set_xlabel('Update Step')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].grid(True)
    
    # Epsilon decay
    axes[1, 0].plot(stats['epsilons'])
    axes[1, 0].set_title('Epsilon Decay')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Epsilon')
    axes[1, 0].grid(True)
    
    # Reward distribution
    axes[1, 1].hist(stats['episode_rewards'], bins=50)
    axes[1, 1].set_title('Reward Distribution')
    axes[1, 1].set_xlabel('Reward')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved training stats to {save_path}")
