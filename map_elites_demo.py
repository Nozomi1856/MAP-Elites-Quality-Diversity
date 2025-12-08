"""
Interactive demo for MAP-Elites Creative Architecture Search
"""
import sys
import os
from datetime import datetime


def print_banner():
    print("\n" + "="*70)
    print("  🎨 MAP-ELITES CREATIVE ARCHITECTURE SEARCH - DEMO 🎨")
    print("="*70 + "\n")


def get_user_choice(prompt, options, default=None):
    """Get user choice from options"""
    print(f"\n{prompt}")
    for i, (key, desc) in enumerate(options.items(), 1):
        default_marker = " (default)" if default == key else ""
        print(f"  {i}. {desc}{default_marker}")
    
    while True:
        choice = input(f"\nEnter choice (1-{len(options)}) [default: {default}]: ").strip()
        
        if choice == '' and default:
            return default
        
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(options):
                return list(options.keys())[idx]
        except:
            pass
        
        print("❌ Invalid choice. Please try again.")


def get_number_input(prompt, min_val, max_val, default):
    """Get numeric input with validation"""
    while True:
        response = input(f"\n{prompt} [{min_val}-{max_val}] (default: {default}): ").strip()
        
        if response == '':
            return default
        
        try:
            value = int(response)
            if min_val <= value <= max_val:
                return value
            else:
                print(f"❌ Please enter a number between {min_val} and {max_val}")
        except:
            print("❌ Please enter a valid number")


def confirm_action(prompt):
    """Get yes/no confirmation"""
    while True:
        response = input(f"\n{prompt} (y/n): ").strip().lower()
        if response in ['y', 'yes']:
            return True
        elif response in ['n', 'no']:
            return False
        print("❌ Please enter 'y' or 'n'")


def main():
    print_banner()
    
    print("MAP-Elites is a quality-diversity algorithm that discovers")
    print("diverse, high-quality architectures across different behaviors.\n")
    print("⏱️  Estimated time: 3-10 minutes depending on your settings")
    print("🚀 Simpler and faster than Deep RL approaches!")
    
    # Step 1: Choose dataset
    dataset_options = {
        'mnist': 'MNIST - Handwritten digits (fastest, ~2-3 min)',
        'fashion': 'Fashion-MNIST - Clothing items (fast, ~3-5 min)',
        'cifar10': 'CIFAR-10 - Natural images (slower, ~5-10 min)'
    }
    
    dataset = get_user_choice(
        "📊 Step 1: Choose dataset:",
        dataset_options,
        default='mnist'
    )
    
    # Step 2: Number of iterations
    print("\n🔄 Step 2: MAP-Elites iterations")
    print("   More iterations = better coverage but longer time")
    iterations = get_number_input(
        "   How many iterations?",
        min_val=100,
        max_val=500,
        default=200
    )
    
    # Step 3: Behavior space resolution
    print("\n🎯 Step 3: Behavior space resolution")
    print("   Higher resolution = more diverse results but slower")
    resolution_options = {
        'low': 'Low (3x3x3 = 27 cells, fastest)',
        'medium': 'Medium (5x5x4 = 100 cells, balanced)',
        'high': 'High (7x7x5 = 245 cells, most diverse)'
    }
    
    resolution = get_user_choice(
        "   Choose resolution:",
        resolution_options,
        default='medium'
    )
    
    # Map resolution to bins
    bin_config = {
        'low': (3, 3, 3),
        'medium': (5, 5, 4),
        'high': (7, 7, 5)
    }
    depth_bins, width_bins, skip_bins = bin_config[resolution]
    
    # Step 4: Number of top architectures to save
    print("\n💾 Step 4: Results to save")
    top_k = get_number_input(
        "   How many top architectures to fully evaluate?",
        min_val=5,
        max_val=20,
        default=10
    )
    
    # Estimate time
    time_estimates = {
        'mnist': iterations * 0.015 + top_k * 0.15,
        'fashion': iterations * 0.02 + top_k * 0.2,
        'cifar10': iterations * 0.03 + top_k * 0.4
    }
    time_estimate = time_estimates[dataset]
    
    # Summary
    print("\n" + "="*70)
    print("📋 CONFIGURATION SUMMARY")
    print("="*70)
    print(f"  Algorithm:          MAP-Elites (Quality-Diversity)")
    print(f"  Dataset:            {dataset_options[dataset].split('-')[0].strip()}")
    print(f"  Iterations:         {iterations}")
    print(f"  Behavior Space:     {depth_bins}×{width_bins}×{skip_bins} = {depth_bins*width_bins*skip_bins} cells")
    print(f"  Top K:              {top_k}")
    print(f"  Estimated Time:     ~{time_estimate:.1f} minutes")
    print("="*70)
    
    if not confirm_action("🚀 Ready to start?"):
        print("\n👋 Goodbye!")
        return
    
    # Run MAP-Elites
    print("\n" + "="*70)
    print("🎬 STARTING MAP-ELITES SEARCH")
    print("="*70 + "\n")
    
    import torch
    from map_elites import MAPElites, BehaviorSpace, MutationOperator
    from architecture import ArchitectureState
    from evaluation import train_architecture
    from utils import save_all_results
    from visualize import create_results_report
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🖥️  Using device: {device}")
    
    if device == 'cpu':
        print("⚠️  Warning: No GPU detected. Training will be slower.")
        if not confirm_action("   Continue anyway?"):
            return
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f'demo_results_mapelites/{dataset}_{timestamp}'
    os.makedirs(output_dir, exist_ok=True)
    
    # Configure dataset
    import evaluation
    import torchvision
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader, Subset
    import numpy as np
    
    if dataset == 'mnist':
        def get_mnist_loaders(batch_size=128, subset_size=None):
            transform = transforms.Compose([
                transforms.Resize(32),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                transforms.Normalize((0.1307,)*3, (0.3081,)*3)
            ])
            
            trainset = torchvision.datasets.MNIST(
                root='./data', train=True, download=True, transform=transform
            )
            testset = torchvision.datasets.MNIST(
                root='./data', train=False, download=True, transform=transform
            )
            
            if subset_size:
                indices = np.random.choice(len(trainset), subset_size, replace=False)
                trainset = Subset(trainset, indices)
            
            return (DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2),
                   DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2))
        
        evaluation.get_cifar10_loaders = get_mnist_loaders
    
    elif dataset == 'fashion':
        def get_fashion_loaders(batch_size=128, subset_size=None):
            transform = transforms.Compose([
                transforms.Resize(32),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                transforms.Normalize((0.2860,)*3, (0.3530,)*3)
            ])
            
            trainset = torchvision.datasets.FashionMNIST(
                root='./data', train=True, download=True, transform=transform
            )
            testset = torchvision.datasets.FashionMNIST(
                root='./data', train=False, download=True, transform=transform
            )
            
            if subset_size:
                indices = np.random.choice(len(trainset), subset_size, replace=False)
                trainset = Subset(trainset, indices)
            
            return (DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2),
                   DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2))
        
        evaluation.get_cifar10_loaders = get_fashion_loaders
    
    # Initialize MAP-Elites
    print("\n📦 Initializing MAP-Elites...")
    
    behavior_space = BehaviorSpace(
        depth_bins=depth_bins,
        width_bins=width_bins,
        skip_bins=skip_bins
    )
    
    mutation_operator = MutationOperator()
    
    map_elites = MAPElites(
        behavior_space=behavior_space,
        mutation_operator=mutation_operator,
        operation_strategy='diverse'
    )
    
    print(f"✅ Behavior space: {behavior_space.get_total_cells()} cells")
    
    # Define evaluation function
    def evaluate_architecture(arch: ArchitectureState) -> float:
        try:
            accuracy = train_architecture(
                arch,
                epochs=3,
                device=device,
                subset_size=10000
            )
            return accuracy
        except Exception as e:
            return 0.1
    
    # Run MAP-Elites
    print(f"\n🎯 Running MAP-Elites for {iterations} iterations...")
    print("   (Watch the coverage increase! 📈)\n")
    
    archive = map_elites.run(
        num_iterations=iterations,
        evaluate_fn=evaluate_architecture,
        verbose=True
    )
    
    # Statistics
    stats = archive.get_stats()
    print(f"\n✅ MAP-Elites complete!")
    print(f"   Coverage: {stats['coverage']:.2%}")
    print(f"   Cells filled: {stats['num_filled']}/{behavior_space.get_total_cells()}")
    print(f"   Best performance: {stats['best_performance']:.4f}")
    
    # Final evaluation
    print(f"\n{'='*70}")
    print(f"🎓 FINAL EVALUATION: Top {top_k} Architectures")
    print(f"{'='*70}\n")
    
    all_archs = archive.get_all_architectures()
    all_archs.sort(key=lambda x: x[1], reverse=True)
    
    final_results = []
    for i, (arch, search_perf, behavior) in enumerate(all_archs[:top_k]):
        print(f"\n[{i+1}/{top_k}] Evaluating...")
        print(f"  Behavior: depth={behavior[0]}, width={behavior[1]}, skip={behavior[2]}")
        
        final_acc = train_architecture(
            arch,
            epochs=10,
            device=device,
            subset_size=None
        )
        
        print(f"  ✅ Accuracy: {final_acc:.4f}")
        
        final_results.append({
            'architecture': arch,
            'search_reward': search_perf,
            'final_accuracy': final_acc,
            'trajectory': [],
            'behavior': behavior
        })
    
    # Save results
    print(f"\n{'='*70}")
    print("💾 SAVING RESULTS")
    print(f"{'='*70}\n")
    
    best_archs = [{
        'architecture': item['architecture'],
        'reward': item['performance'],
        'trajectory': [],
        'episode': 0
    } for item in archive.history]
    
    dummy_stats = {
        'episode_rewards': [item['performance'] for item in archive.history],
        'losses': [],
        'epsilons': []
    }
    
    save_all_results(output_dir, best_archs, final_results, dummy_stats, dataset)
    create_results_report(output_dir)
    
    # Best architecture
    best = max(final_results, key=lambda x: x['final_accuracy'])
    
    print(f"\n{'='*70}")
    print("🎉 SUCCESS! RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"📁 Results saved to: {output_dir}/")
    print(f"\n🏆 Best Architecture:")
    print(f"   Accuracy:   {best['final_accuracy']:.4f}")
    print(f"   Behavior:   {best['behavior']}")
    print(f"   Depth:      {best['architecture'].depth}")
    print(f"   Width:      {best['architecture'].avg_width:.1f}")
    print(f"\n📊 Algorithm Statistics:")
    print(f"   Coverage:   {stats['coverage']:.2%}")
    print(f"   Cells:      {stats['num_filled']}/{behavior_space.get_total_cells()}")
    print(f"   Explored:   {stats['total_evaluated']} architectures")
    print(f"\n📦 Files saved:")
    print(f"   - Models (.pth): {output_dir}/models/")
    print(f"   - Architectures (.json): {output_dir}/architectures/")
    print(f"   - Report: {output_dir}/report.html")
    print(f"{'='*70}\n")
    
    print("✨ Thank you for using MAP-Elites! ✨")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user. Exiting...")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
