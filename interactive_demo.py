"""
Interactive demo UI for Creative NAS

Allows users to quickly explore creative architecture search
with limited compute budget (demo mode)
"""
import sys
import os
from datetime import datetime


def print_banner():
    print("\n" + "="*70)
    print("  🎨 CREATIVE NEURAL ARCHITECTURE SEARCH - INTERACTIVE DEMO 🎨")
    print("="*70 + "\n")


def get_user_choice(prompt, options, default=None):
    """
    Get user choice from options
    """
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
    """
    Get numeric input with validation
    """
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
    """
    Get yes/no confirmation
    """
    while True:
        response = input(f"\n{prompt} (y/n): ").strip().lower()
        if response in ['y', 'yes']:
            return True
        elif response in ['n', 'no']:
            return False
        print("❌ Please enter 'y' or 'n'")


def main():
    print_banner()
    
    print("Welcome! This demo will guide you through discovering creative")
    print("neural network architectures using reinforcement learning.\n")
    print("⏱️  Estimated time: 3-10 minutes depending on your settings")
    
    # Step 1: Choose dataset
    dataset_options = {
        'mnist': 'MNIST - Handwritten digits (fastest, ~2-3 min)',
        'fashion': 'Fashion-MNIST - Clothing items (fast, ~3-5 min)',
        'cifar10': 'CIFAR-10 - Natural images (slower, ~5-10 min)'
    }
    
    dataset = get_user_choice(
        "📊 Step 1: Choose dataset to optimize for:",
        dataset_options,
        default='mnist'
    )
    
    # Step 2: Choose number of episodes
    print("\n🔄 Step 2: Training episodes")
    print("   More episodes = more exploration but longer time")
    episodes = get_number_input(
        "   How many episodes?",
        min_val=50,
        max_val=200,
        default=100
    )
    
    # Step 3: Choose evaluation depth
    print("\n🎯 Step 3: Architecture evaluation")
    print("   More epochs = better accuracy estimates but slower")
    eval_epochs = get_number_input(
        "   Epochs per architecture?",
        min_val=2,
        max_val=5,
        default=3
    )
    
    # Step 4: Number of top architectures to save
    print("\n💾 Step 4: Results to save")
    top_k = get_number_input(
        "   How many top architectures to fully evaluate?",
        min_val=5,
        max_val=20,
        default=10
    )
    
    # Estimate time
    time_estimate = {
        'mnist': episodes * 0.02 + top_k * 0.15,
        'fashion': episodes * 0.025 + top_k * 0.2,
        'cifar10': episodes * 0.04 + top_k * 0.4
    }[dataset]
    
    # Summary
    print("\n" + "="*70)
    print("📋 CONFIGURATION SUMMARY")
    print("="*70)
    print(f"  Dataset:            {dataset_options[dataset].split('-')[0].strip()}")
    print(f"  Training Episodes:  {episodes}")
    print(f"  Evaluation Epochs:  {eval_epochs}")
    print(f"  Top K Architectures: {top_k}")
    print(f"  Estimated Time:     ~{time_estimate:.1f} minutes")
    print("="*70)
    
    if not confirm_action("🚀 Ready to start?"):
        print("\n👋 Goodbye!")
        return
    
    # Run training
    print("\n" + "="*70)
    print("🎬 STARTING CREATIVE ARCHITECTURE SEARCH")
    print("="*70 + "\n")
    
    import torch
    from agent import CreativityDQN
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
    output_dir = f'demo_results/{dataset}_{timestamp}'
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize agent
    print("\n📦 Initializing agent...")
    agent = CreativityDQN(device=device)
    
    # Adjust dataset in evaluation
    import evaluation
    original_get_loaders = evaluation.get_cifar10_loaders
    
    if dataset == 'mnist':
        def get_mnist_loaders(batch_size=128, subset_size=None):
            import torchvision
            import torchvision.transforms as transforms
            from torch.utils.data import DataLoader, Subset
            import numpy as np
            
            transform = transforms.Compose([
                transforms.Resize(32),  # Resize to 32x32
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # Convert to 3 channels
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
            import torchvision
            import torchvision.transforms as transforms
            from torch.utils.data import DataLoader, Subset
            import numpy as np
            
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
    
    # Override train_architecture to use fewer epochs
    original_train = evaluation.train_architecture
    def fast_train(arch, epochs=None, device='cuda', subset_size=10000):
        return original_train(arch, epochs=eval_epochs, device=device, subset_size=subset_size)
    evaluation.train_architecture = fast_train
    
    # Training
    print(f"\n🎯 Training for {episodes} episodes...")
    print("   (This is where the magic happens! 🪄)\n")
    
    best_archs, stats = agent.train(num_episodes=episodes, update_freq=5, eval_freq=25)
    
    # Final evaluation
    print(f"\n{'='*70}")
    print(f"🎓 FINAL EVALUATION: Top {top_k} Architectures")
    print(f"{'='*70}\n")
    
    # Restore original for final eval
    evaluation.train_architecture = original_train
    
    final_results = []
    for i, arch_data in enumerate(best_archs[:top_k]):
        arch = arch_data['architecture']
        print(f"\n[{i+1}/{top_k}] Evaluating architecture...")
        print(f"  Nodes: {len(arch.nodes)}, Depth: {arch.depth}, Width: {arch.avg_width:.1f}")
        
        final_acc = original_train(
            arch,
            epochs=10,  # Longer for final eval
            device=device,
            subset_size=None
        )
        
        print(f"  ✅ Final Accuracy: {final_acc:.4f}")
        
        final_results.append({
            'architecture': arch,
            'search_reward': arch_data['reward'],
            'final_accuracy': final_acc,
            'trajectory': arch_data.get('trajectory', [])
        })
    
    # Save everything
    print(f"\n{'='*70}")
    print("💾 SAVING RESULTS")
    print(f"{'='*70}\n")
    
    save_all_results(output_dir, best_archs, final_results, stats, dataset)
    
    # Create visualizations
    print("\n📊 Creating visualizations...")
    create_results_report(output_dir)
    
    # Final summary
    best = max(final_results, key=lambda x: x['final_accuracy'])
    
    print(f"\n{'='*70}")
    print("🎉 SUCCESS! RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"📁 Results saved to: {output_dir}/")
    print(f"\n🏆 Best Architecture:")
    print(f"   Accuracy:  {best['final_accuracy']:.4f}")
    print(f"   Reward:    {best['search_reward']:.4f}")
    print(f"   Depth:     {best['architecture'].depth}")
    print(f"   Width:     {best['architecture'].avg_width:.1f}")
    print(f"   Nodes:     {len(best['architecture'].nodes)}")
    print(f"\n📊 View detailed report:")
    print(f"   {output_dir}/report.html")
    print(f"\n📦 Model files saved as:")
    print(f"   - PyTorch models (.pth): {output_dir}/models/")
    print(f"   - Architecture JSON: {output_dir}/architectures/")
    print(f"   - Results summary: {output_dir}/results.jsonl")
    print(f"{'='*70}\n")
    
    print("✨ Thank you for using Creative NAS! ✨")


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
