"""
Architecture evaluation functions
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import numpy as np
from architecture import ArchitectureState, OPERATION_POOL
from tqdm import tqdm


class ConvNet(nn.Module):
    """
    Converts ArchitectureState to executable PyTorch model
    """
    def __init__(self, arch: ArchitectureState, num_classes=10):
        super().__init__()
        
        self.arch = arch
        self.num_classes = num_classes
        
        # Build layers
        self.layers = nn.ModuleDict()
        
        for node in arch.nodes:
            op = arch.operations[node]
            channels = arch.channels[node]
            
            # Get input channels (from predecessors)
            predecessors = [src for src, dst in arch.edges if dst == node]
            if len(predecessors) == 0:
                in_channels = 3  # Input image
            else:
                # Average of predecessor channels
                in_channels = int(np.mean([arch.channels[p] for p in predecessors]))
            
            # Create layer based on operation
            if op == 'conv3x3':
                layer = nn.Sequential(
                    nn.Conv2d(in_channels, channels, 3, padding=1),
                    nn.BatchNorm2d(channels),
                    nn.ReLU()
                )
            elif op == 'conv5x5':
                layer = nn.Sequential(
                    nn.Conv2d(in_channels, channels, 5, padding=2),
                    nn.BatchNorm2d(channels),
                    nn.ReLU()
                )
            elif op == 'sep_conv3x3':
                layer = nn.Sequential(
                    nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels),
                    nn.Conv2d(in_channels, channels, 1),
                    nn.BatchNorm2d(channels),
                    nn.ReLU()
                )
            elif op == 'sep_conv5x5':
                layer = nn.Sequential(
                    nn.Conv2d(in_channels, in_channels, 5, padding=2, groups=in_channels),
                    nn.Conv2d(in_channels, channels, 1),
                    nn.BatchNorm2d(channels),
                    nn.ReLU()
                )
            elif op == 'max_pool3x3':
                layer = nn.Sequential(
                    nn.MaxPool2d(3, stride=1, padding=1),
                    nn.Conv2d(in_channels, channels, 1) if in_channels != channels else nn.Identity()
                )
            elif op == 'avg_pool3x3':
                layer = nn.Sequential(
                    nn.AvgPool2d(3, stride=1, padding=1),
                    nn.Conv2d(in_channels, channels, 1) if in_channels != channels else nn.Identity()
                )
            elif op == 'skip_connect':
                if in_channels != channels:
                    layer = nn.Conv2d(in_channels, channels, 1)
                else:
                    layer = nn.Identity()
            else:
                layer = nn.Identity()
            
            self.layers[str(node)] = layer
        
        # Final classifier
        final_channels = arch.channels[arch.output_node]
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(final_channels, num_classes)
        )
        
    def forward(self, x):
        # Store outputs of each node
        outputs = {}
        
        # Topological sort for forward pass
        import networkx as nx
        G = self.arch.to_networkx()
        try:
            order = list(nx.topological_sort(G))
        except:
            # Fallback to node order
            order = sorted(self.arch.nodes)
        
        for node in order:
            # Get inputs
            predecessors = [src for src, dst in self.arch.edges if dst == node]
            
            if len(predecessors) == 0:
                # Input node
                node_input = x
            elif len(predecessors) == 1:
                node_input = outputs[predecessors[0]]
            else:
                # Multiple inputs - average them
                inputs = [outputs[p] for p in predecessors]
                # Make sure all have same shape
                min_size = min(inp.shape[2] for inp in inputs)
                inputs = [F.adaptive_avg_pool2d(inp, min_size) for inp in inputs]
                node_input = torch.stack(inputs).mean(dim=0)
            
            # Apply layer
            outputs[node] = self.layers[str(node)](node_input)
        
        # Get output
        final_output = outputs[self.arch.output_node]
        logits = self.classifier(final_output)
        
        return logits


def get_cifar10_loaders(batch_size=128, subset_size=None):
    """Get CIFAR-10 dataloaders"""
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train
    )
    
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test
    )
    
    # Use subset if specified (for faster training)
    if subset_size:
        indices = np.random.choice(len(trainset), subset_size, replace=False)
        trainset = Subset(trainset, indices)
    
    trainloader = DataLoader(trainset, batch_size=batch_size, 
                            shuffle=True, num_workers=2)
    testloader = DataLoader(testset, batch_size=batch_size, 
                           shuffle=False, num_workers=2)
    
    return trainloader, testloader


def train_architecture(arch: ArchitectureState, epochs=3, 
                       device='cuda', subset_size=10000):
    """
    Train an architecture and return validation accuracy
    
    Args:
        arch: ArchitectureState to evaluate
        epochs: Number of training epochs
        device: 'cuda' or 'cpu'
        subset_size: Size of training subset (for speed)
        
    Returns:
        validation_accuracy: Float between 0 and 1
    """
    try:
        # Create model
        model = ConvNet(arch).to(device)
        
        # Get data
        trainloader, testloader = get_cifar10_loaders(
            batch_size=128,
            subset_size=subset_size
        )
        
        # Optimizer
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, 
                                   momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
        
        # Training loop
        model.train()
        for epoch in range(epochs):
            for inputs, targets in trainloader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = F.cross_entropy(outputs, targets)
                loss.backward()
                optimizer.step()
            
            scheduler.step()
        
        # Evaluation
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in testloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        accuracy = correct / total
        
        return accuracy
        
    except Exception as e:
        print(f"Error training architecture: {e}")
        return 0.1  # Return low accuracy on failure
