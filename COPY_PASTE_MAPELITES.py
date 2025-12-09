# MAP-Elites Creative Neural Architecture Search
# Self-Contained Version - Copy This Into Jupyter/Colab

# ============================================================
# CELL 1: Install Dependencies
# ============================================================
!pip install torch torchvision networkx matplotlib tqdm scipy numpy

# ============================================================
# CELL 2: Imports and Setup
# ============================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import random
import copy
import numpy as np
from typing import List, Tuple, Dict, Optional
from tqdm import tqdm
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

# ============================================================
# CELL 3: Architecture State Class
# ============================================================
OPERATION_POOL = ['conv3x3', 'conv5x5', 'sep_conv3x3', 'sep_conv5x5', 
                  'max_pool3x3', 'avg_pool3x3', 'skip_connect']

class ArchitectureState:
    """Neural architecture as directed acyclic graph"""
    
    def __init__(self):
        self.nodes = []
        self.edges = []
        self.operations = {}
        self.channels = {}
        self.positions = {}
        self.input_node = None
        self.output_node = None
        self._node_counter = 0
    
    @staticmethod
    def initialize_starter():
        arch = ArchitectureState()
        arch.input_node = arch.add_node('input', 3, 0)
        node1 = arch.add_node('conv3x3', 32, 1)
        node2 = arch.add_node('conv3x3', 64, 2)
        arch.output_node = arch.add_node('output', 10, 3)
        arch.add_edge(arch.input_node, node1)
        arch.add_edge(node1, node2)
        arch.add_edge(node2, arch.output_node)
        return arch
    
    def add_node(self, operation: str, channels: int, position: int) -> int:
        node_id = self._node_counter
        self._node_counter += 1
        self.nodes.append(node_id)
        self.operations[node_id] = operation
        self.channels[node_id] = channels
        self.positions[node_id] = position
        return node_id
    
    def add_edge(self, src: int, dst: int):
        if (src, dst) not in self.edges:
            self.edges.append((src, dst))
    
    def remove_node(self, node_id: int):
        if node_id in [self.input_node, self.output_node]:
            return
        predecessors = [src for src, dst in self.edges if dst == node_id]
        successors = [dst for src, dst in self.edges if src == node_id]
        self.edges = [(src, dst) for src, dst in self.edges 
                     if src != node_id and dst != node_id]
        for pred in predecessors:
            for succ in successors:
                self.add_edge(pred, succ)
        self.nodes.remove(node_id)
        del self.operations[node_id]
        del self.channels[node_id]
        del self.positions[node_id]
    
    def remove_edge(self, src: int, dst: int):
        if (src, dst) in self.edges:
            self.edges.remove((src, dst))
    
    def copy(self):
        return copy.deepcopy(self)
    
    @property
    def depth(self) -> int:
        if not self.positions:
            return 0
        return max(self.positions.values()) - min(self.positions.values())
    
    @property
    def avg_width(self) -> float:
        if not self.channels:
            return 0
        return sum(self.channels.values()) / len(self.channels)
    
    @property
    def total_params(self) -> int:
        params = 0
        for src, dst in self.edges:
            params += self.channels[src] * self.channels[dst] * 9
        return params
    
    @property
    def num_skip_connections(self) -> int:
        count = 0
        for src, dst in self.edges:
            if self.positions[dst] - self.positions[src] > 1:
                count += 1
        return count

print("✅ ArchitectureState loaded")

# ============================================================
# CELL 4: Neural Network Model
# ============================================================
class ConvNet(nn.Module):
    def __init__(self, arch: ArchitectureState, num_classes: int = 10):
        super().__init__()
        self.arch = arch
        self.layers = nn.ModuleDict()
        
        for node in arch.nodes:
            if node == arch.input_node:
                continue
            op = arch.operations[node]
            in_ch = self._get_input_channels(node)
            out_ch = arch.channels[node]
            
            if node == arch.output_node:
                self.layers[str(node)] = nn.Linear(in_ch, num_classes)
            else:
                self.layers[str(node)] = self._create_op(op, in_ch, out_ch)
    
    def _get_input_channels(self, node: int) -> int:
        predecessors = [src for src, dst in self.arch.edges if dst == node]
        if not predecessors:
            return 3
        return sum(self.arch.channels[pred] for pred in predecessors)
    
    def _create_op(self, op: str, in_ch: int, out_ch: int):
        if op == 'conv3x3':
            return nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, padding=1),
                               nn.BatchNorm2d(out_ch), nn.ReLU())
        elif op == 'conv5x5':
            return nn.Sequential(nn.Conv2d(in_ch, out_ch, 5, padding=2),
                               nn.BatchNorm2d(out_ch), nn.ReLU())
        elif op == 'max_pool3x3':
            return nn.Sequential(nn.MaxPool2d(3, stride=1, padding=1),
                               nn.Conv2d(in_ch, out_ch, 1),
                               nn.BatchNorm2d(out_ch), nn.ReLU())
        elif op == 'skip_connect':
            if in_ch == out_ch:
                return nn.Identity()
            return nn.Conv2d(in_ch, out_ch, 1)
        else:
            return nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, padding=1),
                               nn.BatchNorm2d(out_ch), nn.ReLU())
    
    def forward(self, x):
        outputs = {self.arch.input_node: x}
        sorted_nodes = sorted(self.arch.nodes, key=lambda n: self.arch.positions[n])
        
        for node in sorted_nodes:
            if node == self.arch.input_node:
                continue
            predecessors = [src for src, dst in self.arch.edges if dst == node]
            if not predecessors:
                continue
            inputs = [outputs[pred] for pred in predecessors if pred in outputs]
            if not inputs:
                continue
            x = torch.cat(inputs, dim=1) if len(inputs) > 1 else inputs[0]
            
            if node == self.arch.output_node:
                x = F.adaptive_avg_pool2d(x, 1)
                x = x.flatten(1)
                outputs[node] = self.layers[str(node)](x)
            else:
                outputs[node] = self.layers[str(node)](x)
        
        return outputs[self.arch.output_node]

print("✅ ConvNet loaded")

# ============================================================
# CELL 5: Training Function
# ============================================================
def train_architecture(arch, epochs=3, dataset='mnist', subset_size=10000):
    try:
        # Data
        if dataset == 'mnist':
            transform = transforms.Compose([
                transforms.Resize(32), transforms.ToTensor(),
                transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                transforms.Normalize((0.1307,)*3, (0.3081,)*3)])
            trainset = torchvision.datasets.MNIST(root='./data', train=True, 
                                                 download=True, transform=transform)
            testset = torchvision.datasets.MNIST(root='./data', train=False, 
                                                download=True, transform=transform)
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
            trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                   download=True, transform=transform)
            testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                                  download=True, transform=transform)
        
        if subset_size:
            indices = np.random.choice(len(trainset), subset_size, replace=False)
            trainset = Subset(trainset, indices)
        
        trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
        testloader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)
        
        # Model
        model = ConvNet(arch).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        
        # Train
        model.train()
        for epoch in range(epochs):
            for inputs, labels in trainloader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
        
        # Evaluate
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        return correct / total
    except Exception as e:
        print(f"Error: {e}")
        return 0.1

print("✅ Training function loaded")

# ============================================================
# CELL 6: MAP-Elites Components
# ============================================================
class BehaviorSpace:
    def __init__(self, depth_bins=5, width_bins=5, skip_bins=4):
        self.depth_bins = depth_bins
        self.width_bins = width_bins
        self.skip_bins = skip_bins
        self.depth_range = (3, 20)
        self.width_range = (16, 256)
        self.skip_range = (0.0, 1.0)
    
    def get_behavior(self, arch) -> Tuple[int, int, int]:
        depth_bin = self._discretize(arch.depth, self.depth_range, self.depth_bins)
        width_bin = self._discretize(arch.avg_width, self.width_range, self.width_bins)
        num_possible = len(arch.nodes) * (len(arch.nodes) - 1) / 2
        skip_ratio = arch.num_skip_connections / (num_possible + 1e-6)
        skip_bin = self._discretize(skip_ratio, self.skip_range, self.skip_bins)
        return (depth_bin, width_bin, skip_bin)
    
    def _discretize(self, value, value_range, num_bins):
        min_val, max_val = value_range
        value = np.clip(value, min_val, max_val)
        normalized = (value - min_val) / (max_val - min_val + 1e-6)
        bin_idx = int(normalized * num_bins)
        return min(bin_idx, num_bins - 1)
    
    def get_total_cells(self):
        return self.depth_bins * self.width_bins * self.skip_bins

class MutationOperator:
    def mutate(self, arch):
        new_arch = arch.copy()
        mutation = random.choice(['add_node', 'remove_node', 'add_edge', 'remove_edge'])
        
        try:
            if mutation == 'add_node' and len(new_arch.nodes) < 20:
                op = random.choice(OPERATION_POOL)
                pos = random.randint(1, max(new_arch.positions.values()))
                ch = random.choice([32, 64, 128])
                new_id = new_arch.add_node(op, ch, pos)
                prev = [n for n in new_arch.nodes if new_arch.positions[n] < pos and n != new_id]
                next_n = [n for n in new_arch.nodes if new_arch.positions[n] >= pos and n != new_id]
                if prev:
                    new_arch.add_edge(random.choice(prev), new_id)
                if next_n:
                    new_arch.add_edge(new_id, random.choice(next_n))
            
            elif mutation == 'remove_node':
                removable = [n for n in new_arch.nodes 
                           if n not in [new_arch.input_node, new_arch.output_node]]
                if removable and len(new_arch.nodes) > 3:
                    new_arch.remove_node(random.choice(removable))
            
            elif mutation == 'add_edge':
                possible = [(s, d) for s in new_arch.nodes for d in new_arch.nodes
                          if s != d and (s, d) not in new_arch.edges 
                          and new_arch.positions[s] < new_arch.positions[d]]
                if possible:
                    new_arch.add_edge(*random.choice(possible))
            
            elif mutation == 'remove_edge' and len(new_arch.edges) > len(new_arch.nodes):
                new_arch.remove_edge(*random.choice(new_arch.edges))
        except:
            pass
        
        return new_arch

class MAPElites:
    def __init__(self, behavior_space, mutation_operator):
        self.behavior_space = behavior_space
        self.mutation_operator = mutation_operator
        self.archive = {}
        self.history = []
    
    def initialize(self, num_random=20):
        for _ in range(num_random):
            arch = ArchitectureState.initialize_starter()
            for _ in range(random.randint(2, 5)):
                arch = self.mutation_operator.mutate(arch)
            behavior = self.behavior_space.get_behavior(arch)
            self.archive[behavior] = (arch, 0.0)
    
    def run(self, num_iterations, evaluate_fn, verbose=True):
        if not self.archive:
            self.initialize()
        
        # Evaluate initial
        for behavior, (arch, _) in list(self.archive.items()):
            perf = evaluate_fn(arch)
            self.archive[behavior] = (arch, perf)
        
        # Main loop
        iterator = tqdm(range(num_iterations)) if verbose else range(num_iterations)
        
        for i in iterator:
            behavior = random.choice(list(self.archive.keys()))
            parent, _ = self.archive[behavior]
            child = self.mutation_operator.mutate(parent)
            performance = evaluate_fn(child)
            child_behavior = self.behavior_space.get_behavior(child)
            
            if child_behavior not in self.archive or performance > self.archive[child_behavior][1]:
                self.archive[child_behavior] = (child, performance)
            
            self.history.append({'architecture': child, 'performance': performance})
            
            if verbose and i % 10 == 0:
                coverage = len(self.archive) / self.behavior_space.get_total_cells()
                best = max(self.archive.values(), key=lambda x: x[1])[1]
                iterator.set_postfix({'coverage': f'{coverage:.2%}', 'best': f'{best:.4f}'})
        
        return self
    
    def get_all_architectures(self):
        return [(arch.copy(), perf, behavior) 
                for behavior, (arch, perf) in self.archive.items()]

print("✅ MAP-Elites loaded")

# ============================================================
# CELL 7: Configuration
# ============================================================
DATASET = 'mnist'
ITERATIONS = 100
DEPTH_BINS = 5
WIDTH_BINS = 5
SKIP_BINS = 4

print(f"Config: {DATASET}, {ITERATIONS} iterations, {DEPTH_BINS}x{WIDTH_BINS}x{SKIP_BINS} cells")

# ============================================================
# CELL 8: Run MAP-Elites
# ============================================================
behavior_space = BehaviorSpace(DEPTH_BINS, WIDTH_BINS, SKIP_BINS)
mutation_operator = MutationOperator()
map_elites = MAPElites(behavior_space, mutation_operator)

def evaluate(arch):
    return train_architecture(arch, epochs=3, dataset=DATASET, subset_size=10000)

print("🚀 Running MAP-Elites...")
map_elites.run(ITERATIONS, evaluate, verbose=True)

# ============================================================
# CELL 9: Results
# ============================================================
all_archs = map_elites.get_all_architectures()
all_archs.sort(key=lambda x: x[1], reverse=True)

print(f"\n🏆 Top 10 Architectures:\n")
for i, (arch, perf, behavior) in enumerate(all_archs[:10]):
    print(f"{i+1}. Acc: {perf:.4f} | Behavior: {behavior} | Nodes: {len(arch.nodes)}")

best_arch, best_perf, best_behavior = all_archs[0]
print(f"\n🥇 Best: {best_perf:.4f} accuracy, Behavior: {best_behavior}")
