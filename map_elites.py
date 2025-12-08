"""
MAP-Elites Algorithm for Creative Neural Architecture Search

Quality-Diversity optimization that maintains an archive of diverse,
high-quality architectures across different behavioral characteristics.
"""
import numpy as np
import random
from typing import Dict, Tuple, List, Optional
from collections import defaultdict
import copy

from architecture import ArchitectureState, ActionSpace, OPERATION_POOL


class BehaviorSpace:
    """
    Defines the behavior space for MAP-Elites archive.
    Each architecture is characterized by multiple behavioral dimensions.
    """
    def __init__(self, 
                 depth_bins=5,
                 width_bins=5, 
                 skip_bins=4):
        """
        Args:
            depth_bins: Number of bins for architecture depth
            width_bins: Number of bins for average width
            skip_bins: Number of bins for skip connection density
        """
        self.depth_bins = depth_bins
        self.width_bins = width_bins
        self.skip_bins = skip_bins
        
        # Define bin ranges
        self.depth_range = (3, 20)  # Min and max depth
        self.width_range = (16, 256)  # Min and max average width
        self.skip_range = (0.0, 1.0)  # Skip connection ratio
        
    def get_behavior(self, arch: ArchitectureState) -> Tuple[int, int, int]:
        """
        Compute behavior descriptor (bin indices) for an architecture.
        
        Returns:
            (depth_bin, width_bin, skip_bin)
        """
        # Depth bin
        depth = arch.depth
        depth_bin = self._discretize(depth, self.depth_range, self.depth_bins)
        
        # Width bin
        avg_width = arch.avg_width
        width_bin = self._discretize(avg_width, self.width_range, self.width_bins)
        
        # Skip connection density bin
        num_possible_skips = len(arch.nodes) * (len(arch.nodes) - 1) / 2
        skip_ratio = arch.num_skip_connections / (num_possible_skips + 1e-6)
        skip_bin = self._discretize(skip_ratio, self.skip_range, self.skip_bins)
        
        return (depth_bin, width_bin, skip_bin)
    
    def _discretize(self, value: float, value_range: Tuple[float, float], 
                   num_bins: int) -> int:
        """Discretize a continuous value into a bin index."""
        min_val, max_val = value_range
        value = np.clip(value, min_val, max_val)
        
        # Compute bin index
        normalized = (value - min_val) / (max_val - min_val + 1e-6)
        bin_idx = int(normalized * num_bins)
        bin_idx = min(bin_idx, num_bins - 1)  # Ensure in range
        
        return bin_idx
    
    def get_total_cells(self) -> int:
        """Get total number of cells in the behavior space."""
        return self.depth_bins * self.width_bins * self.skip_bins


class MutationOperator:
    """
    Defines mutation operators for architecture modification.
    """
    def __init__(self):
        self.mutation_types = [
            'add_node',
            'remove_node', 
            'add_edge',
            'remove_edge',
            'increase_channels',
            'decrease_channels',
            'replace_operation'
        ]
        
    def mutate(self, arch: ArchitectureState, 
               operation_strategy: str = 'diverse') -> Optional[ArchitectureState]:
        """
        Apply a random mutation to the architecture.
        
        Args:
            arch: Architecture to mutate
            operation_strategy: How to select operations when adding nodes
            
        Returns:
            Mutated architecture or None if mutation failed
        """
        # Try mutations until one succeeds
        attempts = 0
        max_attempts = 10
        
        while attempts < max_attempts:
            mutation_type = random.choice(self.mutation_types)
            new_arch = self._apply_mutation(arch, mutation_type, operation_strategy)
            
            if new_arch is not None:
                return new_arch
            
            attempts += 1
        
        # If all mutations fail, return copy of original
        return arch.copy()
    
    def _apply_mutation(self, arch: ArchitectureState, mutation_type: str,
                       operation_strategy: str) -> Optional[ArchitectureState]:
        """Apply a specific mutation type."""
        new_arch = arch.copy()
        
        try:
            if mutation_type == 'add_node':
                return self._add_node(new_arch, operation_strategy)
            elif mutation_type == 'remove_node':
                return self._remove_node(new_arch)
            elif mutation_type == 'add_edge':
                return self._add_edge(new_arch)
            elif mutation_type == 'remove_edge':
                return self._remove_edge(new_arch)
            elif mutation_type == 'increase_channels':
                return self._increase_channels(new_arch)
            elif mutation_type == 'decrease_channels':
                return self._decrease_channels(new_arch)
            elif mutation_type == 'replace_operation':
                return self._replace_operation(new_arch)
        except Exception as e:
            return None
        
        return None
    
    def _add_node(self, arch: ArchitectureState, 
                  operation_strategy: str) -> Optional[ArchitectureState]:
        """Add a new node to the architecture."""
        if len(arch.nodes) >= 20:
            return None
        
        # Choose operation
        if operation_strategy == 'diverse':
            op_counts = {}
            for op in OPERATION_POOL:
                op_counts[op] = sum(1 for o in arch.operations.values() if o == op)
            operation = min(OPERATION_POOL, key=lambda x: op_counts.get(x, 0))
        else:
            operation = random.choice(OPERATION_POOL)
        
        # Choose position
        positions = sorted(set(arch.positions.values()))
        if len(positions) > 1:
            insert_pos = random.choice(positions[1:])
        else:
            insert_pos = 1
        
        # Choose channels
        avg_channels = int(np.mean(list(arch.channels.values())))
        channels = random.choice([avg_channels // 2, avg_channels, avg_channels * 2])
        channels = np.clip(channels, 16, 512)
        
        # Add node
        new_id = arch.add_node(operation, int(channels), insert_pos)
        
        # Connect to previous and next nodes
        prev_nodes = [n for n in arch.nodes 
                     if arch.positions[n] < insert_pos and n != new_id]
        next_nodes = [n for n in arch.nodes 
                     if arch.positions[n] >= insert_pos and n != new_id]
        
        if prev_nodes:
            arch.add_edge(random.choice(prev_nodes), new_id)
        if next_nodes:
            arch.add_edge(new_id, random.choice(next_nodes))
        
        return arch
    
    def _remove_node(self, arch: ArchitectureState) -> Optional[ArchitectureState]:
        """Remove a node from the architecture."""
        removable = [n for n in arch.nodes 
                    if n not in [arch.input_node, arch.output_node]]
        
        if len(removable) == 0 or len(arch.nodes) <= 3:
            return None
        
        node_to_remove = random.choice(removable)
        arch.remove_node(node_to_remove)
        
        return arch
    
    def _add_edge(self, arch: ArchitectureState) -> Optional[ArchitectureState]:
        """Add a skip connection."""
        # Find possible edges
        possible_edges = []
        for src in arch.nodes:
            for dst in arch.nodes:
                if (src != dst and 
                    (src, dst) not in arch.edges and
                    arch.positions[src] < arch.positions[dst]):
                    possible_edges.append((src, dst))
        
        if not possible_edges:
            return None
        
        src, dst = random.choice(possible_edges)
        arch.add_edge(src, dst)
        
        return arch
    
    def _remove_edge(self, arch: ArchitectureState) -> Optional[ArchitectureState]:
        """Remove an edge."""
        removable_edges = [e for e in arch.edges 
                          if not arch._would_disconnect(e[0], e[1])]
        
        if not removable_edges:
            return None
        
        src, dst = random.choice(removable_edges)
        arch.remove_edge(src, dst)
        
        return arch
    
    def _increase_channels(self, arch: ArchitectureState) -> Optional[ArchitectureState]:
        """Increase channels at a random node."""
        modifiable = [n for n in arch.nodes if arch.channels[n] < 512]
        if not modifiable:
            return None
        
        node = random.choice(modifiable)
        arch.increase_channels(node)
        
        return arch
    
    def _decrease_channels(self, arch: ArchitectureState) -> Optional[ArchitectureState]:
        """Decrease channels at a random node."""
        modifiable = [n for n in arch.nodes if arch.channels[n] > 16]
        if not modifiable:
            return None
        
        node = random.choice(modifiable)
        arch.decrease_channels(node)
        
        return arch
    
    def _replace_operation(self, arch: ArchitectureState) -> Optional[ArchitectureState]:
        """Replace operation at a random node."""
        modifiable = [n for n in arch.nodes 
                     if n not in [arch.input_node, arch.output_node]]
        if not modifiable:
            return None
        
        node = random.choice(modifiable)
        current_op = arch.operations[node]
        
        # Choose different operation
        other_ops = [op for op in OPERATION_POOL if op != current_op]
        new_op = random.choice(other_ops)
        
        arch.operations[node] = new_op
        
        return arch


class QualityDiversityArchive:
    """
    Archive that maintains the best architecture in each behavior cell.
    """
    def __init__(self, behavior_space: BehaviorSpace):
        self.behavior_space = behavior_space
        self.archive = {}  # behavior -> (arch, performance)
        self.history = []  # All evaluated architectures
        
    def add(self, arch: ArchitectureState, performance: float) -> bool:
        """
        Add architecture to archive if it's the best in its cell.
        
        Returns:
            True if architecture was added/updated
        """
        behavior = self.behavior_space.get_behavior(arch)
        
        # Store in history
        self.history.append({
            'architecture': arch.copy(),
            'performance': performance,
            'behavior': behavior
        })
        
        # Check if this is the best in its cell
        if behavior not in self.archive or performance > self.archive[behavior][1]:
            self.archive[behavior] = (arch.copy(), performance)
            return True
        
        return False
    
    def sample(self) -> Optional[ArchitectureState]:
        """Sample a random architecture from the archive."""
        if not self.archive:
            return None
        
        behavior = random.choice(list(self.archive.keys()))
        return self.archive[behavior][0].copy()
    
    def get_best(self) -> Tuple[ArchitectureState, float]:
        """Get the best architecture overall."""
        if not self.archive:
            return None, 0.0
        
        best_behavior = max(self.archive.keys(), 
                          key=lambda b: self.archive[b][1])
        return self.archive[best_behavior]
    
    def get_stats(self) -> Dict:
        """Get statistics about the archive."""
        if not self.archive:
            return {
                'coverage': 0.0,
                'num_filled': 0,
                'best_performance': 0.0,
                'mean_performance': 0.0
            }
        
        performances = [perf for _, perf in self.archive.values()]
        
        return {
            'coverage': len(self.archive) / self.behavior_space.get_total_cells(),
            'num_filled': len(self.archive),
            'best_performance': max(performances),
            'mean_performance': np.mean(performances),
            'total_evaluated': len(self.history)
        }
    
    def get_all_architectures(self) -> List[Tuple[ArchitectureState, float, Tuple]]:
        """Get all architectures in archive."""
        return [(arch.copy(), perf, behavior) 
                for behavior, (arch, perf) in self.archive.items()]


class MAPElites:
    """
    MAP-Elites algorithm for creative architecture search.
    """
    def __init__(self,
                 behavior_space: BehaviorSpace,
                 mutation_operator: MutationOperator,
                 operation_strategy: str = 'diverse'):
        
        self.behavior_space = behavior_space
        self.mutation_operator = mutation_operator
        self.operation_strategy = operation_strategy
        self.archive = QualityDiversityArchive(behavior_space)
        
    def initialize_archive(self, num_random: int = 20) -> None:
        """
        Initialize archive with random architectures.
        
        Args:
            num_random: Number of random architectures to generate
        """
        print(f"Initializing archive with {num_random} random architectures...")
        
        for i in range(num_random):
            # Start from simple architecture
            arch = ArchitectureState.initialize_starter(self.operation_strategy)
            
            # Apply random mutations
            num_mutations = random.randint(2, 8)
            for _ in range(num_mutations):
                arch = self.mutation_operator.mutate(arch, self.operation_strategy)
            
            # Add to archive (performance will be evaluated later)
            # For now, just add with dummy performance
            self.archive.add(arch, 0.0)
        
        print(f"Archive initialized with {len(self.archive.archive)} architectures")
    
    def run(self, 
            num_iterations: int,
            evaluate_fn,
            verbose: bool = True) -> QualityDiversityArchive:
        """
        Run MAP-Elites for a number of iterations.
        
        Args:
            num_iterations: Number of iterations to run
            evaluate_fn: Function that takes architecture and returns performance
            verbose: Print progress
            
        Returns:
            Final archive
        """
        from tqdm import tqdm
        
        # Initialize if empty
        if len(self.archive.archive) == 0:
            self.initialize_archive()
        
        # Evaluate initial population
        if verbose:
            print("Evaluating initial population...")
        for behavior, (arch, _) in list(self.archive.archive.items()):
            performance = evaluate_fn(arch)
            self.archive.archive[behavior] = (arch, performance)
        
        # Main loop
        iterator = tqdm(range(num_iterations), desc="MAP-Elites") if verbose else range(num_iterations)
        
        for iteration in iterator:
            # Sample parent from archive
            parent = self.archive.sample()
            if parent is None:
                continue
            
            # Mutate
            child = self.mutation_operator.mutate(parent, self.operation_strategy)
            
            # Evaluate
            performance = evaluate_fn(child)
            
            # Add to archive
            added = self.archive.add(child, performance)
            
            # Update progress bar
            if verbose and iteration % 10 == 0:
                stats = self.archive.get_stats()
                iterator.set_postfix({
                    'coverage': f"{stats['coverage']:.2%}",
                    'best': f"{stats['best_performance']:.4f}",
                    'mean': f"{stats['mean_performance']:.4f}"
                })
        
        return self.archive
