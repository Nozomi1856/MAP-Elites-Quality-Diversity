"""
Architecture representation and manipulation
"""
import torch
import networkx as nx
import numpy as np
from typing import List, Dict, Tuple, Optional
from torch_geometric.data import Data
import copy

# Fixed operation pool
OPERATION_POOL = [
    'conv3x3',
    'conv5x5',
    'sep_conv3x3',
    'sep_conv5x5',
    'max_pool3x3',
    'avg_pool3x3',
    'skip_connect'
]

OP_TO_IDX = {op: idx for idx, op in enumerate(OPERATION_POOL)}

# Action space
class ActionSpace:
    ADD_NODE = 0
    REMOVE_NODE = 1
    ADD_EDGE = 2
    REMOVE_EDGE = 3
    INCREASE_CHANNELS = 4
    DECREASE_CHANNELS = 5
    STOP_BUILDING = 6
    
    NUM_ACTIONS = 7


class ArchitectureState:
    """
    Represents a neural architecture as a directed graph
    """
    def __init__(self):
        # Graph structure
        self.nodes = []  # List of node IDs
        self.edges = []  # List of (src, dst) tuples
        
        # Node attributes
        self.operations = {}  # node_id -> operation name
        self.channels = {}    # node_id -> channel count
        self.positions = {}   # node_id -> depth position
        
        # Graph metadata
        self.node_counter = 0
        self.input_node = None
        self.output_node = None
        
    @classmethod
    def initialize_starter(cls, operation_strategy='diverse'):
        """
        Create a simple starter architecture: input -> conv -> output
        """
        arch = cls()
        
        # Input node
        input_id = arch.add_node('skip_connect', channels=3, position=0)
        arch.input_node = input_id
        
        # First hidden layer
        if operation_strategy == 'diverse':
            op = 'conv3x3'
        else:
            op = np.random.choice(OPERATION_POOL[:4])  # Only convs for first layer
        hidden_id = arch.add_node(op, channels=64, position=1)
        
        # Output node
        output_id = arch.add_node('skip_connect', channels=64, position=2)
        arch.output_node = output_id
        
        # Connect them
        arch.add_edge(input_id, hidden_id)
        arch.add_edge(hidden_id, output_id)
        
        return arch
    
    def add_node(self, operation: str, channels: int, position: int) -> int:
        """Add a new node to the architecture"""
        node_id = self.node_counter
        self.node_counter += 1
        
        self.nodes.append(node_id)
        self.operations[node_id] = operation
        self.channels[node_id] = channels
        self.positions[node_id] = position
        
        return node_id
    
    def remove_node(self, node_id: int):
        """Remove a node (and its edges)"""
        if node_id in [self.input_node, self.output_node]:
            return  # Can't remove input/output
        
        self.nodes.remove(node_id)
        del self.operations[node_id]
        del self.channels[node_id]
        del self.positions[node_id]
        
        # Remove connected edges
        self.edges = [(s, d) for s, d in self.edges 
                      if s != node_id and d != node_id]
    
    def add_edge(self, src: int, dst: int):
        """Add an edge (skip connection)"""
        if (src, dst) not in self.edges and src != dst:
            self.edges.append((src, dst))
    
    def remove_edge(self, src: int, dst: int):
        """Remove an edge"""
        if (src, dst) in self.edges:
            # Don't remove if it would disconnect the graph
            if not self._would_disconnect(src, dst):
                self.edges.remove((src, dst))
    
    def _would_disconnect(self, src: int, dst: int) -> bool:
        """Check if removing edge would disconnect graph"""
        temp_edges = [e for e in self.edges if e != (src, dst)]
        
        # Check if there's still a path from input to output
        graph = nx.DiGraph()
        graph.add_edges_from(temp_edges)
        
        try:
            nx.shortest_path(graph, self.input_node, self.output_node)
            return False
        except nx.NetworkXNoPath:
            return True
    
    def increase_channels(self, node_id: int):
        """Double channels at a node"""
        if node_id in self.channels:
            self.channels[node_id] = min(self.channels[node_id] * 2, 512)
    
    def decrease_channels(self, node_id: int):
        """Halve channels at a node"""
        if node_id in self.channels:
            self.channels[node_id] = max(self.channels[node_id] // 2, 16)
    
    def get_valid_actions(self) -> List[Tuple[int, Optional[int]]]:
        """
        Get list of valid (action_type, target_node) pairs
        
        Returns:
            List of tuples: (action_type, target_node_id or None)
        """
        valid = []
        
        # ADD_NODE - always valid if not too large
        if len(self.nodes) < 20:
            valid.append((ActionSpace.ADD_NODE, None))
        
        # REMOVE_NODE - can remove non-input/output nodes
        removable = [n for n in self.nodes 
                     if n not in [self.input_node, self.output_node]]
        if len(removable) > 0 and len(self.nodes) > 3:
            for node in removable:
                valid.append((ActionSpace.REMOVE_NODE, node))
        
        # ADD_EDGE - can add between any non-connected pairs
        for src in self.nodes:
            for dst in self.nodes:
                if src != dst and (src, dst) not in self.edges:
                    # Only forward edges (src position < dst position)
                    if self.positions[src] < self.positions[dst]:
                        valid.append((ActionSpace.ADD_EDGE, (src, dst)))
        
        # REMOVE_EDGE - can remove if doesn't disconnect
        for src, dst in self.edges:
            if not self._would_disconnect(src, dst):
                valid.append((ActionSpace.REMOVE_EDGE, (src, dst)))
        
        # INCREASE/DECREASE_CHANNELS - can modify any node
        for node in self.nodes:
            if self.channels[node] < 512:
                valid.append((ActionSpace.INCREASE_CHANNELS, node))
            if self.channels[node] > 16:
                valid.append((ActionSpace.DECREASE_CHANNELS, node))
        
        # STOP - always valid if minimum size
        if len(self.nodes) >= 3:
            valid.append((ActionSpace.STOP_BUILDING, None))
        
        return valid
    
    def apply_action(self, action_type: int, target, 
                     operation_strategy: str = 'diverse') -> 'ArchitectureState':
        """
        Apply an action and return new state
        
        Args:
            action_type: Action from ActionSpace
            target: Target node(s) or None
            operation_strategy: How to assign operations ('diverse', 'random', 'cyclic')
        """
        new_state = self.copy()
        
        if action_type == ActionSpace.ADD_NODE:
            # Choose operation based on strategy
            if operation_strategy == 'diverse':
                # Choose least-used operation
                op_counts = {}
                for op in OPERATION_POOL:
                    op_counts[op] = sum(1 for o in new_state.operations.values() if o == op)
                operation = min(OPERATION_POOL, key=lambda x: op_counts[x])
            elif operation_strategy == 'cyclic':
                idx = len(new_state.nodes) % len(OPERATION_POOL)
                operation = OPERATION_POOL[idx]
            else:  # random
                operation = np.random.choice(OPERATION_POOL)
            
            # Determine position (between random existing nodes)
            positions = sorted(set(new_state.positions.values()))
            if len(positions) > 1:
                insert_pos = np.random.choice(positions[1:])
            else:
                insert_pos = 1
            
            # Determine channels (average of neighbors)
            avg_channels = int(np.mean(list(new_state.channels.values())))
            
            # Add node
            new_id = new_state.add_node(operation, avg_channels, insert_pos)
            
            # Connect to a previous node and a later node
            prev_nodes = [n for n in new_state.nodes 
                         if new_state.positions[n] < insert_pos and n != new_id]
            next_nodes = [n for n in new_state.nodes 
                         if new_state.positions[n] >= insert_pos and n != new_id]
            
            if prev_nodes:
                new_state.add_edge(np.random.choice(prev_nodes), new_id)
            if next_nodes:
                new_state.add_edge(new_id, np.random.choice(next_nodes))
        
        elif action_type == ActionSpace.REMOVE_NODE:
            new_state.remove_node(target)
        
        elif action_type == ActionSpace.ADD_EDGE:
            src, dst = target
            new_state.add_edge(src, dst)
        
        elif action_type == ActionSpace.REMOVE_EDGE:
            src, dst = target
            new_state.remove_edge(src, dst)
        
        elif action_type == ActionSpace.INCREASE_CHANNELS:
            new_state.increase_channels(target)
        
        elif action_type == ActionSpace.DECREASE_CHANNELS:
            new_state.decrease_channels(target)
        
        elif action_type == ActionSpace.STOP_BUILDING:
            pass  # No change, just signal to stop
        
        return new_state
    
    def to_networkx(self) -> nx.DiGraph:
        """Convert to NetworkX graph for analysis"""
        G = nx.DiGraph()
        
        for node in self.nodes:
            G.add_node(node, 
                      operation=self.operations[node],
                      channels=self.channels[node],
                      position=self.positions[node])
        
        G.add_edges_from(self.edges)
        
        return G
    
    def to_pyg_data(self) -> Data:
        """Convert to PyTorch Geometric Data object"""
        # Node features: [op_onehot(7), channels(1), position(1)] = 9D
        node_features = []
        node_list = sorted(self.nodes)
        node_to_idx = {n: i for i, n in enumerate(node_list)}
        
        for node in node_list:
            op_idx = OP_TO_IDX[self.operations[node]]
            op_onehot = [0] * len(OPERATION_POOL)
            op_onehot[op_idx] = 1
            
            features = op_onehot + [
                self.channels[node] / 512.0,  # Normalize
                self.positions[node] / 20.0    # Normalize
            ]
            node_features.append(features)
        
        x = torch.tensor(node_features, dtype=torch.float)
        
        # Edge index
        edge_index = []
        for src, dst in self.edges:
            edge_index.append([node_to_idx[src], node_to_idx[dst]])
        
        if len(edge_index) > 0:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t()
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
        
        return Data(x=x, edge_index=edge_index)
    
    def copy(self) -> 'ArchitectureState':
        """Deep copy of the architecture"""
        return copy.deepcopy(self)
    
    @property
    def depth(self) -> int:
        """Get architecture depth"""
        if len(self.nodes) == 0:
            return 0
        return max(self.positions.values()) + 1
    
    @property
    def avg_width(self) -> float:
        """Get average width (channels)"""
        if len(self.channels) == 0:
            return 0
        return np.mean(list(self.channels.values()))
    
    @property
    def total_params(self) -> int:
        """Estimate total parameters (rough approximation)"""
        total = 0
        for node in self.nodes:
            op = self.operations[node]
            c = self.channels[node]
            
            if 'conv3x3' in op:
                total += 9 * c * c
            elif 'conv5x5' in op:
                total += 25 * c * c
            elif 'sep_conv' in op:
                total += 9 * c + c * c
            # Pooling and skip have no params
        
        return total
    
    @property
    def num_skip_connections(self) -> int:
        """Count skip connections (edges spanning >1 layer)"""
        count = 0
        for src, dst in self.edges:
            if self.positions[dst] - self.positions[src] > 1:
                count += 1
        return count
    
    def __repr__(self) -> str:
        return f"Architecture(nodes={len(self.nodes)}, edges={len(self.edges)}, depth={self.depth})"
