"""
GNN encoder and Q-network
"""
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool


class ArchitectureEncoder(nn.Module):
    """
    GNN encoder for architecture graphs
    Uses simple GCN layers (no attention for speed)
    """
    def __init__(self, node_features=9, hidden_dim=128, num_layers=3):
        super().__init__()
        
        self.num_layers = num_layers
        
        # GCN layers
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(node_features, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        # Batch normalization
        self.bns = nn.ModuleList()
        for _ in range(num_layers):
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        
        # Dropout
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x, edge_index, batch):
        """
        Args:
            x: Node features [num_nodes, node_features]
            edge_index: Edge connections [2, num_edges]
            batch: Batch assignment [num_nodes]
            
        Returns:
            Graph embedding [batch_size, hidden_dim]
        """
        # Apply GCN layers
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            x = conv(x, edge_index)
            x = bn(x)
            x = torch.relu(x)
            if i < self.num_layers - 1:
                x = self.dropout(x)
        
        # Global pooling (both mean and max)
        mean_pool = global_mean_pool(x, batch)
        max_pool = global_max_pool(x, batch)
        
        # Concatenate both poolings
        graph_embedding = mean_pool + max_pool
        
        return graph_embedding


class DQNetwork(nn.Module):
    """
    Q-Network that outputs Q-values for actions
    """
    def __init__(self, graph_embed_dim=128, hidden_dim=256, 
                 num_action_types=7):
        super().__init__()
        
        self.encoder = ArchitectureEncoder(hidden_dim=graph_embed_dim)
        
        # Q-value head
        self.q_head = nn.Sequential(
            nn.Linear(graph_embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_action_types)
        )
        
    def forward(self, x, edge_index, batch):
        """
        Args:
            x: Node features
            edge_index: Edge connections
            batch: Batch assignment
            
        Returns:
            Q-values [batch_size, num_action_types]
        """
        # Encode graph
        graph_embed = self.encoder(x, edge_index, batch)
        
        # Compute Q-values
        q_values = self.q_head(graph_embed)
        
        return q_values
