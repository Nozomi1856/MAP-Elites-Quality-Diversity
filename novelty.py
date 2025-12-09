"""
Novelty metrics for architecture creativity
"""
import numpy as np
import networkx as nx
from scipy.stats import entropy
from typing import List
from architecture import ArchitectureState


class TopologicalNovelty:
    """
    Measures how topologically different an architecture is
    """
    def __init__(self, max_archive_size=500):
        self.archive = []
        self.max_archive_size = max_archive_size
        
    def compute_novelty(self, arch: ArchitectureState) -> float:
        """
        Compute topological novelty using graph edit distance
        """
        arch_graph = arch.to_networkx()
        
        if len(self.archive) == 0:
            self.archive.append(arch_graph)
            return 1.0
        
        # Compute distances to archive
        distances = []
        # Use last N for speed
        recent_archive = self.archive[-min(100, len(self.archive)):]
        
        for past_graph in recent_archive:
            ged = self._graph_edit_distance(arch_graph, past_graph)
            distances.append(ged)
        
        # K-nearest neighbors novelty
        k = min(15, len(distances))
        k_nearest = sorted(distances)[:k]
        novelty = np.mean(k_nearest)
        
        # Normalize by graph size
        avg_size = (arch_graph.number_of_nodes() + 
                   np.mean([g.number_of_nodes() for g in recent_archive]))
        novelty = novelty / (avg_size + 1)
        
        # Add to archive
        self.archive.append(arch_graph)
        if len(self.archive) > self.max_archive_size:
            self.archive.pop(0)
        
        return float(np.clip(novelty, 0, 1))
    
    def _graph_edit_distance(self, g1: nx.DiGraph, g2: nx.DiGraph) -> float:
        """
        Approximate graph edit distance
        (Exact GED is NP-hard, so we use a fast heuristic)
        """
        # Node count difference
        node_diff = abs(g1.number_of_nodes() - g2.number_of_nodes())
        
        # Edge count difference
        edge_diff = abs(g1.number_of_edges() - g2.number_of_edges())
        
        # Degree sequence difference
        deg1 = sorted([d for n, d in g1.degree()])
        deg2 = sorted([d for n, d in g2.degree()])
        
        # Pad shorter sequence
        max_len = max(len(deg1), len(deg2))
        deg1 += [0] * (max_len - len(deg1))
        deg2 += [0] * (max_len - len(deg2))
        
        deg_diff = sum(abs(d1 - d2) for d1, d2 in zip(deg1, deg2))
        
        # Clustering coefficient difference
        try:
            cluster1 = nx.average_clustering(g1.to_undirected())
            cluster2 = nx.average_clustering(g2.to_undirected())
            cluster_diff = abs(cluster1 - cluster2) * 10
        except:
            cluster_diff = 0
        
        # Graph diameter difference (measure of connectivity)
        try:
            if nx.is_weakly_connected(g1) and nx.is_weakly_connected(g2):
                diam1 = nx.diameter(g1.to_undirected())
                diam2 = nx.diameter(g2.to_undirected())
                diam_diff = abs(diam1 - diam2)
            else:
                diam_diff = 5  # Penalty for disconnected graphs
        except:
            diam_diff = 0
        
        total_distance = (node_diff + edge_diff + deg_diff + 
                         cluster_diff + diam_diff)
        
        return total_distance


class ScaleNovelty:
    """
    Measures how unusual the depth/width patterns are
    """
    def __init__(self, warmup_size=20):
        self.depth_history = []
        self.width_history = []
        self.param_history = []
        self.width_variance_history = []
        self.warmup_size = warmup_size
        
    def compute_novelty(self, arch: ArchitectureState) -> float:
        """
        Compute scale novelty based on depth, width, and parameter patterns
        """
        depth = arch.depth
        avg_width = arch.avg_width
        total_params = arch.total_params
        
        # Width variance (how much layers differ in size)
        if len(arch.channels) > 0:
            width_variance = np.std(list(arch.channels.values()))
        else:
            width_variance = 0
        
        novelties = []
        
        # During warmup, give medium novelty
        if len(self.depth_history) < self.warmup_size:
            self.depth_history.append(depth)
            self.width_history.append(avg_width)
            self.param_history.append(total_params)
            self.width_variance_history.append(width_variance)
            return 0.5
        
        # 1. Depth novelty
        depth_mean = np.mean(self.depth_history)
        depth_std = np.std(self.depth_history) + 1e-6
        depth_novelty = abs(depth - depth_mean) / depth_std
        novelties.append(depth_novelty)
        
        # 2. Width novelty
        width_mean = np.mean(self.width_history)
        width_std = np.std(self.width_history) + 1e-6
        width_novelty = abs(avg_width - width_mean) / width_std
        novelties.append(width_novelty)
        
        # 3. Depth/width ratio novelty
        ratio = depth / (avg_width + 1)
        historical_ratios = [d / (w + 1) for d, w in 
                            zip(self.depth_history, self.width_history)]
        ratio_mean = np.mean(historical_ratios)
        ratio_std = np.std(historical_ratios) + 1e-6
        ratio_novelty = abs(ratio - ratio_mean) / ratio_std
        novelties.append(ratio_novelty)
        
        # 4. Width variance novelty (unconventional tapering)
        var_mean = np.mean(self.width_variance_history)
        var_std = np.std(self.width_variance_history) + 1e-6
        var_novelty = abs(width_variance - var_mean) / var_std
        novelties.append(var_novelty)
        
        # Update history
        self.depth_history.append(depth)
        self.width_history.append(avg_width)
        self.param_history.append(total_params)
        self.width_variance_history.append(width_variance)
        
        # Clip outliers and combine
        clipped = [np.clip(n, 0, 3) for n in novelties]
        combined_novelty = np.mean(clipped) / 3  # Normalize to 0-1
        
        return float(combined_novelty)


class RewardFunction:
    """
    Combined reward: performance + topological + scale novelty
    """
    def __init__(self, alpha=0.5, beta=0.35, gamma=0.15):
        """
        Args:
            alpha: Performance weight
            beta: Topological novelty weight
            gamma: Scale novelty weight
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
        self.topo_novelty = TopologicalNovelty()
        self.scale_novelty = ScaleNovelty()
        
    def compute_reward(self, arch: ArchitectureState, 
                      performance: float) -> tuple:
        """
        Compute multi-objective reward
        
        Returns:
            (total_reward, components_dict)
        """
        # Compute novelty scores
        topo_nov = self.topo_novelty.compute_novelty(arch)
        scale_nov = self.scale_novelty.compute_novelty(arch)
        
        # Combine rewards
        total_reward = (
            self.alpha * performance +
            self.beta * topo_nov +
            self.gamma * scale_nov
        )
        
        components = {
            'performance': performance,
            'topological_novelty': topo_nov,
            'scale_novelty': scale_nov,
            'total_reward': total_reward
        }
        
        return total_reward, components
