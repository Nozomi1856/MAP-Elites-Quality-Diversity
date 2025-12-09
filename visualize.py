"""
Visualization tools for architectures
"""
import matplotlib.pyplot as plt
import networkx as nx
import json
from pathlib import Path


def visualize_architecture(arch, save_path=None, title="Architecture"):
    """
    Visualize architecture as a graph
    """
    G = arch.to_networkx()
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Position nodes by depth
    pos = {}
    depths = {}
    for node in arch.nodes:
        depth = arch.positions[node]
        if depth not in depths:
            depths[depth] = []
        depths[depth].append(node)
    
    for depth, nodes in depths.items():
        for i, node in enumerate(nodes):
            pos[node] = (depth, i - len(nodes)/2)
    
    # Color nodes by operation
    from architecture import OPERATION_POOL
    colors = plt.cm.Set3(range(len(OPERATION_POOL)))
    node_colors = []
    for node in G.nodes():
        op = arch.operations[node]
        color_idx = OPERATION_POOL.index(op) if op in OPERATION_POOL else 0
        node_colors.append(colors[color_idx])
    
    # Draw graph
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                          node_size=1000, alpha=0.9, ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color='gray', 
                          arrows=True, arrowsize=20, ax=ax)
    
    # Labels
    labels = {}
    for node in G.nodes():
        op = arch.operations[node]
        ch = arch.channels[node]
        labels[node] = f"{op}\n{ch}ch"
    
    nx.draw_networkx_labels(G, pos, labels, font_size=8, ax=ax)
    
    # Legend
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                 markerfacecolor=colors[i], markersize=10,
                                 label=op) 
                      for i, op in enumerate(OPERATION_POOL)]
    ax.legend(handles=legend_elements, loc='upper left', 
             bbox_to_anchor=(1, 1))
    
    ax.set_title(f"{title}\nNodes: {len(arch.nodes)}, "
                f"Edges: {len(arch.edges)}, Depth: {arch.depth}")
    ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()


def create_results_report(output_dir: str):
    """
    Create HTML report of all results
    """
    from utils import load_architecture_json
    
    # Load summary
    summary_path = Path(output_dir) / 'summary.json'
    with open(summary_path) as f:
        summary = json.load(f)
    
    # Load results
    results_path = Path(output_dir) / 'results.jsonl'
    results = []
    with open(results_path) as f:
        for line in f:
            results.append(json.loads(line))
    
    # Sort by accuracy
    results.sort(key=lambda x: x['final_accuracy'], reverse=True)
    
    # Generate visualizations
    viz_dir = Path(output_dir) / 'visualizations'
    viz_dir.mkdir(exist_ok=True)
    
    arch_dir = Path(output_dir) / 'architectures'
    for result in results[:10]:  # Top 10
        arch_id = result['arch_id']
        json_path = arch_dir / f"{arch_id}.json"
        arch = load_architecture_json(str(json_path))
        
        viz_path = viz_dir / f"{arch_id}.png"
        title = f"{arch_id} - Acc: {result['final_accuracy']:.4f}"
        visualize_architecture(arch, save_path=str(viz_path), title=title)
    
    # Create HTML report
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Creative NAS Results</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .summary {{ background: #f0f0f0; padding: 20px; border-radius: 5px; }}
            .arch {{ border: 1px solid #ddd; margin: 20px 0; padding: 15px; }}
            img {{ max-width: 800px; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #4CAF50; color: white; }}
        </style>
    </head>
    <body>
        <h1>Creative NAS Results</h1>
        
        <div class="summary">
            <h2>Summary</h2>
            <p><strong>Dataset:</strong> {summary['dataset']}</p>
            <p><strong>Total Episodes:</strong> {summary['total_episodes']}</p>
            <p><strong>Architectures Explored:</strong> {summary['architectures_explored']}</p>
            <p><strong>Best Accuracy:</strong> {summary['best_accuracy']:.4f}</p>
            <p><strong>Average Accuracy:</strong> {summary['avg_accuracy']:.4f}</p>
        </div>
        
        <h2>Top Architectures</h2>
        <table>
            <tr>
                <th>Rank</th>
                <th>ID</th>
                <th>Accuracy</th>
                <th>Reward</th>
                <th>Depth</th>
                <th>Avg Width</th>
                <th>Parameters</th>
            </tr>
    """
    
    for i, result in enumerate(results[:20], 1):
        html += f"""
            <tr>
                <td>{i}</td>
                <td>{result['arch_id']}</td>
                <td>{result['final_accuracy']:.4f}</td>
                <td>{result['search_reward']:.4f}</td>
                <td>{result['depth']}</td>
                <td>{result['avg_width']:.1f}</td>
                <td>{result['total_params']:,}</td>
            </tr>
        """
    
    html += """
        </table>
        
        <h2>Architecture Visualizations (Top 10)</h2>
    """
    
    for result in results[:10]:
        arch_id = result['arch_id']
        html += f"""
        <div class="arch">
            <h3>{arch_id} - Accuracy: {result['final_accuracy']:.4f}</h3>
            <img src="visualizations/{arch_id}.png" alt="{arch_id}">
            <p><strong>Topological Novelty:</strong> {result.get('topological_novelty', 0):.3f}</p>
            <p><strong>Scale Novelty:</strong> {result.get('scale_novelty', 0):.3f}</p>
        </div>
        """
    
    html += """
    </body>
    </html>
    """
    
    report_path = Path(output_dir) / 'report.html'
    with open(report_path, 'w') as f:
        f.write(html)
    
    print(f"\nCreated HTML report: {report_path}")
    print(f"Open it in a browser to view results!")
