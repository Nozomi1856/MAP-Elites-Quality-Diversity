# 🎨 Creative Neural Architecture Search (NAS)

Discover novel neural network architectures through reinforcement learning that optimizes for both **performance** and **creativity** (topological and scale novelty).

## 📋 Overview

This system uses Deep Q-Learning to automatically design neural network architectures by:
- **Adding/removing layers and connections** (topological creativity)
- **Adjusting channel dimensions** (scale creativity)  
- **Balancing performance with novelty** through multi-objective rewards

Unlike traditional NAS that searches a fixed space, this system **generates** architectures through iterative refinement.

---

## 🚀 Quick Start (Interactive Demo)

**Estimated time: 3-10 minutes**

```bash
# Install dependencies
pip install -r requirements.txt

# Run interactive demo
python interactive_demo.py
```

The demo will guide you through:
1. Choosing a dataset (MNIST, Fashion-MNIST, or CIFAR-10)
2. Setting training episodes (50-200 for demo)
3. Configuring evaluation depth
4. Selecting how many top architectures to save

**Output:**
- PyTorch models (`.pth` files)
- Architecture descriptions (`.json` files)
- Interactive HTML report with visualizations
- Complete results in `.jsonl` format

---

## 📦 Installation

### Requirements
- Python 3.8+
- PyTorch 2.0+
- CUDA (optional, but recommended)

### Setup

```bash
# Create environment
conda create -n creative_nas python=3.10
conda activate creative_nas

# Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install torch-geometric networkx scipy matplotlib tqdm

# Or use requirements.txt
pip install -r requirements.txt
```

---

## 🎯 Usage

### Option 1: Interactive Demo (Recommended for First-Time Users)

```bash
python interactive_demo.py
```

**Features:**
- ✅ Guided step-by-step setup
- ✅ Reasonable defaults for quick demos
- ✅ Automatic result saving and visualization
- ✅ Estimated completion time

**Example Session:**
```
🎨 CREATIVE NEURAL ARCHITECTURE SEARCH - INTERACTIVE DEMO 🎨

📊 Step 1: Choose dataset to optimize for:
  1. MNIST - Handwritten digits (fastest, ~2-3 min)
  2. Fashion-MNIST - Clothing items (fast, ~3-5 min)
  3. CIFAR-10 - Natural images (slower, ~5-10 min)

Enter choice (1-3) [default: mnist]: 1

🔄 Step 2: Training episodes
   How many episodes? [50-200] (default: 100): 100

... (continues interactively)
```

### Option 2: Command-Line Training

```bash
# Basic training
python train.py

# Custom configuration
python train.py --episodes 2000 --dataset mnist --eval_top_k 30

# CPU training (no GPU)
python train.py --device cpu --episodes 500
```

**Arguments:**
- `--episodes`: Number of training episodes (default: 1000)
- `--dataset`: Dataset choice: `mnist`, `fashion`, `cifar10` (default: cifar10)
- `--device`: `cuda` or `cpu` (default: cuda)
- `--eval_top_k`: How many top architectures to fully evaluate (default: 20)
- `--output_dir`: Where to save results (default: results/)

---

## 📂 Output Structure

After training, results are saved in organized directories:

```
demo_results/mnist_20241203_143022/
├── models/                      # PyTorch model files
│   ├── arch_000.pth
│   ├── arch_001.pth
│   └── ...
├── architectures/               # Architecture descriptions
│   ├── arch_000.json
│   ├── arch_001.json
│   └── ...
├── visualizations/              # Architecture graphs
│   ├── arch_000.png
│   ├── arch_001.png
│   └── ...
├── results.jsonl               # All results (one per line)
├── summary.json                # Overview statistics
├── report.html                 # Interactive report ⭐
└── agent.pt                    # Trained RL agent
```

### Key Files

**`report.html`**: Open in browser for:
- Summary statistics
- Top architecture rankings
- Visual graphs of best architectures
- Detailed metrics (accuracy, novelty, parameters)

**`results.jsonl`**: Machine-readable results
```json
{"arch_id": "arch_000", "final_accuracy": 0.9847, "search_reward": 0.723, ...}
{"arch_id": "arch_001", "final_accuracy": 0.9801, "search_reward": 0.698, ...}
```

**`models/*.pth`**: PyTorch models ready to load
```python
import torch
checkpoint = torch.load('models/arch_000.pth')
model_state = checkpoint['model_state_dict']
metadata = checkpoint['metadata']
```

**`architectures/*.json`**: Architecture definitions
```json
{
  "nodes": [0, 1, 2, 3],
  "edges": [[0, 1], [1, 2], [0, 2], [2, 3]],
  "operations": {"0": "skip_connect", "1": "conv3x3", ...},
  "channels": {"0": 3, "1": 64, "2": 128, "3": 64},
  "metadata": {"final_accuracy": 0.9847, ...}
}
```

---

## 🔧 Advanced Usage

### Loading and Using Saved Models

```python
import torch
from utils import load_architecture_json
from evaluation import ConvNet

# Load architecture from JSON
arch = load_architecture_json('architectures/arch_000.json')

# Create model
model = ConvNet(arch, num_classes=10)

# Load trained weights
checkpoint = torch.load('models/arch_000.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# Use for inference
model.eval()
with torch.no_grad():
    output = model(input_tensor)
```

### Visualizing Custom Architectures

```python
from visualize import visualize_architecture
from utils import load_architecture_json

# Load architecture
arch = load_architecture_json('architectures/arch_000.json')

# Visualize
visualize_architecture(arch, save_path='my_arch.png', 
                      title='Custom Architecture')
```

### Analyzing Results Programmatically

```python
import json

# Load all results
results = []
with open('results.jsonl') as f:
    for line in f:
        results.append(json.loads(line))

# Sort by accuracy
results.sort(key=lambda x: x['final_accuracy'], reverse=True)

# Find most novel
most_novel = max(results, key=lambda x: x['topological_novelty'])

# Find best depth/width ratio
best_ratio = max(results, 
                key=lambda x: x['depth'] / (x['avg_width'] + 1))
```

---

## 🎓 How It Works

### 1. **Reinforcement Learning Setup**

- **State**: Current architecture represented as a graph
- **Actions**: Add/remove nodes/edges, adjust channels, stop building
- **Reward**: Weighted combination of:
  - Performance (accuracy on validation set)
  - Topological novelty (graph edit distance to previously seen architectures)
  - Scale novelty (unusual depth/width patterns)

### 2. **Architecture Representation**

Each architecture is a directed acyclic graph (DAG) where:
- **Nodes** = Operations (conv3x3, conv5x5, pooling, skip connections)
- **Edges** = Data flow between operations
- **Attributes** = Channel dimensions, positions (depth)

### 3. **Novelty Metrics**

**Topological Novelty**: How different is the connectivity pattern?
- Uses graph edit distance to compare with archive
- K-nearest neighbors averaging

**Scale Novelty**: How unusual are the dimensions?
- Measures deviation in depth, width, depth/width ratio
- Tracks width variance (tapering patterns)

### 4. **Training Process**

1. **Exploration Phase**: Randomly generate architectures
2. **Experience Replay**: Learn from past architecture evaluations
3. **Policy Refinement**: DQN learns to select good actions
4. **Diversity Maintenance**: Novelty rewards prevent premature convergence

---

## 📊 Performance Tips

### For Fastest Results (Demo Mode)
```bash
python interactive_demo.py
# Choose: MNIST, 50 episodes, 2 epochs, 5 architectures
# Time: ~2-3 minutes
```

### For Best Architectures (Research Mode)
```bash
python train.py --episodes 2000 --dataset cifar10 --eval_top_k 50
# Time: ~60-80 hours on T4 GPU
```

### GPU vs CPU
- **GPU (T4)**: 50-60 hours for 1000 episodes on CIFAR-10
- **CPU**: 5-10x slower, only recommended for demos with MNIST

---

## 🐛 Troubleshooting

### CUDA Out of Memory
```bash
# Reduce batch size in evaluation.py
# Change batch_size=128 to batch_size=64
```

### ImportError for torch_geometric
```bash
pip install torch-geometric
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
```

### No GPU detected
The system will automatically use CPU, but training will be much slower. For demos, use MNIST dataset with fewer episodes.

---

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@software{creative_nas_2024,
  title = {Creative Neural Architecture Search},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/creative-nas}
}
```

---

## 📄 License

MIT License - see LICENSE file for details

---

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

---

## 💬 Support

- 📧 Email: support@example.com
- 🐛 Issues: GitHub Issues page
- 📖 Docs: Full documentation at docs/

---

**Happy Architecture Hunting! 🎨✨**
