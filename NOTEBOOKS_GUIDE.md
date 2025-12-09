# 📓 Jupyter Notebooks Guide

Three notebooks are provided for different use cases:

---

## 🎯 Which Notebook Should I Use?

### 1. **creative_nas_demo.ipynb** ⭐ **START HERE**

**Best for:** First-time users, quick demos, learning the system

**Time:** 3-10 minutes

**What it does:**
- Interactive demo with easy configuration
- Guided step-by-step execution
- Trains a small number of architectures (5-10)
- Creates visualizations and reports
- Perfect for validating your setup

**When to use:**
- First time using the system
- Want to see results quickly
- Testing if everything works
- Demonstrating to others
- Learning how the system works

**Quick start:**
```python
# Just set these at the top:
DATASET = 'mnist'      # Fastest option
EPISODES = 50          # Quick run
EVAL_EPOCHS = 2        # Fast evaluation
TOP_K = 5              # Small number of architectures
```

---

### 2. **creative_nas_complete.ipynb**

**Best for:** Full training runs, research, customization

**Time:** 1-60+ hours depending on settings

**What it does:**
- Complete system with all components
- Full control over configuration
- Longer training for better results
- Comprehensive analysis tools
- Production-quality outputs

**When to use:**
- After successfully running the demo
- Want publication-quality results
- Need to customize components
- Running overnight/weekend training
- Serious research work

**Configuration example:**
```python
CONFIG = {
    'dataset': 'cifar10',
    'episodes': 2000,         # Long run
    'eval_epochs': 3,
    'final_epochs': 20,       # Full evaluation
    'top_k': 50,              # Many architectures
}
```

---

### 3. **architecture_explorer.ipynb**

**Best for:** Analyzing saved results, loading models, exploration

**Time:** 5-15 minutes

**What it does:**
- Load previously saved architectures
- Visualize and compare results
- Load trained model weights
- Test inference
- Export models for deployment
- Analyze patterns across architectures

**When to use:**
- After training is complete
- Want to explore saved results
- Need to load a trained model
- Comparing different training runs
- Exporting models for use elsewhere
- Analyzing what worked best

**Usage:**
```python
# Point to your results directory
RESULTS_DIR = 'demo_results/mnist_20241203_143022'

# Then explore all saved architectures
```

---

## 🚀 Recommended Workflow

### First Time Users:

```
1. creative_nas_demo.ipynb (3 minutes)
   ↓
2. Check results look good
   ↓
3. architecture_explorer.ipynb (5 minutes)
   ↓
4. Understand outputs and visualizations
   ↓
5. creative_nas_complete.ipynb (longer run)
```

### Experienced Users:

```
1. creative_nas_complete.ipynb (full training)
   ↓
2. architecture_explorer.ipynb (analyze results)
   ↓
3. Export best models for deployment
```

---

## 📊 Comparison Table

| Feature | Demo | Complete | Explorer |
|---------|------|----------|----------|
| **Time** | 3-10 min | 1-60 hrs | 5-15 min |
| **Difficulty** | ⭐ Easy | ⭐⭐ Medium | ⭐ Easy |
| **Customization** | Limited | Full | N/A |
| **Output Quality** | Good | Excellent | N/A |
| **Use Case** | Learning | Research | Analysis |
| **GPU Required** | Optional | Recommended | No |

---

## 💡 Tips

### For Demo Notebook:
- Start with MNIST dataset (fastest)
- Use 50-100 episodes for quick validation
- Increase settings after you verify it works
- GPU optional but recommended

### For Complete Notebook:
- Use CIFAR-10 for best results
- 1000-2000 episodes for publication quality
- Run overnight or over weekend
- GPU highly recommended
- Save checkpoints frequently

### For Explorer Notebook:
- Works on CPU (no training needed)
- Great for collaborative analysis
- Export models for deployment
- Compare multiple training runs
- Create presentations with visualizations

---

## 🎓 Learning Path

**Day 1: Getting Started**
```
1. Run creative_nas_demo.ipynb with MNIST (3 min)
2. Open report.html to see results
3. Run architecture_explorer.ipynb on demo results
```

**Day 2: First Real Run**
```
1. Run creative_nas_demo.ipynb with Fashion-MNIST (10 min)
2. Compare results with Day 1
3. Analyze patterns in explorer notebook
```

**Week 1: Production Run**
```
1. Run creative_nas_complete.ipynb with CIFAR-10
2. Let it train overnight (8-12 hours)
3. Analyze all results in explorer
4. Export best models
```

---

## ⚙️ Configuration Quick Reference

### Quick Demo (2-3 minutes):
```python
DATASET = 'mnist'
EPISODES = 50
EVAL_EPOCHS = 2
TOP_K = 5
```

### Medium Run (10-15 minutes):
```python
DATASET = 'fashion'
EPISODES = 100
EVAL_EPOCHS = 3
TOP_K = 10
```

### Full Run (60+ hours):
```python
DATASET = 'cifar10'
EPISODES = 2000
EVAL_EPOCHS = 3
FINAL_EPOCHS = 20
TOP_K = 50
```

---

## 🆘 Troubleshooting

### Demo won't run?
- Check all Python files are in same directory
- Install dependencies: `pip install -r requirements.txt`
- Try CPU mode if GPU fails

### Complete notebook too slow?
- Reduce episodes to 200-500 for testing
- Use MNIST instead of CIFAR-10
- Check GPU is being used

### Explorer can't find results?
- Check RESULTS_DIR path is correct
- Make sure training completed
- Look in both demo_results/ and results/ folders

---

## 📞 Getting Help

1. Check error messages carefully
2. Verify all dependencies installed
3. Try demo notebook first to validate setup
4. Check GPU is detected: `torch.cuda.is_available()`
5. Reduce settings if running out of memory

---

## ✨ Pro Tips

1. **Always start with demo** - validates your setup in minutes
2. **Save early, save often** - checkpoints are your friend
3. **Use explorer liberally** - helps understand what's happening
4. **Start small, scale up** - 50 episodes → 100 → 500 → 2000
5. **Compare runs** - use explorer to analyze multiple experiments

---

**Happy exploring! 🎨✨**
