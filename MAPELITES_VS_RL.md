# MAP-Elites vs RL - Key Differences

## 🔄 Import Differences

### RL Version (DQN + GNN):
```python
from architecture import ArchitectureState, ActionSpace, OPERATION_POOL
from agent import CreativityDQN
from gnn_models import ArchitectureEncoder, DQNetwork
```

**Why ActionSpace?**
- RL needs discrete actions: ADD_NODE, REMOVE_NODE, ADD_EDGE, etc.
- Agent learns which actions to take via Q-learning

---

### MAP-Elites Version:
```python
from architecture import ArchitectureState, OPERATION_POOL
from map_elites import MAPElites, BehaviorSpace, MutationOperator
```

**No ActionSpace needed!**
- Uses evolutionary mutations instead
- No learned decision-making
- Just random mutations + keep best in each cell

---

## 🧬 How Each Approach Modifies Architectures

### **RL + GNN:**
```python
# Uses ActionSpace
valid_actions = arch.get_valid_actions()  # Returns ActionSpace.ADD_NODE, etc.
action = agent.select_action(state)       # Learn which action
new_arch = arch.apply_action(action)      # Modify architecture
```

### **MAP-Elites:**
```python
# Uses MutationOperator
mutation_type = random.choice(['add_node', 'remove_node', ...])
new_arch = mutator.mutate(parent_arch)    # Random mutation
```

---

## 📦 What Each File Needs

### architecture.py:
```python
# Defines:
- ArchitectureState: Graph representation of architectures
- ActionSpace: Action types (ADD_NODE, REMOVE_NODE, etc.) [RL ONLY]
- OPERATION_POOL: List of operations (conv3x3, conv5x5, etc.) [BOTH]
```

### For MAP-Elites:
- ✅ Needs: `ArchitectureState`, `OPERATION_POOL`
- ❌ Doesn't need: `ActionSpace` (no RL actions)

### For RL Version:
- ✅ Needs: `ArchitectureState`, `ActionSpace`, `OPERATION_POOL`

---

## 🔧 Fixed Imports

I've already fixed `map_elites.py` to only import what it needs:

```python
# map_elites.py (CORRECTED)
from architecture import ArchitectureState, OPERATION_POOL
# No ActionSpace needed!
```

---

## 💡 Why the Confusion?

Both approaches:
1. Work with `ArchitectureState` (same representation)
2. Use `OPERATION_POOL` (same operations)
3. Modify architectures (different methods)

But:
- **RL**: Learns which actions to take → needs `ActionSpace`
- **MAP-Elites**: Random mutations → doesn't need `ActionSpace`

---

## ✅ Quick Check

**If you see this error:**
```
NameError: name 'ActionSpace' is not defined
```

**In MAP-Elites files, that's correct!** 
- MAP-Elites doesn't use `ActionSpace`
- It only appears in RL files (agent.py, train.py, creative_nas_demo.ipynb)

**You're good to go!** 🚀

---

## 📁 File Dependencies

```
map_elites.py
├── architecture.py (ArchitectureState, OPERATION_POOL)
└── NO agent.py, NO gnn_models.py

train_map_elites.py
├── map_elites.py
├── evaluation.py
├── utils.py
└── visualize.py

agent.py (RL only)
├── architecture.py (ALL: ArchitectureState, ActionSpace, OPERATION_POOL)
├── gnn_models.py
├── novelty.py
└── evaluation.py
```

---

## 🎯 Summary

**MAP-Elites = Simpler!**
- Fewer imports
- No neural networks in search
- No ActionSpace needed
- Just mutations + archive

**You have both versions now:**
1. `train.py` / `agent.py` - RL version (uses ActionSpace)
2. `train_map_elites.py` / `map_elites.py` - MAP-Elites version (no ActionSpace)

Choose based on your needs! 🎨
