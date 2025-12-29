# Code-Vergleich: CFRSolverWithTree vs TensorCFRSolver

## 1. Datenstrukturen Initialisierung

### CFRSolverWithTree (Dict-basiert)
```python
# cfr_solver_with_tree.py
self.regret_sum = {}  # {infoset_key: {action: value}}
self.strategy_sum = {}  # {infoset_key: {action: value}}
self.nodes = {}  # {node_id: Node}
self.infoset_to_nodes = defaultdict(list)  # {infoset_key: [node_ids]}
```

### TensorCFRSolver (Tensor-basiert)
```python
# tensor_cfr_solver.py
num_infosets = 1000
num_actions = 4

self.regret_sum = torch.zeros((num_infosets, num_actions), device=device)
self.strategy_sum = torch.zeros((num_infosets, num_actions), device=device)

# Node-Daten als Arrays:
self.node_types = torch.tensor([...])  # [num_nodes]
self.players = torch.tensor([...])    # [num_nodes]
self.infosets = torch.tensor([...])    # [num_nodes]
self.children = torch.tensor([...])   # [num_nodes, num_actions]
self.payoffs = torch.tensor([...])     # [num_nodes, 2]
```

---

## 2. Tree Traversierung

### CFRSolverWithTree (Rekursiv)
```python
def cfr_iteration(self):
    for root_id in self.root_nodes:
        reach_probs = [1.0, 1.0]
        self.traverse_tree(root_id, 0, reach_probs)  # ← Rekursiv
        self.traverse_tree(root_id, 1, reach_probs)  # ← Rekursiv

def traverse_tree(self, node_id, player_id, reach_probabilities):
    node = self.nodes[node_id]  # ← Einzelner Node
    
    if node.type == 'terminal':
        return node.payoffs[player_id]
    
    # ... verarbeite diesen Node ...
    for action in node.legal_actions:
        child_id = node.children[action]
        # ← Rekursiver Aufruf für jeden Child
        utility = self.traverse_tree(child_id, player_id, new_reach_probs)
    
    return utility
```

### TensorCFRSolver (Layer-by-Layer)
```python
def _cfr_iteration(self, num_nodes):
    # Forward Pass: Layer für Layer (Top-Down)
    nodes_reach = torch.zeros((num_nodes, 2), device=device)
    nodes_reach[self.roots_tensor] = self.root_prob
    
    for layer_idx in self.layer_indices:  # ← Layer 1, dann 2, dann 3...
        decision_nodes = layer_idx[decision_mask]  # ← ALLE Nodes dieser Schicht
        
        # ← Batch-Operationen für alle Nodes gleichzeitig
        inf_ids = self.infosets[decision_nodes]  # Batch-Lookup
        node_strat = current_strategy[inf_ids]    # Batch-Lookup
        
        # ← Propagate Reach für alle Children auf einmal
        nodes_reach.index_add_(0, target_indices, source_vals)
    
    # Backward Pass: Layer für Layer (Bottom-Up)
    nodes_values = torch.zeros((num_nodes, 2), device=device)
    
    for layer_idx in reversed(self.layer_indices):  # ← Layer 5, dann 4, dann 3...
        # ← Alle Nodes dieser Schicht gleichzeitig verarbeiten
        nodes_values[dec_nodes] = ev  # Batch-Update
```

---

## 3. Reach Probability Update

### CFRSolverWithTree (Sequenziell)
```python
def traverse_tree(self, node_id, player_id, reach_probabilities):
    node = self.nodes[node_id]
    
    if current_player != player_id:
        opponent_strategy = self.get_current_strategy(...)
        
        for action in node.legal_actions:  # ← Einzelne Aktion
            action_prob = opponent_strategy[action]
            child_id = node.children[action]
            
            new_reach_probs = reach_probabilities.copy()
            new_reach_probs[opponent] *= action_prob
            
            # ← Rekursiv für jeden Child einzeln
            state_value += action_prob * self.traverse_tree(
                child_id, player_id, new_reach_probs
            )
```

### TensorCFRSolver (Batch)
```python
# In Forward Pass:
decision_nodes = layer_idx[decision_mask]  # ← z.B. [1, 2, 6, 7] (4 Nodes)

# ← Hole ALLE Strategies auf einmal
inf_ids = self.infosets[decision_nodes]  # [5, 5, 12, 12]
node_strat = current_strategy[inf_ids]   # [4 Nodes, 4 Actions] Tensor!

# ← Berechne Reach für ALLE Children auf einmal
multipliers = torch.ones((len(decision_nodes), num_actions, 2))
multipliers[mask0, :, 0] = node_strat[mask0]  # Batch-Update für P0
multipliers[mask1, :, 1] = node_strat[mask1]  # Batch-Update für P1

child_reach_vals = parent_reach_expanded * multipliers  # [4, 4, 2]

# ← Scatter zu allen Children auf einmal
nodes_reach.index_add_(0, target_indices, source_vals)
# ↑ Aktualisiert HUNDERTE Children gleichzeitig!
```

---

## 4. Utility Berechnung

### CFRSolverWithTree (Rekursiv)
```python
def traverse_tree(self, node_id, player_id, reach_probabilities):
    # Player's node
    action_utilities = {}
    
    for action in node.legal_actions:  # ← Einzelne Aktion
        child_id = node.children[action]
        
        # ← Rekursiv für jeden Child einzeln
        action_utilities[action] = self.traverse_tree(
            child_id, player_id, new_reach_probs
        )
    
    # ← Berechne Utility für diesen Node
    current_utility = sum(
        current_strategy[action] * action_utilities[action]
        for action in node.legal_actions
    )
    
    return current_utility
```

### TensorCFRSolver (Batch)
```python
# In Backward Pass:
dec_nodes = layer_idx[dec_mask]  # ← z.B. [1, 2, 6, 7] (4 Nodes)

# ← Hole ALLE Children auf einmal
child_ids = self.children[dec_nodes]  # [4 Nodes, 4 Actions]

# ← Hole ALLE Child Values auf einmal (Batch-Lookup!)
safe_child_ids = child_ids.clone()
safe_child_ids[~valid_mask] = 0
c_vals = nodes_values[safe_child_ids]  # [4 Nodes, 4 Actions, 2 Players]

# ← Berechne Expected Value für ALLE Nodes auf einmal
strat = current_strategy[self.infosets[dec_nodes]]  # [4, 4]
ev = (strat.unsqueeze(2) * c_vals).sum(dim=1)  # [4, 2]
# ↑ Matrix-Multiplikation über alle Nodes gleichzeitig!

nodes_values[dec_nodes] = ev  # Batch-Update
```

---

## 5. Regret Update

### CFRSolverWithTree (Einzeln)
```python
def update_regrets(self, info_set_key, legal_actions, 
                   action_utilities, current_utility, 
                   counterfactual_weight):
    for action in legal_actions:  # ← Einzelne Aktion
        instantaneous_regret = counterfactual_weight * (
            action_utilities[action] - current_utility
        )
        # ← Einzelner Update
        self.regret_sum[info_set_key][action] += instantaneous_regret
```

### TensorCFRSolver (Batch)
```python
# In Backward Pass:
dec_nodes = layer_idx[dec_mask]  # ← z.B. [1, 2, 6, 7] (4 Nodes)

# ← Berechne Instantaneous Regret für ALLE Nodes und ALLE Actions
opp_ids = 1 - p_ids
opp_reach = nodes_reach[dec_nodes].gather(1, opp_ids.unsqueeze(1)).squeeze(1)

q_vals = c_vals.gather(2, p_idx_expanded).squeeze(2)  # [4, 4]
v_vals = ev.gather(1, p_ids.unsqueeze(1)).squeeze(1)  # [4]

# ← Instantaneous Regret für ALLE Nodes und ALLE Actions gleichzeitig
inst_regret = opp_reach.unsqueeze(1) * (q_vals - v_vals.unsqueeze(1))
# Shape: [4 Nodes, 4 Actions]

# ← Batch-Update für alle Infosets gleichzeitig
self.delta_regret.index_add_(0, inf_ids, inst_regret)

# Am Ende der Iteration:
self.regret_sum += self.delta_regret  # Batch-Update!
```

---

## 6. Strategy Update

### CFRSolverWithTree (Einzeln)
```python
def update_strategy_sum(self, info_set_key, legal_actions, 
                        current_strategy, player_reach):
    for action in legal_actions:  # ← Einzelne Aktion
        # ← Einzelner Update
        self.strategy_sum[info_set_key][action] += (
            player_reach * current_strategy[action]
        )
```

### TensorCFRSolver (Batch)
```python
# In Forward Pass:
decision_nodes = layer_idx[decision_mask]  # ← z.B. [1, 2, 6, 7]

# ← Reach des spielenden Spielers für alle Nodes
reach_p = nodes_reach[decision_nodes].gather(1, p_ids.unsqueeze(1)).squeeze(1)

# ← Beitrag für Strategy Sum (für alle Nodes gleichzeitig)
contrib = node_strat * reach_p.unsqueeze(1)  # [4, 4]

if self.algorithm == 'cfr_plus':
    contrib *= self.t

# ← Batch-Update für alle Infosets gleichzeitig
self.strategy_sum.index_add_(0, inf_ids, contrib)
```

---

## 7. Regret Matching (Strategy Berechnung)

### CFRSolverWithTree (Einzeln pro Infoset)
```python
def get_current_strategy(self, info_set_key, legal_actions):
    # ← Für EIN Infoset
    regrets = {a: self.regret_sum[info_set_key][a] 
               for a in legal_actions}
    
    positive_regrets = {a: max(regrets[a], 0) 
                        for a in legal_actions}
    sum_pos = sum(positive_regrets.values())
    
    if sum_pos > 0:
        return {a: positive_regrets[a] / sum_pos 
                for a in legal_actions}
    else:
        return {a: 1.0 / len(legal_actions) 
                for a in legal_actions}
```

### TensorCFRSolver (Batch für alle Infosets)
```python
def _cfr_iteration(self, num_nodes):
    # ← Mask invalid regrets (für ALLE Infosets gleichzeitig)
    self.regret_sum.masked_fill_(~self.infoset_valid_actions, -1e9)
    
    # ← Clamp to positive (für ALLE Infosets gleichzeitig)
    positive_regrets = torch.clamp(self.regret_sum, min=0)
    sum_pos = torch.sum(positive_regrets, dim=1, keepdim=True)  # [Infosets, 1]
    
    # ← Initialize strategy (für ALLE Infosets gleichzeitig)
    current_strategy = torch.zeros_like(self.regret_sum)
    has_pos = (sum_pos > 1e-12).squeeze()
    
    # ← Positive regrets case (für ALLE Infosets gleichzeitig!)
    current_strategy[has_pos] = positive_regrets[has_pos] / sum_pos[has_pos]
    
    # ← Uniform case (für ALLE Infosets gleichzeitig!)
    uniform_probs = 1.0 / torch.clamp(self.infoset_valid_counts, min=1)
    no_pos_mask = ~has_pos
    uniform_contrib = uniform_probs * self.infoset_valid_actions.float()
    current_strategy[no_pos_mask] = uniform_contrib[no_pos_mask]
    
    current_strategy.masked_fill_(~self.infoset_valid_actions, 0.0)
```

---

## 8. Kompletter CFR Iteration Flow

### CFRSolverWithTree
```
cfr_iteration()
    ↓
for root in root_nodes:
    traverse_tree(root, player=0)  ← Rekursiv
        ↓
    traverse_tree(root, player=1)  ← Rekursiv
        ↓
    (Node für Node, sequenziell)
```

### TensorCFRSolver
```
_cfr_iteration()
    ↓
1. Regret Matching (alle Infosets parallel)
    ↓
2. Forward Pass (Layer-by-Layer Top-Down)
    Layer 1 → Layer 2 → Layer 3 → ...
    (Alle Nodes einer Schicht parallel)
    ↓
3. Backward Pass (Layer-by-Layer Bottom-Up)
    Layer 5 → Layer 4 → Layer 3 → ...
    (Alle Nodes einer Schicht parallel)
    ↓
4. Regret Update (alle Infosets parallel)
```

---

## 9. Performance-Vergleich

### CFRSolverWithTree
```python
# Beispiel: 1000 Nodes, 4 Layers
# Zeit pro Iteration: ~1.0s

# Operationen:
- 1000 rekursive Aufrufe
- 1000 Dictionary-Lookups
- 1000 einzelne Updates
- CPU-only (1-8 Cores)
```

### TensorCFRSolver
```python
# Beispiel: 1000 Nodes, 4 Layers
# Zeit pro Iteration: ~0.04s (25× schneller!)

# Operationen:
- 4 Layer-Iterationen
- Batch-Lookups (100-300 Nodes gleichzeitig)
- Batch-Updates (100-300 Nodes gleichzeitig)
- GPU (1000+ Cores parallel)
```

---

## 10. Wichtige Tensor-Operationen erklärt

### `index_add_`: Batch-Update
```python
# Statt:
for i in [1, 2, 3]:
    array[i] += values[i]

# Tensor:
array.index_add_(0, [1, 2, 3], values)
# ↑ Aktualisiert Indizes 1, 2, 3 gleichzeitig
```

### `gather`: Batch-Lookup
```python
# Statt:
result = []
for node_id in [1, 2, 3]:
    result.append(array[infosets[node_id]])

# Tensor:
inf_ids = infosets[[1, 2, 3]]  # [5, 5, 12]
result = array[inf_ids]  # Batch-Lookup!
# ↑ Holt alle 3 Werte gleichzeitig
```

### Masking: Batch-Filtering
```python
# Statt:
decision_nodes = []
for node_id in layer:
    if node_types[node_id] == 1:
        decision_nodes.append(node_id)

# Tensor:
decision_mask = (node_types[layer] == 1)  # [True, False, True, ...]
decision_nodes = layer[decision_mask]  # Batch-Filter!
# ↑ Filtert alle Nodes gleichzeitig
```

### Broadcasting: Automatische Dimension-Erweiterung
```python
# Statt:
for i in range(len(nodes)):
    result[i] = array[i] * scalar

# Tensor:
result = array * scalar  # Broadcasting!
# ↑ Multipliziert alle Elemente gleichzeitig
```

---

## Zusammenfassung: Die Kernunterschiede

| Operation | CFRSolverWithTree | TensorCFRSolver |
|-----------|------------------|-----------------|
| **Traversierung** | Rekursiv, Node für Node | Layer-by-Layer, Batch |
| **Lookups** | Dictionary (langsam) | Tensor Indexing (schnell) |
| **Updates** | Einzeln, sequenziell | Batch, parallel |
| **Verarbeitung** | CPU, 1-8 Cores | GPU, 1000+ Cores |
| **Geschwindigkeit** | ~1s/Iteration | ~0.04s/Iteration |

**Der Hauptunterschied:**
- **CFRSolverWithTree**: "Gehe durch den Tree wie ein Mensch" (sequenziell)
- **TensorCFRSolver**: "Verarbeite alle Nodes einer Tiefe wie eine Maschine" (parallel)


