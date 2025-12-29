# Tensor CFR Solver - Detaillierte Erklärung

## 1. Was sind Tensoren?

**Tensoren = Mehrdimensionale Arrays (wie NumPy, aber auf GPU)**

Stell dir vor:
- **Python Dictionary**: `regret_sum['infoset_key']['action'] = 5.0`
- **Tensor**: `regret_sum[infoset_id, action_id] = 5.0`

### Beispiel aus dem Code:

```python
# Statt Dictionary:
self.regret_sum = {
    'K|check': {'bet': 2.5, 'check': 1.0},
    'Q|bet': {'call': 3.0, 'fold': 0.0}
}

# Tensor (2D Array):
self.regret_sum = torch.zeros((num_infosets, num_actions))
# Shape: [1000 Infosets, 4 Aktionen] = 4000 Werte in einem Array
# Zugriff: regret_sum[infoset_id, action_id]
```

**Vorteil**: Alle 4000 Werte können **gleichzeitig** auf der GPU berechnet werden!

---

## 2. Layer-by-Layer: Warum und Wie?

### Problem mit rekursiver Traversierung:

```
Dein cfr_solver_with_tree.py:
├─ Node 1 (Depth 1)
│  ├─ Node 2 (Depth 2) ← verarbeite einzeln
│  │  ├─ Node 5 (Depth 3) ← verarbeite einzeln
│  │  └─ Node 6 (Depth 3) ← verarbeite einzeln
│  └─ Node 3 (Depth 2) ← verarbeite einzeln
└─ Node 4 (Depth 1)
```

**Problem**: Du gehst Node für Node durch → **sequenziell**, langsam!

### Lösung: Layer-by-Layer (Batch Processing)

```
Layer 1: [Node 1, Node 4]           ← alle gleichzeitig verarbeiten!
Layer 2: [Node 2, Node 3]           ← alle gleichzeitig verarbeiten!
Layer 3: [Node 5, Node 6]           ← alle gleichzeitig verarbeiten!
```

**Vorteil**: Alle Nodes einer Tiefe werden **parallel** verarbeitet!

---

## 3. Wie wird der Tree in Layers organisiert?

### Schritt 1: Tree bauen (wie vorher)

```python
# Aus tensor_game_tree.py
def traverse(depth):
    # ... Node erstellen ...
    flat_nodes.append((type, player, infoset_id, depth, ...))
    # depth wird gespeichert!
```

### Schritt 2: Nach Depth gruppieren

```python
# Aus tensor_cfr_solver.py, Zeile 171-182
max_depth = np.max(depths)  # z.B. 5

self.layer_indices = []
for d in range(1, max_depth + 1):
    # Finde alle Nodes mit depth == d
    indices = np.where(depths == d)[0]
    # z.B. Layer 1: [0, 5, 10] (3 Nodes)
    #     Layer 2: [1, 2, 6, 7, 11, 12] (6 Nodes)
    self.layer_indices.append(torch.tensor(indices))
```

**Ergebnis**: `layer_indices[0]` = alle Nodes in Tiefe 1, `layer_indices[1]` = alle in Tiefe 2, etc.

---

## 4. Forward Pass: Reach Probabilities Layer-by-Layer

### Was passiert?

Wir berechnen für jeden Node: "Wie wahrscheinlich ist es, dass wir hier ankommen?"

### Rekursiv (dein Code):

```python
def traverse_tree(node_id, player_id, reach_probs):
    node = self.nodes[node_id]
    # ... berechne für DIESEN Node ...
    for action in node.legal_actions:
        child_id = node.children[action]
        traverse_tree(child_id, ...)  # Rekursiv weiter
```

**Problem**: Ein Node nach dem anderen, sequenziell.

### Tensor Layer-by-Layer:

```python
# Zeile 166-214 in tensor_cfr_solver.py

# 1. Initialisiere Roots
nodes_reach[self.roots_tensor, 0] = self.root_prob  # P0
nodes_reach[self.roots_tensor, 1] = self.root_prob  # P1

# 2. Gehe Layer für Layer durch (Top-Down)
for layer_idx in self.layer_indices:  # Layer 1, dann Layer 2, dann Layer 3...
    
    # Hole ALLE Decision Nodes dieser Schicht
    decision_nodes = layer_idx[decision_mask]  # z.B. [1, 2, 6, 7] (4 Nodes)
    
    # Hole ALLE Infosets dieser Nodes auf einmal
    inf_ids = self.infosets[decision_nodes]  # [5, 5, 12, 12]
    
    # Hole ALLE Strategies auf einmal
    node_strat = current_strategy[inf_ids]  # Shape: [4 Nodes, 4 Actions]
    # Das ist ein Batch-Lookup! Alle 4 Nodes gleichzeitig!
    
    # Berechne Reach für ALLE Children auf einmal
    child_reach_vals = parent_reach_expanded * multipliers  # [4 Nodes, 4 Actions, 2 Players]
    
    # Scatter zu Children (alle auf einmal)
    nodes_reach.index_add_(0, target_indices, source_vals)
    # ↑ Das aktualisiert HUNDERTE Nodes gleichzeitig!
```

**Der Trick**: 
- Statt `for node in nodes: process(node)` → `process(all_nodes_at_once)`
- Tensor-Operationen arbeiten auf **ganzen Arrays** gleichzeitig

---

## 5. Backward Pass: Utilities Layer-by-Layer

### Was passiert?

Wir berechnen rückwärts: "Was ist der erwartete Wert von diesem Node?"

### Rekursiv (dein Code):

```python
def traverse_tree(node_id, ...):
    # ... gehe zu Children ...
    action_utilities = {}
    for action in legal_actions:
        child_id = node.children[action]
        action_utilities[action] = traverse_tree(child_id, ...)  # Rekursiv
    # Berechne Utility
    return utility
```

### Tensor Layer-by-Layer (Bottom-Up):

```python
# Zeile 220-269 in tensor_cfr_solver.py

# Gehe rückwärts durch Layers (von tiefstem zu oberstem)
for layer_idx in reversed(self.layer_indices):  # Layer 5, dann 4, dann 3...
    
    # 1. Terminal Nodes: Setze Payoffs
    term_nodes = layer_idx[term_mask]
    nodes_values[term_nodes] = self.payoffs[term_nodes]
    # ↑ Alle Terminal Nodes dieser Schicht auf einmal!
    
    # 2. Decision Nodes: Berechne aus Children
    dec_nodes = layer_idx[dec_mask]
    
    # Hole ALLE Children auf einmal
    child_ids = self.children[dec_nodes]  # [Batch, 4 Actions]
    # z.B. für Node 1: [2, 3, -1, -1] (Children bei Action 0,1,2,3)
    
    # Hole ALLE Child Values auf einmal
    c_vals = nodes_values[safe_child_ids]  # [Batch, 4 Actions, 2 Players]
    # ↑ Das ist ein Batch-Lookup! Alle Child-Values gleichzeitig!
    
    # Berechne Expected Value für ALLE Nodes auf einmal
    ev = (strat.unsqueeze(2) * c_vals).sum(dim=1)  # [Batch, 2 Players]
    # ↑ Matrix-Multiplikation über alle Nodes gleichzeitig!
    
    nodes_values[dec_nodes] = ev
```

**Der Trick**: 
- Statt rekursiv von unten nach oben → **Layer für Layer von unten nach oben**
- Alle Nodes einer Schicht werden **parallel** verarbeitet

---

## 6. Konkrete Zahlen-Beispiele

### Beispiel: 1000 Nodes, 4 Layers

**Rekursiv (cfr_solver_with_tree.py):**
```
Node 1 → Node 2 → Node 5 → Node 6 → Node 2 fertig → Node 3 → ...
Zeit: 1000 Nodes × 0.001s = 1.0s pro Iteration
```

**Tensor Layer-by-Layer:**
```
Layer 1: [250 Nodes] → alle parallel → 0.01s
Layer 2: [300 Nodes] → alle parallel → 0.01s  
Layer 3: [300 Nodes] → alle parallel → 0.01s
Layer 4: [150 Nodes] → alle parallel → 0.01s
Zeit: 0.04s pro Iteration (25× schneller!)
```

**Warum schneller?**
- GPU kann **Hunderte von Operationen parallel** ausführen
- Statt 1000 einzelne Operationen → 4 Batch-Operationen

---

## 7. Die wichtigsten Tensor-Operationen

### `index_add_`: Batch-Update

```python
# Statt:
for node_id in [1, 2, 3]:
    nodes_reach[node_id] += contribution[node_id]

# Tensor:
nodes_reach.index_add_(0, [1, 2, 3], contributions)
# ↑ Aktualisiert Nodes 1, 2, 3 gleichzeitig!
```

### `gather`: Batch-Lookup

```python
# Statt:
for node_id in [1, 2, 3]:
    strategy = current_strategy[infosets[node_id]]

# Tensor:
inf_ids = self.infosets[[1, 2, 3]]  # [5, 5, 12]
node_strat = current_strategy[inf_ids]  # [3 Nodes, 4 Actions]
# ↑ Holt alle 3 Strategies gleichzeitig!
```

### Masking: Batch-Filtering

```python
# Statt:
decision_nodes = []
for node_id in layer:
    if node_types[node_id] == 1:
        decision_nodes.append(node_id)

# Tensor:
decision_mask = (node_types_layer == 1)  # [True, False, True, ...]
decision_nodes = layer_idx[decision_mask]  # Batch-Filter!
```

---

## 8. Zusammenfassung: Warum ist das schneller?

| Aspekt | Rekursiv (dein Code) | Tensor Layer-by-Layer |
|--------|---------------------|----------------------|
| **Verarbeitung** | Node für Node | Alle Nodes einer Schicht parallel |
| **Speicher** | Python Objekte (langsam) | Tensors (GPU-optimiert) |
| **Operationen** | 1000 einzelne Updates | 4 Batch-Updates |
| **Hardware** | CPU (1-8 Cores) | GPU (1000+ Cores) |
| **Geschwindigkeit** | ~1s/Iteration | ~0.04s/Iteration |

**Der Hauptunterschied**: 
- **Rekursiv**: "Gehe durch den Tree wie ein Mensch" (sequenziell)
- **Tensor Layer-by-Layer**: "Verarbeite alle Nodes einer Tiefe wie eine Maschine" (parallel)

---

## 9. Code-Flow Diagramm

```
CFR Iteration Start
    ↓
1. Regret Matching (alle Infosets parallel)
    ↓
2. Forward Pass (Layer-by-Layer Top-Down)
    ├─ Layer 1: [100 Nodes] → alle parallel verarbeiten
    ├─ Layer 2: [200 Nodes] → alle parallel verarbeiten
    ├─ Layer 3: [300 Nodes] → alle parallel verarbeiten
    └─ Layer 4: [400 Nodes] → alle parallel verarbeiten
    ↓
3. Backward Pass (Layer-by-Layer Bottom-Up)
    ├─ Layer 4: [400 Nodes] → alle parallel verarbeiten
    ├─ Layer 3: [300 Nodes] → alle parallel verarbeiten
    ├─ Layer 2: [200 Nodes] → alle parallel verarbeiten
    └─ Layer 1: [100 Nodes] → alle parallel verarbeiten
    ↓
4. Regret Update (alle Infosets parallel)
    ↓
Iteration fertig!
```

---

## 10. Praktisches Beispiel: Reach Probability Update

### Rekursiv (dein Code):

```python
def traverse_tree(node_id, reach_probs):
    node = self.nodes[node_id]
    strategy = get_strategy(node.infoset_key)
    
    for action in node.legal_actions:
        child_id = node.children[action]
        new_reach = reach_probs.copy()
        new_reach[node.player] *= strategy[action]
        traverse_tree(child_id, new_reach)  # Rekursiv
```

**Problem**: Für jeden Node einzeln, sequenziell.

### Tensor Layer-by-Layer:

```python
# Alle Nodes einer Schicht auf einmal:
decision_nodes = layer_idx[decision_mask]  # [1, 2, 3, 4, 5, ...] (100 Nodes)

# Hole alle Strategies auf einmal
inf_ids = self.infosets[decision_nodes]  # [5, 5, 12, 12, ...]
node_strat = current_strategy[inf_ids]  # [100 Nodes, 4 Actions] Tensor!

# Berechne Child Reach für ALLE Nodes und ALLE Actions auf einmal
multipliers = torch.ones((100, 4, 2))  # [100 Nodes, 4 Actions, 2 Players]
multipliers[mask0, :, 0] = node_strat[mask0]  # Batch-Update für P0
multipliers[mask1, :, 1] = node_strat[mask1]  # Batch-Update für P1

child_reach_vals = parent_reach_expanded * multipliers  # [100, 4, 2]

# Scatter zu allen Children auf einmal
nodes_reach.index_add_(0, target_indices, source_vals)
# ↑ Aktualisiert HUNDERTE Children gleichzeitig!
```

**Vorteil**: Statt 100× rekursiv → **1× Batch-Operation**!

---

Das ist der Kern: **Statt sequenziell durch den Tree zu gehen, verarbeitet man alle Nodes einer Tiefe parallel als Batch!**
