# Migrations-Guide: Von CFRSolverWithTree zu TensorCFRSolver

## Übersicht: Was ändert sich?

| Aspekt | CFRSolverWithTree | TensorCFRSolver |
|--------|-------------------|-----------------|
| **Datenstruktur** | Python Dicts + Node-Objekte | PyTorch Tensors |
| **Tree-Traversierung** | Rekursiv (Node für Node) | Layer-by-Layer (Batch) |
| **Regret/Strategy** | `{infoset_key: {action: value}}` | `Tensor[infoset_id, action_id]` |
| **Verarbeitung** | Sequenziell | Parallel (GPU) |

---

## Schritt 1: Datenstrukturen konvertieren

### 1.1 Regret Sum & Strategy Sum

**Vorher (Dict-basiert):**
```python
# cfr_solver_with_tree.py
self.regret_sum = {}  # {infoset_key: {action: value}}
self.strategy_sum = {}  # {infoset_key: {action: value}}

# Zugriff:
regret = self.regret_sum['K|check']['bet']
```

**Nachher (Tensor-basiert):**
```python
# tensor_cfr_solver.py
num_infosets = 1000  # Anzahl eindeutiger Infosets
num_actions = 4      # check, bet, call, fold

self.regret_sum = torch.zeros((num_infosets, num_actions), device=device)
self.strategy_sum = torch.zeros((num_infosets, num_actions), device=device)

# Zugriff:
# Zuerst: infoset_key → infoset_id mappen
infoset_id = infoset_map['K|check']  # z.B. 42
action_id = action_to_idx['bet']     # z.B. 1
regret = self.regret_sum[infoset_id, action_id]
```

**Migration:**
```python
# Schritt 1: Infoset-Mapping erstellen
infoset_map = {}  # {infoset_key: infoset_id}
next_infoset_id = 0

# Beim Tree-Bauen:
for node in nodes:
    if node.infoset_key not in infoset_map:
        infoset_map[node.infoset_key] = next_infoset_id
        next_infoset_id += 1

# Schritt 2: Tensor initialisieren
num_infosets = len(infoset_map)
self.regret_sum = torch.zeros((num_infosets, num_actions))
```

---

### 1.2 Node-Struktur

**Vorher (Objekt-basiert):**
```python
class Node:
    def __init__(self, node_id):
        self.node_id = node_id
        self.type = None  # 'terminal' oder 'decision'
        self.player = None
        self.infoset_key = None
        self.legal_actions = []
        self.children = {}  # {action: child_node_id}
        self.payoffs = None

# Zugriff:
node = self.nodes[node_id]
child_id = node.children['bet']
```

**Nachher (Tensor-basiert):**
```python
# Statt Node-Objekte: Flache Arrays
self.node_types = torch.tensor([0, 1, 1, 0, ...])  # 0=terminal, 1=decision
self.players = torch.tensor([-1, 0, 1, -1, ...])   # -1 für terminal
self.infosets = torch.tensor([-1, 5, 12, -1, ...])  # -1 für terminal
self.children = torch.tensor([
    [-1, -1, -1, -1],  # Node 0: keine Children (terminal)
    [2, 3, -1, -1],    # Node 1: child bei action 0,1
    ...
])  # Shape: [num_nodes, num_actions]
self.payoffs = torch.tensor([
    [1.0, -1.0],  # Node 0: Payoffs
    [0.0, 0.0],   # Node 1: Decision (keine Payoffs)
    ...
])  # Shape: [num_nodes, 2]

# Zugriff:
node_id = 1
child_id = self.children[node_id, action_id]  # action_id = 0 für 'check'
```

**Migration:**
```python
# Beim Tree-Bauen: Statt Node-Objekte → Arrays füllen
num_nodes = len(nodes)
node_types = np.zeros(num_nodes, dtype=np.int8)
children = np.full((num_nodes, num_actions), -1, dtype=np.int32)

for node_id, node in nodes.items():
    node_types[node_id] = 0 if node.type == 'terminal' else 1
    
    if node.type == 'decision':
        for action in node.legal_actions:
            action_idx = action_to_idx[action]
            children[node_id, action_idx] = node.children[action]
```

---

## Schritt 2: Tree nach Depth gruppieren (Layer-Indices)

**Das ist der Schlüssel für Layer-by-Layer Processing!**

### Vorher:
```python
# Keine Gruppierung - einfach rekursiv durchgehen
def traverse_tree(node_id, ...):
    node = self.nodes[node_id]
    # ... verarbeite diesen Node ...
    for action in node.legal_actions:
        child_id = node.children[action]
        traverse_tree(child_id, ...)  # Rekursiv
```

### Nachher:
```python
# tensor_cfr_solver.py, Zeile 68-75
max_depth = int(np.max(self.tree.depths))
self.layer_indices = []

for d in range(1, max_depth + 1):
    # Finde ALLE Nodes mit depth == d
    indices = np.where(self.tree.depths == d)[0]
    # z.B. Layer 1: [0, 5, 10] (3 Nodes)
    #     Layer 2: [1, 2, 6, 7, 11, 12] (6 Nodes)
    self.layer_indices.append(torch.tensor(indices, device=device))
```

**Migration:**
```python
# Beim Tree-Bauen: Depth speichern
for node_id, node in nodes.items():
    depths[node_id] = node.depth

# Nach dem Bauen: Gruppieren
max_depth = max(depths.values())
layer_indices = []
for d in range(1, max_depth + 1):
    layer_nodes = [nid for nid, node in nodes.items() if node.depth == d]
    layer_indices.append(layer_nodes)
```

---

## Schritt 3: Forward Pass - Reach Probabilities

### Vorher (Rekursiv):
```python
# cfr_solver_with_tree.py, traverse_tree()
def traverse_tree(self, node_id, player_id, reach_probabilities):
    node = self.nodes[node_id]
    
    if node.type == 'terminal':
        return node.payoffs[player_id]
    
    # Opponent's node
    if current_player != player_id:
        opponent_strategy = self.get_current_strategy(...)
        state_value = 0.0
        for action in node.legal_actions:
            action_prob = opponent_strategy[action]
            child_id = node.children[action]
            
            new_reach_probs = reach_probabilities.copy()
            new_reach_probs[opponent] *= action_prob
            
            state_value += action_prob * self.traverse_tree(child_id, ...)
        return state_value
    
    # Player's node
    # ... ähnlich ...
```

**Problem**: Sequenziell, Node für Node.

### Nachher (Layer-by-Layer):
```python
# tensor_cfr_solver.py, Zeile 159-214

# 1. Initialisiere Reach für Roots
nodes_reach = torch.zeros((num_nodes, 2), device=device)
nodes_reach[self.roots_tensor, 0] = self.root_prob  # P0
nodes_reach[self.roots_tensor, 1] = self.root_prob  # P1

# 2. Gehe Layer für Layer durch (Top-Down)
for layer_idx in self.layer_indices:  # Layer 1, dann 2, dann 3...
    if len(layer_idx) == 0: continue
    
    # Hole ALLE Decision Nodes dieser Schicht
    decision_mask = (self.node_types[layer_idx] == 1)
    decision_nodes = layer_idx[decision_mask]
    # z.B. [1, 2, 6, 7] - 4 Nodes gleichzeitig!
    
    if len(decision_nodes) == 0: continue
    
    # Hole ALLE Infosets auf einmal
    inf_ids = self.infosets[decision_nodes]  # [5, 5, 12, 12]
    
    # Hole ALLE Strategies auf einmal (Batch-Lookup!)
    node_strat = current_strategy[inf_ids]  # Shape: [4, 4]
    # ↑ Das ist der Trick: Alle 4 Nodes gleichzeitig!
    
    # Hole Player IDs
    p_ids = self.players[decision_nodes]  # [0, 0, 1, 1]
    
    # Hole Children
    child_ids = self.children[decision_nodes]  # [4 Nodes, 4 Actions]
    
    # Berechne Reach für ALLE Children auf einmal
    multipliers = torch.ones((len(decision_nodes), num_actions, 2))
    multipliers[mask0, :, 0] = node_strat[mask0]  # P0 Nodes
    multipliers[mask1, :, 1] = node_strat[mask1]  # P1 Nodes
    
    parent_reach = nodes_reach[decision_nodes].unsqueeze(1)  # [4, 1, 2]
    child_reach_vals = parent_reach * multipliers  # [4, 4, 2]
    
    # Scatter zu Children (alle auf einmal!)
    valid = (child_ids != -1)
    nodes_reach.index_add_(0, child_ids[valid], child_reach_vals[valid])
    # ↑ Aktualisiert HUNDERTE Children gleichzeitig!
```

**Migration Schritt-für-Schritt:**

```python
# Schritt 1: Statt rekursiv → Layer-Liste
# Statt:
def traverse_tree(node_id, reach_probs):
    # rekursiv...

# Mache:
reach_probs = {}  # {node_id: [reach_p0, reach_p1]}
reach_probs[root_id] = [1.0, 1.0]

# Schritt 2: Layer für Layer durchgehen
for layer in layer_indices:
    for node_id in layer:
        node = nodes[node_id]
        if node.type == 'decision':
            # Berechne Reach für Children
            # ...

# Schritt 3: Vektorisieren
# Statt for-Schleife → Tensor-Operationen
decision_nodes = torch.tensor([...])  # Alle Nodes dieser Schicht
# Alle gleichzeitig verarbeiten!
```

---

## Schritt 4: Backward Pass - Utilities

### Vorher (Rekursiv):
```python
# cfr_solver_with_tree.py
def traverse_tree(self, node_id, player_id, reach_probabilities):
    # ...
    # Player's node
    action_utilities = {}
    for action in node.legal_actions:
        child_id = node.children[action]
        action_utilities[action] = self.traverse_tree(child_id, ...)
    
    current_utility = sum(strategy[a] * action_utilities[a] for a in ...)
    return current_utility
```

### Nachher (Layer-by-Layer Bottom-Up):
```python
# tensor_cfr_solver.py, Zeile 217-254

nodes_values = torch.zeros((num_nodes, 2), device=device)

# Gehe rückwärts durch Layers (von tiefstem zu oberstem)
for layer_idx in reversed(self.layer_indices):  # Layer 5, dann 4, dann 3...
    
    # 1. Terminal Nodes: Setze Payoffs
    term_mask = (self.node_types[layer_idx] == 0)
    term_nodes = layer_idx[term_mask]
    nodes_values[term_nodes] = self.payoffs[term_nodes]
    # ↑ Alle Terminal Nodes dieser Schicht auf einmal!
    
    # 2. Decision Nodes: Berechne aus Children
    dec_mask = (self.node_types[layer_idx] == 1)
    dec_nodes = layer_idx[dec_mask]
    
    # Hole ALLE Children auf einmal
    child_ids = self.children[dec_nodes]  # [Batch, 4 Actions]
    
    # Hole ALLE Child Values auf einmal (Batch-Lookup!)
    safe_child_ids = child_ids.clone()
    safe_child_ids[~valid_mask] = 0
    c_vals = nodes_values[safe_child_ids]  # [Batch, 4 Actions, 2 Players]
    c_vals[~valid_mask] = 0.0
    
    # Berechne Expected Value für ALLE Nodes auf einmal
    strat = current_strategy[self.infosets[dec_nodes]]  # [Batch, 4 Actions]
    ev = (strat.unsqueeze(2) * c_vals).sum(dim=1)  # [Batch, 2 Players]
    # ↑ Matrix-Multiplikation über alle Nodes gleichzeitig!
    
    nodes_values[dec_nodes] = ev
```

**Migration:**
```python
# Schritt 1: Statt rekursiv → Bottom-Up durch Layers
# Statt:
utility = traverse_tree(node_id, ...)

# Mache:
values = {}  # {node_id: [value_p0, value_p1]}

# Schritt 2: Von tiefstem Layer starten
for layer in reversed(layer_indices):
    for node_id in layer:
        if nodes[node_id].type == 'terminal':
            values[node_id] = nodes[node_id].payoffs
        else:
            # Berechne aus Children
            # ...

# Schritt 3: Vektorisieren
# Statt for-Schleife → Tensor-Operationen
dec_nodes = torch.tensor([...])
child_ids = self.children[dec_nodes]
c_vals = nodes_values[child_ids]  # Batch-Lookup!
ev = (strat * c_vals).sum(dim=1)  # Batch-Berechnung!
```

---

## Schritt 5: Regret Updates

### Vorher:
```python
# cfr_solver_with_tree.py
def update_regrets(self, info_set_key, legal_actions, action_utilities, 
                   current_utility, counterfactual_weight):
    for action in legal_actions:
        instantaneous_regret = counterfactual_weight * (
            action_utilities[action] - current_utility
        )
        self.regret_sum[info_set_key][action] += instantaneous_regret
```

### Nachher:
```python
# tensor_cfr_solver.py, Zeile 256-269

# Berechne Instantaneous Regret für ALLE Nodes auf einmal
opp_ids = 1 - p_ids  # Opponent IDs
opp_reach = nodes_reach[dec_nodes].gather(1, opp_ids.unsqueeze(1)).squeeze(1)

# Q-Values (Action Utilities) für den aktuellen Spieler
q_vals = c_vals.gather(2, p_idx_expanded).squeeze(2)  # [Batch, Actions]

# V-Values (Node Utility)
v_vals = ev.gather(1, p_ids.unsqueeze(1)).squeeze(1)  # [Batch]

# Instantaneous Regret für ALLE Nodes und ALLE Actions
inst_regret = opp_reach.unsqueeze(1) * (q_vals - v_vals.unsqueeze(1))
# Shape: [Batch, Actions]

# Update Regret Sum (Batch-Update!)
self.delta_regret.index_add_(0, inf_ids, inst_regret)
# ↑ Aktualisiert alle Infosets gleichzeitig!

# Am Ende der Iteration:
self.regret_sum += self.delta_regret
if self.algorithm == 'cfr_plus':
    self.regret_sum = torch.clamp(self.regret_sum, min=0)
```

**Migration:**
```python
# Schritt 1: Statt einzelne Updates → Batch sammeln
delta_regret = {}  # {infoset_id: {action: delta}}

# Schritt 2: Während Backward Pass sammeln
for node_id in dec_nodes:
    infoset_id = infoset_map[nodes[node_id].infoset_key]
    for action in legal_actions:
        inst_regret = ...
        if infoset_id not in delta_regret:
            delta_regret[infoset_id] = {}
        delta_regret[infoset_id][action] = inst_regret

# Schritt 3: Am Ende: Batch-Update
for infoset_id, action_deltas in delta_regret.items():
    for action, delta in action_deltas.items():
        self.regret_sum[infoset_id][action] += delta

# Schritt 4: Vektorisieren
# Statt Dict → Tensor
delta_regret_tensor = torch.zeros((num_infosets, num_actions))
# ... sammle Updates ...
self.regret_sum += delta_regret_tensor  # Batch-Update!
```

---

## Schritt 6: Strategy Updates

### Vorher:
```python
# cfr_solver_with_tree.py
def update_strategy_sum(self, info_set_key, legal_actions, 
                        current_strategy, player_reach):
    for action in legal_actions:
        self.strategy_sum[info_set_key][action] += (
            player_reach * current_strategy[action]
        )
```

### Nachher:
```python
# tensor_cfr_solver.py, Zeile 183-191

# Während Forward Pass:
reach_p = nodes_reach[decision_nodes].gather(1, p_ids.unsqueeze(1)).squeeze(1)
# ↑ Reach des spielenden Spielers für alle Nodes

contrib = node_strat * reach_p.unsqueeze(1)  # [Batch, Actions]
# ↑ Beitrag für Strategy Sum

if self.algorithm == 'cfr_plus':
    contrib *= self.t  # CFR+ Weighting

self.strategy_sum.index_add_(0, inf_ids, contrib)
# ↑ Batch-Update für alle Infosets gleichzeitig!
```

---

## Schritt 7: Regret Matching (Strategy Berechnung)

### Vorher:
```python
# cfr_solver_with_tree.py
def get_current_strategy(self, info_set_key, legal_actions):
    regrets = {a: self.regret_sum[info_set_key][a] for a in legal_actions}
    positive_regrets = {a: max(regrets[a], 0) for a in legal_actions}
    sum_pos = sum(positive_regrets.values())
    
    if sum_pos > 0:
        return {a: positive_regrets[a] / sum_pos for a in legal_actions}
    else:
        return {a: 1.0 / len(legal_actions) for a in legal_actions}
```

### Nachher:
```python
# tensor_cfr_solver.py, Zeile 134-156

# Mask invalid regrets
self.regret_sum.masked_fill_(~self.infoset_valid_actions, -1e9)

# Clamp to positive
positive_regrets = torch.clamp(self.regret_sum, min=0)
sum_pos = torch.sum(positive_regrets, dim=1, keepdim=True)  # [Infosets, 1]

# Initialize strategy
current_strategy = torch.zeros_like(self.regret_sum)
has_pos = (sum_pos > 1e-12).squeeze()

# Positive regrets case (für ALLE Infosets gleichzeitig!)
current_strategy[has_pos] = positive_regrets[has_pos] / sum_pos[has_pos]

# Uniform case (für ALLE Infosets gleichzeitig!)
uniform_probs = 1.0 / torch.clamp(self.infoset_valid_counts, min=1)
no_pos_mask = ~has_pos
uniform_contrib = uniform_probs * self.infoset_valid_actions.float()
current_strategy[no_pos_mask] = uniform_contrib[no_pos_mask]

# Ensure invalid actions are 0
current_strategy.masked_fill_(~self.infoset_valid_actions, 0.0)
```

**Der Trick**: Statt für jedes Infoset einzeln → **alle Infosets gleichzeitig**!

---

## Praktischer Migrationsplan

### Phase 1: Datenstrukturen vorbereiten
1. ✅ Infoset-Mapping erstellen (Key → ID)
2. ✅ Action-Mapping erstellen (Action → ID)
3. ✅ Node-Daten in Arrays konvertieren
4. ✅ Depth-Information speichern

### Phase 2: Layer-Indices erstellen
1. ✅ Nodes nach Depth gruppieren
2. ✅ `layer_indices` Liste erstellen

### Phase 3: Forward Pass migrieren
1. ✅ Statt rekursiv → Layer für Layer (Top-Down)
2. ✅ Reach Probabilities in Tensor speichern
3. ✅ Batch-Operationen für Strategy Updates

### Phase 4: Backward Pass migrieren
1. ✅ Statt rekursiv → Layer für Layer (Bottom-Up)
2. ✅ Utilities in Tensor speichern
3. ✅ Batch-Operationen für Regret Updates

### Phase 5: Optimieren
1. ✅ Auf GPU verschieben
2. ✅ `torch.no_grad()` für Training
3. ✅ Memory-Effizienz optimieren

---

## Code-Vergleich: Kompletter CFR Iteration

### Vorher (Rekursiv):
```python
def cfr_iteration(self):
    for root_id in self.root_nodes:
        reach_probs = [1.0, 1.0]
        self.traverse_tree(root_id, 0, reach_probs)  # Rekursiv
        self.traverse_tree(root_id, 1, reach_probs)  # Rekursiv

def traverse_tree(self, node_id, player_id, reach_probs):
    node = self.nodes[node_id]
    if node.type == 'terminal':
        return node.payoffs[player_id]
    
    # ... rekursiv weiter ...
    for action in node.legal_actions:
        child_id = node.children[action]
        utility = self.traverse_tree(child_id, ...)  # Rekursiv
    # ...
```

### Nachher (Layer-by-Layer):
```python
def _cfr_iteration(self, num_nodes):
    # 1. Regret Matching (alle Infosets parallel)
    current_strategy = self._compute_strategy()
    
    # 2. Forward Pass (Layer-by-Layer Top-Down)
    nodes_reach = torch.zeros((num_nodes, 2), device=device)
    nodes_reach[self.roots_tensor] = self.root_prob
    
    for layer_idx in self.layer_indices:  # Layer 1, 2, 3...
        decision_nodes = layer_idx[decision_mask]
        # ... Batch-Operationen für alle Nodes dieser Schicht ...
        nodes_reach.index_add_(0, target_indices, source_vals)
    
    # 3. Backward Pass (Layer-by-Layer Bottom-Up)
    nodes_values = torch.zeros((num_nodes, 2), device=device)
    
    for layer_idx in reversed(self.layer_indices):  # Layer 5, 4, 3...
        # Terminal Nodes
        nodes_values[term_nodes] = self.payoffs[term_nodes]
        
        # Decision Nodes
        c_vals = nodes_values[child_ids]  # Batch-Lookup!
        ev = (strat * c_vals).sum(dim=1)  # Batch-Berechnung!
        nodes_values[dec_nodes] = ev
        
        # Regret Updates
        inst_regret = ...  # Batch-Berechnung!
        self.delta_regret.index_add_(0, inf_ids, inst_regret)
    
    # 4. Apply Regrets
    self.regret_sum += self.delta_regret
```

---

## Wichtige Tensor-Operationen zum Verstehen

### `index_add_`: Batch-Update
```python
# Statt:
for i in [1, 2, 3]:
    array[i] += values[i]

# Tensor:
array.index_add_(0, [1, 2, 3], values)
```

### `gather`: Batch-Lookup
```python
# Statt:
for node_id in [1, 2, 3]:
    value = array[infosets[node_id]]

# Tensor:
inf_ids = infosets[[1, 2, 3]]
values = array[inf_ids]  # Batch-Lookup!
```

### Masking: Batch-Filtering
```python
# Statt:
result = []
for x in array:
    if x > 0:
        result.append(x)

# Tensor:
mask = (array > 0)
result = array[mask]  # Batch-Filter!
```

---

## Zusammenfassung

**Der Hauptunterschied:**
- **Vorher**: Rekursiv, Node für Node, sequenziell
- **Nachher**: Layer-by-Layer, alle Nodes einer Schicht parallel, vektorisiert

**Die Migration:**
1. Datenstrukturen: Dicts → Tensors
2. Traversierung: Rekursiv → Layer-by-Layer
3. Operationen: Einzeln → Batch
4. Hardware: CPU → GPU (optional, aber empfohlen)

**Der Gewinn:**
- 10-100× schneller (je nach Tree-Größe)
- Skaliert besser für große Bäume
- Nutzt GPU-Parallelisierung
