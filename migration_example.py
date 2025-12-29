"""
Praktisches Beispiel: Migration von CFRSolverWithTree zu TensorCFRSolver

Dieses Script zeigt Schritt für Schritt, wie man die wichtigsten
Operationen von der rekursiven zur Tensor-Version migriert.
"""

import torch
import numpy as np
from collections import defaultdict

# ============================================================================
# SCHRITT 1: Datenstrukturen konvertieren
# ============================================================================

def step1_convert_data_structures():
    """
    Konvertiert Dict-basierte Datenstrukturen zu Tensors.
    """
    print("=" * 60)
    print("SCHRITT 1: Datenstrukturen konvertieren")
    print("=" * 60)
    
    # VORHER: Dict-basiert
    print("\n--- VORHER (Dict-basiert) ---")
    regret_sum_dict = {
        'K|check': {'bet': 2.5, 'check': 1.0},
        'Q|bet': {'call': 3.0, 'fold': 0.0}
    }
    print(f"regret_sum_dict = {regret_sum_dict}")
    print(f"Zugriff: regret_sum_dict['K|check']['bet'] = {regret_sum_dict['K|check']['bet']}")
    
    # NACHHER: Tensor-basiert
    print("\n--- NACHHER (Tensor-basiert) ---")
    
    # 1. Infoset-Mapping erstellen
    infoset_map = {
        'K|check': 0,
        'Q|bet': 1
    }
    action_to_idx = {'check': 0, 'bet': 1, 'call': 2, 'fold': 3}
    
    # 2. Tensor initialisieren
    num_infosets = len(infoset_map)
    num_actions = len(action_to_idx)
    regret_sum_tensor = torch.zeros((num_infosets, num_actions))
    
    # 3. Werte setzen
    regret_sum_tensor[0, 1] = 2.5  # 'K|check', 'bet'
    regret_sum_tensor[0, 0] = 1.0  # 'K|check', 'check'
    regret_sum_tensor[1, 2] = 3.0  # 'Q|bet', 'call'
    regret_sum_tensor[1, 3] = 0.0  # 'Q|bet', 'fold'
    
    print(f"regret_sum_tensor.shape = {regret_sum_tensor.shape}")
    print(f"regret_sum_tensor = \n{regret_sum_tensor}")
    print(f"Zugriff: regret_sum_tensor[0, 1] = {regret_sum_tensor[0, 1]}")
    
    return regret_sum_tensor, infoset_map, action_to_idx


# ============================================================================
# SCHRITT 2: Layer-Indices erstellen
# ============================================================================

def step2_create_layer_indices():
    """
    Zeigt, wie man Nodes nach Depth gruppiert.
    """
    print("\n" + "=" * 60)
    print("SCHRITT 2: Layer-Indices erstellen")
    print("=" * 60)
    
    # Beispiel: Nodes mit verschiedenen Depths
    nodes = {
        0: {'depth': 1, 'type': 'decision'},
        1: {'depth': 2, 'type': 'decision'},
        2: {'depth': 2, 'type': 'decision'},
        3: {'depth': 3, 'type': 'terminal'},
        4: {'depth': 3, 'type': 'terminal'},
        5: {'depth': 1, 'type': 'decision'},
    }
    
    print("\n--- Nodes mit Depths ---")
    for node_id, node in nodes.items():
        print(f"Node {node_id}: depth={node['depth']}, type={node['type']}")
    
    # Gruppiere nach Depth
    print("\n--- Nach Depth gruppiert ---")
    max_depth = max(n['depth'] for n in nodes.values())
    layer_indices = []
    
    for d in range(1, max_depth + 1):
        layer_nodes = [nid for nid, node in nodes.items() if node['depth'] == d]
        layer_indices.append(layer_nodes)
        print(f"Layer {d}: {layer_nodes}")
    
    # Konvertiere zu Tensor
    print("\n--- Als Tensor ---")
    layer_indices_tensor = []
    for layer in layer_indices:
        if len(layer) > 0:
            layer_indices_tensor.append(torch.tensor(layer))
            print(f"Layer {len(layer_indices_tensor)}: {layer_indices_tensor[-1]}")
    
    return layer_indices_tensor


# ============================================================================
# SCHRITT 3: Forward Pass - Reach Probabilities
# ============================================================================

def step3_forward_pass():
    """
    Zeigt den Unterschied zwischen rekursivem und Layer-by-Layer Forward Pass.
    """
    print("\n" + "=" * 60)
    print("SCHRITT 3: Forward Pass - Reach Probabilities")
    print("=" * 60)
    
    # Beispiel: 4 Nodes in 2 Layers
    # Layer 1: Nodes 0, 1
    # Layer 2: Nodes 2, 3 (Children von 0, 1)
    
    print("\n--- VORHER (Rekursiv) ---")
    print("""
def traverse_tree(node_id, reach_probs):
    node = nodes[node_id]
    strategy = get_strategy(node.infoset_key)
    
    for action in node.legal_actions:
        child_id = node.children[action]
        new_reach = reach_probs.copy()
        new_reach[node.player] *= strategy[action]
        traverse_tree(child_id, new_reach)  # ← Rekursiv
    """)
    
    print("\n--- NACHHER (Layer-by-Layer) ---")
    
    # Simuliere Layer-by-Layer
    num_nodes = 4
    nodes_reach = torch.zeros((num_nodes, 2))
    
    # Layer 1: Roots initialisieren
    roots = torch.tensor([0, 1])
    nodes_reach[roots, 0] = 1.0  # P0 reach
    nodes_reach[roots, 1] = 1.0  # P1 reach
    print(f"\nNach Root-Initialisierung:")
    print(f"nodes_reach = \n{nodes_reach}")
    
    # Layer 1: Decision Nodes verarbeiten
    layer_1 = torch.tensor([0, 1])
    decision_nodes = layer_1
    inf_ids = torch.tensor([5, 12])  # Beispiel-Infoset-IDs
    
    # Simuliere Strategy (normalerweise aus regret_sum berechnet)
    current_strategy = torch.tensor([
        [0.5, 0.5, 0.0, 0.0],  # Infoset 5: [check, bet, call, fold]
        [0.0, 0.0, 0.7, 0.3],  # Infoset 12: [check, bet, call, fold]
    ])
    
    node_strat = current_strategy  # [2 Nodes, 4 Actions]
    print(f"\nStrategies für Layer 1 Nodes:")
    print(f"node_strat = \n{node_strat}")
    
    # Berechne Child Reach
    p_ids = torch.tensor([0, 1])  # Player IDs
    multipliers = torch.ones((2, 4, 2))
    multipliers[0, :, 0] = node_strat[0]  # P0
    multipliers[1, :, 1] = node_strat[1]  # P1
    
    parent_reach = nodes_reach[decision_nodes].unsqueeze(1)  # [2, 1, 2]
    child_reach_vals = parent_reach * multipliers  # [2, 4, 2]
    
    print(f"\nChild Reach Values:")
    print(f"child_reach_vals.shape = {child_reach_vals.shape}")
    print(f"child_reach_vals = \n{child_reach_vals}")
    
    # Scatter zu Children (vereinfacht)
    child_ids = torch.tensor([
        [2, 3, -1, -1],  # Node 0: Children bei Action 0,1
        [2, 3, -1, -1],  # Node 1: Children bei Action 0,1
    ])
    
    valid = (child_ids != -1)
    target_indices = child_ids[valid]  # [2, 3, 2, 3]
    source_vals = child_reach_vals[valid]  # [4, 2]
    
    nodes_reach.index_add_(0, target_indices, source_vals)
    print(f"\nNach Propagation zu Layer 2:")
    print(f"nodes_reach = \n{nodes_reach}")
    
    print("\n✓ Alle Nodes einer Schicht werden parallel verarbeitet!")


# ============================================================================
# SCHRITT 4: Backward Pass - Utilities
# ============================================================================

def step4_backward_pass():
    """
    Zeigt den Unterschied zwischen rekursivem und Layer-by-Layer Backward Pass.
    """
    print("\n" + "=" * 60)
    print("SCHRITT 4: Backward Pass - Utilities")
    print("=" * 60)
    
    print("\n--- VORHER (Rekursiv) ---")
    print("""
def traverse_tree(node_id, ...):
    # Player's node
    action_utilities = {}
    for action in node.legal_actions:
        child_id = node.children[action]
        action_utilities[action] = traverse_tree(child_id, ...)  # ← Rekursiv
    
    current_utility = sum(strategy[a] * action_utilities[a] for a in ...)
    return current_utility
    """)
    
    print("\n--- NACHHER (Layer-by-Layer Bottom-Up) ---")
    
    # Beispiel: 4 Nodes, Layer 2 (Terminal) → Layer 1 (Decision)
    num_nodes = 4
    nodes_values = torch.zeros((num_nodes, 2))
    
    # Layer 2: Terminal Nodes
    layer_2 = torch.tensor([2, 3])
    payoffs = torch.tensor([
        [1.0, -1.0],  # Node 2: P0 gewinnt 1, P1 verliert 1
        [-1.0, 1.0],  # Node 3: P0 verliert 1, P1 gewinnt 1
    ])
    
    nodes_values[layer_2] = payoffs
    print(f"\nLayer 2 (Terminal):")
    print(f"nodes_values = \n{nodes_values}")
    
    # Layer 1: Decision Nodes
    layer_1 = torch.tensor([0, 1])
    dec_nodes = layer_1
    
    # Children
    child_ids = torch.tensor([
        [2, 3, -1, -1],  # Node 0: Children bei Action 0,1
        [2, 3, -1, -1],  # Node 1: Children bei Action 0,1
    ])
    
    # Hole Child Values (Batch-Lookup!)
    valid_mask = (child_ids != -1)
    safe_child_ids = child_ids.clone()
    safe_child_ids[~valid_mask] = 0
    
    c_vals = nodes_values[safe_child_ids]  # [2 Nodes, 4 Actions, 2 Players]
    c_vals[~valid_mask] = 0.0
    print(f"\nChild Values (Batch-Lookup):")
    print(f"c_vals.shape = {c_vals.shape}")
    print(f"c_vals = \n{c_vals}")
    
    # Strategy
    current_strategy = torch.tensor([
        [0.5, 0.5, 0.0, 0.0],  # Node 0
        [0.0, 0.0, 0.7, 0.3],  # Node 1
    ])
    
    # Expected Value (Batch-Berechnung!)
    ev = (current_strategy.unsqueeze(2) * c_vals).sum(dim=1)  # [2, 2]
    print(f"\nExpected Values (Batch-Berechnung):")
    print(f"ev = \n{ev}")
    
    nodes_values[dec_nodes] = ev
    print(f"\nNach Layer 1 Update:")
    print(f"nodes_values = \n{nodes_values}")
    
    print("\n✓ Alle Nodes einer Schicht werden parallel verarbeitet!")


# ============================================================================
# SCHRITT 5: Regret Update
# ============================================================================

def step5_regret_update():
    """
    Zeigt den Unterschied zwischen einzelnen und Batch-Regret-Updates.
    """
    print("\n" + "=" * 60)
    print("SCHRITT 5: Regret Update")
    print("=" * 60)
    
    print("\n--- VORHER (Einzeln) ---")
    print("""
def update_regrets(info_set_key, legal_actions, action_utilities, 
                   current_utility, counterfactual_weight):
    for action in legal_actions:  # ← Einzelne Aktion
        instantaneous_regret = counterfactual_weight * (
            action_utilities[action] - current_utility
        )
        self.regret_sum[info_set_key][action] += instantaneous_regret
    """)
    
    print("\n--- NACHHER (Batch) ---")
    
    # Beispiel: 2 Decision Nodes
    dec_nodes = torch.tensor([0, 1])
    inf_ids = torch.tensor([5, 12])  # Infoset IDs
    p_ids = torch.tensor([0, 1])  # Player IDs
    
    # Simuliere Values
    nodes_values = torch.tensor([
        [0.5, -0.5],  # Node 0: EV für P0, P1
        [-0.3, 0.3],  # Node 1: EV für P0, P1
    ])
    
    # Child Values (vereinfacht)
    c_vals = torch.tensor([
        [[1.0, -1.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],  # Node 0
        [[0.0, 0.0], [0.0, 0.0], [-1.0, 1.0], [0.0, 0.0]],  # Node 1
    ])
    
    # Opponent Reach (vereinfacht)
    opp_reach = torch.tensor([1.0, 1.0])
    
    # Q-Values für den aktuellen Spieler
    p_idx_expanded = p_ids.view(-1, 1, 1).expand(-1, 4, 1)
    q_vals = c_vals.gather(2, p_idx_expanded).squeeze(2)  # [2, 4]
    print(f"Q-Values (Action Utilities):")
    print(f"q_vals = \n{q_vals}")
    
    # V-Values (Node Utility)
    v_vals = nodes_values.gather(1, p_ids.unsqueeze(1)).squeeze(1)  # [2]
    print(f"V-Values (Node Utility):")
    print(f"v_vals = {v_vals}")
    
    # Instantaneous Regret (Batch-Berechnung!)
    inst_regret = opp_reach.unsqueeze(1) * (q_vals - v_vals.unsqueeze(1))
    print(f"\nInstantaneous Regret (Batch):")
    print(f"inst_regret.shape = {inst_regret.shape}")
    print(f"inst_regret = \n{inst_regret}")
    
    # Batch-Update
    regret_sum = torch.zeros((100, 4))  # Beispiel: 100 Infosets
    regret_sum.index_add_(0, inf_ids, inst_regret)
    print(f"\nRegret Sum nach Update (für Infosets 5 und 12):")
    print(f"regret_sum[5] = {regret_sum[5]}")
    print(f"regret_sum[12] = {regret_sum[12]}")
    
    print("\n✓ Alle Regret-Updates werden parallel verarbeitet!")


# ============================================================================
# SCHRITT 6: Regret Matching (Strategy Berechnung)
# ============================================================================

def step6_regret_matching():
    """
    Zeigt den Unterschied zwischen einzelnen und Batch-Strategy-Berechnungen.
    """
    print("\n" + "=" * 60)
    print("SCHRITT 6: Regret Matching (Strategy Berechnung)")
    print("=" * 60)
    
    print("\n--- VORHER (Einzeln pro Infoset) ---")
    print("""
def get_current_strategy(info_set_key, legal_actions):
    regrets = {a: self.regret_sum[info_set_key][a] for a in legal_actions}
    positive_regrets = {a: max(regrets[a], 0) for a in legal_actions}
    sum_pos = sum(positive_regrets.values())
    
    if sum_pos > 0:
        return {a: positive_regrets[a] / sum_pos for a in legal_actions}
    else:
        return {a: 1.0 / len(legal_actions) for a in legal_actions}
    """)
    
    print("\n--- NACHHER (Batch für alle Infosets) ---")
    
    # Beispiel: 3 Infosets
    num_infosets = 3
    num_actions = 4
    
    regret_sum = torch.tensor([
        [2.0, 1.0, -0.5, 0.0],  # Infoset 0: positive regrets
        [-1.0, -0.5, -0.2, 0.0],  # Infoset 1: keine positiven regrets
        [0.0, 0.0, 0.0, 0.0],  # Infoset 2: keine regrets
    ])
    
    infoset_valid_actions = torch.tensor([
        [True, True, False, False],  # Infoset 0: check, bet
        [False, False, True, True],  # Infoset 1: call, fold
        [True, True, True, True],  # Infoset 2: alle
    ])
    
    print(f"Regret Sum:")
    print(f"regret_sum = \n{regret_sum}")
    
    # Mask invalid regrets
    regret_sum.masked_fill_(~infoset_valid_actions, -1e9)
    
    # Clamp to positive
    positive_regrets = torch.clamp(regret_sum, min=0)
    sum_pos = torch.sum(positive_regrets, dim=1, keepdim=True)  # [3, 1]
    
    print(f"\nPositive Regrets:")
    print(f"positive_regrets = \n{positive_regrets}")
    print(f"sum_pos = \n{sum_pos}")
    
    # Initialize strategy
    current_strategy = torch.zeros_like(regret_sum)
    has_pos = (sum_pos > 1e-12).squeeze()
    
    # Positive regrets case (Batch!)
    current_strategy[has_pos] = positive_regrets[has_pos] / sum_pos[has_pos]
    
    # Uniform case (Batch!)
    infoset_valid_counts = infoset_valid_actions.sum(dim=1, keepdim=True)
    uniform_probs = 1.0 / torch.clamp(infoset_valid_counts, min=1)
    no_pos_mask = ~has_pos
    uniform_contrib = uniform_probs * infoset_valid_actions.float()
    current_strategy[no_pos_mask] = uniform_contrib[no_pos_mask]
    
    # Ensure invalid actions are 0
    current_strategy.masked_fill_(~infoset_valid_actions, 0.0)
    
    print(f"\nCurrent Strategy (Batch-Berechnung für alle 3 Infosets):")
    print(f"current_strategy = \n{current_strategy}")
    
    print("\n✓ Alle Infosets werden parallel verarbeitet!")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("MIGRATION BEISPIEL: CFRSolverWithTree → TensorCFRSolver")
    print("=" * 60)
    
    # Schritt 1: Datenstrukturen
    regret_sum_tensor, infoset_map, action_to_idx = step1_convert_data_structures()
    
    # Schritt 2: Layer-Indices
    layer_indices = step2_create_layer_indices()
    
    # Schritt 3: Forward Pass
    step3_forward_pass()
    
    # Schritt 4: Backward Pass
    step4_backward_pass()
    
    # Schritt 5: Regret Update
    step5_regret_update()
    
    # Schritt 6: Regret Matching
    step6_regret_matching()
    
    print("\n" + "=" * 60)
    print("MIGRATION BEISPIEL ABGESCHLOSSEN")
    print("=" * 60)
    print("\nZusammenfassung:")
    print("- Dicts → Tensors: Batch-Operationen möglich")
    print("- Rekursiv → Layer-by-Layer: Parallel-Verarbeitung")
    print("- Einzeln → Batch: 10-100× schneller")
    print("\nSiehe MIGRATION_GUIDE.md für detaillierte Anleitung!")


