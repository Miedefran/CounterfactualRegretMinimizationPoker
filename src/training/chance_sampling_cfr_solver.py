"""
Chance Sampling CFR Solver mit vorher gebautem Game Tree.

Chance Sampling CFR sammelt nur Zufallsaktionen (chance actions).
Statt alle Kombinationen zu durchlaufen, wird pro Iteration nur eine
zufällige Kombination gesampelt.

Basierend auf: Waugh et al. (2009) - Monte Carlo Sampling for Regret 
Minimization in Extensive Games
"""

import random

from training.cfr_solver_with_tree import CFRSolverWithTree


class ChanceSamplingCFRSolver(CFRSolverWithTree):
    """
    Chance Sampling CFR Solver der den Game Tree einmal vorher baut.
    
    Unterschied zu CFRSolverWithTree:
    - Statt alle Kombinationen zu durchlaufen, wird pro Iteration nur
      eine zufällige Kombination gesampelt
    - Die Updates werden mit der Anzahl der Kombinationen gewichtet,
      um den Erwartungswert beizubehalten
    """
    
    def __init__(self, game, combination_generator, game_name=None, load_tree=True):
        # Rufe Basisklassen-__init__ auf
        super().__init__(game, combination_generator, game_name=game_name, load_tree=load_tree)
        
        # Zusätzliche Attribute für Chance Sampling
        self.num_combinations = len(self.combinations)
        self.combination_to_root = {}  # Mapping von Kombination zu Root-Node-ID
        
        # Initialisiere combination_to_root Mapping
        self._initialize_combination_mapping()
    
    def _initialize_combination_mapping(self):
        """Initialisiert das Mapping von Kombinationen zu Root-Node-IDs"""
        self.combination_to_root = {}
        for i, root_id in enumerate(self.root_nodes):
            if i < len(self.combinations):
                # Kombinationen sind normalerweise Tuples, die hashable sind
                combo = self.combinations[i]
                self.combination_to_root[combo] = root_id
    
    def _convert_game_tree_to_internal(self, game_tree):
        """Konvertiert ein GameTree Objekt zu interner Struktur"""
        # Rufe Basisklassen-Methode auf
        super()._convert_game_tree_to_internal(game_tree)
        
        # Initialisiere alle InfoSets, die im Tree existieren
        # (wichtig für Sampling-Strategien, damit alle InfoSets definiert sind)
        for infoset_key, node_ids in self.infoset_to_nodes.items():
            if node_ids:
                # Verwende die legal_actions vom ersten Node dieses InfoSets
                first_node = self.nodes[node_ids[0]]
                self.ensure_init(infoset_key, first_node.legal_actions)
        
        # Initialisiere combination_to_root Mapping
        self._initialize_combination_mapping()
    
    def cfr_iteration(self):
        """
        Eine Chance Sampling CFR Iteration.
        
        Statt alle Kombinationen zu durchlaufen, wird nur eine zufällige
        Kombination gesampelt. Die Updates werden mit der Anzahl der Kombinationen
        gewichtet, um den Erwartungswert beizubehalten.
        
        Basierend auf: Waugh et al. (2009) - Chance Sampling MCCFR
        """
        # Sample eine zufällige Kombination (Chance-Outcome)
        # Die Wahrscheinlichkeit jeder Kombination ist 1/num_combinations
        sampled_combination = random.choice(self.combinations)
        root_id = self.combination_to_root.get(sampled_combination)
        
        if root_id is None:
            # Fallback: Verwende Index-basiertes Mapping
            combo_index = self.combinations.index(sampled_combination)
            if combo_index < len(self.root_nodes):
                root_id = self.root_nodes[combo_index]
            else:
                # Letzter Fallback: Verwende ersten Root-Node
                root_id = self.root_nodes[0]
        
        reach_probs = [1.0, 1.0]
        
        # Traverse für beide Spieler
        # Die Gewichtung erfolgt in update_regrets und update_strategy_sum
        self.traverse_tree(root_id, 0, reach_probs)
        self.traverse_tree(root_id, 1, reach_probs)
    
    def update_regrets(self, info_set_key, legal_actions, action_utilities, current_utility, counterfactual_weight):
        """
        Aktualisiert Regret Sum mit Gewichtung für Chance Sampling.
        
        Da wir nur eine Kombination pro Iteration samplen, müssen wir die Updates
        mit der Anzahl der Kombinationen gewichten, um den Erwartungswert beizubehalten.
        """
        # Gewichtung: Anzahl der Kombinationen (entspricht 1 / Wahrscheinlichkeit einer Kombination)
        sampling_weight = self.num_combinations
        
        for action in legal_actions:
            instantaneous_regret = counterfactual_weight * (action_utilities[action] - current_utility)
            # Gewichte den Regret mit der Sampling-Wahrscheinlichkeit
            self.regret_sum[info_set_key][action] += sampling_weight * instantaneous_regret
    
    def update_strategy_sum(self, info_set_key, legal_actions, current_strategy, player_reach):
        """
        Aktualisiert Strategy Sum mit Gewichtung für Chance Sampling.
        """
        # Gewichtung: Anzahl der Kombinationen
        sampling_weight = self.num_combinations
        
        for action in legal_actions:
            self.strategy_sum[info_set_key][action] += sampling_weight * player_reach * current_strategy[action]
