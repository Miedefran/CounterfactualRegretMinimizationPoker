"""
Chance Sampling CFR Solver mit vorher gebautem Game Tree.

Chance Sampling CFR sammelt nur Zufallsaktionen (chance actions).
Statt alle Kombinationen zu durchlaufen, wird pro Iteration nur eine
zufällige Kombination gesampelt.

Basierend auf: Waugh et al. (2009) - Monte Carlo Sampling for Regret 
Minimization in Extensive Games
"""

import random
import numpy as np

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
        
        reach_probs = np.array([1.0, 1.0], dtype=np.float64)
        
        # Traverse für Spieler 0
        self.traverse_tree(root_id, 0, reach_probs)
        
        # Policy Update nach Spieler 0 (wichtig: damit Spieler 1 die aktualisierte Policy verwendet)
        self._update_all_policies()
        
        # Traverse für Spieler 1 (mit aktualisierter Policy von Spieler 0)
        self.traverse_tree(root_id, 1, reach_probs)
        
        # Policy Update nach Spieler 1
        self._update_all_policies()
    
    def traverse_tree(self, node_id, player, reach_probabilities):
        """
        Traversiert den Tree und berechnet Counterfactual Regret für einen Spieler.
        Verwendet Chance Sampling Gewichtung in den Updates.
        
        Args:
            node_id: ID des aktuellen Nodes
            reach_probabilities: np.array([reach_p0, reach_p1])
            player: Spieler für den wir CFR durchführen (0 oder 1)
        
        Returns:
            Utility für player
        """
        node = self.nodes[node_id]
        
        # Terminal Node: Payoff zurückgeben
        if node.type == 'terminal':
            return node.payoffs[player]
        
        # Decision Node
        current_player = node.player
        info_state = node.infoset_key
        
        # Early exit wenn Reach Probabilities 0 sind
        if np.all(reach_probabilities[:2] == 0):
            return 0.0
        
        self.ensure_init(info_state, node.legal_actions)
        
        # Hole aktuelle Policy für dieses InfoSet
        policy = self._get_policy(info_state)
        
        # Berechne Utilities für alle Aktionen
        action_utilities = {}
        state_value = 0.0
        
        for action in node.legal_actions:
            action_prob = policy.get(action, 0.0)
            child_id = node.children[action]
            
            # Neue Reach Probabilities für diesen Pfad
            new_reach_probs = reach_probabilities.copy()
            new_reach_probs[current_player] *= action_prob
            
            # Rekursiv traversieren
            child_utility = self.traverse_tree(child_id, player, new_reach_probs)
            
            action_utilities[action] = child_utility
            state_value += action_prob * child_utility
        
        # Wenn wir nicht für den aktuellen Spieler updaten, nur Wert zurückgeben
        if current_player != player:
            return state_value
        
        # Regret Updates für den aktuellen Spieler mit Chance Sampling Gewichtung
        reach_prob = reach_probabilities[current_player]
        
        # Counterfactual Reach Probability = Produkt aller anderen Spieler
        counterfactual_reach = 1.0
        for p in range(len(reach_probabilities)):
            if p != current_player:
                counterfactual_reach *= reach_probabilities[p]
        
        # Gewichtung: Anzahl der Kombinationen (entspricht 1 / Wahrscheinlichkeit einer Kombination)
        sampling_weight = self.num_combinations
        
        # Akkumuliere Regrets mit Sampling-Gewichtung
        for action in node.legal_actions:
            # Instantaneous Regret
            instantaneous_regret = counterfactual_reach * (action_utilities[action] - state_value)
            # Gewichte den Regret mit der Sampling-Wahrscheinlichkeit
            self.cumulative_regret[info_state][action] += sampling_weight * instantaneous_regret
        
        # Akkumuliere Policy mit Sampling-Gewichtung
        for action, action_prob in policy.items():
            # Uniform averaging: reach_prob * action_prob, gewichtet mit sampling_weight
            self.cumulative_policy[info_state][action] += sampling_weight * reach_prob * action_prob
        
        return state_value
    
    def update_regrets(self, info_set_key, legal_actions, action_utilities, current_utility, counterfactual_weight):
        """
        Wird nicht verwendet in dieser Implementierung.
        Die Updates erfolgen direkt in traverse_tree.
        Behalten für Kompatibilität mit Basisklasse.
        """
        pass
    
    def update_strategy_sum(self, info_set_key, legal_actions, current_strategy, player_reach):
        """
        Wird nicht verwendet in dieser Implementierung.
        Die Updates erfolgen direkt in traverse_tree.
        Behalten für Kompatibilität mit Basisklasse.
        """
        pass
