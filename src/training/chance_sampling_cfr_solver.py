"""
Chance Sampling CFR Solver mit vorher gebautem Game Tree.

Chance Sampling CFR sammelt nur Zufallsaktionen (chance actions).
Statt alle Chance-Outcomes zu expandieren, werden an Chance-Nodes Outcomes
gemäß ihrer Chance-Probability gesampelt. Player-Decision-Nodes werden
vollständig expandiert (full-width).

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
    - Chance Nodes werden gesampelt (statt expandiert)
    - Updates werden per Importance-Weighting mit 1/prob(sampled chance path)
      korrigiert, damit der Erwartungswert erhalten bleibt.
    """
    
    def __init__(self, game, combination_generator, game_name=None, load_tree=True):
        # Rufe Basisklassen-__init__ auf
        super().__init__(game, combination_generator, game_name=game_name, load_tree=load_tree)
    
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
    
    def cfr_iteration(self):
        """
        Eine Chance Sampling CFR Iteration.
        
        Chance Nodes werden gesampelt; Decision Nodes full-width traversiert.
        """
        root_id = self.root_nodes[0]
        
        reach_probs = np.array([1.0, 1.0], dtype=np.float64)

        # Sample ONE consistent chance sequence per CFR iteration and reuse it
        # across all chance nodes that correspond to the same "deal index".
        # This matches poker-style chance-sampled CFR (sample full deal once),
        # and avoids sampling different public cards for different betting histories.
        sampled_chance_by_index = {}
        # Debug/diagnostics: record which chance outcome was used per index, and
        # every chance-node encounter (to verify consistency across histories).
        self._debug_last_sampled_chance_by_index = sampled_chance_by_index
        self._debug_last_chance_encounters = []  # list[(chance_index, node_id, outcome)]
        
        # Traverse für Spieler 0
        self._traverse_chance_sample(
            root_id,
            player=0,
            reach_probabilities=reach_probs,
            chance_index=0,
            sampled_chance_by_index=sampled_chance_by_index,
        )
        
        # Policy Update nach Spieler 0 (wichtig: damit Spieler 1 die aktualisierte Policy verwendet)
        self._update_all_policies()
        
        # Traverse für Spieler 1 (mit aktualisierter Policy von Spieler 0)
        self._traverse_chance_sample(
            root_id,
            player=1,
            reach_probabilities=reach_probs,
            chance_index=0,
            sampled_chance_by_index=sampled_chance_by_index,
        )
        
        # Policy Update nach Spieler 1
        self._update_all_policies()
    
    def _sample_chance_outcome(self, node):
        """Sample a chance outcome according to node.chance_probs."""
        probs = node.chance_probs or {}
        outcomes = list(node.legal_actions)
        if not outcomes:
            return None, 0.0
        weights = [float(probs.get(o, 0.0)) for o in outcomes]
        s = sum(weights)
        if s <= 0:
            # fallback uniform
            o = random.choice(outcomes)
            return o, 1.0 / len(outcomes)
        # normalize
        r = random.random() * s
        acc = 0.0
        for o, w in zip(outcomes, weights):
            acc += w
            if r <= acc:
                return o, float(w) / s
        return outcomes[-1], float(weights[-1]) / s

    def _traverse_chance_sample(self, node_id, player, reach_probabilities, chance_index: int, sampled_chance_by_index: dict):
        """
        Traversiert den Tree für Chance Sampling CFR:
        - Chance Nodes: sample one outcome
        - Decision Nodes: full-width expansion
        Chance outcomes are sampled ONCE per chance_index and reused across all
        chance nodes at that index (e.g. same public card for all betting histories).
        
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

        # Chance Node: sample one chance outcome
        if node.type == 'chance':
            # Reuse a consistent sampled outcome for this deal index.
            outcome = sampled_chance_by_index.get(chance_index)
            if outcome is None or outcome not in node.children:
                outcome, _ = self._sample_chance_outcome(node)
                if outcome is None:
                    return 0.0
                sampled_chance_by_index[chance_index] = outcome
            child_id = node.children.get(outcome)
            if child_id is None:
                return 0.0
            # Record encounter for diagnosis
            if hasattr(self, "_debug_last_chance_encounters"):
                self._debug_last_chance_encounters.append((chance_index, node_id, outcome))
            # For chance-sampling CFR (Waugh et al. 2009; Zinkevich et al. chance-sampled CFR),
            # we do NOT apply an additional 1/probability importance weight here. Sampling chance
            # outcomes from the true chance distribution is already unbiased; the chance
            # probabilities cancel in the sampled counterfactual value (Eq. 6, paper).
            return self._traverse_chance_sample(
                child_id,
                player,
                reach_probabilities,
                chance_index=chance_index + 1,
                sampled_chance_by_index=sampled_chance_by_index,
            )
        
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
            child_utility = self._traverse_chance_sample(
                child_id,
                player,
                new_reach_probs,
                chance_index=chance_index,
                sampled_chance_by_index=sampled_chance_by_index,
            )
            
            action_utilities[action] = child_utility
            state_value += action_prob * child_utility
        
        # Wenn wir nicht für den aktuellen Spieler updaten, nur Wert zurückgeben
        if current_player != player:
            return state_value
        
        reach_prob = reach_probabilities[current_player]
        counterfactual_reach = reach_probabilities[1 - current_player]
        
        # Akkumuliere Regrets (ohne zusätzliche Importance-Gewichtung)
        for action in node.legal_actions:
            # Instantaneous Regret
            instantaneous_regret = counterfactual_reach * (action_utilities[action] - state_value)
            self.regret_sum[info_state][action] += instantaneous_regret
        
        # Akkumuliere Policy (Standard CFR Average Strategy: π_i(I) * σ_i(a|I))
        for action, action_prob in policy.items():
            self.strategy_sum[info_state][action] += reach_prob * action_prob
        
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
