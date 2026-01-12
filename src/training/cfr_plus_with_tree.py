"""
CFR+ mit vorher gebautem Game Tree.

Hauptunterschiede zu normalem CFR:
- Negative Regrets werden auf 0 gesetzt (Regret Matching+)
- Linear averaging statt uniform averaging
- Alternierende Updates pro Iteration

Referenz: Tammelin et al. (2015)
"""

import gzip
import pickle as pkl
import numpy as np

from training.cfr_solver_with_tree import CFRSolverWithTree


class CFRPlusWithTree(CFRSolverWithTree):
    """
    CFR+ Implementation die den Tree vorher baut.
    
    Erbt von CFRSolverWithTree, überschreibt aber:
    - Regret Matching+ (negative regrets -> 0)
    - Linear averaging
    - Alternierende Updates
    """
    
    def __init__(self, game, combination_generator, game_name=None, load_tree=True):
        super().__init__(game, combination_generator, game_name=game_name, load_tree=load_tree)
        
        # Regrets können hier negativ sein, werden später auf 0 geklappt
        self.cumulative_regret = {}
        
        # Für durchschnittliche Strategie
        self.cumulative_policy = {}
        
        # Policy Cache damit wir nicht jedes mal neu berechnen müssen
        self._policy_cache = {}
        
        # Aktuelle Iteration für linear averaging (1-indexiert)
        self._current_iteration = 0
    
    def ensure_init(self, info_set_key, legal_actions):
        """Initialisiert die Dictionaries falls noch nicht vorhanden"""
        if info_set_key not in self.cumulative_regret:
            self.cumulative_regret[info_set_key] = {a: 0.0 for a in legal_actions}
        if info_set_key not in self.cumulative_policy:
            self.cumulative_policy[info_set_key] = {a: 0.0 for a in legal_actions}
    
    def cfr_iteration(self):
        """
        Eine CFR+ Iteration.
        
        Macht alternierende Updates: erst Spieler 0, dann Spieler 1.
        Nach jedem Spieler werden negative Regrets auf 0 geklappt und
        die Policy neu berechnet.
        """
        self._current_iteration = self.iteration_count + 1
        
        # Spieler 0 für alle Kombinationen
        for root_id in self.root_nodes:
            reach_probs = np.array([1.0, 1.0], dtype=np.float64)
            self._traverse_for_player(root_id, reach_probs, player=0)
        
        # Negative Regrets auf 0 setzen und Policy updaten
        self._reset_negative_regrets()
        self._update_all_policies()
        
        # Spieler 1 für alle Kombinationen
        for root_id in self.root_nodes:
            reach_probs = np.array([1.0, 1.0], dtype=np.float64)
            self._traverse_for_player(root_id, reach_probs, player=1)
        
        # Nochmal negative Regrets auf 0 setzen und Policy updaten
        self._reset_negative_regrets()
        self._update_all_policies()
    
    def _traverse_for_player(self, node_id, reach_probabilities, player):
        """
        Traversiert den Tree und sammelt Regrets für einen Spieler.
        
        node_id: aktueller Node
        reach_probabilities: [reach_p0, reach_p1]
        player: für welchen Spieler wir CFR machen (0 oder 1)
        """
        node = self.nodes[node_id]
        
        if node.type == 'terminal':
            return node.payoffs[player]
        
        current_player = node.player
        info_state = node.infoset_key
        
        # Wenn reach probs 0 sind können wir früher abbrechen
        if np.all(reach_probabilities[:2] == 0):
            return 0.0
        
        self.ensure_init(info_state, node.legal_actions)
        
        # Aktuelle Policy holen
        policy = self._get_policy(info_state)
        
        # Utilities für alle Aktionen berechnen
        action_utilities = {}
        state_value = 0.0
        
        for action in node.legal_actions:
            action_prob = policy.get(action, 0.0)
            child_id = node.children[action]
            
            # Neue reach probs für diesen Pfad
            new_reach_probs = reach_probabilities.copy()
            new_reach_probs[current_player] *= action_prob
            
            # Rekursiv weiter
            child_utility = self._traverse_for_player(child_id, new_reach_probs, player)
            
            action_utilities[action] = child_utility
            state_value += action_prob * child_utility
        
        # Wenn wir nicht für diesen Spieler updaten, einfach Wert zurückgeben
        if current_player != player:
            return state_value
        
        # Regret Updates für aktuellen Spieler
        reach_prob = reach_probabilities[current_player]
        
        # Counterfactual reach = produkt aller anderen Spieler
        counterfactual_reach = 1.0
        for p in range(len(reach_probabilities)):
            if p != current_player:
                counterfactual_reach *= reach_probabilities[p]
        
        # Regrets und Policy akkumulieren
        for action, action_prob in policy.items():
            regret = counterfactual_reach * (action_utilities[action] - state_value)
            
            # Regret akkumulieren (kann negativ sein)
            self.cumulative_regret[info_state][action] += regret
            
            # Linear averaging: iteration * reach_prob * action_prob
            self.cumulative_policy[info_state][action] += (
                self._current_iteration * reach_prob * action_prob)
        
        return state_value
    
    def _reset_negative_regrets(self):
        """Setzt alle negativen Regrets auf 0 (Regret Matching+)"""
        for info_state in self.cumulative_regret:
            for action in self.cumulative_regret[info_state]:
                if self.cumulative_regret[info_state][action] < 0:
                    self.cumulative_regret[info_state][action] = 0.0
    
    def _update_all_policies(self):
        """
        Aktualisiert alle Policies basierend auf aktuellen Regrets.
        
        Wird nach jedem Spieler-Update gemacht, damit die nächste
        Traversierung die neue Policy verwendet.
        """
        for info_state in self.cumulative_regret:
            node_ids = self.infoset_to_nodes.get(info_state, [])
            if not node_ids:
                continue
            
            node = self.nodes[node_ids[0]]
            legal_actions = node.legal_actions
            
            # Neue Policy berechnen
            policy = self._regret_matching(info_state, legal_actions)
            
            # In Cache speichern
            self._policy_cache[info_state] = policy
    
    def _get_policy(self, info_state):
        """Gibt die aktuelle Policy zurück, berechnet sie falls nötig"""
        if info_state in self._policy_cache:
            return self._policy_cache[info_state]
        
        node_ids = self.infoset_to_nodes.get(info_state, [])
        if not node_ids:
            return {}
        
        node = self.nodes[node_ids[0]]
        legal_actions = node.legal_actions
        
        # Policy neu berechnen
        policy = self._regret_matching(info_state, legal_actions)
        
        # Cachen
        self._policy_cache[info_state] = policy
        
        return policy
    
    def _regret_matching(self, info_state, legal_actions):
        """
        Berechnet Policy mit Regret Matching.
        
        Nur positive Regrets werden verwendet, dann normalisiert.
        Falls keine positiven Regrets vorhanden, gleichverteilung.
        """
        regrets = self.cumulative_regret.get(info_state, {})
        
        positive_regrets = {}
        total_positive = 0.0
        
        for action in legal_actions:
            regret = regrets.get(action, 0.0)
            positive_regret = max(0.0, regret)
            positive_regrets[action] = positive_regret
            total_positive += positive_regret
        
        if total_positive > 0:
            return {action: positive_regrets[action] / total_positive 
                   for action in legal_actions}
        else:
            # Gleichverteilung wenn keine positiven Regrets
            uniform_prob = 1.0 / len(legal_actions)
            return {action: uniform_prob for action in legal_actions}
    
    def get_current_strategy(self, info_set_key, legal_actions):
        """Gibt aktuelle Strategie zurück (für Basisklasse)"""
        policy = self._get_policy(info_set_key)
        result = {}
        for action in legal_actions:
            result[action] = policy.get(action, 0.0)
        return result
    
    def update_regrets(self, info_set_key, legal_actions, action_utilities, current_utility, counterfactual_weight):
        """Nicht verwendet, nur für Kompatibilität"""
        pass
    
    def update_strategy_sum(self, info_set_key, legal_actions, current_strategy, player_reach):
        """Nicht verwendet, nur für Kompatibilität"""
        pass
    
    def get_average_strategy(self):
        """
        Berechnet durchschnittliche Strategie mit linear averaging.
        
        Bei linear averaging wird einfach durch die Summe geteilt.
        """
        average_strategy = {}
        
        for info_state, policy_dict in self.cumulative_policy.items():
            total = sum(policy_dict.values())
            
            if total == 0:
                # Gleichverteilung wenn nichts akkumuliert wurde
                node_ids = self.infoset_to_nodes.get(info_state, [])
                if not node_ids:
                    continue
                node = self.nodes[node_ids[0]]
                num_actions = len(node.legal_actions)
                average_strategy[info_state] = {
                    action: 1.0 / num_actions for action in node.legal_actions
                }
            else:
                # Normalisieren
                average_strategy[info_state] = {
                    action: action_sum / total
                    for action, action_sum in policy_dict.items()
                }
        
        return average_strategy
    
    def save_gzip(self, filepath):
        """Speichert die Daten"""
        data = {
            'cumulative_regret': self.cumulative_regret,
            'cumulative_policy': self.cumulative_policy,
            'average_strategy': self.average_strategy,
            'iteration_count': self._current_iteration,
            'training_time': self.training_time
        }
        
        with gzip.open(filepath, 'wb') as f:
            pkl.dump(data, f)
        
        print(f"Saved to {filepath}")
    
    def load_gzip(self, filepath):
        """Lädt die Daten"""
        with gzip.open(filepath, 'rb') as f:
            data = pkl.load(f)
        
        self.cumulative_regret = data['cumulative_regret']
        self.cumulative_policy = data['cumulative_policy']
        self.average_strategy = data.get('average_strategy', {})
        self._current_iteration = data.get('iteration_count', 0)
        self.training_time = data.get('training_time', 0)
        
        # iteration_count ist 0-indexiert, _current_iteration ist 1-indexiert
        self.iteration_count = max(0, self._current_iteration - 1)
        
        # Policy Cache neu aufbauen
        self._policy_cache = {}
        for info_state in self.cumulative_regret.keys():
            self._get_policy(info_state)
        
        print(f"Loaded from {filepath}")
