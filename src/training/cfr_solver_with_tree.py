"""
CFR Solver mit vorher gebautem Game Tree.

Unterschied zum normalen cfr_solver:
- Tree wird einmal vorher gebaut statt bei jeder Iteration neu durchlaufen
- Kein game.step()/step_back() mehr nötig
- Einfach über die Datenstruktur iterieren
"""

import pickle as pkl
import gzip
import time
import numpy as np
from collections import defaultdict

from training.build_game_tree import load_game_tree, build_game_tree, save_game_tree, GameTree


class Node:
    """Ein Node im Game Tree"""
    def __init__(self, node_id):
        self.node_id = node_id
        self.type = None  # 'terminal' oder 'decision'
        self.player = None  # 0 oder 1
        self.infoset_key = None
        self.legal_actions = []
        self.children = {}  # {action: child_node_id}
        self.payoffs = None  # [payoff_p0, payoff_p1]
        self.depth = 0


class CFRSolverWithTree:
    """
    CFR Solver der den Tree vorher baut.
    
    Vorteile:
    - Kein wiederholtes game.step()/step_back()
    - Schnellere Lookups
    - Klarere Struktur
    """
    
    def __init__(self, game, combination_generator, game_name=None, load_tree=True, alternating_updates=True):
        self.game = game
        self.combination_generator = combination_generator
        self.combinations = combination_generator.get_all_combinations()
        self.alternating_updates = alternating_updates
        
        # CFR Datenstrukturen
        self.cumulative_regret = {}
        self.cumulative_policy = {}
        self.iteration_count = 0
        self.training_time = 0
        
        # Policy Cache damit wir nicht jedes mal neu berechnen müssen
        self._policy_cache = {}
        
        # Tree Datenstrukturen
        self.nodes = {}
        self.next_node_id = 0
        self.infoset_to_nodes = defaultdict(list)
        self.root_nodes = []
        
        # Tree laden oder bauen
        if load_tree and game_name:
            try:
                print(f"Attempting to load game tree for {game_name}...")
                game_tree = load_game_tree(game_name)
                self._convert_game_tree_to_internal(game_tree)
                print(f"Tree loaded: {len(self.nodes)} nodes, {len(self.infoset_to_nodes)} unique infosets")
            except FileNotFoundError:
                print(f"Tree file not found for {game_name}, building tree...")
                game_tree = build_game_tree(self.game, self.combination_generator, game_name=game_name)
                self._convert_game_tree_to_internal(game_tree)
                save_game_tree(game_tree, game_name)
                print(f"Tree built and saved: {len(self.nodes)} nodes, {len(self.infoset_to_nodes)} unique infosets")
        else:
            print("Building game tree...")
            game_tree = build_game_tree(self.game, self.combination_generator)
            self._convert_game_tree_to_internal(game_tree)
            print(f"Tree built: {len(self.nodes)} nodes, {len(self.infoset_to_nodes)} unique infosets")
    
    def _convert_game_tree_to_internal(self, game_tree):
        """Konvertiert GameTree zu interner Struktur"""
        self.nodes = {}
        self.infoset_to_nodes = defaultdict(list)
        self.root_nodes = game_tree.root_nodes
        
        for node_id, node_data in game_tree.nodes.items():
            node = Node(node_data.node_id)
            node.type = node_data.type
            node.player = node_data.player
            node.infoset_key = node_data.infoset_key
            node.legal_actions = node_data.legal_actions
            node.children = node_data.children
            node.payoffs = node_data.payoffs
            node.depth = node_data.depth
            self.nodes[node_id] = node
            
            if node.infoset_key is not None:
                self.infoset_to_nodes[node.infoset_key].append(node_id)
        
        if self.nodes:
            self.next_node_id = max(self.nodes.keys()) + 1
        else:
            self.next_node_id = 0
    
    def ensure_init(self, info_set_key, legal_actions):
        """Initialisiert die Dictionaries falls noch nicht vorhanden"""
        if info_set_key not in self.cumulative_regret:
            self.cumulative_regret[info_set_key] = {a: 0.0 for a in legal_actions}
        if info_set_key not in self.cumulative_policy:
            self.cumulative_policy[info_set_key] = {a: 0.0 for a in legal_actions}
    
    def train(self, iterations, br_tracker=None):
        """
        Training mit vorher gebautem Tree.
        
        iterations: Anzahl Iterationen
        br_tracker: Optional für Best Response Evaluation
        """
        start_time = time.time()
        
        for i in range(iterations):
            self.cfr_iteration()
            self.iteration_count += 1
            
            if i % 100 == 0:
                print(f"Iteration {i}")
            
            # Best Response Evaluation
            if br_tracker is not None and br_tracker.should_evaluate(i + 1):
                current_avg_strategy = self.get_average_strategy()
                br_tracker.evaluate_and_add(current_avg_strategy, i + 1, start_time=start_time)
                br_tracker.last_eval_iteration = i + 1
        
        # Finale Best Response Evaluation
        if br_tracker is not None:
            current_avg_strategy = self.get_average_strategy()
            br_tracker.evaluate_and_add(current_avg_strategy, iterations, start_time=start_time)
        
        total_time = time.time() - start_time
        
        # Best Response Zeit abziehen
        if br_tracker is not None:
            br_time = br_tracker.get_total_br_time()
            self.training_time = total_time - br_time
            if br_time > 0:
                print(f"Best Response Evaluation Zeit: {br_time:.2f}s")
        else:
            self.training_time = total_time
        
        if self.training_time >= 60:
            minutes = self.training_time / 60
            print(f"Training completed in {minutes:.2f} minutes (ohne Best Response Evaluation)")
        else:
            print(f"Training completed in {self.training_time:.2f} seconds (ohne Best Response Evaluation)")
        
        self.average_strategy = self.get_average_strategy()
    
    def cfr_iteration(self):
        """
        Eine CFR Iteration.
        
        Wenn alternating_updates=True (Standard):
        - Erst Spieler 0 für alle Kombinationen traversieren
        - Policy updaten
        - Dann Spieler 1 für alle Kombinationen traversieren
        - Policy updaten
        
        Wenn alternating_updates=False:
        - Beide Spieler gleichzeitig traversieren
        - Policy updaten
        """
        if self.alternating_updates:
            # Alternierende Updates
            # Spieler 0 für alle Kombinationen
            for root_id in self.root_nodes:
                reach_probs = np.array([1.0, 1.0], dtype=np.float64)
                self._traverse_for_player(root_id, reach_probs, player=0)
            
            # Policy Update nach Spieler 0
            self._update_all_policies()
            
            # Spieler 1 für alle Kombinationen
            for root_id in self.root_nodes:
                reach_probs = np.array([1.0, 1.0], dtype=np.float64)
                self._traverse_for_player(root_id, reach_probs, player=1)
            
            # Policy Update nach Spieler 1
            self._update_all_policies()
        else:
            # Simultane Updates
            # Beide Spieler gleichzeitig für alle Kombinationen
            for root_id in self.root_nodes:
                reach_probs = np.array([1.0, 1.0], dtype=np.float64)
                self._traverse_for_player(root_id, reach_probs, player=0)
                self._traverse_for_player(root_id, reach_probs, player=1)
            
            # Policy Update nach beiden Spielern
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
            
            # Regret akkumulieren (kann negativ sein bei Vanilla CFR)
            self.cumulative_regret[info_state][action] += regret
            
            # Uniform averaging: reach_prob * action_prob
            self.cumulative_policy[info_state][action] += reach_prob * action_prob
        
        return state_value
    
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
        Berechnet durchschnittliche Strategie mit uniform averaging.
        
        Bei uniform averaging wird einfach durch die Summe geteilt.
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
            'iteration_count': self.iteration_count,
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
        self.iteration_count = data.get('iteration_count', 0)
        self.training_time = data.get('training_time', 0)
        
        # Policy Cache neu aufbauen
        self._policy_cache = {}
        for info_state in self.cumulative_regret.keys():
            self._get_policy(info_state)
        
        print(f"Loaded from {filepath}")
