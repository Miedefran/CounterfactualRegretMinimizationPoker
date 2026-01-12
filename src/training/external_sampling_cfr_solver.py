"""
External Sampling CFR mit vorher gebautem Game Tree.

Bei External Sampling werden nur die Aktionen des Gegners und Chance gesampelt.
Die eigenen Aktionen werden vollständig durchlaufen (kein Sampling).

Referenz: Waugh et al. (2009) - Monte Carlo Sampling for Regret 
Minimization in Extensive Games
"""

import pickle as pkl
import gzip
import time
import random
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


class ExternalSamplingCFRSolver:
    """
    External Sampling CFR der den Tree vorher baut.
    
    Unterschied zu normalem CFR:
    - Bei Gegner-Nodes: nur eine Aktion sampeln
    - Bei eigenen Nodes: alle Aktionen durchlaufen
    - Regret Updates ohne Counterfactual Reach Probability
    - Policy Updates am Gegner-Node ohne Reach Probability Gewichtung
    """
    
    def __init__(self, game, combination_generator, game_name=None, load_tree=True):
        self.game = game
        self.combination_generator = combination_generator
        self.combinations = combination_generator.get_all_combinations()
        
        # CFR Datenstrukturen
        self.cumulative_regret = {}
        self.cumulative_policy = {}
        self.iteration_count = 0
        self.training_time = 0
        
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
        
        # Alle InfoSets initialisieren
        for infoset_key, node_ids in self.infoset_to_nodes.items():
            if node_ids:
                first_node = self.nodes[node_ids[0]]
                self.ensure_init(infoset_key, first_node.legal_actions)
    
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
        Training mit External Sampling.
        
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
        Eine External Sampling CFR Iteration.
        
        Für jeden Spieler wird UpdateRegrets aufgerufen, beginnend
        vom Root State. Bei External Sampling wird nur EINE Kombination
        pro Iteration gesampelt.
        """
        # Für jeden Spieler
        for player in range(2):
            # Eine zufällige Kombination sampeln
            sampled_root_id = random.choice(self.root_nodes)
            self._update_regrets(sampled_root_id, player)
    
    def _update_regrets(self, node_id, player):
        """
        UpdateRegrets für External Sampling.
        
        node_id: aktueller Node
        player: für welchen Spieler wir CFR machen (0 oder 1)
        """
        node = self.nodes[node_id]
        
        if node.type == 'terminal':
            return node.payoffs[player]
        
        current_player = node.player
        info_set_key = node.infoset_key
        
        self.ensure_init(info_set_key, node.legal_actions)
        
        # Aktuelle Policy berechnen
        current_policy = self._get_current_policy(info_set_key, node.legal_actions)
        
        value = 0.0
        child_values = {}
        
        if current_player != player:
            # Gegner-Node: eine Aktion sampeln
            sampled_action = self._sample_action(current_policy, node.legal_actions)
            child_id = node.children[sampled_action]
            value = self._update_regrets(child_id, player)
        else:
            # Eigener Node: alle Aktionen durchlaufen
            for action in node.legal_actions:
                child_id = node.children[action]
                child_value = self._update_regrets(child_id, player)
                child_values[action] = child_value
                value += current_policy.get(action, 0.0) * child_value
        
        # Regret Updates nur am eigenen Node
        if current_player == player:
            for action in node.legal_actions:
                # WICHTIG: Bei External Sampling OHNE Counterfactual Reach Probability!
                regret = child_values[action] - value
                self.cumulative_regret[info_set_key][action] += regret
        
        # Policy Updates am Gegner-Node (Simple Average)
        opponent = (player + 1) % 2
        if current_player == opponent:
            # WICHTIG: Bei External Sampling OHNE Reach Probability Gewichtung!
            for action in node.legal_actions:
                action_prob = current_policy.get(action, 0.0)
                self.cumulative_policy[info_set_key][action] += action_prob
        
        return value
    
    def _sample_action(self, policy, legal_actions):
        """
        Sample eine Aktion basierend auf Policy.
        
        Falls Policy nicht normalisiert ist, wird normalisiert.
        Falls keine Wahrscheinlichkeiten vorhanden, gleichverteilung.
        """
        actions = list(legal_actions)
        probabilities = [policy.get(action, 0.0) for action in actions]
        
        # Normalisieren falls nötig
        total = sum(probabilities)
        if total > 0:
            probabilities = [p / total for p in probabilities]
        else:
            # Gleichverteilung wenn keine Wahrscheinlichkeiten
            probabilities = [1.0 / len(actions)] * len(actions)
        
        return random.choices(actions, weights=probabilities)[0]
    
    def _get_current_policy(self, info_set_key, legal_actions):
        """
        Berechnet aktuelle Policy mit Regret Matching.
        
        Nur positive Regrets werden verwendet, dann normalisiert.
        Falls keine positiven Regrets vorhanden, gleichverteilung.
        """
        regrets = self.cumulative_regret.get(info_set_key, {})
        
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
        return self._get_current_policy(info_set_key, legal_actions)
    
    def get_average_strategy(self):
        """
        Berechnet durchschnittliche Strategie.
        
        Bei External Sampling (Simple Average) einfach durch die Summe teilen.
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
        
        print(f"Loaded from {filepath}")
