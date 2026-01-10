"""
Chance Sampling CFR Solver mit vorher gebautem Game Tree.

Chance Sampling CFR sammelt nur Zufallsaktionen (chance actions).
Statt alle Kombinationen zu durchlaufen, wird pro Iteration nur eine
zufällige Kombination gesampelt.

Basierend auf: Waugh et al. (2009) - Monte Carlo Sampling for Regret 
Minimization in Extensive Games
"""

import pickle as pkl
import gzip
import time
import random
from collections import defaultdict

from training.build_game_tree import load_game_tree, build_game_tree, save_game_tree, GameTree


class Node:
    """Repräsentiert einen Node im Game Tree"""
    def __init__(self, node_id):
        self.node_id = node_id
        self.type = None  # 'terminal' oder 'decision'
        self.player = None  # 0 oder 1 (nur bei decision nodes)
        self.infoset_key = None  # InfoSet Key (nur bei decision nodes)
        self.legal_actions = []  # Liste von legalen Aktionen
        self.children = {}  # {action: child_node_id}
        self.payoffs = None  # [payoff_p0, payoff_p1] (nur bei terminal nodes)
        self.depth = 0


class ChanceSamplingCFRSolver:
    """
    Chance Sampling CFR Solver der den Game Tree einmal vorher baut.
    
    Unterschied zu CFRSolverWithTree:
    - Statt alle Kombinationen zu durchlaufen, wird pro Iteration nur
      eine zufällige Kombination gesampelt
    - Die Updates werden mit der Anzahl der Kombinationen gewichtet,
      um den Erwartungswert beizubehalten
    """
    
    def __init__(self, game, combination_generator, game_name=None, load_tree=True):
        self.game = game
        self.combination_generator = combination_generator
        self.combinations = combination_generator.get_all_combinations()
        self.num_combinations = len(self.combinations)
        
        # CFR Datenstrukturen
        self.regret_sum = {}
        self.strategy_sum = {}
        self.iteration_count = 0
        self.training_time = 0
        
        # Tree Datenstrukturen
        self.nodes = {}  # {node_id: Node}
        self.next_node_id = 0
        self.infoset_to_nodes = defaultdict(list)  # {infoset_key: [node_ids]}
        self.root_nodes = []  # Liste von root node IDs (eine pro Kombination)
        self.combination_to_root = {}  # Mapping von Kombination zu Root-Node-ID
        
        # Versuche Tree zu laden, sonst baue ihn
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
        """Konvertiert ein GameTree Objekt zu interner Struktur"""
        self.nodes = {}
        self.infoset_to_nodes = defaultdict(list)
        self.root_nodes = game_tree.root_nodes
        
        # Erstelle Mapping von Kombination zu Root-Node
        # Wir müssen die Kombinationen in der gleichen Reihenfolge wie beim Bauen durchgehen
        # Die Kombinationen werden in build_game_tree in der gleichen Reihenfolge verarbeitet
        self.combination_to_root = {}
        for i, root_id in enumerate(game_tree.root_nodes):
            if i < len(self.combinations):
                # Kombinationen sind normalerweise Tuples, die hashable sind
                combo = self.combinations[i]
                self.combination_to_root[combo] = root_id
        
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
        
        # Initialisiere alle InfoSets, die im Tree existieren
        # (wichtig für Sampling-Strategien, damit alle InfoSets definiert sind)
        for infoset_key, node_ids in self.infoset_to_nodes.items():
            if node_ids:
                # Verwende die legal_actions vom ersten Node dieses InfoSets
                first_node = self.nodes[node_ids[0]]
                self.ensure_init(infoset_key, first_node.legal_actions)
    
    def ensure_init(self, info_set_key, legal_actions):
        """Initialisiert Regret und Strategy Sum für ein InfoSet"""
        if info_set_key not in self.regret_sum:
            self.regret_sum[info_set_key] = {a: 0.0 for a in legal_actions}
        if info_set_key not in self.strategy_sum:
            self.strategy_sum[info_set_key] = {a: 0.0 for a in legal_actions}
    
    def train(self, iterations, br_tracker=None):
        """
        Training mit vorher gebautem Tree und Chance Sampling.
        
        Args:
            iterations: Anzahl der Training-Iterationen
            br_tracker: Optionaler BestResponseTracker für Best Response Evaluation
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
                br_tracker.evaluate_and_add(current_avg_strategy, i + 1)
                br_tracker.last_eval_iteration = i + 1
        
        # Finale Best Response Evaluation
        if br_tracker is not None:
            current_avg_strategy = self.get_average_strategy()
            br_tracker.evaluate_and_add(current_avg_strategy, iterations)
        
        total_time = time.time() - start_time
        
        # Ziehe Best Response Zeit von der Trainingszeit ab
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
    
    def traverse_tree(self, node_id, player_id, reach_probabilities):
        """
        Traversiert den vorher gebauten Tree.
        
        Args:
            node_id: ID des aktuellen Nodes
            player_id: Spieler für den wir CFR durchführen (0 oder 1)
            reach_probabilities: [reach_p0, reach_p1]
        
        Returns:
            Utility für player_id
        """
        node = self.nodes[node_id]
        
        # Terminal Node: Payoff zurückgeben
        if node.type == 'terminal':
            return node.payoffs[player_id]
        
        # Decision Node
        current_player = node.player
        
        # Opponent's node: Keine Regret Updates
        if current_player != player_id:
            opponent = 1 - player_id
            opponent_info_set = node.infoset_key
            self.ensure_init(opponent_info_set, node.legal_actions)
            opponent_strategy = self.get_current_strategy(opponent_info_set, node.legal_actions)
            
            state_value = 0.0
            for action in node.legal_actions:
                action_prob = opponent_strategy[action]
                child_id = node.children[action]
                
                new_reach_probs = reach_probabilities.copy()
                new_reach_probs[opponent] *= action_prob
                
                state_value += action_prob * self.traverse_tree(child_id, player_id, new_reach_probs)
            
            return state_value
        
        # Player's node: Regret Updates
        info_set_key = node.infoset_key
        self.ensure_init(info_set_key, node.legal_actions)
        current_strategy = self.get_current_strategy(info_set_key, node.legal_actions)
        
        action_utilities = {}
        for action in node.legal_actions:
            child_id = node.children[action]
            
            new_reach_probs = reach_probabilities.copy()
            new_reach_probs[player_id] *= current_strategy[action]
            
            action_utilities[action] = self.traverse_tree(child_id, player_id, new_reach_probs)
        
        current_utility = sum(current_strategy[action] * action_utilities[action] for action in node.legal_actions)
        
        counterfactual_weight = reach_probabilities[1 - player_id]
        player_reach = reach_probabilities[player_id]
        
        self.update_regrets(info_set_key, node.legal_actions, action_utilities, current_utility, counterfactual_weight)
        self.update_strategy_sum(info_set_key, node.legal_actions, current_strategy, player_reach)
        
        return current_utility
    
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
    
    def get_current_strategy(self, info_set_key, legal_actions):
        """Regret Matching - Gleichung 8 aus Zinkevich et al. (2007)"""
        regrets = {a: self.regret_sum[info_set_key][a] for a in legal_actions}
        
        # Nur positive Regrets verwenden
        positive_regrets = {a: max(regrets[a], 0) for a in legal_actions}
        sum_pos = sum(positive_regrets.values())
        
        if sum_pos > 0:
            return {a: positive_regrets[a] / sum_pos for a in legal_actions}
        else:
            # Wenn keine positiven Regrets: Gleichverteilung
            return {a: 1.0 / len(legal_actions) for a in legal_actions}
    
    def get_average_strategy(self):
        """Gleichung 4 aus Zinkevich et al. (2007)"""
        average_strategy = {}
        
        for info_set_key in self.strategy_sum:
            total = sum(self.strategy_sum[info_set_key].values())
            if total > 0:
                average_strategy[info_set_key] = {
                    action: self.strategy_sum[info_set_key][action] / total
                    for action in self.strategy_sum[info_set_key]
                }
            else:
                num_actions = len(self.strategy_sum[info_set_key])
                average_strategy[info_set_key] = {
                    action: 1.0 / num_actions
                    for action in self.strategy_sum[info_set_key]
                }
        
        return average_strategy
    
    """Storage Methods"""
    
    def save_pickle(self, filepath):
        data = {
            'regret_sum': self.regret_sum,
            'strategy_sum': self.strategy_sum,
            'iteration_count': self.iteration_count
        }
        
        with open(filepath, 'wb') as f:
            pkl.dump(data, f)
        
        print(f"Saved to {filepath}")
    
    def load_pickle(self, filepath):
        with open(filepath, 'rb') as f:
            data = pkl.load(f)
        
        self.regret_sum = data['regret_sum']
        self.strategy_sum = data['strategy_sum']
        self.iteration_count = data['iteration_count']
        
        print(f"Loaded from {filepath}")
    
    def save_gzip(self, filepath):
        data = {
            'regret_sum': self.regret_sum,
            'strategy_sum': self.strategy_sum,
            'average_strategy': self.average_strategy,
            'iteration_count': self.iteration_count,
            'training_time': self.training_time
        }
        
        with gzip.open(filepath, 'wb') as f:
            pkl.dump(data, f)
        
        print(f"Saved to {filepath}")
    
    def load_gzip(self, filepath):
        with gzip.open(filepath, 'rb') as f:
            data = pkl.load(f)
        
        self.regret_sum = data['regret_sum']
        self.strategy_sum = data['strategy_sum']
        self.average_strategy = data['average_strategy']
        self.iteration_count = data['iteration_count']
        self.training_time = data.get('training_time', 0)
        
        print(f"Loaded from {filepath}")
