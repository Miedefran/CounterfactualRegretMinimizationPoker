"""
Outcome Sampling CFR Solver mit vorher gebautem Game Tree.

Bei Outcome Sampling wird in jeder Iteration nur eine einzige Terminal History
gesampelt. Dies ist die extremste Form des Samplings, da pro Iteration nur
ein einziger Pfad durch den Game Tree betrachtet wird.

Die Sampling Policy wird verwendet, um eine Terminal History zu sampeln.
Die Regret Updates werden mit Importance Sampling gewichtet, um den
Erwartungswert beizubehalten.

Referenz: Waugh et al. (2009) - Monte Carlo Sampling for Regret 
Minimization in Extensive Games

Implementierung folgt exakt der OpenSpiel Logik.
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


class OutcomeSamplingCFRSolver:
    """
    Outcome Sampling CFR Solver der den Tree vorher baut.
    
    Implementierung folgt exakt der OpenSpiel Logik.
    """
    
    def __init__(self, game, combination_generator, game_name=None, load_tree=True, epsilon=0.6):
        """
        Initialisiert den Outcome Sampling CFR Solver.
        
        Args:
            game: Das Spiel-Objekt
            combination_generator: Generator für Kartenkombinationen
            game_name: Name des Spiels (für Tree-Laden)
            load_tree: Ob der Tree geladen werden soll
            epsilon: Epsilon für epsilon-greedy Sampling Policy (Standard: 0.6)
        """
        self.game = game
        self.combination_generator = combination_generator
        self.combinations = combination_generator.get_all_combinations()
        self.epsilon = epsilon
        
        # CFR Datenstrukturen
        self.regret_sum = {}
        self.strategy_sum = {}
        self.iteration_count = 0
        self.training_time = 0
        
        # Tree Datenstrukturen
        self.nodes = {}
        self.next_node_id = 0
        self.infoset_to_nodes = defaultdict(list)
        self.root_nodes = []
        
        # Bestimme ob Suit Abstraction verwendet wird (basierend auf Combination Generator Typ)
        from utils.poker_utils import LeducHoldemCombinationsAbstracted, TwelveCardPokerCombinationsAbstracted
        use_suit_abstraction = isinstance(combination_generator, 
            (LeducHoldemCombinationsAbstracted, TwelveCardPokerCombinationsAbstracted))
        
        # Tree laden oder bauen
        if load_tree and game_name:
            try:
                print(f"Attempting to load game tree for {game_name}...")
                game_tree = load_game_tree(game_name, abstract_suits=use_suit_abstraction)
                self._convert_game_tree_to_internal(game_tree)
                print(f"Tree loaded: {len(self.nodes)} nodes, {len(self.infoset_to_nodes)} unique infosets")
            except FileNotFoundError:
                print(f"Tree file not found for {game_name}, building tree...")
                game_tree = build_game_tree(self.game, self.combination_generator, game_name=game_name, abstract_suits=use_suit_abstraction)
                self._convert_game_tree_to_internal(game_tree)
                save_game_tree(game_tree, game_name, abstract_suits=use_suit_abstraction)
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
        if info_set_key not in self.regret_sum:
            self.regret_sum[info_set_key] = {a: 0.0 for a in legal_actions}
        if info_set_key not in self.strategy_sum:
            self.strategy_sum[info_set_key] = {a: 0.0 for a in legal_actions}
    
    def train(self, iterations, br_tracker=None, print_interval=100):
        """
        Training mit Outcome Sampling.
        
        iterations: Anzahl Iterationen
        br_tracker: Optional für Best Response Evaluation
        print_interval: Intervall für Print-Statements (Standard: 100)
        """
        start_time = time.time()
        
        for i in range(iterations):
            self.cfr_iteration()
            self.iteration_count += 1
            
            if (i + 1) % print_interval == 0:
                print(f"Iteration {i + 1}")
            
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
        Eine Outcome Sampling CFR Iteration.
        
        Bei Outcome Sampling wird nur eine einzige Terminal History gesampelt.
        Die Sampling Policy wird verwendet, um einen Pfad durch den Tree zu sampeln.
        """
        # Für jeden Spieler
        for player in range(2):
            # Eine zufällige Kombination sampeln (Chance Node)
            sampled_root_id = random.choice(self.root_nodes)
            num_roots = len(self.root_nodes)
            root_prob = 1.0 / num_roots if num_roots > 0 else 1.0
            
            # Starte mit my_reach=1.0, opp_reach=root_prob, sample_reach=root_prob
            # (Root Node ist bereits gesampelt)
            self._episode(sampled_root_id, player, my_reach=1.0, 
                         opp_reach=root_prob, sample_reach=root_prob)
    
    def _baseline(self, node_id, info_set_key, aidx):
        """Baseline für baseline-corrected outcome sampling (Standard: 0)"""
        return 0.0
    
    def _baseline_corrected_child_value(self, sampled_aidx, aidx, child_value, sample_prob):
        """
        Berechnet baseline-corrected child value.
        
        Applies Eq. 9 of Schmid et al. '19
        """
        baseline = self._baseline(None, None, aidx)
        if aidx == sampled_aidx:
            return baseline + (child_value - baseline) / sample_prob
        else:
            return baseline
    
    def _episode(self, node_id, update_player, my_reach, opp_reach, sample_reach):
        """
        Führt eine Episode von Outcome Sampling durch.
        
        Implementierung folgt exakt der OpenSpiel-Logik.
        
        Args:
            node_id: aktueller Node
            update_player: Spieler für den wir CFR machen (0 oder 1)
            my_reach: Reach Probability des update_player
            opp_reach: Reach Probability aller anderen Spieler (inkl. Chance)
            sample_reach: Reach Probability der Sampling Policy
        
        Returns:
            value_estimate: geschätzter Wert für den update_player
        """
        node = self.nodes[node_id]
        
        if node.type == 'terminal':
            # Terminal Node: Return payoff
            return node.payoffs[update_player]
        
        current_player = node.player
        info_set_key = node.infoset_key
        legal_actions = node.legal_actions
        
        self.ensure_init(info_set_key, legal_actions)
        
        # Aktuelle Policy berechnen
        current_policy = self._get_current_policy(info_set_key, legal_actions)
        policy_array = np.array([current_policy.get(action, 0.0) for action in legal_actions], dtype=np.float64)
        
        # Epsilon-greedy Sampling Policy (nur für update_player)
        if current_player == update_player:
            uniform_policy = np.ones(len(legal_actions), dtype=np.float64) / len(legal_actions)
            sample_policy = self.epsilon * uniform_policy + (1.0 - self.epsilon) * policy_array
        else:
            sample_policy = policy_array.copy()
        
        # Normalisiere sample_policy
        sample_policy_sum = np.sum(sample_policy)
        if sample_policy_sum > 1e-10:
            sample_policy = sample_policy / sample_policy_sum
        else:
            sample_policy = np.ones(len(legal_actions), dtype=np.float64) / len(legal_actions)
        
        # Sample eine Aktion
        sampled_aidx = np.random.choice(len(legal_actions), p=sample_policy)
        sampled_action = legal_actions[sampled_aidx]
        
        # Aktualisiere Reach Probabilities
        # WICHTIG: my_reach verwendet die aktuelle Policy (policy_array), nicht die Sampling-Policy
        # sample_reach verwendet die Sampling-Policy (sample_policy)
        if current_player == update_player:
            new_my_reach = my_reach * policy_array[sampled_aidx]
            new_opp_reach = opp_reach
        else:
            new_my_reach = my_reach
            new_opp_reach = opp_reach * policy_array[sampled_aidx]
        
        new_sample_reach = sample_reach * sample_policy[sampled_aidx]
        
        # Rekursiver Aufruf
        child_id = node.children[sampled_action]
        child_value = self._episode(child_id, update_player, new_my_reach, 
                                   new_opp_reach, new_sample_reach)
        
        # Compute each of the child estimated values (baseline-corrected)
        child_values = np.zeros(len(legal_actions), dtype=np.float64)
        for aidx in range(len(legal_actions)):
            child_values[aidx] = self._baseline_corrected_child_value(
                sampled_aidx, aidx, child_value, sample_policy[aidx])
        
        # Value estimate = gewichtete Summe
        value_estimate = np.sum(policy_array * child_values)
        
        # Update regrets and avg strategies (nur für update_player)
        if current_player == update_player:
            # Estimate for the counterfactual value of the policy
            if sample_reach > 1e-10:
                cf_value = value_estimate * opp_reach / sample_reach
            else:
                cf_value = 0.0
            
            # Update regrets
            for aidx, action in enumerate(legal_actions):
                # Estimate for the counterfactual value of the policy replaced by always
                # choosing action at this information state
                if sample_reach > 1e-10:
                    cf_action_value = child_values[aidx] * opp_reach / sample_reach
                else:
                    cf_action_value = 0.0
                
                regret = cf_action_value - cf_value
                self.regret_sum[info_set_key][action] += regret
            
            # Update the average policy
            # WICHTIG: Bei Outcome Sampling werden Policy Updates für ALLE Aktionen gemacht
            # (wie in OpenSpiel). Die Berechnung ist: increment = my_reach * policy[aidx] / sample_reach
            # Dabei ist sample_reach die Sampling-Reach bis zu diesem Node (enthält sample_policy[sampled_aidx])
            # Für nicht-gesampelte Aktionen ist increment sehr klein, aber nicht 0
            for aidx, action in enumerate(legal_actions):
                if sample_reach > 1e-10:
                    increment = my_reach * policy_array[aidx] / sample_reach
                else:
                    increment = 0.0
                self.strategy_sum[info_set_key][action] += increment
        
        return value_estimate
    
    def _get_current_policy(self, info_set_key, legal_actions):
        """
        Berechnet aktuelle Policy mit Regret Matching.
        
        Nur positive Regrets werden verwendet, dann normalisiert.
        Falls keine positiven Regrets vorhanden, gleichverteilung.
        """
        regrets = self.regret_sum.get(info_set_key, {})
        
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
    
    def _regret_matching(self, regret_array, num_legal_actions):
        """
        Wendet Regret Matching an, um eine Policy zu berechnen.
        
        Args:
            regret_array: numpy array mit Regrets für jede Aktion
            num_legal_actions: Anzahl der legalen Aktionen
        
        Returns:
            numpy array mit Policy-Wahrscheinlichkeiten
        """
        positive_regrets = np.maximum(regret_array, np.zeros(num_legal_actions, dtype=np.float64))
        sum_pos_regret = positive_regrets.sum()
        if sum_pos_regret <= 0:
            return np.ones(num_legal_actions, dtype=np.float64) / num_legal_actions
        else:
            return positive_regrets / sum_pos_regret
    
    def get_current_strategy(self, info_set_key, legal_actions):
        """Gibt aktuelle Strategie zurück (für Basisklasse)"""
        regrets = self.regret_sum.get(info_set_key, {})
        regret_array = np.array([regrets.get(action, 0.0) for action in legal_actions], dtype=np.float64)
        policy = self._regret_matching(regret_array, len(legal_actions))
        return {action: policy[i] for i, action in enumerate(legal_actions)}
    
    def get_average_strategy(self):
        """
        Berechnet durchschnittliche Strategie (wie OpenSpiel).
        """
        average_strategy = {}
        
        for info_state, policy_dict in self.strategy_sum.items():
            policy_array = np.array([policy_dict.get(action, 0.0) for action in 
                                   sorted(policy_dict.keys())], dtype=np.float64)
            total = policy_array.sum()
            
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
                # Normalisieren (wie OpenSpiel)
                normalized = policy_array / total
                actions = sorted(policy_dict.keys())
                average_strategy[info_state] = {
                    actions[i]: float(normalized[i]) for i in range(len(actions))
                }
        
        return average_strategy
    
    def save_gzip(self, filepath):
        """Speichert die Daten"""
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
        """Lädt die Daten"""
        with gzip.open(filepath, 'rb') as f:
            data = pkl.load(f)
        
        self.regret_sum = data['regret_sum']
        self.strategy_sum = data['strategy_sum']
        self.average_strategy = data.get('average_strategy', {})
        self.iteration_count = data.get('iteration_count', 0)
        self.training_time = data.get('training_time', 0)
        
        print(f"Loaded from {filepath}")
