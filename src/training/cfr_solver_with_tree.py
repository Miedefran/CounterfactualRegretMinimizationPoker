"""
CFR Solver mit vorher gebautem Game Tree.

Der Unterschied zum normalen cfr_solver.py:
- Statt bei jeder Iteration den Tree neu zu durchlaufen (mit game.step/step_back),
  wird der Tree einmal vorher gebaut und als Datenstruktur gespeichert
- Bei jeder Iteration wird nur noch über diese Struktur iteriert
"""

import pickle as pkl
import gzip
import time
import numpy as np
from collections import defaultdict

from training.build_game_tree import load_game_tree, build_game_tree, save_game_tree, GameTree


class CFRSolverWithTree:
    """
    CFR Solver der den Game Tree einmal vorher baut.
    
    Vorteile:
    - Kein wiederholtes game.step()/step_back() bei jeder Iteration
    - Schnellere Lookups über Dictionary
    - Klarere Struktur
    """
    
    def __init__(
        self,
        game,
        combination_generator,
        game_name=None,
        load_tree=True,
        alternating_updates=True,
        partial_pruning=False,
    ):
        self.game = game
        self.combination_generator = combination_generator
        self.combinations = combination_generator.get_all_combinations()
        self.alternating_updates = alternating_updates
        # Kleine Optimierung: wenn alle Reach-Probs 0 sind, breche die Rekursion ab
        # (kann optional aktiviert werden, um „pruning an/aus“ vergleichen zu können)
        self.partial_pruning = partial_pruning
        
        # CFR Datenstrukturen
        self.regret_sum = {}  # {info_set_key: {action: float}}
        self.strategy_sum = {}  # {info_set_key: {action: float}}
        self.iteration_count = 0
        self.training_time = 0
        # 1-indexierte Iteration (hilfreich für CFR+ / DCFR Ableitungen)
        self._current_iteration = 0
        
        # Cache für aktuelle Policy (wird nach jedem Update aktualisiert)
        self._policy_cache = {}  # {info_set_key: {action: prob}}
        
        # Tree Datenstrukturen
        # Hinweis: Nodes stammen aus `training.build_game_tree.Node`
        self.nodes = {}  # {node_id: Node}
        self.next_node_id = 0
        self.infoset_to_nodes = defaultdict(list)  # {infoset_key: [node_ids]}
        self.root_nodes = []  # Liste von root node IDs (eine pro Kombination)
        
        # Bestimme ob Suit Abstraction verwendet wird (basierend auf Combination Generator Typ)
        from utils.poker_utils import LeducHoldemCombinationsAbstracted, TwelveCardPokerCombinationsAbstracted
        use_suit_abstraction = isinstance(combination_generator, 
            (LeducHoldemCombinationsAbstracted, TwelveCardPokerCombinationsAbstracted))
        
        # Versuche Tree zu laden, sonst baue ihn
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
    
    def _convert_game_tree_to_internal(self, game_tree):
        """
        Übernimmt (adoptiert) ein GameTree Objekt als interne Struktur.

        WICHTIG: Wir erzeugen KEINE zweite Node-Welt mehr, sondern referenzieren
        direkt die Nodes aus `training.build_game_tree`.
        """
        # Optional: halte Referenz (hilfreich für Debugging / Ownership)
        self.game_tree = game_tree

        # Direkt übernehmen: keine Kopie, kein RAM-Peak durch doppelte Nodes
        self.nodes = game_tree.nodes
        self.infoset_to_nodes = game_tree.infoset_to_nodes
        self.root_nodes = game_tree.root_nodes

        self.next_node_id = (max(self.nodes.keys()) + 1) if self.nodes else 0
    
    def ensure_init(self, info_set_key, legal_actions):
        """Initialisiert regret_sum und strategy_sum für ein InfoSet"""
        if info_set_key not in self.regret_sum:
            self.regret_sum[info_set_key] = {a: 0.0 for a in legal_actions}
        if info_set_key not in self.strategy_sum:
            self.strategy_sum[info_set_key] = {a: 0.0 for a in legal_actions}
    
    def train(self, iterations, br_tracker=None, print_interval=100):
        """
        Training mit vorher gebautem Tree.
        
        Args:
            iterations: Anzahl der Training-Iterationen
            br_tracker: Optionaler BestResponseTracker für Best Response Evaluation
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
                # Zeit wird automatisch in evaluate_and_add berechnet wenn start_time gegeben
                br_tracker.evaluate_and_add(current_avg_strategy, i + 1, start_time=start_time)
                br_tracker.last_eval_iteration = i + 1
        
        # Finale Best Response Evaluation
        if br_tracker is not None:
            current_avg_strategy = self.get_average_strategy()
            # Zeit wird automatisch in evaluate_and_add berechnet wenn start_time gegeben
            br_tracker.evaluate_and_add(current_avg_strategy, iterations, start_time=start_time)
        
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
        Eine CFR Iteration.
        
        Wenn alternating_updates=True (Standard):
        1. Für alle Kombinationen: Spieler 0 traversieren, Regrets akkumulieren
        2. Policy aktualisieren
        3. Für alle Kombinationen: Spieler 1 traversieren, Regrets akkumulieren
        4. Policy aktualisieren
        
        Wenn alternating_updates=False (simultane Updates):
        1. Für alle Kombinationen: Beide Spieler gleichzeitig traversieren
        2. Policy aktualisieren
        """
        self._current_iteration = self.iteration_count + 1
        if self.alternating_updates:
            # Alternierende Updates (Standard)
            # Zuerst Spieler 0 für alle Kombinationen
            for root_id in self.root_nodes:
                reach_probs = np.array([1.0, 1.0], dtype=np.float64)
                self._traverse_for_player(root_id, reach_probs, player=0)
            
            # Hook für algorithmusspezifische Nachbearbeitung (z.B. CFR+ Clamp, DCFR Discounting)
            self.after_player_traversal(player=0)

            # Policy Update nach Spieler 0
            self._update_all_policies()
            
            # Dann Spieler 1 für alle Kombinationen (mit aktualisierter Policy von Spieler 0)
            for root_id in self.root_nodes:
                reach_probs = np.array([1.0, 1.0], dtype=np.float64)
                self._traverse_for_player(root_id, reach_probs, player=1)
            
            self.after_player_traversal(player=1)

            # Policy Update nach Spieler 1
            self._update_all_policies()
        else:
            # Simultane Updates (wie original CFR Paper)
            # Beide Spieler gleichzeitig für alle Kombinationen
            for root_id in self.root_nodes:
                reach_probs = np.array([1.0, 1.0], dtype=np.float64)
                # Traverse für beide Spieler mit derselben Policy
                self._traverse_for_player(root_id, reach_probs, player=0)
                self._traverse_for_player(root_id, reach_probs, player=1)
            
            self.after_simultaneous_traversal()

            # Policy Update nach beiden Spielern
            self._update_all_policies()

    def after_player_traversal(self, player: int):
        """
        Hook: wird nach der Traversierung aller Root-Nodes für einen Spieler aufgerufen,
        aber vor dem Policy-Update.

        Ableitungen können hier z.B.:
        - CFR+: negative Regrets clamped to 0
        - DCFR: Discounting auf Regrets anwenden
        """
        return

    def after_simultaneous_traversal(self):
        """
        Hook: wird nach simultanen Updates (beide Spieler traversiert) aufgerufen,
        aber vor dem Policy-Update.
        """
        return
    
    def _traverse_for_player(self, node_id, reach_probabilities, player):
        """
        Traversiert den Tree und berechnet Counterfactual Regret für einen Spieler.
        
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
        if self.partial_pruning and np.all(reach_probabilities[:2] == 0):
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
            child_utility = self._traverse_for_player(child_id, new_reach_probs, player)
            
            action_utilities[action] = child_utility
            state_value += action_prob * child_utility
        
        # Wenn wir nicht für den aktuellen Spieler updaten, nur Wert zurückgeben
        if current_player != player:
            return state_value
        
        # Spieler-Node: Regrets & Strategy-Sum updaten (über Hook-Methoden),
        # analog zur dynamischen CFRSolver-Implementierung.
        reach_prob = reach_probabilities[current_player]
        counterfactual_weight = reach_probabilities[1 - current_player]

        self.update_regrets(
            info_state,
            node.legal_actions,
            action_utilities,
            state_value,
            counterfactual_weight,
        )
        self.update_strategy_sum(
            info_state,
            node.legal_actions,
            policy,
            reach_prob,
        )
        
        return state_value
    
    def _update_all_policies(self):
        """
        Aktualisiert die Policy für alle InfoSets basierend auf aktuellen Regrets.
        
        Die Policy wird nach jedem Spieler-Update neu berechnet, damit alle Nodes
        in der nächsten Traversierung die aktualisierte Policy verwenden.
        """
        for info_state in self.regret_sum:
            # Hole legal_actions
            node_ids = self.infoset_to_nodes.get(info_state, [])
            if not node_ids:
                continue
            
            node = self.nodes[node_ids[0]]
            legal_actions = node.legal_actions
            
            # Berechne neue Policy mit Regret Matching
            policy = self._regret_matching(info_state, legal_actions)
            
            # Cache für schnelleren Zugriff
            self._policy_cache[info_state] = policy
    
    def _get_policy(self, info_state):
        """
        Gibt die aktuelle Policy für ein InfoSet zurück.
        
        Falls nicht im Cache, wird sie neu berechnet.
        """
        if info_state in self._policy_cache:
            return self._policy_cache[info_state]
        
        # Hole legal_actions
        node_ids = self.infoset_to_nodes.get(info_state, [])
        if not node_ids:
            return {}
        
        node = self.nodes[node_ids[0]]
        legal_actions = node.legal_actions
        
        # Berechne Policy neu
        policy = self._regret_matching(info_state, legal_actions)
        
        # Cache
        self._policy_cache[info_state] = policy
        
        return policy
    
    def _regret_matching(self, info_state, legal_actions):
        """
        Regret Matching: Berechnet Policy basierend auf positiven Regrets.
        
        Args:
            info_state: InfoSet Key
            legal_actions: Liste von legalen Aktionen
        
        Returns:
            {action: prob} Dictionary
        """
        regrets = self.regret_sum.get(info_state, {})
        
        # Berechne positive Regrets
        positive_regrets = {}
        total_positive = 0.0
        
        for action in legal_actions:
            regret = regrets.get(action, 0.0)
            positive_regret = max(0.0, regret)
            positive_regrets[action] = positive_regret
            total_positive += positive_regret
        
        # Normalisiere
        if total_positive > 0:
            return {action: positive_regrets[action] / total_positive 
                   for action in legal_actions}
        else:
            # Gleichverteilung wenn keine positiven Regrets
            uniform_prob = 1.0 / len(legal_actions)
            return {action: uniform_prob for action in legal_actions}
    
    def get_current_strategy(self, info_set_key, legal_actions):
        """
        Gibt die aktuelle Strategie für ein InfoSet zurück.
        
        Wird von der Basisklasse verwendet.
        """
        policy = self._get_policy(info_set_key)
        # Stelle sicher, dass alle legal_actions enthalten sind
        result = {}
        for action in legal_actions:
            result[action] = policy.get(action, 0.0)
        return result
    
    def update_regrets(self, info_set_key, legal_actions, action_utilities, current_utility, counterfactual_weight):
        for action in legal_actions:
            instantaneous_regret = counterfactual_weight * (action_utilities[action] - current_utility)
            self.regret_sum[info_set_key][action] += instantaneous_regret
    
    def update_strategy_sum(self, info_set_key, legal_actions, current_strategy, player_reach):
        for action in legal_actions:
            self.strategy_sum[info_set_key][action] += player_reach * current_strategy[action]
    
    def get_average_strategy(self):
        """
        Berechnet die durchschnittliche Strategie mit uniform averaging.
        
        Bei uniform averaging ist die durchschnittliche Strategie:
        average_policy[info_state][action] = strategy_sum[info_state][action] / sum(strategy_sum[info_state].values())
        """
        average_strategy = {}
        
        for info_state, policy_dict in self.strategy_sum.items():
            total = sum(policy_dict.values())
            
            if total == 0:
                # Gleichverteilung wenn keine Policy akkumuliert
                node_ids = self.infoset_to_nodes.get(info_state, [])
                if not node_ids:
                    continue
                node = self.nodes[node_ids[0]]
                num_actions = len(node.legal_actions)
                average_strategy[info_state] = {
                    action: 1.0 / num_actions for action in node.legal_actions
                }
            else:
                # Normalisiere
                average_strategy[info_state] = {
                    action: action_sum / total
                    for action, action_sum in policy_dict.items()
                }
        
        return average_strategy
    
    def save_gzip(self, filepath):
        """Speichert CFR Daten"""
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
        """Lädt CFR Daten"""
        with gzip.open(filepath, 'rb') as f:
            data = pkl.load(f)
        
        self.regret_sum = data['regret_sum']
        self.strategy_sum = data['strategy_sum']
        self.average_strategy = data.get('average_strategy', {})
        self.iteration_count = data.get('iteration_count', 0)
        self.training_time = data.get('training_time', 0)
        
        # Rebuild policy cache
        self._policy_cache = {}
        for info_state in self.regret_sum.keys():
            self._get_policy(info_state)
        
        print(f"Loaded from {filepath}")
