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
        self.type = None  # 'terminal', 'decision', 'chance'
        self.player = None  # 0/1, -1 for chance
        self.infoset_key = None
        self.legal_actions = []
        self.children = {}  # {action: child_node_id}
        self.chance_probs = None  # {outcome: prob} for chance nodes
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
        # With explicit chance nodes, we do not enumerate combinations up front.
        self.combinations = []
        self.num_combinations = 0
        
        # CFR Datenstrukturen
        self.regret_sum = {}
        self.strategy_sum = {}
        self.iteration_count = 0
        self.training_time = 0
        
        # Policy Cache (für schnelleren Zugriff)
        self._policy_cache = {}  # {info_set_key: {action: prob}}
        
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
                has_chance = any(getattr(n, "type", None) == "chance" for n in game_tree.nodes.values())
                if not has_chance:
                    raise FileNotFoundError("Legacy tree format detected (no chance nodes); rebuilding.")
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
            node.chance_probs = getattr(node_data, "chance_probs", None)
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
    
    def train(self, iterations, br_tracker=None, print_interval=100, stop_exploitability_mb=None):
        """
        Training mit External Sampling.
        
        iterations: Anzahl Iterationen
        br_tracker: Optional für Best Response Evaluation
        print_interval: Intervall für Print-Statements (Standard: 100)
        """
        start_time = time.time()
        stopped_early = False
        
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
                if (
                    stop_exploitability_mb is not None
                    and br_tracker.values
                    and float(br_tracker.values[-1][1]) < float(stop_exploitability_mb)
                ):
                    print(
                        f"Early stop: Exploitability {float(br_tracker.values[-1][1]):.6f} mb/g "
                        f"< {float(stop_exploitability_mb):.6f} mb/g (Iteration {i + 1})."
                    )
                    stopped_early = True
                    break
        
        # Finale Best Response Evaluation
        if br_tracker is not None and not stopped_early:
            current_avg_strategy = self.get_average_strategy()
            if br_tracker.last_eval_iteration != self.iteration_count:
                br_tracker.evaluate_and_add(current_avg_strategy, self.iteration_count, start_time=start_time)
        
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
        pro Iteration gesampelt (für beide Spieler gleich).
        
        WICHTIG: Bei External Sampling müssen alle Nodes im selben Informationsset
        die gleiche gesampelte Aktion verwenden (Perfect Recall).
        
        WICHTIG: Die Policy wird WÄHREND der Traversierung am Gegner-Node akkumuliert
        (wie in OpenSpiel), nicht danach.
        
        WICHTIG: Die Kombination (Chance) wird EINMAL pro Iteration gesampelt,
        nicht einmal pro Spieler! Beide Spieler müssen die gleiche Kombination sehen.
        """
        # KRITISCH: Kombination EINMAL pro Iteration sampeln (für beide Spieler gleich)
        # Bei External Sampling ist Chance ein "externer" Faktor, der für beide Spieler
        # gleich sein muss. Die Kombination entspricht einem Chance-Node-Outcome.
        sampled_root_id = self.root_nodes[0]
        
        # Policy-Cache für diese Iteration (wird pro InfoSet einmal berechnet)
        # WICHTIG: Die Policy ändert sich während einer Iteration nicht, da Regrets
        # erst am Ende aktualisiert werden. Daher können wir die Policy einmal pro InfoSet cachen.
        self._policy_cache_this_iteration = {}
        
        # Für jeden Spieler
        for player in range(2):
            # Dictionary für gesampelte Aktionen pro Informationsset (Perfect Recall)
            # Wird für jeden Spieler neu initialisiert (Gegner-Aktionen werden separat gesampelt)
            self._sampled_actions = {}
            
            # Set für Infosets, die bereits Policy-Updates bekommen haben
            # KRITISCH: Jedes Infoset darf nur EINMAL pro Spieler-Update akkumuliert werden!
            # Im Paper: "Due to perfect recall it can never visit more than one history
            # from the same information set during this traversal"
            # In unserem Tree-basierten Ansatz können mehrere Nodes zum selben Infoset gehören,
            # daher müssen wir sicherstellen, dass wir die Policy nur einmal akkumulieren.
            self._policy_updated_infosets = set()
            
            # Verwende die GLEICHE Kombination für beide Spieler
            # Starte mit opponent_reach = 1.0 (am Root ist noch keine Gegner-Aktion gesampelt)
            self._update_regrets(sampled_root_id, player, opponent_reach=1.0)
    
    def _update_regrets(self, node_id, player, opponent_reach=1.0):
        """
        UpdateRegrets für External Sampling.
        
        node_id: aktueller Node
        player: für welchen Spieler wir CFR machen (0 oder 1)
        opponent_reach: Reach Probability des Gegners bis zu diesem Node (wird rekursiv aktualisiert)
        
        WICHTIG: Bei External Sampling müssen alle Nodes im selben Informationsset
        die gleiche gesampelte Aktion verwenden (Perfect Recall).
        
        WICHTIG: Policy-Akkumulation erfolgt WÄHREND der Traversierung am Gegner-Node
        (wie in OpenSpiel), mit der Policy die VOR den Regret-Updates berechnet wurde.
        
        WICHTIG: Bei External Sampling ist q(z) = π_{-i}^σ(z), also die Reach Probability des Gegners.
        Die Gewichtung ist 1/q(z) = 1/π_{-i}^σ(z). Da wir nur eine Kombination sampeln,
        müssen wir zusätzlich mit num_combinations gewichten.
        """
        node = self.nodes[node_id]
        
        if node.type == 'terminal':
            return node.payoffs[player]

        if node.type == 'chance':
            # Sample chance outcome according to chance_probs (true chance distribution)
            probs = node.chance_probs or {}
            outcomes = list(node.legal_actions)
            if not outcomes:
                return 0.0
            weights = np.array([probs.get(o, 0.0) for o in outcomes], dtype=np.float64)
            s = weights.sum()
            if s <= 0:
                outcome = random.choice(outcomes)
            else:
                weights = weights / s
                outcome = outcomes[int(np.random.choice(len(outcomes), p=weights))]
            child_id = node.children[outcome]
            return self._update_regrets(child_id, player, opponent_reach=opponent_reach)
        
        current_player = node.player
        info_set_key = node.infoset_key
        
        self.ensure_init(info_set_key, node.legal_actions)
        
        # WICHTIG: Policy VOR Regret-Updates berechnen (wie in OpenSpiel)
        # Diese Policy wird während der Traversierung verwendet und am Gegner-Node akkumuliert
        current_policy = self._get_current_policy(info_set_key, node.legal_actions)
        
        value = 0.0
        child_values = {}
        
        if current_player != player:
            # Gegner-Node: eine Aktion sampeln
            # WICHTIG: Perfect Recall - verwende bereits gesampelte Aktion falls vorhanden
            if info_set_key not in self._sampled_actions:
                # Erste Begegnung mit diesem Informationsset: sampeln und speichern
                sampled_action = self._sample_action(current_policy, node.legal_actions)
                self._sampled_actions[info_set_key] = sampled_action
            else:
                # Wiederverwende die bereits gesampelte Aktion für dieses Informationsset
                sampled_action = self._sampled_actions[info_set_key]
            
            # Aktualisiere opponent_reach mit der Wahrscheinlichkeit der gesampelten Aktion
            sampled_action_prob = current_policy.get(sampled_action, 0.0)
            new_opponent_reach = opponent_reach * sampled_action_prob
            
            child_id = node.children[sampled_action]
            value = self._update_regrets(child_id, player, new_opponent_reach)
        else:
            # Eigener Node: alle Aktionen durchlaufen
            for action in node.legal_actions:
                child_id = node.children[action]
                child_value = self._update_regrets(child_id, player, opponent_reach)
                child_values[action] = child_value
                value += current_policy.get(action, 0.0) * child_value
        
        # Regret Updates nur am eigenen Node
        if current_player == player:
            # WICHTIG: Bei External Sampling ist die Gewichtung implizit durch das Sampling enthalten.
            # Im Paper (Formel 11): r̃(I, a) = (1 - σ(a|I)) * Σ_{z∈Q∩Z_I} u_i(z) * π_i^σ(z[I]a, z)
            # Es gibt KEINE explizite Gewichtung mit 1/q(z) in dieser Formel.
            # In OpenSpiel wird auch keine explizite Gewichtung verwendet (siehe external_sampling_mccfr.py Zeile 157-158).
            # Die Gewichtung ist implizit durch das Sampling der Chance und Gegner-Aktionen enthalten.
            for action in node.legal_actions:
                # External Sampling Regret-Update:
                # r(I, a) = (1 - σ(a|I)) * Σ u(z) * π_i^σ(z[I]a, z)
                # Vereinfacht: regret = child_value - value
                # Da wir alle eigenen Aktionen durchlaufen, ist value = Σ σ(a|I) * child_value
                regret = child_values[action] - value
                # KEINE explizite Gewichtung (wie in OpenSpiel)
                self.regret_sum[info_set_key][action] += regret
        
        # WICHTIG: Policy-Akkumulation WÄHREND der Traversierung am Gegner-Node
        # (wie in OpenSpiel: Simple Average)
        # Bei 2 Spielern: Gegner = (player + 1) % 2
        # KRITISCH: Jedes Infoset darf nur EINMAL pro Spieler-Update akkumuliert werden!
        # Im Paper: "Due to perfect recall it can never visit more than one history
        # from the same information set during this traversal"
        # In unserem Tree-basierten Ansatz können mehrere Nodes zum selben Infoset gehören,
        # daher müssen wir sicherstellen, dass wir die Policy nur einmal akkumulieren.
        opponent = (player + 1) % 2
        if current_player == opponent and info_set_key not in self._policy_updated_infosets:
            # WICHTIG: Bei External Sampling wird die Policy am Gegner-Node akkumuliert (Simple Average).
            # In OpenSpiel wird auch keine explizite Gewichtung verwendet (siehe external_sampling_mccfr.py Zeile 164-165).
            # Die Gewichtung ist implizit durch das Sampling enthalten.
            # Akkumuliere die Policy, die VOR den Regret-Updates berechnet wurde
            for action in node.legal_actions:
                action_prob = current_policy.get(action, 0.0)
                # KEINE explizite Gewichtung (wie in OpenSpiel)
                self.strategy_sum[info_set_key][action] += action_prob
            
            # Markiere als aktualisiert
            self._policy_updated_infosets.add(info_set_key)
        
        return value
    
    def _sample_action(self, policy, legal_actions):
        """
        Sample eine Aktion basierend auf Policy.
        
        Verwendet np.random.choice wie in OpenSpiel für Konsistenz.
        Falls Policy nicht normalisiert ist, wird normalisiert.
        Falls keine Wahrscheinlichkeiten vorhanden, gleichverteilung.
        """
        actions = list(legal_actions)
        probabilities = np.array([policy.get(action, 0.0) for action in actions], dtype=np.float64)
        
        # Normalisieren falls nötig (np.random.choice normalisiert auch, aber wir machen es explizit)
        total = probabilities.sum()
        if total > 0:
            probabilities = probabilities / total
        else:
            # Gleichverteilung wenn keine Wahrscheinlichkeiten
            probabilities = np.ones(len(actions), dtype=np.float64) / len(actions)
        
        # Verwende np.random.choice wie in OpenSpiel
        action_idx = np.random.choice(len(actions), p=probabilities)
        return actions[action_idx]
    
    def _get_current_policy(self, info_set_key, legal_actions):
        """
        Berechnet aktuelle Policy mit Regret Matching.
        
        PERFORMANCE-OPTIMIERUNG: Die Policy wird pro InfoSet pro Iteration nur einmal berechnet
        und gecacht, da sich die Regrets während einer Iteration nicht ändern.
        """
        # Prüfe ob Policy bereits für diese Iteration gecacht ist
        if info_set_key in self._policy_cache_this_iteration:
            return self._policy_cache_this_iteration[info_set_key]
        
        # Berechne Policy aus aktuellen Regrets
        regrets = self.regret_sum.get(info_set_key, {})
        
        positive_regrets = {}
        total_positive = 0.0
        
        for action in legal_actions:
            regret = regrets.get(action, 0.0)
            positive_regret = max(0.0, regret)
            positive_regrets[action] = positive_regret
            total_positive += positive_regret
        
        if total_positive > 0:
            policy = {action: positive_regrets[action] / total_positive 
                     for action in legal_actions}
        else:
            # Gleichverteilung wenn keine positiven Regrets
            uniform_prob = 1.0 / len(legal_actions)
            policy = {action: uniform_prob for action in legal_actions}
        
        # Cache die Policy für diese Iteration
        self._policy_cache_this_iteration[info_set_key] = policy
        
        return policy
    
    
    def get_current_strategy(self, info_set_key, legal_actions):
        """Gibt aktuelle Strategie zurück (für Basisklasse)"""
        return self._get_current_policy(info_set_key, legal_actions)
    
    def get_average_strategy(self):
        """
        Berechnet durchschnittliche Strategie.
        
        Bei External Sampling (Simple Average) einfach durch die Summe teilen.
        """
        average_strategy = {}
        
        for info_state, policy_dict in self.strategy_sum.items():
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
