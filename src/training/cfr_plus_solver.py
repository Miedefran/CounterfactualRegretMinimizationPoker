import gzip
import pickle as pkl
import numpy as np
from training.cfr_solver import CFRSolver

class CFRPlusSolver(CFRSolver):
    
    def __init__(self, game, combination_generator, alternating_updates=True, partial_pruning=False):
        super().__init__(
            game,
            combination_generator,
            alternating_updates=alternating_updates,
            partial_pruning=partial_pruning,
        )
        self.Q = {}
        # Aktuelle Iteration für linear averaging (1-indexiert)
        self._current_iteration = 0
    
    def ensure_init(self, info_set_key, legal_actions):
        super().ensure_init(info_set_key, legal_actions)
        if info_set_key not in self.Q:
            self.Q[info_set_key] = {a: 0.0 for a in legal_actions}
    
    def cfr_iteration(self):
        """
        Eine CFR+ Iteration.
        
        Macht alternierende Updates: erst Spieler 0, dann Spieler 1.
        Nach jedem Spieler werden negative Regrets auf 0 gesetzt und
        die Policy neu berechnet.
        """
        self._current_iteration = self.iteration_count + 1
        
        if self.alternating_updates:
            # Spieler 0 für alle Kombinationen
            for combination in self.combinations:
                self.combination_generator.setup_game_with_combination(self.game, combination)
                reach_probs = np.array([1.0, 1.0], dtype=np.float64)
                self.traverse_game_tree(0, reach_probs)
            
            # Negative Regrets auf 0 setzen und Policy updaten
            self._reset_negative_regrets()
            self._update_all_policies()
            
            # Spieler 1 für alle Kombinationen
            for combination in self.combinations:
                self.combination_generator.setup_game_with_combination(self.game, combination)
                reach_probs = np.array([1.0, 1.0], dtype=np.float64)
                self.traverse_game_tree(1, reach_probs)
            
            # Nochmal negative Regrets auf 0 setzen und Policy updaten
            self._reset_negative_regrets()
            self._update_all_policies()
        else:
            # Simultane Updates (wie original CFR Paper)
            for combination in self.combinations:
                self.combination_generator.setup_game_with_combination(self.game, combination)
                reach_probs = np.array([1.0, 1.0], dtype=np.float64)
                # Traverse für beide Spieler mit derselben Policy
                self.traverse_game_tree(0, reach_probs)
                self.traverse_game_tree(1, reach_probs)
            
            # Negative Regrets auf 0 setzen und Policy updaten
            self._reset_negative_regrets()
            self._update_all_policies()
    
    def update_regrets(self, info_set_key, legal_actions, action_utilities, current_utility, counterfactual_weight):
        """
        Aktualisiert Q-Werte für CFR+.
        
        Regrets werden akkumuliert (können negativ sein) und dann nach jedem
        Spieler-Update mit _reset_negative_regrets() auf 0 gesetzt.
        """
        for action in legal_actions:
            instantaneous_regret = counterfactual_weight * (action_utilities[action] - current_utility)
            # Akkumuliere Regret (kann negativ sein, wird später auf 0 gesetzt)
            self.Q[info_set_key][action] += instantaneous_regret
    
    def update_strategy_sum(self, info_set_key, legal_actions, current_strategy, player_reach):
        """
        Aktualisiert Strategy Sum mit Iterations-basierter Gewichtung für CFR+.
        
        Gemäß Paper: σ̄pT = 2/(T² + T) Σt=1^T tσpt
        Die Gewichtung t ist die Iterationsnummer (1-indexiert).
        """
        for action in legal_actions:
            # CFR+ verwendet die Iterationsnummer t (1-indexiert) als Gewichtung
            # _current_iteration ist bereits 1-indexiert
            weight = self._current_iteration
            self.strategy_sum[info_set_key][action] += weight * player_reach * current_strategy[action]
    
    def get_current_strategy(self, info_set_key, legal_actions):
        """
        Regret Matching basierend auf Q-Werten (positive Regrets).
        
        CFR+ verwendet Q statt regret_sum für die Strategieberechnung.
        Gemäß Paper: σt(a) = Qt-1(a) / Σb∈A Qt-1(b)
        
        Gibt die aktuelle Strategie für ein InfoSet zurück.
        Falls nicht im Cache, wird sie neu berechnet.
        """
        if info_set_key in self._policy_cache:
            policy = self._policy_cache[info_set_key]
            # Stelle sicher, dass alle legal_actions enthalten sind
            result = {}
            for action in legal_actions:
                result[action] = policy.get(action, 0.0)
            return result
        
        Q_values = {a: self.Q[info_set_key][a] for a in legal_actions}
        sum_Q = sum(Q_values.values())
        
        if sum_Q > 0:
            policy = {a: Q_values[a] / sum_Q for a in legal_actions}
        else:
            # Wenn keine positiven Q-Werte: Gleichverteilung
            policy = {a: 1.0 / len(legal_actions) for a in legal_actions}
        
        # Cache für schnelleren Zugriff
        self._policy_cache[info_set_key] = policy
        return policy
    
    def get_average_strategy(self):
        """
        Berechnet die linear gewichtete durchschnittliche Strategie für CFR+.
        
        Gemäß Paper (Theorem 3): σ̄pT = 2/(T² + T) Σt=1^T tσpt
        
        Da strategy_sum bereits mit Gewicht t akkumuliert wurde, normalisieren wir
        einfach durch die Summe (die reach-gewichtete Summe der Gewichte).
        """
        return self.average_from_strategy_sum(self.strategy_sum)
    
    def save_gzip(self, filepath):
        data = {
            'Q': self.Q,
            'strategy_sum': self.strategy_sum,
            'average_strategy': self.average_strategy,
            'iteration_count': self.iteration_count,
            'training_time': self.training_time
        }
        
        with gzip.open(filepath, 'wb') as f:
            pkl.dump(data, f)
        
        print(f"Saved to {filepath}")
    
    def _reset_negative_regrets(self):
        """Setzt alle negativen Regrets auf 0 (Regret Matching+)"""
        for info_state in self.Q:
            for action in self.Q[info_state]:
                if self.Q[info_state][action] < 0:
                    self.Q[info_state][action] = 0.0
    
    def _update_all_policies(self):
        """
        Aktualisiert alle Policies basierend auf aktuellen Q-Werten.
        
        Wird nach jedem Spieler-Update gemacht, damit die nächste
        Traversierung die neue Policy verwendet.
        """
        for info_set_key in self.Q:
            if info_set_key not in self.strategy_sum:
                continue
            
            legal_actions = list(self.strategy_sum[info_set_key].keys())
            if not legal_actions:
                continue
            
            # Berechne neue Policy mit Regret Matching
            policy = self._regret_matching_plus(info_set_key, legal_actions)
            
            # Cache für schnelleren Zugriff
            self._policy_cache[info_set_key] = policy
    
    def _regret_matching_plus(self, info_set_key, legal_actions):
        """
        Berechnet Policy mit Regret Matching basierend auf Q-Werten.
        
        Args:
            info_set_key: InfoSet Key
            legal_actions: Liste von legalen Aktionen
        
        Returns:
            {action: prob} Dictionary
        """
        Q_values = self.Q.get(info_set_key, {})
        
        total_Q = 0.0
        for action in legal_actions:
            total_Q += Q_values.get(action, 0.0)
        
        if total_Q > 0:
            return {action: Q_values.get(action, 0.0) / total_Q 
                   for action in legal_actions}
        else:
            # Gleichverteilung wenn keine positiven Q-Werte
            uniform_prob = 1.0 / len(legal_actions)
            return {action: uniform_prob for action in legal_actions}
    
    def load_gzip(self, filepath):
        with gzip.open(filepath, 'rb') as f:
            data = pkl.load(f)
        
        self.Q = data['Q']
        self.strategy_sum = data['strategy_sum']
        self.average_strategy = data['average_strategy']
        self.iteration_count = data['iteration_count']
        self.training_time = data.get('training_time', 0)
        
        # _current_iteration ist 1-indexiert, iteration_count ist 0-indexiert
        self._current_iteration = max(1, self.iteration_count + 1)
        
        # Rebuild policy cache
        self._policy_cache = {}
        for info_set_key in self.Q.keys():
            if info_set_key in self.strategy_sum:
                legal_actions = list(self.strategy_sum[info_set_key].keys())
                if legal_actions:
                    policy = self._regret_matching_plus(info_set_key, legal_actions)
                    self._policy_cache[info_set_key] = policy
        
        print(f"Loaded from {filepath}")
