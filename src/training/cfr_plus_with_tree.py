"""
CFR+ Solver mit vorher gebautem Game Tree.

CFR+ ist eine Verbesserung von CFR, die:
- Positive Regrets verwendet (Q-Werte statt regret_sum)
- Negative Regrets auf 0 klappt
- Iterations-basierte Gewichtung für die durchschnittliche Strategie verwendet
- Alternierende Updates durchführt (beide Spieler in einer Iteration, aber separat)

Basierend auf: Tammelin et al. (2015) - Solving Heads-Up Limit Texas Hold'em
"""

import gzip
import pickle as pkl

from training.cfr_solver_with_tree import CFRSolverWithTree


class CFRPlusWithTree(CFRSolverWithTree):
    """
    CFR+ Solver der den Game Tree einmal vorher baut.
    
    Unterschied zu CFRSolverWithTree:
    - Verwendet Q-Werte statt regret_sum (positive Regrets)
    - Klappt negative Regrets auf 0
    - Verwendet (iteration_count + 1) als Gewichtung für strategy_sum (1-indexiert)
    - Alternierende Updates: Beide Spieler werden in einer Iteration aktualisiert, aber separat
      (wie in OpenSpiel implementiert)
    """
    
    def __init__(self, game, combination_generator, game_name=None, load_tree=True):
        # Rufe Basisklassen-__init__ auf
        super().__init__(game, combination_generator, game_name=game_name, load_tree=load_tree)
        
        # CFR+ verwendet Q statt regret_sum
        self.Q = {}
    
    def ensure_init(self, info_set_key, legal_actions):
        """Initialisiert Q und Strategy Sum für ein InfoSet"""
        super().ensure_init(info_set_key, legal_actions)
        if info_set_key not in self.Q:
            self.Q[info_set_key] = {a: 0.0 for a in legal_actions}
    
    def cfr_iteration(self):
        """
        Eine CFR+ Iteration über den vorher gebauten Tree.
        
        Bei alternierenden Updates werden BEIDE Spieler in einer Iteration aktualisiert,
        aber separat (wie in OpenSpiel implementiert).
        
        WICHTIG: Bei alternating updates muss für ALLE Kombinationen zuerst Spieler 0,
        dann für ALLE Kombinationen Spieler 1 traversiert werden, damit die Strategie
        zwischen den Updates aktualisiert wird.
        """
        # Bei alternierenden Updates: Zuerst alle Kombinationen für Spieler 0
        for root_id in self.root_nodes:
            reach_probs = [1.0, 1.0]
            self.traverse_tree(root_id, 0, reach_probs)
        
        # Dann alle Kombinationen für Spieler 1 (mit aktualisierter Strategie von Spieler 0)
        for root_id in self.root_nodes:
            reach_probs = [1.0, 1.0]
            self.traverse_tree(root_id, 1, reach_probs)
    
    def update_regrets(self, info_set_key, legal_actions, action_utilities, current_utility, counterfactual_weight):
        """
        Aktualisiert Q-Werte (positive Regrets) für CFR+.
        
        CFR+ klappt negative Regrets auf 0, daher verwenden wir Q statt regret_sum.
        """
        for action in legal_actions:
            instantaneous_regret = counterfactual_weight * (action_utilities[action] - current_utility)
            # Klappe negative Werte auf 0 (CFR+ Eigenschaft)
            self.Q[info_set_key][action] = max(self.Q[info_set_key][action] + instantaneous_regret, 0)
    
    def update_strategy_sum(self, info_set_key, legal_actions, current_strategy, player_reach):
        """
        Aktualisiert Strategy Sum mit Iterations-basierter Gewichtung für CFR+.
        
        WICHTIG: Verwendet (iteration_count + 1), um 1-indexiert zu sein (konsistent mit Tensor Solver).
        """
        for action in legal_actions:
            # CFR+ verwendet (iteration_count + 1) als Gewichtung (1-indexiert)
            weight = self.iteration_count + 1
            self.strategy_sum[info_set_key][action] += weight * player_reach * current_strategy[action]
    
    def get_current_strategy(self, info_set_key, legal_actions):
        """
        Regret Matching basierend auf Q-Werten (positive Regrets).
        
        CFR+ verwendet Q statt regret_sum für die Strategieberechnung.
        """
        Q_values = {a: self.Q[info_set_key][a] for a in legal_actions}
        sum_Q = sum(Q_values.values())
        
        if sum_Q > 0:
            return {a: Q_values[a] / sum_Q for a in legal_actions}
        else:
            # Wenn keine positiven Q-Werte: Gleichverteilung
            return {a: 1.0 / len(legal_actions) for a in legal_actions}
    
    def save_gzip(self, filepath):
        """Speichert CFR+ Daten (Q statt regret_sum)"""
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
    
    def load_gzip(self, filepath):
        """Lädt CFR+ Daten (Q statt regret_sum)"""
        with gzip.open(filepath, 'rb') as f:
            data = pkl.load(f)
        
        self.Q = data['Q']
        self.strategy_sum = data['strategy_sum']
        self.average_strategy = data['average_strategy']
        self.iteration_count = data['iteration_count']
        self.training_time = data.get('training_time', 0)
        
        print(f"Loaded from {filepath}")
