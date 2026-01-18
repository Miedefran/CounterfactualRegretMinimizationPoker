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

from training.cfr_solver_with_tree import CFRSolverWithTree


class CFRPlusWithTree(CFRSolverWithTree):
    """
    CFR+ Implementation die den Tree vorher baut.
    
    Erbt von CFRSolverWithTree, überschreibt aber:
    - Regret Matching+ (negative regrets -> 0)
    - Linear averaging
    - Alternierende Updates
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
        super().__init__(
            game,
            combination_generator,
            game_name=game_name,
            load_tree=load_tree,
            alternating_updates=alternating_updates,
            partial_pruning=partial_pruning,
        )

    def after_player_traversal(self, player: int):
        # CFR+: negative Regrets nach jedem Spieler-Update auf 0 klappen
        self._reset_negative_regrets()

    def after_simultaneous_traversal(self):
        # CFR+: bei simultanen Updates einmal nach beiden Traversierungen clampen
        self._reset_negative_regrets()

    def update_strategy_sum(self, info_set_key, legal_actions, current_strategy, player_reach):
        """
        Linear averaging (CFR+): iteration * reach_prob * action_prob
        """
        t = self._current_iteration
        for action in legal_actions:
            self.strategy_sum[info_set_key][action] += t * player_reach * current_strategy[action]
    
    def _reset_negative_regrets(self):
        """Setzt alle negativen Regrets auf 0 (Regret Matching+)"""
        for info_state in self.regret_sum:
            for action in self.regret_sum[info_state]:
                if self.regret_sum[info_state][action] < 0:
                    self.regret_sum[info_state][action] = 0.0
    
    def get_average_strategy(self):
        """
        Berechnet durchschnittliche Strategie mit linear averaging.
        
        Bei linear averaging wird einfach durch die Summe geteilt.
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
        
        self.regret_sum = data['regret_sum']
        self.strategy_sum = data['strategy_sum']
        self.average_strategy = data.get('average_strategy', {})
        self._current_iteration = data.get('iteration_count', 0)
        self.training_time = data.get('training_time', 0)
        
        # iteration_count ist 0-indexiert, _current_iteration ist 1-indexiert
        self.iteration_count = max(0, self._current_iteration - 1)
        
        # Policy Cache neu aufbauen
        self._policy_cache = {}
        for info_state in self.regret_sum.keys():
            self._get_policy(info_state)
        
        print(f"Loaded from {filepath}")
