"""
Discounted CFR with Tree (DCFR) mit vorher gebautem Game Tree.

Hauptunterschiede zu normalem CFR:
- Discounting von positiven Regrets: R_t = (R_{t-1} + r_t) * t^α / (t^α + 1)
- Discounting von negativen Regrets: R_t = (R_{t-1} + r_t) * t^β / (t^β + 1)
- Strategy Averaging (wie in OpenSpiel): Σ_t = Σ_{t-1} + π_t * t^γ
- Alternierende Updates pro Iteration

Referenz: Brown & Sandholm (2018) - "Solving Imperfect-Information Games via Discounted Regret Minimization"
"""

import gzip
import pickle as pkl

from training.cfr_solver_with_tree import CFRSolverWithTree


class DiscountedCFRWithTreeSolver(CFRSolverWithTree):
    """
    Discounted CFR with Tree Implementation die den Tree vorher baut.
    
    Erbt von CFRSolverWithTree, überschreibt aber:
    - Discounted Regret Updates (mit alpha und beta)
    - Discounted Strategy Averaging (mit gamma)
    - Alternierende Updates
    """
    
    def __init__(
        self,
        game,
        combination_generator,
        game_name=None,
        load_tree=True,
        alpha=1.5,
        beta=0.0,
        gamma=2.0,
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
        
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        print(f"Discounted CFR with Tree initialized with α={alpha}, β={beta}, γ={gamma}")

    def after_player_traversal(self, player: int):
        # DCFR: Discounting nach jedem Spieler-Update (vor Policy-Update)
        self._apply_regret_discounting(player=player)

    def after_simultaneous_traversal(self):
        # DCFR: Discounting nach simultanen Updates für beide Spieler
        self._apply_regret_discounting(player=None)

    def update_strategy_sum(self, info_set_key, legal_actions, current_strategy, player_reach):
        """
        Strategy averaging mit gamma-Discounting (OpenSpiel): reach_prob * action_prob * t^gamma
        """
        t = self._current_iteration
        for action in legal_actions:
            self.strategy_sum[info_set_key][action] += player_reach * current_strategy[action] * (t ** self.gamma)
    
    def _apply_regret_discounting(self, player):
        """
        Wendet Discounting auf Regrets an (NACH der Traversierung).
        
        Wie in OpenSpiel: Erst wird der Regret addiert, dann wird das Discounting
        auf den gesamten kumulativen Regret angewendet.
        
        Args:
            player: Spieler für den Discounting angewendet wird (0, 1, oder None für beide)
        """
        t = self._current_iteration
        
        # Berechne Discount-Faktoren (wie in OpenSpiel)
        if self.alpha > 0:
            t_alpha = t ** self.alpha
            positive_discount = t_alpha / (t_alpha + 1)
        else:
            positive_discount = 1.0
        
        # Beta Handling: Bei beta = 0 wird t^0 / (t^0 + 1) = 1 / 2 verwendet
        # Bei beta > 0: t^beta / (t^beta + 1)
        if self.beta == 0:
            negative_discount = 0.5
        elif self.beta > 0:
            t_beta = t ** self.beta
            negative_discount = t_beta / (t_beta + 1)
        else:
            # Bei beta < 0 (z.B. -inf für CFR+): kein Discounting
            negative_discount = 1.0
        
        # Wende Discounting auf alle InfoStates an
        for info_state, regret_dict in self.regret_sum.items():
            # Prüfe ob dieses InfoSet zu dem Spieler gehört
            node_ids = self.infoset_to_nodes.get(info_state, [])
            if not node_ids:
                continue
            
            node = self.nodes[node_ids[0]]
            if player is not None and node.player != player:
                continue
            
            # Wende Discounting basierend auf Vorzeichen an
            for action in regret_dict.keys():
                current_regret = self.regret_sum[info_state][action]
                if current_regret >= 0:
                    # Positive Regret: wende alpha-Discounting an
                    self.regret_sum[info_state][action] *= positive_discount
                else:
                    # Negative Regret: wende beta-Discounting an
                    self.regret_sum[info_state][action] *= negative_discount
    
    def get_average_strategy(self):
        """
        Berechnet durchschnittliche Strategie mit Discounted Averaging.
        
        Bei Discounted CFR wird die durchschnittliche Strategie normalisiert
        über die kumulative Policy (die bereits mit gamma discounting gewichtet wurde).
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
            'training_time': self.training_time,
            'alpha': self.alpha,
            'beta': self.beta,
            'gamma': self.gamma
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
        self.alpha = data.get('alpha', 1.5)
        self.beta = data.get('beta', 0.0)
        self.gamma = data.get('gamma', 2.0)
        
        # _current_iteration ist 1-indexiert, iteration_count ist 0-indexiert
        self._current_iteration = max(1, self.iteration_count + 1)
        
        # Policy Cache neu aufbauen
        self._policy_cache = {}
        for info_state in self.regret_sum.keys():
            self._get_policy(info_state)
        
        print(f"Loaded from {filepath}")
