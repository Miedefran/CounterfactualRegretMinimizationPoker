"""
Discounted CFR (DCFR) ohne vorher gebautem Game Tree.

Discounted CFR unterscheidet sich von normalem CFR durch:
- Discounting von positiven Regrets: R_t = (R_{t-1} + r_t) * t^α / (t^α + 1)
- Discounting von negativen Regrets: R_t = (R_{t-1} + r_t) * t^β / (t^β + 1)
- Strategy Averaging (wie in OpenSpiel): Σ_t = Σ_{t-1} + π_t * t^γ

Implementierung folgt OpenSpiel:
- Regret wird erst addiert, dann wird Discounting angewendet
- Strategy Averaging multipliziert den neuen Beitrag mit t^γ

Referenz: Brown & Sandholm (2018) - "Solving Imperfect-Information Games via Discounted Regret Minimization"
Empfohlene Parameter: α=3/2, β=0, γ=2 (DCFR_{3/2,0,2})
"""

import pickle as pkl
import gzip
import time
import numpy as np

from utils.data_models import KeyGenerator
from training.cfr_solver import CFRSolver


class DiscountedCFRSolver(CFRSolver):
    """
    Discounted CFR Implementation ohne vorher gebautem Tree.
    
    Erbt von CFRSolver, überschreibt aber:
    - Discounted Regret Updates (mit alpha und beta)
    - Discounted Strategy Averaging (mit gamma)
    """
    
    def __init__(
        self,
        game,
        combination_generator,
        alternating_updates=True,
        partial_pruning=False,
        alpha=1.5,
        beta=0.0,
        gamma=2.0,
    ):
        """
        Initialisiert Discounted CFR Solver.
        
        Args:
            game: Das Spiel-Objekt
            combination_generator: Generator für Kartenkombinationen
            alternating_updates: Ob alternierende Updates verwendet werden sollen
            alpha: Discounting-Parameter für positive Regrets (Standard: 1.5)
            beta: Discounting-Parameter für negative Regrets (Standard: 0.0)
            gamma: Discounting-Parameter für durchschnittliche Strategie (Standard: 2.0)
        """
        super().__init__(
            game,
            combination_generator,
            alternating_updates=alternating_updates,
            partial_pruning=partial_pruning,
        )
        
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
        # Aktuelle Iteration (1-indexiert für Discounting-Formeln)
        self._current_iteration = 0
        
        print(f"Discounted CFR initialized with α={alpha}, β={beta}, γ={gamma}")
    
    def cfr_iteration(self):
        """
        Eine Discounted CFR Iteration.
        
        Wenn alternating_updates=True:
        1. Für alle Kombinationen: Spieler 0 traversieren, Regrets akkumulieren
        2. Discounting auf Regrets anwenden (für alle InfoStates von Spieler 0)
        3. Policy aktualisieren
        4. Für alle Kombinationen: Spieler 1 traversieren, Regrets akkumulieren
        5. Discounting auf Regrets anwenden (für alle InfoStates von Spieler 1)
        6. Policy aktualisieren
        
        Wenn alternating_updates=False:
        1. Für alle Kombinationen: Beide Spieler gleichzeitig traversieren
        2. Discounting auf Regrets anwenden (für alle InfoStates beider Spieler)
        3. Policy aktualisieren
        """
        self._current_iteration = self.iteration_count + 1
        
        if self.alternating_updates:
            # Alternierende Updates (Standard)
            # Zuerst Spieler 0 für alle Kombinationen
            for combination in self.combinations:
                self.combination_generator.setup_game_with_combination(self.game, combination)
                reach_probs = np.array([1.0, 1.0], dtype=np.float64)
                self.traverse_game_tree(0, reach_probs)
            
            # Discounting auf Regrets von Spieler 0 anwenden
            self._apply_regret_discounting(player=0)
            
            # Policy Update nach Spieler 0
            self._update_all_policies()
            
            # Dann Spieler 1 für alle Kombinationen (mit aktualisierter Policy von Spieler 0)
            for combination in self.combinations:
                self.combination_generator.setup_game_with_combination(self.game, combination)
                reach_probs = np.array([1.0, 1.0], dtype=np.float64)
                self.traverse_game_tree(1, reach_probs)
            
            # Discounting auf Regrets von Spieler 1 anwenden
            self._apply_regret_discounting(player=1)
            
            # Policy Update nach Spieler 1
            self._update_all_policies()
        else:
            # Simultane Updates (wie original CFR Paper)
            # Beide Spieler gleichzeitig für alle Kombinationen
            for combination in self.combinations:
                self.combination_generator.setup_game_with_combination(self.game, combination)
                reach_probs = np.array([1.0, 1.0], dtype=np.float64)
                # Traverse für beide Spieler mit derselben Policy
                self.traverse_game_tree(0, reach_probs)
                self.traverse_game_tree(1, reach_probs)
            
            # Discounting auf Regrets beider Spieler anwenden
            self._apply_regret_discounting(player=None)
            
            # Policy Update nach beiden Spielern
            self._update_all_policies()
    
    def update_strategy_sum(self, info_set_key, legal_actions, current_strategy, player_reach):
        """
        Aktualisiert Strategy Sum mit Discounting für Discounted CFR.
        
        Gemäß Paper: Σ_t = Σ_{t-1} + π_t * t^γ
        Die Gewichtung t^γ wird auf den neuen Beitrag angewendet.
        """
        t = self._current_iteration
        for action in legal_actions:
            # Strategy Averaging mit gamma-Discounting: reach_prob * action_prob * (iteration^gamma)
            self.strategy_sum[info_set_key][action] += (
                player_reach * current_strategy[action] * (t ** self.gamma)
            )
    
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
            # Extrahiere player_id aus dem InfoSet Key
            # Key Format: (private_card, public_cards, clean_history, player_id)
            if len(info_state) >= 4:
                info_set_player = info_state[3]
            else:
                # Fallback: Wenn Key-Format anders ist, überspringe
                continue
            
            # Prüfe ob dieses InfoSet zu dem Spieler gehört
            if player is not None and info_set_player != player:
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
        Berechnet die durchschnittliche Strategie.
        
        Bei Discounted CFR wird die durchschnittliche Strategie normalisiert
        über die kumulative Policy (die bereits mit gamma discounting gewichtet wurde).
        """
        return self.average_from_strategy_sum(self.strategy_sum)
    
    def save_gzip(self, filepath):
        """Speichert Discounted CFR Daten"""
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
        """Lädt Discounted CFR Daten"""
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
        
        # Rebuild policy cache
        self._policy_cache = {}
        for info_set_key in self.regret_sum.keys():
            if info_set_key in self.strategy_sum:
                legal_actions = list(self.strategy_sum[info_set_key].keys())
                if legal_actions:
                    self._get_policy(info_set_key, legal_actions)
        
        print(f"Loaded from {filepath}")
