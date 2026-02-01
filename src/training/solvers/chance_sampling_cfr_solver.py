"""Chance Sampling CFR (dynamic traversal). Samples chance outcomes; decision nodes expanded full-width."""

import random
import numpy as np
from typing import Any

from utils.data_models import KeyGenerator
from training.solvers.cfr_solver import CFRSolver
from training.registry import TrainingSolver
from training.config import TrainingConfig


class ChanceSamplingCFRSolver(CFRSolver):
    """
    Chance Sampling CFR (dynamic traversal).
    - Sample chance outcomes (one per chance_index, consistent).
    - Decision nodes: full-width expansion.
    """

    def __init__(self, game, combination_generator, alternating_updates=False, partial_pruning=False):
        super().__init__(
            game,
            combination_generator,
            alternating_updates=alternating_updates,
            partial_pruning=partial_pruning,
        )
        self._debug_last_sampled_chance_by_index = {}
        self._debug_last_chance_encounters = []

    @staticmethod
    def evaluate_solver(config: TrainingConfig) -> bool:
        return config.algorithm == 'chance_sampling'

    @staticmethod
    def create_solver(config: TrainingConfig, game: Any, combo_gen: Any) -> 'TrainingSolver':
        return ChanceSamplingCFRSolver(game, combo_gen)

    @staticmethod
    def supports_alternating_updates() -> bool:
        return False

    @staticmethod
    def supports_partial_pruning() -> bool:
        return False

    @staticmethod
    def supports_squared_weights() -> bool:
        return False

    def cfr_iteration(self):
        """One chance sampling CFR iteration: sample chance, expand decisions full-width."""
        self._current_iteration = self.iteration_count + 1
        sampled_chance_by_index = {}
        self._debug_last_sampled_chance_by_index = sampled_chance_by_index
        self._debug_last_chance_encounters = []

        reach_probs = np.array([1.0, 1.0], dtype=np.float64)

        self.game.reset(0)
        self._traverse_chance_sample(player_id=0, reach_probabilities=reach_probs,
                                     chance_index=0, sampled_chance_by_index=sampled_chance_by_index)
        self._update_all_policies()

        self.game.reset(0)
        self._traverse_chance_sample(player_id=1, reach_probabilities=reach_probs,
                                     chance_index=0, sampled_chance_by_index=sampled_chance_by_index)
        self._update_all_policies()

    def _sample_chance_outcome(self, outcomes_with_probs):
        """Sample a chance outcome according to outcomes_with_probs (dict outcome -> prob)."""
        if not outcomes_with_probs:
            return None, 0.0
        outcomes = list(outcomes_with_probs.keys())
        weights = [float(outcomes_with_probs.get(o, 0.0)) for o in outcomes]
        s = sum(weights)
        if s <= 0:
            o = random.choice(outcomes)
            return o, 1.0 / len(outcomes)
        r = random.random() * s
        acc = 0.0
        for o, w in zip(outcomes, weights):
            acc += w
            if r <= acc:
                return o, float(w) / s
        return outcomes[-1], float(weights[-1]) / s

    def _traverse_chance_sample(self, player_id, reach_probabilities, chance_index,
                                sampled_chance_by_index, chance_reach: float = 1.0):
        """Traverse with game API: sample chance, expand decisions full-width."""
        # Terminal node: return payoff
        if self.game.done:
            return self.game.get_payoff(player_id)

        # Chance node: sample outcome (consistent per chance_index)
        if hasattr(self.game, 'is_chance_node') and self.game.is_chance_node():
            outcomes_with_probs = self.game.get_chance_outcomes_with_probs()
            if not outcomes_with_probs:
                return 0.0
            # Get or sample chance outcome for this chance_index (consistent across traversals)
            outcome = sampled_chance_by_index.get(chance_index)
            if outcome is None or outcome not in outcomes_with_probs:
                outcome, _ = self._sample_chance_outcome(outcomes_with_probs)
                if outcome is None:
                    return 0.0
                sampled_chance_by_index[chance_index] = outcome
            prob = float(outcomes_with_probs.get(outcome, 0.0))
            if prob <= 0:
                return 0.0
            self.game.step(outcome)
            # Recurse with sampled outcome, update chance_reach
            value = self._traverse_chance_sample(
                player_id, reach_probabilities, chance_index + 1,sampled_chance_by_index, chance_reach * prob)
            self.game.step_back()
            return value

        # Decision node: expand all actions full-width
        current_player = self.game.current_player
        legal_actions = self.game.get_legal_actions()
        info_set_key = KeyGenerator.get_info_set_key(self.game, current_player)
        self.ensure_init(info_set_key, legal_actions)
        policy = self.get_current_strategy(info_set_key, legal_actions)

        # Compute utilities for all actions (full-width)
        action_utilities = {}
        state_value = 0.0
        for action in legal_actions:
            action_prob = policy.get(action, 0.0)
            self.game.step(action)
            new_reach = reach_probabilities.copy()
            new_reach[current_player] *= action_prob
            child_util = self._traverse_chance_sample(
                player_id, new_reach, chance_index, sampled_chance_by_index, chance_reach)
            self.game.step_back()
            action_utilities[action] = child_util
            state_value += action_prob * child_util

        # Opponent's node: return value only, no regret updates
        if current_player != player_id:
            return state_value

        # Player's node: update regrets and strategy sum (counterfactual reach includes chance_reach)
        counterfactual_reach = reach_probabilities[1 - player_id] * chance_reach
        reach_prob = reach_probabilities[player_id] * chance_reach
        for action in legal_actions:
            instantaneous_regret = counterfactual_reach * (action_utilities[action] - state_value)
            self.regret_sum[info_set_key][action] += instantaneous_regret
        for action, action_prob in policy.items():
            self.strategy_sum[info_set_key][action] += reach_prob * action_prob

        return state_value

    def update_regrets(self, info_set_key, legal_actions, action_utilities, current_utility, counterfactual_weight):
        """No-op: updates occur in _traverse_chance_sample."""
        pass

    def update_strategy_sum(self, info_set_key, legal_actions, current_strategy, player_reach):
        """No-op: updates occur in _traverse_chance_sample."""
        pass
