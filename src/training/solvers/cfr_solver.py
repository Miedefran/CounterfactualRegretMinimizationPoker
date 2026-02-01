import pickle as pkl
import gzip
import time
import numpy as np
from typing import Any

from utils.data_models import KeyGenerator
from training.registry import TrainingSolver
from training.config import TrainingConfig


class CFRSolver(TrainingSolver):
    """
    Vanilla CFR (dynamic traversal via game API).
    - Regret matching; uniform strategy averaging (weight 1.0).
    - Alternating or simultaneous updates; hooks after each pass for subclasses.
    """

    def __init__(self, game, combination_generator, alternating_updates=True, partial_pruning=False):
        self.game = game
        self.combination_generator = combination_generator

        self.combinations = []
        self.regret_sum = {}
        self.strategy_sum = {}
        self.iteration_count = 0
        self.training_time = 0
        self.alternating_updates = alternating_updates
        self.partial_pruning = partial_pruning
        self._current_iteration = 0
        self._policy_cache = {}

    @staticmethod
    def evaluate_solver(config: TrainingConfig) -> bool:
        return config.algorithm == 'cfr'

    @staticmethod
    def create_solver(config: TrainingConfig, game: Any, combo_gen: Any) -> 'TrainingSolver':
        return CFRSolver(game, combo_gen, alternating_updates=config.alternating_updates,
                         partial_pruning=config.partial_pruning)

    @staticmethod
    def supports_alternating_updates() -> bool:
        return True

    @staticmethod
    def supports_partial_pruning() -> bool:
        return True

    @staticmethod
    def supports_squared_weights() -> bool:
        return False

    def ensure_init(self, info_set_key, legal_actions):
        if info_set_key not in self.regret_sum:
            self.regret_sum[info_set_key] = {a: 0.0 for a in legal_actions}
        if info_set_key not in self.strategy_sum:
            self.strategy_sum[info_set_key] = {a: 0.0 for a in legal_actions}

    def train(self, iterations, br_tracker=None, print_interval=100, stop_exploitability_mb=None):
        """
        Train the CFR solver for a given number of iterations.
        
        Args:
            iterations: Number of training iterations
            br_tracker: Optional BestResponseTracker for best response evaluation
            print_interval: Interval for print statements (default: 100)
            stop_exploitability_mb: Early stop if exploitability drops below this value (millibets per game)
        """
        start_time = time.time()
        stopped_early = False

        for i in range(iterations):
            self.cfr_iteration()
            self.iteration_count += 1

            if (i + 1) % print_interval == 0:
                print(f"Iteration {i + 1}")

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

        if br_tracker is not None and not stopped_early:
            current_avg_strategy = self.get_average_strategy()
            if br_tracker.last_eval_iteration != self.iteration_count:
                br_tracker.evaluate_and_add(current_avg_strategy, self.iteration_count, start_time=start_time)

        total_time = time.time() - start_time

        if br_tracker is not None:
            br_time = br_tracker.get_total_br_time()
            self.training_time = total_time - br_time
            if br_time > 0:
                print(f"Best Response Evaluation time: {br_time:.2f}s")
        else:
            self.training_time = total_time

        if self.training_time >= 60:
            minutes = self.training_time / 60
            print(f"Training completed in {minutes:.2f} minutes (excluding best response evaluation)")
        else:
            print(f"Training completed in {self.training_time:.2f} seconds (excluding best response evaluation)")

        self.average_strategy = self.get_average_strategy()

    def cfr_iteration(self):
        """One CFR iteration: traverse for player(s), run hooks, update policies."""
        self._current_iteration = self.iteration_count + 1

        if self.alternating_updates:
            self.game.reset(0)
            reach_probs = np.array([1.0, 1.0], dtype=np.float64)
            self.traverse_game_tree(0, reach_probs)
            self.after_player_traversal(player=0)
            self._update_all_policies()

            self.game.reset(0)
            reach_probs = np.array([1.0, 1.0], dtype=np.float64)
            self.traverse_game_tree(1, reach_probs)
            self.after_player_traversal(player=1)
            self._update_all_policies()
        else:
            self.game.reset(0)
            reach_probs = np.array([1.0, 1.0], dtype=np.float64)
            self.traverse_game_tree(0, reach_probs)
            self.traverse_game_tree(1, reach_probs)
            self.after_simultaneous_traversal()
            self._update_all_policies()

    def _get_strategy_sum_weight(self):
        """Return weight for strategy sum (default 1.0)."""
        return 1.0

    def after_player_traversal(self, player: int):
        """Called after traversing one player."""
        return

    def after_simultaneous_traversal(self):
        """Called after simultaneous traversal."""
        return

    def traverse_game_tree(self, player_id, reach_probabilities, chance_reach: float = 1.0):
        """
        Traverse game tree and compute counterfactual regret for one player.
        
        Args:
            player_id: Player to compute CFR for (0 or 1)
            reach_probabilities: np.array([reach_p0, reach_p1])
            chance_reach: Cumulative chance reach probability
        
        Returns:
            Utility for player_id
        """
        if self.game.done:
            return self.game.get_payoff(player_id)

        # Chance node: expectation over chance outcomes (no regret updates)
        if hasattr(self.game, 'is_chance_node') and self.game.is_chance_node():
            outcomes_with_probs = self.game.get_chance_outcomes_with_probs()
            if not outcomes_with_probs:
                return 0.0
            value = 0.0
            for outcome, prob in outcomes_with_probs.items():
                if prob == 0:
                    continue
                self.game.step(outcome)
                value += prob * self.traverse_game_tree(player_id, reach_probabilities, chance_reach * prob)
                self.game.step_back()
            return value

        current_player = self.game.current_player

        # Opponent's node, don't update regrets
        if current_player != player_id:
            legal_actions = self.game.get_legal_actions()
            opponent = 1 - player_id
            opponent_info_set = KeyGenerator.get_info_set_key(self.game, opponent)
            self.ensure_init(opponent_info_set, legal_actions)
            opponent_strategy = self.get_current_strategy(opponent_info_set, legal_actions)

            state_value = 0.0
            for action in legal_actions:
                action_prob = opponent_strategy[action]
                self.game.step(action)

                new_reach_probs = reach_probabilities.copy()
                new_reach_probs[opponent] *= action_prob

                state_value += action_prob * self.traverse_game_tree(player_id, new_reach_probs, chance_reach)
                self.game.step_back()

            return state_value

        # Player's node, update regrets
        info_set_key = KeyGenerator.get_info_set_key(self.game, player_id)
        legal_actions = self.game.get_legal_actions()
        self.ensure_init(info_set_key, legal_actions)
        current_strategy = self.get_current_strategy(info_set_key, legal_actions)

        action_utilities = {}
        for action in legal_actions:
            self.game.step(action)

            new_reach_probs = reach_probabilities.copy()
            new_reach_probs[player_id] *= current_strategy[action]

            action_utilities[action] = self.traverse_game_tree(player_id, new_reach_probs, chance_reach)
            self.game.step_back()

        current_utility = sum(current_strategy[action] * action_utilities[action] for action in legal_actions)

        counterfactual_weight = reach_probabilities[1 - player_id] * chance_reach
        player_reach = reach_probabilities[player_id] * chance_reach

        self.update_regrets(info_set_key, legal_actions, action_utilities, current_utility, counterfactual_weight)
        self.update_strategy_sum(info_set_key, legal_actions, current_strategy, player_reach)

        return current_utility

    def update_regrets(self, info_set_key, legal_actions, action_utilities, current_utility, counterfactual_weight):
        for action in legal_actions:
            instantaneous_regret = counterfactual_weight * (action_utilities[action] - current_utility)
            self.regret_sum[info_set_key][action] += instantaneous_regret

    def update_strategy_sum(self, info_set_key, legal_actions, current_strategy, player_reach):
        weight = self._get_strategy_sum_weight()
        for action in legal_actions:
            self.strategy_sum[info_set_key][action] += weight * player_reach * current_strategy[action]

    def _get_infosets_to_update(self):
        """Return dict of infosets used for policy update."""
        return self.regret_sum

    def _get_values_for_current_policy(self, info_set_key, legal_actions):
        """Return action values used for regret matching (e.g. max(0, regret))."""
        regrets = self.regret_sum.get(info_set_key, {})
        return {a: max(0.0, regrets.get(a, 0.0)) for a in legal_actions}

    def _compute_policy_for_infoset(self, info_set_key, legal_actions):
        """Compute policy from values (normalize or uniform)."""
        values = self._get_values_for_current_policy(info_set_key, legal_actions)
        total = sum(values.values())
        if total > 0:
            return {a: values[a] / total for a in legal_actions}
        return {a: 1.0 / len(legal_actions) for a in legal_actions}

    def get_current_strategy(self, info_set_key, legal_actions):
        """Current strategy for infoset (from cache or regret matching)."""
        if info_set_key in self._policy_cache:
            policy = self._policy_cache[info_set_key]
            return {a: policy.get(a, 0.0) for a in legal_actions}
        policy = self._compute_policy_for_infoset(info_set_key, legal_actions)
        self._policy_cache[info_set_key] = policy
        return policy

    def _update_all_policies(self):
        """Refresh policy cache for all infosets."""
        for info_set_key in self._get_infosets_to_update():
            if info_set_key not in self.strategy_sum:
                continue
            legal_actions = list(self.strategy_sum[info_set_key].keys())
            if not legal_actions:
                continue
            self._policy_cache[info_set_key] = self._compute_policy_for_infoset(info_set_key, legal_actions)

    @staticmethod
    def average_from_strategy_sum(strategy_sum):
        average_strategy = {}

        for info_set_key in strategy_sum:
            total = sum(strategy_sum[info_set_key].values())
            if total > 0:
                average_strategy[info_set_key] = {
                    action: strategy_sum[info_set_key][action] / total
                    for action in strategy_sum[info_set_key]
                }
            else:
                num_actions = len(strategy_sum[info_set_key])
                average_strategy[info_set_key] = {
                    action: 1.0 / num_actions
                    for action in strategy_sum[info_set_key]
                }

        return average_strategy

    def get_average_strategy(self):
        """Average strategy (normalized strategy_sum)."""
        return self.average_from_strategy_sum(self.strategy_sum)

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

        # Rebuild policy cache
        self._policy_cache = {}
        for info_set_key in self.regret_sum.keys():
            if info_set_key in self.strategy_sum:
                legal_actions = list(self.strategy_sum[info_set_key].keys())
                if legal_actions:
                    self._get_policy(info_set_key, legal_actions)

        print(f"Loaded from {filepath}")

    def _get_policy(self, info_set_key, legal_actions):
        """Return current policy for infoset (from cache or computed)."""
        if info_set_key in self._policy_cache:
            return self._policy_cache[info_set_key]

        policy = self._compute_policy_for_infoset(info_set_key, legal_actions)
        self._policy_cache[info_set_key] = policy
        return policy
