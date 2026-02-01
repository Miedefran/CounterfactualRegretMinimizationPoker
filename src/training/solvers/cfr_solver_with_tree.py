"""CFR solver with pre-built game tree. Traverses a stored tree instead of stepping the game each iteration."""

import pickle as pkl
import gzip
import time
import numpy as np
from collections import defaultdict
from typing import Any

from training.build_game_tree import load_game_tree, build_game_tree, save_game_tree, GameTree
from training.registry import TrainingSolver
from training.config import TrainingConfig


class CFRSolverWithTree(TrainingSolver):
    """
    CFR with pre-built game tree.
    - Same as vanilla CFR but traverses a pre-built tree structure.
    - Alternating or simultaneous updates; hooks for subclasses.
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
        # With explicit chance nodes, we do not enumerate combinations upfront.
        self.combinations = []
        self.alternating_updates = alternating_updates
        #stop if reach probs of opponent are 0
        self.partial_pruning = partial_pruning

        # CFR data structures
        self.regret_sum = {}  # {info_set_key: {action: float}}
        self.strategy_sum = {}  # {info_set_key: {action: float}}
        self.iteration_count = 0
        self.training_time = 0
        self._current_iteration = 0  # 1-indexed for CFR+/DCFR

        # Cache for current policy (updated after each iteration)
        self._policy_cache = {}  # {info_set_key: {action: prob}}

        # Tree data structures; nodes from build_game_tree.Node
        self.nodes = {}  # {node_id: Node}
        self.next_node_id = 0
        self.infoset_to_nodes = defaultdict(list)  # {infoset_key: [node_ids]}
        self.root_nodes = []  # List of root node IDs (one per combination)

        # Check if suit abstraction is used
        from utils.poker_utils import LeducHoldemCombinationsAbstracted, TwelveCardPokerCombinationsAbstracted
        use_suit_abstraction = isinstance(combination_generator,
                                          (LeducHoldemCombinationsAbstracted, TwelveCardPokerCombinationsAbstracted))

        # Try to load tree, otherwise build
        if load_tree and game_name:
            try:
                print(f"Attempting to load game tree for {game_name}...")
                game_tree = load_game_tree(game_name, abstract_suits=use_suit_abstraction)
                # Detect legacy trees (built from enumerated combinations, no explicit chance nodes)
                has_chance = any(getattr(n, 'type', None) == 'chance' for n in game_tree.nodes.values())
                if not has_chance:
                    raise FileNotFoundError("Legacy tree format detected (no chance nodes); rebuilding.")
                self._convert_game_tree_to_internal(game_tree)
                print(f"Tree loaded: {len(self.nodes)} nodes, {len(self.infoset_to_nodes)} unique infosets")
            except FileNotFoundError:
                print(f"Tree file not found for {game_name}, building tree...")
                game_tree = build_game_tree(self.game, self.combination_generator, game_name=game_name,
                                            abstract_suits=use_suit_abstraction)
                self._convert_game_tree_to_internal(game_tree)
                save_game_tree(game_tree, game_name, abstract_suits=use_suit_abstraction)
                print(f"Tree built and saved: {len(self.nodes)} nodes, {len(self.infoset_to_nodes)} unique infosets")
        else:
            print("Building game tree...")
            game_tree = build_game_tree(self.game, self.combination_generator)
            self._convert_game_tree_to_internal(game_tree)
            print(f"Tree built: {len(self.nodes)} nodes, {len(self.infoset_to_nodes)} unique infosets")

    @staticmethod
    def evaluate_solver(config: TrainingConfig) -> bool:
        return config.algorithm == 'cfr_with_tree'

    @staticmethod
    def create_solver(config: TrainingConfig, game: Any, combo_gen: Any) -> 'TrainingSolver':
        return CFRSolverWithTree(
            game,
            combo_gen,
            game_name=config.game,
            alternating_updates=config.alternating_updates,
            partial_pruning=config.partial_pruning,
        )

    @staticmethod
    def supports_alternating_updates() -> bool:
        return True

    @staticmethod
    def supports_partial_pruning() -> bool:
        return True

    @staticmethod
    def supports_squared_weights() -> bool:
        return False

    def _convert_game_tree_to_internal(self, game_tree):
        """Adopt GameTree as internal structure (reference, no copy)."""
        self.game_tree = game_tree

        # Use reference, no copy
        self.nodes = game_tree.nodes
        self.infoset_to_nodes = game_tree.infoset_to_nodes
        self.root_nodes = game_tree.root_nodes

        self.next_node_id = (max(self.nodes.keys()) + 1) if self.nodes else 0

    def ensure_init(self, info_set_key, legal_actions):
        """Initialize regret_sum and strategy_sum for an infoset."""
        if info_set_key not in self.regret_sum:
            self.regret_sum[info_set_key] = {a: 0.0 for a in legal_actions}
        if info_set_key not in self.strategy_sum:
            self.strategy_sum[info_set_key] = {a: 0.0 for a in legal_actions}

    def train(self, iterations, br_tracker=None, print_interval=100, stop_exploitability_mb=None):
        """
        Train using pre-built tree.
        
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
                # Time is computed in evaluate_and_add when start_time is given
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

        # Final best response evaluation (if not already evaluated)
        if br_tracker is not None and not stopped_early:
            current_avg_strategy = self.get_average_strategy()
            # Time is computed in evaluate_and_add when start_time is given
            if br_tracker.last_eval_iteration != self.iteration_count:
                br_tracker.evaluate_and_add(current_avg_strategy, self.iteration_count, start_time=start_time)

        total_time = time.time() - start_time

        # Subtract best response time from training time
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
        """One CFR iteration. Alternating: traverse player 0, hook, policy update, then player 1; simultaneous: traverse both, hook, policy update."""
        self._current_iteration = self.iteration_count + 1
        if self.alternating_updates:
            # Alternating updates
            for root_id in self.root_nodes:
                reach_probs = np.array([1.0, 1.0], dtype=np.float64)
                self._traverse_for_player(root_id, reach_probs, player=0)

            # Hook for algorithm-specific post-processing (e.g., CFR+ clamp, DCFR discounting)
            self.after_player_traversal(player=0)

            # Policy update after player 0
            self._update_all_policies()

            for root_id in self.root_nodes:
                reach_probs = np.array([1.0, 1.0], dtype=np.float64)
                self._traverse_for_player(root_id, reach_probs, player=1)

            self.after_player_traversal(player=1)

            # Policy update after player 1
            self._update_all_policies()
        else:
            # Simultaneous updates (both players in one pass)
            for root_id in self.root_nodes:
                reach_probs = np.array([1.0, 1.0], dtype=np.float64)
                # Traverse for both players with same policy
                self._traverse_for_player(root_id, reach_probs, player=0)
                self._traverse_for_player(root_id, reach_probs, player=1)

            self.after_simultaneous_traversal()

            # Policy update after both players
            self._update_all_policies()

    def after_player_traversal(self, player: int):
        """Called after traversal for one player, before policy update. Override for CFR+ clamp or DCFR discounting."""
        return

    def after_simultaneous_traversal(self):
        """Called after simultaneous updates (both players traversed), before policy update."""
        return

    def _traverse_for_player(self, node_id, reach_probabilities, player, chance_reach: float = 1.0):
        """Traverse tree and compute counterfactual regret for one player. Args: node_id, reach_probabilities, player. Returns: utility for player."""
        node = self.nodes[node_id]

        # Terminal node: return payoff
        if node.type == 'terminal':
            return node.payoffs[player]

        # Chance node: expected value over outcomes (no updates)
        if node.type == 'chance':
            if not node.legal_actions:
                return 0.0
            probs = node.chance_probs or {}
            value = 0.0
            for outcome in node.legal_actions:
                prob = probs.get(outcome, 0.0)
                if prob == 0.0:
                    continue
                child_id = node.children[outcome]
                value += prob * self._traverse_for_player(child_id, reach_probabilities, player, chance_reach * prob)
            return value

        # Decision node
        current_player = node.player
        info_state = node.infoset_key

        # Early exit when reach probabilities are zero
        if self.partial_pruning and np.all(reach_probabilities[:2] == 0):
            return 0.0

        self.ensure_init(info_state, node.legal_actions)

        # Get current policy for this infoset
        policy = self._get_policy(info_state)

        # Compute utilities for all actions
        action_utilities = {}
        state_value = 0.0

        for action in node.legal_actions:
            action_prob = policy.get(action, 0.0)
            child_id = node.children[action]

            # New reach probabilities for this path
            new_reach_probs = reach_probabilities.copy()
            new_reach_probs[current_player] *= action_prob

            # Recurse
            child_utility = self._traverse_for_player(child_id, new_reach_probs, player, chance_reach)

            action_utilities[action] = child_utility
            state_value += action_prob * child_utility

        # If not updating for current player, return value only
        if current_player != player:
            return state_value

        # Player node: update regrets and strategy sum (CFR weight includes chance reach)
        reach_prob = reach_probabilities[current_player] * chance_reach
        counterfactual_weight = reach_probabilities[1 - current_player] * chance_reach

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
        """Refresh policy cache for all infosets (from _get_infosets_to_update via _compute_policy_for_infoset)."""
        for info_state in self._get_infosets_to_update():
            node_ids = self.infoset_to_nodes.get(info_state, [])
            if not node_ids:
                continue
            node = self.nodes[node_ids[0]]
            legal_actions = node.legal_actions
            self._policy_cache[info_state] = self._compute_policy_for_infoset(info_state, legal_actions)

    def _get_policy(self, info_state):
        if info_state in self._policy_cache:
            return self._policy_cache[info_state]
        node_ids = self.infoset_to_nodes.get(info_state, [])
        if not node_ids:
            return {}
        node = self.nodes[node_ids[0]]
        legal_actions = node.legal_actions
        policy = self._compute_policy_for_infoset(info_state, legal_actions)
        self._policy_cache[info_state] = policy
        return policy

    def get_current_strategy(self, info_set_key, legal_actions):
        """Current strategy for infoset (from cache or regret matching)."""
        policy = self._get_policy(info_set_key)
        # Ensure all legal_actions are included
        result = {}
        for action in legal_actions:
            result[action] = policy.get(action, 0.0)
        return result

    def _get_strategy_sum_weight(self):
        """Weight for strategy_sum (CFR: 1.0, CFR+: t/t², DCFR: t^gamma)."""
        return 1.0

    def _get_infosets_to_update(self):
        """Source of infosets for policy update (default: regret_sum)."""
        return self.regret_sum

    def _get_values_for_current_policy(self, info_state, legal_actions):
        """Values for regret matching (default: max(0, regret_sum))."""
        regrets = self.regret_sum.get(info_state, {})
        return {a: max(0.0, regrets.get(a, 0.0)) for a in legal_actions}

    def _compute_policy_for_infoset(self, info_state, legal_actions):
        """Compute policy from _get_values_for_current_policy (normalize or uniform)."""
        values = self._get_values_for_current_policy(info_state, legal_actions)
        total = sum(values.values())
        if total > 0:
            return {a: values[a] / total for a in legal_actions}
        return {a: 1.0 / len(legal_actions) for a in legal_actions}

    def update_regrets(self, info_set_key, legal_actions, action_utilities, current_utility, counterfactual_weight):
        for action in legal_actions:
            instantaneous_regret = counterfactual_weight * (action_utilities[action] - current_utility)
            self.regret_sum[info_set_key][action] += instantaneous_regret

    def update_strategy_sum(self, info_set_key, legal_actions, current_strategy, player_reach):
        weight = self._get_strategy_sum_weight()
        for action in legal_actions:
            self.strategy_sum[info_set_key][action] += weight * player_reach * current_strategy[action]

    def get_average_strategy(self):
        """Average strategy (normalized strategy_sum)."""
        average_strategy = {}

        for info_state, policy_dict in self.strategy_sum.items():
            total = sum(policy_dict.values())

            if total == 0:
                # Uniform if no strategy accumulated
                node_ids = self.infoset_to_nodes.get(info_state, [])
                if not node_ids:
                    continue
                node = self.nodes[node_ids[0]]
                num_actions = len(node.legal_actions)
                average_strategy[info_state] = {
                    action: 1.0 / num_actions for action in node.legal_actions
                }
            else:
                # Normalize
                average_strategy[info_state] = {
                    action: action_sum / total
                    for action, action_sum in policy_dict.items()
                }

        return average_strategy

    def save_gzip(self, filepath):
        """Save solver state to gzip file."""
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
        """Load solver state from gzip file."""
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
