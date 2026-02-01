from __future__ import annotations

import gzip
import os
import pickle as pkl
import time
from typing import Optional, Any

import numpy as np

from training.flat_game_tree import (
    FlatTree,
    ACTION_ORDER,
    NUM_ACTIONS,
    FLOAT_DTYPE,
    determine_use_suit_abstraction,
    get_flat_tree_cache_path,
    load_flat_tree_cache,
    save_flat_tree_cache,
    build_flat_tree_directly_from_game,
)
from training.registry import TrainingSolver
from training.config import TrainingConfig


class CFRSolverWithFlatTree(TrainingSolver):
    """
    CFR Solver on a flattened tree (numpy arrays).
    """

    def __init__(
            self,
            game,
            combination_generator,
            game_name: Optional[str] = None,
            load_tree: bool = True,
            alternating_updates: bool = True,
            partial_pruning: bool = False,
            validate_flat_tree: bool = False,
    ):
        self.game = game
        self.combination_generator = combination_generator
        self.alternating_updates = alternating_updates
        self.partial_pruning = partial_pruning

        self.iteration_count = 0
        self.training_time = 0.0
        self._current_iteration = 0  # 1-indexed

        use_suit_abstraction = determine_use_suit_abstraction(game, combination_generator)

        # Fast path: load cached flat-tree if available.
        self.flat = None
        if load_tree and game_name:
            flat_path = get_flat_tree_cache_path(game_name, abstract_suits=use_suit_abstraction)
            if os.path.exists(flat_path):
                try:
                    print(f"Loading flat tree cache for {game_name}...")
                    print(f"  Flat tree cache path: {flat_path}")
                    self.flat = load_flat_tree_cache(flat_path)
                    if validate_flat_tree:
                        self._validate_flat_tree_sampled(self.flat)
                    print(
                        f"Flat tree cache loaded: {len(self.flat.node_type)} nodes, "
                        f"{len(self.flat.infoset_id_to_key)} infosets"
                    )
                except Exception as e:
                    print(f"Failed to load flat tree cache (will rebuild): {e}")
                    self.flat = None

        if self.flat is None:
            # Build tree directly from game API (no Python object tree in RAM)
            print("Building flat tree directly from game environment...")
            t0 = time.time()
            self.flat = build_flat_tree_directly_from_game(game)
            print(
                f"Flat tree ready: {len(self.flat.node_type)} nodes, "
                f"{len(self.flat.infoset_id_to_key)} infosets ({time.time() - t0:.3f}s)"
            )
            if validate_flat_tree:
                self._validate_flat_tree_sampled(self.flat)

            # If we have a game_name, also write the flat-tree cache for next time.
            if load_tree and game_name:
                flat_path = get_flat_tree_cache_path(game_name, abstract_suits=use_suit_abstraction)
                try:
                    save_flat_tree_cache(self.flat, flat_path)
                    print(f"Flat tree cache saved to: {flat_path}")
                except Exception as e:
                    print(f"Warning: failed to save flat tree cache: {e}")

        # CFR arrays (float32 for reduced memory bandwidth)
        self.num_infosets = len(self.flat.infoset_id_to_key)
        self.regret_sum = np.zeros((self.num_infosets, NUM_ACTIONS), dtype=FLOAT_DTYPE)
        self.strategy_sum = np.zeros((self.num_infosets, NUM_ACTIONS), dtype=FLOAT_DTYPE)

        # Precompute uniform strategy per infoset (over valid actions)
        valid = self.flat.infoset_valid_actions.astype(FLOAT_DTYPE)
        counts = np.maximum(valid.sum(axis=1, keepdims=True), FLOAT_DTYPE(1.0))
        self._uniform_strategy = valid / counts

        # Work buffers
        self._reach = np.zeros((len(self.flat.node_type), 2), dtype=FLOAT_DTYPE)
        self._values = np.zeros((len(self.flat.node_type), 2), dtype=FLOAT_DTYPE)

        self.average_strategy = {}

    # --- Registry/Factory Methods ---
    @staticmethod
    def evaluate_solver(config: TrainingConfig) -> bool:
        return config.algorithm == 'cfr_with_flat_tree'

    @staticmethod
    def create_solver(config: TrainingConfig, game: Any, combo_gen: Any) -> 'TrainingSolver':
        return CFRSolverWithFlatTree(
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

    # --- Public API ---
    def train(self, iterations, br_tracker=None, print_interval=100, stop_exploitability_mb=None):
        start_time = time.time()
        stopped_early = False
        for i in range(iterations):
            self.cfr_iteration()

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

        # Final best response evaluation (if not already evaluated last)
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
            print(f"Training completed in {self.training_time / 60:.2f} minutes (excluding best response evaluation)")
        else:
            print(f"Training completed in {self.training_time:.2f} seconds (excluding best response evaluation)")

        self.average_strategy = self.get_average_strategy()

    def cfr_iteration(self):
        """One CFR iteration: compute strategy, forward pass, backward pass, update policies."""
        self._current_iteration = self.iteration_count + 1
        sigma = self._compute_current_strategy()

        # Forward reach under current sigma
        self._forward_reach(sigma)

        if self.alternating_updates:
            # Player 0 pass
            self._backward_values_and_updates(
                sigma=sigma,
                updating_player=0,
                strategy_sum_weight=self._get_strategy_sum_weight(),
            )
            self.after_player_traversal(player=0)
            sigma = self._compute_current_strategy()
            self._forward_reach(sigma)

            # Player 1 pass (with updated sigma)
            self._backward_values_and_updates(
                sigma=sigma,
                updating_player=1,
                strategy_sum_weight=self._get_strategy_sum_weight(),
            )
            self.after_player_traversal(player=1)
        else:
            # Simultaneous: one backward pass for both players
            self._backward_values_and_updates(
                sigma=sigma,
                updating_player=None,
                strategy_sum_weight=self._get_strategy_sum_weight(),
            )
            self.after_simultaneous_traversal()

        self.iteration_count += 1

    # --- Strategy Computation ---
    def _compute_current_strategy(self) -> np.ndarray:
        """
        Regret matching over positive regrets, restricted to valid actions.
        Returns:
          sigma: float32 array [num_infosets, 4]
        """
        # Keep everything in float32 to avoid upcasting.
        pos = np.maximum(self.regret_sum, FLOAT_DTYPE(0.0)) * self.flat.infoset_valid_actions
        s = pos.sum(axis=1, keepdims=True)
        sigma = np.empty_like(pos)
        has_pos = (s[:, 0] > 1e-15)
        sigma[has_pos] = pos[has_pos] / s[has_pos]
        sigma[~has_pos] = self._uniform_strategy[~has_pos]
        return sigma

    def _get_strategy_sum_weight(self) -> float:
        """Return weight for strategy sum (default 1.0 for uniform averaging)."""
        return 1.0

    # --- Forward Pass (Top-Down Reach Propagation) ---
    def _forward_reach(self, sigma: np.ndarray):
        """
        Forward Pass (top-down):
        - Input: sigma[infoset, action]
        - Output: reach[node, player] for both players (chance reach is included automatically).
        """
        ft = self.flat
        self._reach.fill(FLOAT_DTYPE(0.0))
        self._reach[ft.root, 0] = FLOAT_DTYPE(1.0)
        self._reach[ft.root, 1] = FLOAT_DTYPE(1.0)

        # Per depth: use precomputed lists (layer_*_nodes)
        for depth in range(ft.max_depth + 1):
            if self.partial_pruning:
                # Still compute pruned subsets per type, but skip scanning all nodes by type.
                self._forward_process_chance_nodes(ft.layer_chance_nodes[depth], prune=True)
                self._forward_process_decision_nodes(ft.layer_decision_nodes[depth], sigma, prune=True)
            else:
                self._forward_process_chance_nodes(ft.layer_chance_nodes[depth], prune=False)
                self._forward_process_decision_nodes(ft.layer_decision_nodes[depth], sigma, prune=False)

    def _forward_process_chance_nodes(self, chance_nodes: np.ndarray, prune: bool):
        """
        Forward reach propagation for all chance nodes in `layer`.

        For each chance node n and each outcome edge (n -> c) with probability p:
          reach[c,0] += reach[n,0] * p
          reach[c,1] += reach[n,1] * p
        """
        ft = self.flat
        if chance_nodes.size == 0:
            return
        if prune:
            # Skip chance nodes with reach == 0 for both players.
            chance_nodes = chance_nodes[(self._reach[chance_nodes, 0] + self._reach[chance_nodes, 1]) > 0]
            if chance_nodes.size == 0:
                return
        for n in chance_nodes:
            s = int(ft.chance_offsets[n])
            e = int(ft.chance_offsets[n + 1])
            if e <= s:
                continue
            kids = ft.chance_children[s:e]
            probs = ft.chance_probs[s:e]
            r0, r1 = self._reach[n, 0], self._reach[n, 1]
            self._reach[kids, 0] += r0 * probs
            self._reach[kids, 1] += r1 * probs

    def _forward_process_decision_nodes(self, dec_nodes: np.ndarray, sigma: np.ndarray, prune: bool):
        """
        Forward reach propagation for all decision nodes in `layer`.

        Decision node semantics:
        - active player's reach is multiplied by sigma(action)
        - inactive player's reach is copied unchanged
        """
        ft = self.flat
        if dec_nodes.size == 0:
            return
        if prune:
            dec_nodes = dec_nodes[(self._reach[dec_nodes, 0] + self._reach[dec_nodes, 1]) > 0]
            if dec_nodes.size == 0:
                return

        node_players = ft.player[dec_nodes].astype(np.int8)
        node_infosets = ft.infoset_id[dec_nodes].astype(np.int32)
        parent_reach = self._reach[dec_nodes]
        node_strategy = sigma[node_infosets]
        kids = ft.children[dec_nodes]

        for a in range(NUM_ACTIONS):
            child_a = kids[:, a]
            valid = (child_a != -1)
            if not np.any(valid):
                continue

            m0 = valid & (node_players == 0)
            if np.any(m0):
                c = child_a[m0]
                self._reach[c, 0] += parent_reach[m0, 0] * node_strategy[m0, a]
                self._reach[c, 1] += parent_reach[m0, 1]

            m1 = valid & (node_players == 1)
            if np.any(m1):
                c = child_a[m1]
                self._reach[c, 1] += parent_reach[m1, 1] * node_strategy[m1, a]
                self._reach[c, 0] += parent_reach[m1, 0]

    # --- Backward Pass (Bottom-Up Value Computation & Updates) ---
    def _backward_values_and_updates(
            self,
            sigma: np.ndarray,
            updating_player: Optional[int],
            strategy_sum_weight: float,
    ):
        """
        Backward Pass (bottom-up):
        - Input: sigma[infoset, action], reach[node, player]
        - Computes: values[node, player]
        - Updates: regret_sum and strategy_sum.

        If `updating_player` is 0/1: updates regrets and strategy_sum only for that player (alternating pass).
        If `updating_player` is None: simultaneous update for both players (updates both).
        """
        ft = self.flat
        self._values.fill(FLOAT_DTYPE(0.0))
        delta_regret = np.zeros_like(self.regret_sum)

        # Bottom-up over precomputed layers
        for depth in range(ft.max_depth, -1, -1):
            self._backward_set_terminal_values(ft.layer_terminal_nodes[depth])
            self._backward_set_chance_values(ft.layer_chance_nodes[depth])
            self._backward_process_decision_layer(
                dec_nodes=ft.layer_decision_nodes[depth],
                sigma=sigma,
                updating_player=updating_player,
                strategy_sum_weight=strategy_sum_weight,
                delta_regret=delta_regret,
            )

        self.regret_sum += delta_regret

    def _backward_set_terminal_values(self, term_nodes: np.ndarray):
        """Set terminal node values from payoffs."""
        if term_nodes.size > 0:
            self._values[term_nodes] = self.flat.payoffs[term_nodes]

    def _backward_set_chance_values(self, chance_nodes: np.ndarray):
        """
        Chance nodes are evaluated as expectation over their children:
          values[n] = Σ_o P(o) * values[child(o)]
        """
        ft = self.flat
        if chance_nodes.size == 0:
            return
        for n in chance_nodes:
            s = int(ft.chance_offsets[n])
            e = int(ft.chance_offsets[n + 1])
            if e <= s:
                continue
            kids = ft.chance_children[s:e]
            probs = ft.chance_probs[s:e]
            self._values[n] = (self._values[kids] * probs[:, None]).sum(axis=0)

    def _backward_process_decision_layer(
            self,
            dec_nodes: np.ndarray,
            sigma: np.ndarray,
            updating_player: Optional[int],
            strategy_sum_weight: float,
            delta_regret: np.ndarray,
    ):
        """Process decision nodes: compute values and accumulate updates."""
        ft = self.flat
        if dec_nodes.size == 0:
            return

        node_players = ft.player[dec_nodes].astype(np.int8)
        node_infosets = ft.infoset_id[dec_nodes].astype(np.int32)
        node_children = ft.children[dec_nodes]

        valid_actions = (node_children != -1)
        safe_children = np.where(valid_actions, node_children, 0)
        child_vals = self._values[safe_children] * valid_actions[:, :, None]

        node_strategy = sigma[node_infosets]
        node_ev = (child_vals * node_strategy[:, :, None]).sum(axis=1)
        self._values[dec_nodes] = node_ev

        self._accumulate_strategy_sum_for_decisions(
            dec_nodes=dec_nodes,
            node_players=node_players,
            node_infosets=node_infosets,
            node_strategy=node_strategy,
            valid_actions=valid_actions,
            updating_player=updating_player,
            strategy_sum_weight=strategy_sum_weight,
        )

        self._accumulate_regrets_for_decisions(
            dec_nodes=dec_nodes,
            node_players=node_players,
            node_infosets=node_infosets,
            node_ev=node_ev,
            child_vals=child_vals,
            valid_actions=valid_actions,
            updating_player=updating_player,
            delta_regret=delta_regret,
        )

    def _accumulate_strategy_sum_for_decisions(
            self,
            dec_nodes: np.ndarray,
            node_players: np.ndarray,
            node_infosets: np.ndarray,
            node_strategy: np.ndarray,
            valid_actions: np.ndarray,
            updating_player: Optional[int],
            strategy_sum_weight: float,
    ):
        """Accumulate reach * sigma (with weight) into strategy_sum per infoset."""
        if updating_player is None:
            reach_p = self._reach[dec_nodes, node_players]
            contrib = node_strategy * reach_p[:, None] * strategy_sum_weight
            contrib *= valid_actions
            np.add.at(self.strategy_sum, node_infosets, contrib)
            return

        up_mask = (node_players == updating_player)
        if not np.any(up_mask):
            return
        nodes_u = dec_nodes[up_mask]
        infosets_u = node_infosets[up_mask]
        reach_p = self._reach[nodes_u, updating_player]
        contrib = node_strategy[up_mask] * reach_p[:, None] * strategy_sum_weight
        contrib *= valid_actions[up_mask]
        np.add.at(self.strategy_sum, infosets_u, contrib)

    def _accumulate_regrets_for_decisions(
            self,
            dec_nodes: np.ndarray,
            node_players: np.ndarray,
            node_infosets: np.ndarray,
            node_ev: np.ndarray,
            child_vals: np.ndarray,
            valid_actions: np.ndarray,
            updating_player: Optional[int],
            delta_regret: np.ndarray,
    ):
        """
        Accumulate instantaneous regrets: opp_reach * (q(a) - v)
        where opp_reach = opponent reach, q(a) = child value, v = EV at node.
        """
        if updating_player is None:
            m0 = (node_players == 0)
            if np.any(m0):
                opp_reach = self._reach[dec_nodes[m0], 1]
                v = node_ev[m0, 0]
                q = child_vals[m0, :, 0]
                inst = opp_reach[:, None] * (q - v[:, None])
                inst *= valid_actions[m0]
                np.add.at(delta_regret, node_infosets[m0], inst)

            m1 = (node_players == 1)
            if np.any(m1):
                opp_reach = self._reach[dec_nodes[m1], 0]
                v = node_ev[m1, 1]
                q = child_vals[m1, :, 1]
                inst = opp_reach[:, None] * (q - v[:, None])
                inst *= valid_actions[m1]
                np.add.at(delta_regret, node_infosets[m1], inst)
            return

        up_mask = (node_players == updating_player)
        if not np.any(up_mask):
            return
        nodes_u = dec_nodes[up_mask]
        infosets_u = node_infosets[up_mask]
        opp = 1 - updating_player
        opp_reach = self._reach[nodes_u, opp]
        v = node_ev[up_mask, updating_player]
        q = child_vals[up_mask, :, updating_player]
        inst = opp_reach[:, None] * (q - v[:, None])
        inst *= valid_actions[up_mask]
        np.add.at(delta_regret, infosets_u, inst)

    # --- Hooks (for subclasses) ---
    def after_player_traversal(self, player: int):
        """Called after traversing one player."""
        return

    def after_simultaneous_traversal(self):
        """Called after simultaneous traversal."""
        return

    # --- Strategy Extraction ---
    def get_average_strategy(self):
        """Extract average strategy from strategy_sum."""
        avg_strategy = {}
        ss = self.strategy_sum
        valid = self.flat.infoset_valid_actions

        for iid, key in enumerate(self.flat.infoset_id_to_key):
            row = ss[iid].copy()
            row *= valid[iid]
            total = float(row.sum())

            if total > 0:
                probs = row / total
            else:
                probs = self._uniform_strategy[iid]

            action_dict = {}
            for a_idx, a in enumerate(ACTION_ORDER):
                if valid[iid, a_idx]:
                    action_dict[a] = float(probs[a_idx])
            avg_strategy[key] = action_dict

        return avg_strategy

    # --- Persistence ---
    def save_gzip(self, filepath):
        data = {
            "regret_sum": self.regret_sum,
            "strategy_sum": self.strategy_sum,
            "average_strategy": self.average_strategy,
            "iteration_count": self.iteration_count,
            "training_time": self.training_time,
        }
        with gzip.open(filepath, "wb") as f:
            pkl.dump(data, f)
        print(f"Saved to {filepath}")

    def load_gzip(self, filepath):
        with gzip.open(filepath, "rb") as f:
            data = pkl.load(f)
        self.regret_sum = data["regret_sum"]
        self.strategy_sum = data["strategy_sum"]
        self.average_strategy = data.get("average_strategy", {})
        self.iteration_count = data.get("iteration_count", 0)
        self.training_time = data.get("training_time", 0.0)
        # _current_iteration is 1-indexed
        self._current_iteration = self.iteration_count
        print(f"Loaded from {filepath}")

    # --- Validation ---
    @staticmethod
    def _validate_flat_tree_sampled(ft: FlatTree, max_nodes: int = 2000):
        """
        Sample-based validation: chance_offsets length/monotonicity,
        depth consistency (child.depth == parent.depth + 1) for decision/chance edges.
        """
        num_nodes = int(len(ft.node_type))
        if num_nodes == 0:
            raise ValueError("FlatTree is empty")
        if ft.root < 0 or ft.root >= num_nodes:
            raise ValueError(f"Invalid root index {ft.root}")

        if int(len(ft.chance_offsets)) != num_nodes + 1:
            raise ValueError("chance_offsets must have length num_nodes+1")
        if int(ft.chance_offsets[0]) != 0:
            raise ValueError("chance_offsets[0] must be 0")
        if np.any(ft.chance_offsets[1:] < ft.chance_offsets[:-1]):
            raise ValueError("chance_offsets must be non-decreasing")

        total_edges = int(ft.chance_offsets[-1])
        if total_edges != int(len(ft.chance_children)) or total_edges != int(len(ft.chance_probs)):
            raise ValueError("chance edge arrays length mismatch")

        # Depth check for a sample of decision edges
        decision_nodes = np.where(ft.node_type == 1)[0][:max_nodes]
        for n in decision_nodes:
            pd = int(ft.depth[n])
            for c in ft.children[n]:
                c = int(c)
                if c == -1:
                    continue
                if int(ft.depth[c]) != pd + 1:
                    raise ValueError(f"Decision edge depth mismatch: {n}->{c} ({pd} -> {int(ft.depth[c])})")

        # Depth check for a sample of chance edges
        chance_nodes = np.where(ft.node_type == 2)[0][:max_nodes]
        for n in chance_nodes:
            pd = int(ft.depth[n])
            s = int(ft.chance_offsets[n])
            e = int(ft.chance_offsets[n + 1])
            if e < s:
                raise ValueError(f"Invalid chance offset range for node {n}: {s}..{e}")
            kids = ft.chance_children[s:e]
            for c in kids[:50]:  # sample within node as well
                c = int(c)
                if int(ft.depth[c]) != pd + 1:
                    raise ValueError(f"Chance edge depth mismatch: {n}->{c} ({pd} -> {int(ft.depth[c])})")
