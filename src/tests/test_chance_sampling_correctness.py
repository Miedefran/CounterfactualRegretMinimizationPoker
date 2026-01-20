import os
import sys

import numpy as np

# Ensure src is on path when running via pytest from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.leduc_holdem.game import LeducHoldemGame
from utils.poker_utils import GAME_CONFIGS, LeducHoldemCombinations
from training.chance_sampling_cfr_solver import ChanceSamplingCFRSolver


def test_chance_sampling_uses_one_consistent_deal_per_iteration_leduc():
    """
    Chance-sampled CFR (poker variant) must sample ONE consistent chance sequence per iteration:
    - private deal p0 (index 0)
    - private deal p1 (index 1)
    - public deal (index 2)

    Because the tree contains many public-deal chance nodes (one per betting history that ends round 0),
    the solver must reuse the SAME sampled public card across all of them within the iteration.
    """
    cfg = dict(GAME_CONFIGS["leduc"])
    game = LeducHoldemGame(**cfg, abstract_suits=False)
    combo = LeducHoldemCombinations()
    solver = ChanceSamplingCFRSolver(game, combo, game_name="leduc", load_tree=True)

    # Make sampling deterministic for the test.
    import random

    random.seed(123)
    np.random.seed(123)

    solver.cfr_iteration()

    sampled = getattr(solver, "_debug_last_sampled_chance_by_index", None)
    encounters = getattr(solver, "_debug_last_chance_encounters", None)
    assert isinstance(sampled, dict)
    assert isinstance(encounters, list)
    assert len(encounters) > 0

    # We must have at least the 3 deal indices present in Leduc.
    assert 0 in sampled
    assert 1 in sampled
    assert 2 in sampled

    # For each chance_index, ALL encountered chance nodes must use the same outcome.
    by_idx = {}
    for idx, node_id, outcome in encounters:
        by_idx.setdefault(idx, set()).add(outcome)

    for idx, outs in by_idx.items():
        assert len(outs) == 1, f"Chance index {idx} used multiple outcomes in one iteration: {outs}"
        # And it must match the sampled mapping.
        assert sampled.get(idx) in outs

