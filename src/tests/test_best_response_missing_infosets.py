import os
import sys
import math
import pytest

# Ensure src is on path when running via pytest from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.leduc_holdem.game import LeducHoldemGame
from utils.poker_utils import GAME_CONFIGS
from evaluation.build_public_state_tree_v2 import build_public_state_tree
from evaluation.best_response_agent_v2 import compute_best_response_value


def _build_leduc_public_tree():
    cfg = dict(GAME_CONFIGS["leduc"])
    # Ensure we test the NOT-abstracted tree (matches typical BR runs shown in logs)
    return build_public_state_tree(LeducHoldemGame, cfg, use_cache=False, abstract_suits=False)


def _uniform_avg_strategy_from_public_tree(pst_tree):
    """
    Build a *complete* uniform average strategy mapping for all infosets that
    appear in the Public State Tree.

    This mirrors OpenSpiel's "undefined policy -> uniform" fallback, but does so
    explicitly so we can compare against missing-key behavior.
    """
    avg = {}
    # Track expected legal action sets per infoset to catch inconsistencies.
    infoset_actions = {}

    for _, node in pst_tree["public_states"].items():
        if node.get("type") != "choice":
            continue
        pid = node.get("player")
        children = node.get("children", {})
        legal_actions = list(children.keys())
        if not legal_actions:
            continue

        p = 1.0 / len(legal_actions)
        infosets = node.get(f"player{pid}_info_sets", [])
        for I in infosets:
            if I in infoset_actions:
                assert infoset_actions[I] == set(legal_actions), (
                    f"Infoset {I} appears with different legal_actions: "
                    f"{sorted(infoset_actions[I])} vs {sorted(set(legal_actions))}"
                )
                continue
            infoset_actions[I] = set(legal_actions)
            avg[I] = {a: p for a in legal_actions}

    return avg


@pytest.mark.parametrize("br_player", [0, 1])
def test_best_response_missing_infosets_falls_back_to_uniform(br_player):
    """
    If the evaluated avg_strategy does not define every infoset, BR evaluation must still
    be well-defined. The expected behavior (matching OpenSpiel policy fallback) is:
    missing infoset => uniform over legal actions at that infoset.
    """
    pst = _build_leduc_public_tree()
    avg_uniform = _uniform_avg_strategy_from_public_tree(pst)

    v_uniform = compute_best_response_value("leduc", br_player, pst, avg_uniform, root_hist=())
    v_empty = compute_best_response_value("leduc", br_player, pst, {}, root_hist=())

    assert math.isfinite(v_uniform)
    assert math.isfinite(v_empty)
    assert abs(v_uniform - v_empty) < 1e-9

