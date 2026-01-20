import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.leduc_holdem.game import LeducHoldemGame
from utils.poker_utils import GAME_CONFIGS, LeducHoldemCombinationsAbstracted
from training.build_game_tree import build_game_tree


def test_leduc_suit_abstracted_infoset_count_288():
    """
    Regression target from SUIT_ABSTRACTION_FIX.md:
    Leduc suit-abstracted should have exactly 288 infosets.
    """
    cfg = dict(GAME_CONFIGS["leduc"])
    game = LeducHoldemGame(**cfg, abstract_suits=True)
    combo = LeducHoldemCombinationsAbstracted()

    tree = build_game_tree(game, combo, game_name="leduc", abstract_suits=True)
    assert len(tree.infoset_to_nodes) == 288

