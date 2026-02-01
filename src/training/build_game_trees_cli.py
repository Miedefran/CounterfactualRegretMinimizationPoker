"""CLI: Explicitly build and save game trees (for Tree-CFR solvers). Optionally overwrites existing files."""

from __future__ import annotations

import argparse
from pathlib import Path

from training.build_game_tree import build_game_tree, save_game_tree
from utils.poker_utils import (
    find_game_class_for_abstraction,
    KuhnPokerCombinations,
    LeducHoldemCombinations,
    LeducHoldemCombinationsAbstracted,
    SmallIslandHoldemCombinations,
    TwelveCardPokerCombinations,
    TwelveCardPokerCombinationsAbstracted,
)


def _default_tree_path(game_name: str, abstract_suits: bool) -> Path:
    subdir = "abstracted" if abstract_suits else "normal"
    return Path("data") / "trees" / "game_trees" / subdir / f"{game_name}_game_tree.pkl.gz"


def build_one(game: str, abstract_suits: bool, force: bool) -> None:
    game_class = find_game_class_for_abstraction(game, abstract_suits)
    if game_class is None:
        raise ValueError(f"Unsupported game for game-tree build: {game}")

    game_obj = game_class()

    if game == "kuhn_case2":
        combo_gen = KuhnPokerCombinations()
        save_name = "kuhn_case2"
    elif game == "small_island_holdem":
        combo_gen = SmallIslandHoldemCombinations()
        save_name = "small_island_holdem"
    elif game == "leduc":
        combo_gen = LeducHoldemCombinationsAbstracted() if abstract_suits else LeducHoldemCombinations()
        save_name = "leduc"
    elif game == "twelve_card_poker":
        combo_gen = TwelveCardPokerCombinationsAbstracted() if abstract_suits else TwelveCardPokerCombinations()
        save_name = "twelve_card_poker"
    else:
        raise ValueError(f"Unsupported game for game-tree build: {game}")

    out_path = _default_tree_path(save_name, abstract_suits)
    if out_path.exists() and not force:
        print(f"SKIP: {game} (exists: {out_path})")
        return

    abstraction_str = " (suit abstracted)" if abstract_suits else " (NOT abstracted)"
    if out_path.exists() and force:
        print(f"FORCE: overwriting existing game tree: {out_path}")

    print(f"\n=== BUILD GAME TREE: {game}{abstraction_str} ===")
    tree = build_game_tree(game_obj, combo_gen, game_name=save_name, abstract_suits=abstract_suits)
    save_game_tree(tree, save_name, abstract_suits=abstract_suits)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and save game trees")
    parser.add_argument(
        "games",
        nargs="+",
        choices=["kuhn_case2", "leduc", "twelve_card_poker", "small_island_holdem"],
        help="Which game trees to build",
    )
    parser.add_argument(
        "--no-suit-abstraction",
        action="store_true",
        help="Build NOT-abstracted game trees (default).",
    )
    parser.add_argument(
        "--suit-abstraction",
        action="store_true",
        help="Build suit-abstracted game trees.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing game tree files.",
    )

    args = parser.parse_args()

    # Default: NOT abstracted
    abstract_suits = False
    if args.suit_abstraction:
        abstract_suits = True
    if args.no_suit_abstraction:
        abstract_suits = False

    for g in args.games:
        build_one(g, abstract_suits=abstract_suits, force=args.force)


if __name__ == "__main__":
    main()
