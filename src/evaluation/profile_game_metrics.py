"""
Profiling-Skript für Kapitel 6: pro Spiel Text-Zusammenfassung (Game-Tree-/Public-Tree-Größen, BR-Dauer).
Baut Trees nicht automatisch; erwartet i.d.R. existierende Strategie-Datei (*.pkl.gz).
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class GameTreeStats:
    path: str
    nodes: int
    infosets: int


@dataclass
class PublicTreeStats:
    path: str
    states: int
    chance_nodes: int
    choice_nodes: int
    terminal_nodes: int


@dataclass
class StrategyStats:
    infosets: int


def _safe_mkdir_for_file(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def try_load_game_tree(game: str, abstract_suits: bool) -> Optional[GameTreeStats]:
    try:
        from training.build_game_tree import load_game_tree

        tree = load_game_tree(game, abstract_suits=abstract_suits)
        return GameTreeStats(
            path="(siehe Konsole: load_game_tree gibt Pfad aus)",
            nodes=len(tree.nodes),
            infosets=len(tree.infoset_to_nodes),
        )
    except FileNotFoundError:
        return None


def try_load_public_tree(game: str, abstract_suits: bool) -> Optional[PublicTreeStats]:
    try:
        from training.best_response_evaluator import get_public_state_tree_path
        from evaluation.best_response_agent_v2 import load_public_tree

        tree_path = get_public_state_tree_path(game, abstract_suits=abstract_suits)
        tree = load_public_tree(tree_path)
        states = tree.get("public_states", {})
        chance = sum(1 for s in states.values() if s.get("type") == "chance")
        choice = sum(1 for s in states.values() if s.get("type") == "choice")
        terminal = sum(1 for s in states.values() if s.get("type") == "terminal")
        return PublicTreeStats(
            path=tree_path,
            states=len(states),
            chance_nodes=chance,
            choice_nodes=choice,
            terminal_nodes=terminal,
        )
    except FileNotFoundError:
        return None


def measure_best_response_seconds(game: str, player: int, public_tree_path: str, strategy_path: str) -> float:
    from evaluation.best_response_agent_v2 import load_public_tree, load_average_strategy, compute_best_response_value

    tree = load_public_tree(public_tree_path)
    avg_strategy = load_average_strategy(strategy_path)

    t0 = time.time()
    _ = compute_best_response_value(game, player, tree, avg_strategy, root_hist=())
    return time.time() - t0


def load_strategy_stats(strategy_path: str) -> StrategyStats:
    from evaluation.best_response_agent_v2 import load_average_strategy

    avg_strategy = load_average_strategy(strategy_path)
    return StrategyStats(infosets=len(avg_strategy))


def format_report(
        game: str,
        player: int,
        strategy_path: str,
        abstract_suits: bool,
        game_tree_stats: Optional[GameTreeStats],
        public_tree_stats: Optional[PublicTreeStats],
        strategy_stats: StrategyStats,
        br_seconds: Optional[float],
) -> str:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    lines: list[str] = []
    lines.append(f"Profiling Report ({ts})")
    lines.append(f"game = {game}")
    lines.append(f"player_id_for_br = {player}")
    lines.append(f"strategy_path = {strategy_path}")
    lines.append(f"suit_abstraction = {abstract_suits}")
    lines.append(f"average_strategy_infosets = {strategy_stats.infosets}")
    lines.append("")

    lines.append("== Game Tree ==")
    if game_tree_stats is None:
        lines.append("game_tree: NOT FOUND")
        lines.append("hint: Tree wird nur für *_with_tree Solver automatisch gebaut/geladen.")
    else:
        lines.append(f"game_tree_nodes = {game_tree_stats.nodes}")
        lines.append(f"game_tree_infosets = {game_tree_stats.infosets}")
        lines.append(f"game_tree_path = {game_tree_stats.path}")
    lines.append("")

    lines.append("== Public State Tree ==")
    if public_tree_stats is None:
        lines.append("public_tree: NOT FOUND")
        lines.append("hint: erst Public State Tree bauen/speichern, dann erneut ausführen.")
    else:
        lines.append(f"public_tree_path = {public_tree_stats.path}")
        lines.append(f"public_tree_states = {public_tree_stats.states}")
        lines.append(f"public_tree_choice_nodes = {public_tree_stats.choice_nodes}")
        lines.append(f"public_tree_chance_nodes = {public_tree_stats.chance_nodes}")
        lines.append(f"public_tree_terminal_nodes = {public_tree_stats.terminal_nodes}")
    lines.append("")

    lines.append("== Best Response (1 Spieler) ==")
    if br_seconds is None:
        lines.append("br_eval_seconds_player = N/A (public tree missing?)")
    else:
        lines.append(f"br_eval_seconds_player = {br_seconds:.6f}")
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Profiling: Tree-Größen + BR-Zeit (pro Spiel)")
    parser.add_argument("--game", required=True, type=str)
    parser.add_argument("--strategy", required=True, type=str,
                        help="Pfad zu einer *.pkl.gz Model-Datei (mit average_strategy)")
    parser.add_argument("--player", type=int, default=0, choices=[0, 1], help="Spieler für BR-Zeitmessung (default: 0)")
    parser.add_argument("--output", type=str, default=None,
                        help="Optionaler Output-Pfad (*.txt). Wenn nicht gesetzt: nur stdout.")
    parser.add_argument(
        "--abstract-suits",
        action="store_true",
        help="Verwendet Suit Abstraction (Dateinamen/Pfade der Trees). Default: False",
    )
    args = parser.parse_args()

    strategy_path = os.path.abspath(args.strategy)
    if not os.path.exists(strategy_path):
        raise FileNotFoundError(f"Strategy file not found: {strategy_path}")

    game_tree_stats = try_load_game_tree(args.game, abstract_suits=args.abstract_suits)
    public_tree_stats = try_load_public_tree(args.game, abstract_suits=args.abstract_suits)

    br_seconds: Optional[float] = None
    if public_tree_stats is not None:
        br_seconds = measure_best_response_seconds(
            args.game,
            args.player,
            public_tree_stats.path,
            strategy_path,
        )

    report = format_report(
        game=args.game,
        player=args.player,
        strategy_path=strategy_path,
        abstract_suits=args.abstract_suits,
        game_tree_stats=game_tree_stats,
        public_tree_stats=public_tree_stats,
        strategy_stats=load_strategy_stats(strategy_path),
        br_seconds=br_seconds,
    )

    if args.output:
        out_path = os.path.abspath(args.output)
        _safe_mkdir_for_file(out_path)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(report)
            f.write("\n")
        print(f"Wrote report: {out_path}")
    else:
        print(report)


if __name__ == "__main__":
    main()
