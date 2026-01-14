"""
Script zum Bauen und Speichern von Game Trees.

Usage:
    python src/training/build_and_save_tree.py <game_name>
    
Beispiel:
    python src/training/build_and_save_tree.py leduc
    python src/training/build_and_save_tree.py twelve_card_poker
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.build_game_tree import build_game_tree, save_game_tree
from envs.kuhn_poker.game import KuhnPokerGame
from envs.leduc_holdem.game import LeducHoldemGame
from envs.rhode_island.game import RhodeIslandGame
from envs.twelve_card_poker.game import TwelveCardPokerGame
from envs.royal_holdem.game import RoyalHoldemGame
from envs.limit_holdem.game import LimitHoldemGame

from utils.poker_utils import (
    GAME_CONFIGS,
    KuhnPokerCombinations,
    LeducHoldemCombinations,
    LeducHoldemCombinationsAbstracted,
    RhodeIslandCombinations,
    TwelveCardPokerCombinations,
    TwelveCardPokerCombinationsAbstracted,
    RoyalHoldemCombinations,
    LimitHoldemCombinations,
)


def main():
    if len(sys.argv) < 2:
        print("Usage: python build_and_save_tree.py <game_name> [--no-suit-abstraction]")
        print("Available games: kuhn_case1, kuhn_case2, kuhn_case3, kuhn_case4, leduc, rhode_island, twelve_card_poker, royal_holdem, limit_holdem")
        print("Note: leduc and twelve_card_poker use suit abstraction by default. Use --no-suit-abstraction to build normal tree.")
        sys.exit(1)
    
    game_name = sys.argv[1]
    no_suit_abstraction = '--no-suit-abstraction' in sys.argv
    
    if game_name not in GAME_CONFIGS:
        print(f"Unknown game: {game_name}")
        sys.exit(1)
    
    config = GAME_CONFIGS[game_name]
    
    # Bestimme ob Suit Abstraction verwendet werden soll
    # Standardmäßig für leduc und twelve_card_poker, außer wenn --no-suit-abstraction gesetzt ist
    use_suit_abstraction = False
    if game_name in ['leduc', 'twelve_card_poker']:
        use_suit_abstraction = not no_suit_abstraction
    
    # Erstelle Game und Combination Generator
    if game_name.startswith('kuhn'):
        game = KuhnPokerGame(ante=config['ante'], bet_size=config['bet_size'])
        combo_gen = KuhnPokerCombinations()
    elif game_name == 'leduc':
        game = LeducHoldemGame(ante=config['ante'], bet_sizes=config['bet_sizes'], bet_limit=config['bet_limit'])
        if use_suit_abstraction:
            combo_gen = LeducHoldemCombinationsAbstracted()
        else:
            combo_gen = LeducHoldemCombinations()
    elif game_name == 'rhode_island':
        game = RhodeIslandGame(ante=config['ante'], bet_sizes=config['bet_sizes'], bet_limit=config['bet_limit'])
        combo_gen = RhodeIslandCombinations()
    elif game_name == 'twelve_card_poker':
        game = TwelveCardPokerGame(ante=config['ante'], bet_sizes=config['bet_sizes'], bet_limit=config['bet_limit'])
        if use_suit_abstraction:
            combo_gen = TwelveCardPokerCombinationsAbstracted()
        else:
            combo_gen = TwelveCardPokerCombinations()
    elif game_name == 'royal_holdem':
        game = RoyalHoldemGame(ante=config['ante'], bet_sizes=config['bet_sizes'], bet_limit=config['bet_limit'])
        combo_gen = RoyalHoldemCombinations()
    elif game_name == 'limit_holdem':
        game = LimitHoldemGame(small_blind=config['small_blind'], big_blind=config['big_blind'], 
                              bet_sizes=config['bet_sizes'], bet_limit=config['bet_limit'])
        combo_gen = LimitHoldemCombinations()
    
    # Baue Tree
    tree = build_game_tree(game, combo_gen, game_name=game_name, game_config=config, abstract_suits=use_suit_abstraction)
    
    # Speichere Tree
    save_game_tree(tree, game_name, abstract_suits=use_suit_abstraction)
    
    abstraction_str = " (suit abstracted)" if use_suit_abstraction else ""
    print(f"\nGame tree for {game_name}{abstraction_str} successfully built and saved!")


if __name__ == "__main__":
    main()
