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
    RhodeIslandCombinations,
    TwelveCardPokerCombinations,
    RoyalHoldemCombinations,
    LimitHoldemCombinations,
)


def main():
    if len(sys.argv) < 2:
        print("Usage: python build_and_save_tree.py <game_name>")
        print("Available games: kuhn_case1, kuhn_case2, kuhn_case3, kuhn_case4, leduc, rhode_island, twelve_card_poker, royal_holdem, limit_holdem")
        sys.exit(1)
    
    game_name = sys.argv[1]
    
    if game_name not in GAME_CONFIGS:
        print(f"Unknown game: {game_name}")
        sys.exit(1)
    
    config = GAME_CONFIGS[game_name]
    
    # Erstelle Game und Combination Generator
    if game_name.startswith('kuhn'):
        game = KuhnPokerGame(ante=config['ante'], bet_size=config['bet_size'])
        combo_gen = KuhnPokerCombinations()
    elif game_name == 'leduc':
        game = LeducHoldemGame(ante=config['ante'], bet_sizes=config['bet_sizes'], bet_limit=config['bet_limit'])
        combo_gen = LeducHoldemCombinations()
    elif game_name == 'rhode_island':
        game = RhodeIslandGame(ante=config['ante'], bet_sizes=config['bet_sizes'], bet_limit=config['bet_limit'])
        combo_gen = RhodeIslandCombinations()
    elif game_name == 'twelve_card_poker':
        game = TwelveCardPokerGame(ante=config['ante'], bet_sizes=config['bet_sizes'], bet_limit=config['bet_limit'])
        combo_gen = TwelveCardPokerCombinations()
    elif game_name == 'royal_holdem':
        game = RoyalHoldemGame(ante=config['ante'], bet_sizes=config['bet_sizes'], bet_limit=config['bet_limit'])
        combo_gen = RoyalHoldemCombinations()
    elif game_name == 'limit_holdem':
        game = LimitHoldemGame(small_blind=config['small_blind'], big_blind=config['big_blind'], 
                              bet_sizes=config['bet_sizes'], bet_limit=config['bet_limit'])
        combo_gen = LimitHoldemCombinations()
    
    # Baue Tree
    tree = build_game_tree(game, combo_gen, game_name=game_name, game_config=config)
    
    # Speichere Tree
    save_game_tree(tree, game_name)
    
    print(f"\nGame tree for {game_name} successfully built and saved!")


if __name__ == "__main__":
    main()
