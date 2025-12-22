import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.cfr_solver import CFRSolver
from training.cfr_plus_solver import CFRPlusSolver
from training.fold_solver import AlwaysFoldSolver
from training.mccfr_solver import MCCFRSolver
from training.tensor_cfr_solver import TensorCFRSolver
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
    get_model_path
)

def main():
    parser = argparse.ArgumentParser(description='Train CFR for Poker')
    parser.add_argument('game', type=str,
                       choices=['kuhn_case1', 'kuhn_case2', 'kuhn_case3', 'kuhn_case4', 'leduc', 'rhode_island', 'twelve_card_poker', 'royal_holdem', 'limit_holdem'],
                       help='Poker variant to train on')
    parser.add_argument('iterations', type=int,
                       help='Number of CFR iterations')
    parser.add_argument('algorithm', type=str,
                       choices=['fold', 'cfr', 'cfr_plus', 'mccfr', 'tensor_cfr'],
                       nargs='?',
                       default='cfr',
                       help='Algorithm to use (default: cfr)')
    args = parser.parse_args()
    config = GAME_CONFIGS[args.game]
    
    print(f"Training {args.game} for {args.iterations} iterations")
    
    if args.game.startswith('kuhn'):
        game = KuhnPokerGame(ante=config['ante'], bet_size=config['bet_size'])
        combo_gen = KuhnPokerCombinations()
    elif args.game.startswith('leduc'):
        game = LeducHoldemGame(ante=config['ante'], bet_sizes=config['bet_sizes'], bet_limit=config['bet_limit'])
        combo_gen = LeducHoldemCombinations()
    elif args.game.startswith('rhode'):
        game = RhodeIslandGame(ante=config['ante'], bet_sizes=config['bet_sizes'], bet_limit=config['bet_limit'])
        combo_gen = RhodeIslandCombinations()
    elif args.game == 'twelve_card_poker':
        game = TwelveCardPokerGame(ante=config['ante'], bet_sizes=config['bet_sizes'], bet_limit=config['bet_limit'])
        combo_gen = TwelveCardPokerCombinations()
    elif args.game == 'royal_holdem':
        game = RoyalHoldemGame(ante=config['ante'], bet_sizes=config['bet_sizes'], bet_limit=config['bet_limit'])
        combo_gen = RoyalHoldemCombinations()
    elif args.game == 'limit_holdem':
        game = LimitHoldemGame(small_blind=config['small_blind'], big_blind=config['big_blind'], bet_sizes=config['bet_sizes'], bet_limit=config['bet_limit'])
        combo_gen = LimitHoldemCombinations()
    
    if args.algorithm == 'cfr':
        solver = CFRSolver(game, combo_gen)
    elif args.algorithm == 'cfr_plus':
        solver = CFRPlusSolver(game, combo_gen)
    elif args.algorithm == 'mccfr':
        solver = MCCFRSolver(game, combo_gen)
    elif args.algorithm == 'fold':
        solver = AlwaysFoldSolver(game, combo_gen)
    elif args.algorithm == 'tensor_cfr':
        solver = TensorCFRSolver(game, combo_gen)
    
    solver.train(args.iterations)
    
    filepath = get_model_path(args.game, args.iterations, args.algorithm)
    solver.save_gzip(filepath)
    
    avg_strategy = solver.average_strategy
    print(f"{len(avg_strategy)} information sets")
    
    if args.game.startswith('kuhn'):
        print("\nStrategy:")
        for key, actions in sorted(avg_strategy.items()):
            # Updated to match KeyGenerator format: (private, public, history, pid)
            card, public_cards, history, player = key
            print(f"\nCard: {card}, History: {history}, Player: {player}")
            for action, prob in actions.items():
                print(f"  {action}: {prob:.4f}")

if __name__ == "__main__":
    main()