import argparse
from training.cfr_solver import CFRSolver
from envs.kuhn_poker.game import KuhnPokerGame
from envs.leduc_holdem.game import LeducHoldemGame

from utils.poker_utils import (
    GAME_CONFIGS,
    KuhnPokerCombinations,
    LeducHoldemCombinations,
    get_model_path
)

def main():
    parser = argparse.ArgumentParser(description='Train CFR for Poker')
    parser.add_argument('game', type=str,
                       choices=['kuhn_case1', 'kuhn_case2', 'kuhn_case3', 'kuhn_case4', 'leduc'],
                       help='Poker variant to train on')
    parser.add_argument('iterations', type=int,
                       help='Number of CFR iterations')
    args = parser.parse_args()
    config = GAME_CONFIGS[args.game]
    
    print(f"Training {args.game} for {args.iterations} iterations")
    
    if args.game.startswith('kuhn'):
        game = KuhnPokerGame(ante=config['ante'], bet_size=config['bet_size'])
        combo_gen = KuhnPokerCombinations()
    elif args.game.startswith('leduc'):
        game = LeducHoldemGame(ante=config['ante'], bet_sizes=config['bet_sizes'], bet_limit=config['bet_limit'])
        combo_gen = LeducHoldemCombinations()
    
    solver = CFRSolver(game, combo_gen)
    solver.train(args.iterations)
    
    filepath = get_model_path(args.game, args.iterations)
    solver.save_gzip(filepath)
    
    avg_strategy = solver.average_strategy
    print(f"{len(avg_strategy)} information sets")
    
    if args.game.startswith('kuhn'):
        print("\nStrategy:")
        for key, actions in sorted(avg_strategy.items()):
            card, history, player = key
            print(f"\nCard: {card}, History: {history}, Player: {player}")
            for action, prob in actions.items():
                print(f"  {action}: {prob:.4f}")

if __name__ == "__main__":
    main()