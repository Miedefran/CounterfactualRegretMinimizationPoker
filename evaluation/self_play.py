import sys
import os
import pickle as pkl
import gzip
import argparse
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.kuhn_poker.game import KuhnPokerGame
from envs.leduc_holdem.game import LeducHoldemGame
from agents.strategy_agent import StrategyAgent
from utils.poker_utils import GAME_CONFIGS
from utils.test_logger import log_performance, extract_iterations_from_filename

def play_single_game(agent0, agent1, game):
    game.reset(0)
    game.players[0].private_card = game.dealer.deal_card()
    game.players[1].private_card = game.dealer.deal_card()
    
    while not game.done:
        current_player = game.current_player
        state = game.get_state(current_player)
        
        if current_player == 0:
            action = agent0.get_action(state)
        else:
            action = agent1.get_action(state)
        
        result = game.step(action)
        
        if result:
            return result
    
    return None

def run_self_play(game, strategy_file, num_games):
    with gzip.open(strategy_file, 'rb') as f:
        data = pkl.load(f)
    
    avg_strategy = data['average_strategy']
    
    agent0 = StrategyAgent(avg_strategy, 0)
    agent1 = StrategyAgent(avg_strategy, 1)
    
    results_p0 = []
    
    for _ in range(num_games):
        payoffs = play_single_game(agent0, agent1, game)
        results_p0.append(payoffs[0])
    
    avg_p0 = sum(results_p0) / len(results_p0)
    avg_p1 = -avg_p0
    return avg_p0, avg_p1

def main():
    parser = argparse.ArgumentParser(description='Self-Play Evaluation for CFR Strategy')
    parser.add_argument('game', type=str,
                       choices=['kuhn_case1', 'kuhn_case2', 'kuhn_case3', 'kuhn_case4', 'leduc'],
                       help='Poker variant')
    parser.add_argument('strategy_file', type=str,
                       help='Path to strategy pickle file')
    parser.add_argument('--games', type=int, default=1000,
                       help='Number of games to play')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducibility')
    parser.add_argument('--save', action='store_true',
                       help='Save results to CSV')
    
    args = parser.parse_args()
    
    if args.seed is not None:
        random.seed(args.seed)
    config = GAME_CONFIGS[args.game]
    
    if args.game.startswith('kuhn'):
        game = KuhnPokerGame(ante=config['ante'], bet_size=config['bet_size'])
    elif args.game == 'leduc':
        game = LeducHoldemGame(ante=config['ante'], bet_sizes=config['bet_sizes'], bet_limit=config['bet_limit'])
    
    print(f"Self-play: {args.game}, {args.games} games")
    print(f"Strategy: {args.strategy_file}\n")
    
    avg_p0, avg_p1 = run_self_play(game, args.strategy_file, args.games)
    print(f"Player 0: {avg_p0:.6f}")
    print(f"Player 1: {avg_p1:.6f}")
    
    if args.save:
        iterations = extract_iterations_from_filename(args.strategy_file)
        
        if iterations:
            log_performance('vanilla_cfr', args.game, iterations,
                          'self_play_ev_p0', avg_p0,
                          seed=args.seed,
                          games=args.games)
        else:
            print("Warning: Could not extract iterations from filename, skipping save")

if __name__ == "__main__":
    main()