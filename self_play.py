import pickle as pkl
import gzip
import argparse
from envs.kuhn_poker.game import KuhnPokerGame
from agents.strategy_agent import StrategyAgent

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

def main():
    parser = argparse.ArgumentParser(description='Self-Play Evaluation for CFR Strategy')
    parser.add_argument('strategy_file', type=str,
                       help='Path to strategy pickle file')
    parser.add_argument('--games', type=int, default=1000,
                       help='Number of games to play')
    
    args = parser.parse_args()
    
    print(f"Self-play: {args.strategy_file}, {args.games} games\n")
    
    with gzip.open(args.strategy_file, 'rb') as f:
        data = pkl.load(f)
    
    strategy_sum = data['strategy_sum']
    avg_strategy = {}
    
    for info_set_key in strategy_sum:
        total = sum(strategy_sum[info_set_key].values())
        if total > 0:
            avg_strategy[info_set_key] = {
                action: strategy_sum[info_set_key][action] / total
                for action in strategy_sum[info_set_key]
            }
    
    game = KuhnPokerGame(ante=1, bet_size=1)
    agent0 = StrategyAgent(avg_strategy, 0)
    agent1 = StrategyAgent(avg_strategy, 1)
    
    results_p0 = []
    
    for i in range(args.games):
        payoffs = play_single_game(agent0, agent1, game)
        results_p0.append(payoffs[0])
    
    avg_p0 = sum(results_p0) / len(results_p0)
    avg_p1 = -avg_p0
    expected = -1/18
    
    print(f"Player 0: {avg_p0:.6f}")
    print(f"Player 1: {avg_p1:.6f}")
    print(f"Expected: {expected:.6f} (-1/18)")
    print(f"Difference: {abs(avg_p0 - expected):.6f}")

if __name__ == "__main__":
    main()