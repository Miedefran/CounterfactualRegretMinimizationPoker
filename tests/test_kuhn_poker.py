from envs.kuhn_poker.game import KuhnPokerGame
from agents.random_agent import RandomAgent
import random

def test_single_game():
    print("Testing Kuhn Poker")
    
    game = KuhnPokerGame(ante=1, bet_size=1)
    agent0 = RandomAgent(0)
    agent1 = RandomAgent(1)
    
    starting_player = random.randint(0, 1)
    game.reset(starting_player)
    
    game.players[0].private_card = game.dealer.deal_card()
    game.players[1].private_card = game.dealer.deal_card()
    
    print(f"Starting player: {starting_player}")
    print(f"Player 0 hand: {game.players[0].private_card}")
    print(f"Player 1 hand: {game.players[1].private_card}")
    
    step_count = 0
    while not game.done:
        current_state = game.get_state(game.current_player)
        print(f"\nStep {step_count}: Player {game.current_player}'s turn")
        print(f"Current state: {current_state}")
        
        if game.current_player == 0:
            action = agent0.get_action(current_state)
        else:
            action = agent1.get_action(current_state)
            
        print(f"Player {game.current_player} chooses: {action}")
        
        result = game.step(action)
        step_count += 1
        
        if result:
            print(f"\nFinal Pot: {game.pot}, Player Bets: {game.player_bets}")
            print(f"Result: {result}")
            break
    
    print(f"Total steps: {step_count}\n")

if __name__ == "__main__":
    test_single_game()
