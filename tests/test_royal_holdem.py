from envs.royal_holdem.game import RoyalHoldemGame
from agents.random_agent import RandomAgent
import random

def test_single_game():
    print("Testing Royal Hold'em")
    
    game = RoyalHoldemGame()
    agent0 = RandomAgent(0)
    agent1 = RandomAgent(1)
    
    starting_player = random.randint(0, 1)
    game.reset(starting_player)
    
    game.dealer.reset()
    game.dealer.shuffle()
    game.players[0].set_private_cards(game.dealer.deal_card(), game.dealer.deal_card())
    game.players[1].set_private_cards(game.dealer.deal_card(), game.dealer.deal_card())
    
    print(f"Starting player: {starting_player}")
    print(f"Player 0 private cards: {game.players[0].private_cards}")
    print(f"Player 1 private cards: {game.players[1].private_cards}")
    
    step_count = 0
    betting_round = 0
    
    while not game.done:
        current_player = game.current_player
        current_state = game.get_state(current_player)
        print(f"\nStep {step_count}: Player {current_player}'s turn (Betting Round {game.betting_round})")
        print(f"Legal actions: {current_state['legal_actions']}")
        
        if current_player == 0:
            action = agent0.get_action(current_state)
        else:
            action = agent1.get_action(current_state)
            
        print(f"Player {current_player} chooses: {action}")
        
        result = game.step(action)
        step_count += 1
        
        if game.betting_round != betting_round:
            betting_round = game.betting_round
            if game.public_cards:
                print(f"\nPublic cards: {game.public_cards}")
                print(f"Player 0 hand: {game.players[0].private_cards} + {game.public_cards}")
                print(f"Player 1 hand: {game.players[1].private_cards} + {game.public_cards}")
        
        if result:
            print(f"\nFinal Pot: {game.pot}, Player Bets: {game.total_bets}")
            print(f"Result: {result}")
            break
    
    print(f"Total steps: {step_count}\n")

if __name__ == "__main__":
    test_single_game()

