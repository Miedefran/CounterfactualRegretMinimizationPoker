import pytest
import random
from envs.twelve_card_poker.game import TwelveCardPokerGame
from agents.random_agent import RandomAgent

def test_twelve_card_poker_random_game():
    print("Testing Twelve Card Poker")
    
    game = TwelveCardPokerGame()
    agent0 = RandomAgent(0)
    agent1 = RandomAgent(1)
    
    starting_player = random.randint(0, 1)
    game.reset(starting_player)
    
    game.players[0].private_card = game.dealer.deal_card()
    game.players[1].private_card = game.dealer.deal_card()
    
    step_count = 0
    
    while not game.done:
        current_player = game.current_player
        current_state = game.get_state(current_player)
        
        if current_player == 0:
            action = agent0.get_action(current_state)
        else:
            action = agent1.get_action(current_state)
            
        result = game.step(action)
        step_count += 1
        
        if result:
            break
    
    assert game.done
    assert step_count > 0
    assert game.get_payoff(0) + game.get_payoff(1) == 0