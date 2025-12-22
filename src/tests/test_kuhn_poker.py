import pytest
import random
from envs.kuhn_poker.game import KuhnPokerGame
from agents.random_agent import RandomAgent

def test_kuhn_poker_random_game():
    print("Testing Kuhn Poker")
    
    game = KuhnPokerGame(ante=1, bet_size=1)
    agent0 = RandomAgent(0)
    agent1 = RandomAgent(1)
    
    starting_player = random.randint(0, 1)
    game.reset(starting_player)
    
    game.players[0].private_card = game.dealer.deal_card()
    game.players[1].private_card = game.dealer.deal_card()
    
    step_count = 0
    while not game.done:
        current_state = game.get_state(game.current_player)
        
        if game.current_player == 0:
            action = agent0.get_action(current_state)
        else:
            action = agent1.get_action(current_state)
            
        result = game.step(action)
        step_count += 1
        
        if result:
            break
    
    assert game.done
    assert step_count > 0
    # Zero sum check
    payoff0 = game.get_payoff(0)
    payoff1 = game.get_payoff(1)
    assert payoff0 + payoff1 == 0