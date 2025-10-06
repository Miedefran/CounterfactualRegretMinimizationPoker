import random

class StrategyAgent:
    
    def __init__(self, strategy, player_id):
        self.strategy = strategy
        self.player_id = player_id
    
    def get_action(self, state):
        card = state['hand']
        history = tuple(state['history'])
        current_player = state['current_player']
        
        info_set_key = (card, history, current_player)
        
        if info_set_key not in self.strategy:
            legal_actions = state['legal_actions']
            return random.choice(legal_actions)
        
        action_probs = self.strategy[info_set_key]
        actions = list(action_probs.keys())
        probs = list(action_probs.values())
        
        return random.choices(actions, weights=probs)[0]

