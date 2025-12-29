from envs.kuhn_poker.dealer import KuhnPokerDealer
from envs.kuhn_poker.player import KuhnPokerPlayer
from envs.kuhn_poker.judger import KuhnPokerJudger

class KuhnPokerGame:
    
    def __init__(self, ante, bet_size):
        self.ante = ante
        self.bet_size = bet_size
        self.pot = 0
        self.player_bets = [0, 0]
        self.dealer = KuhnPokerDealer()
        self.players = [KuhnPokerPlayer(0), KuhnPokerPlayer(1)]
        self.judger = KuhnPokerJudger()
        self.history = []
        self.state_stack = []
        self.reset(0)
        
    
    def reset(self, starting_player):
        self.state_stack.clear()
        self.history.clear()
        self.dealer.reset()
        self.dealer.shuffle()
        self.players[0].reset()
        self.players[1].reset()
        self.current_player = starting_player
        self.done = False
        self.pot = 2 * self.ante
        self.player_bets = [self.ante, self.ante]
        
        
    def step(self, action):
            
        self.state_stack.append(self.save_state())
        self.history.append(action)
        
        if action == 'bet':
            self.player_bets[self.current_player] += self.bet_size
            self.pot += self.bet_size
        elif action == 'call':
            self.player_bets[self.current_player] += self.bet_size
            self.pot += self.bet_size
        
        if action == 'fold' or self.history[-2:] == ['bet', 'call'] or self.history[-2:] == ['check','check']:
            self.done = True
            if action == 'fold':
                return self.judger.judge(self.players, self.history, 1 - self.current_player, self.pot, self.player_bets)
            else:
                return self.judger.judge(self.players, self.history, self.current_player, self.pot, self.player_bets) 
        else:
            self.current_player = 1 - self.current_player
            return None

    def get_legal_actions(self):
        if not self.history:
            return ['check', 'bet']
        last_action = self.history[-1]
        if last_action == 'check':
            return ['check', 'bet']
        elif last_action == 'bet':
            return ['call', 'fold']
        return [] 

    """Game Tree Traversal Methods"""
    
    def get_state(self, player_id):
        return {
            'hand': self.players[player_id].private_card, 
            'history': list(self.history),
            'current_player': self.current_player,
            'done': self.done,
            'pot': self.pot,
            'player_bets': list(self.player_bets),
            'legal_actions': self.get_legal_actions()
        }
        
    def save_state(self):
        return {
            'player_hands': [self.players[0].private_card, self.players[1].private_card],
            'history': self.history.copy(),
            'current_player': self.current_player,
            'done': self.done,
            'pot': self.pot,
            'player_bets': self.player_bets.copy()
        }
    
    def restore_state(self, saved_state):
        self.players[0].private_card = saved_state['player_hands'][0]
        self.players[1].private_card = saved_state['player_hands'][1]
        self.history = saved_state['history']
        self.current_player = saved_state['current_player']
        self.done = saved_state['done']
        self.pot = saved_state['pot']
        self.player_bets = saved_state['player_bets'].copy()
    
    def step_back(self):
        if self.state_stack:
            previous_state = self.state_stack.pop()
            self.restore_state(previous_state)
        else:
            raise ValueError("No saved state available for step_back")
    
    def get_info_set_key(self, player_id):
        return (
            self.players[player_id].private_card,
            tuple(self.history),
            self.current_player
        )
    
    def get_payoff(self, player_id):
        if not self.done:
            raise ValueError("Game isnt done yet")
        
        payoffs = self.judger.judge(self.players, self.history, self.current_player, self.pot, self.player_bets)
        return payoffs[player_id]