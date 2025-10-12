from envs.kuhn_poker.game import KuhnPokerGame
from envs.leduc_holdem.dealer import LeducHoldemDealer
from envs.leduc_holdem.player import LeducHoldemPlayer
from envs.leduc_holdem.judger import LeducHoldemJudger
from envs.leduc_holdem.round import LeducHoldemRound

class LeducHoldemGame(KuhnPokerGame):
   
    def __init__(self, ante=1, bet_sizes=[2, 4], bet_limit=2):
        self.ante = ante
        self.bet_size = bet_sizes[0]
        self.dealer = LeducHoldemDealer()
        self.players = [LeducHoldemPlayer(0), LeducHoldemPlayer(1)]
        self.judger = LeducHoldemJudger()
        self.round = LeducHoldemRound(bet_sizes=bet_sizes, bet_limit=bet_limit)
        self.history = []
        self.state_stack = []
        self.done = False
        self.betting_round = 0  
        self.public_card = None
        self.current_player = 0
        self.pot = 0
        self.total_bets = [0, 0]
        self.starting_player = 0 

    
    def reset(self, starting_player):
        super().reset(starting_player)
        self.betting_round = 0
        self.public_card = None
        self.starting_player = starting_player 
        self.current_player = starting_player
        self.pot = 2
        self.total_bets = [1, 1]
        
        self.round.start_new_round(starting_player, self.betting_round) 
        
    def step(self, action):
        
        self.state_stack.append(self.save_state())
        self.history.append(action)
    
        self.round.proceed_round(self, action)
        
        if action == 'fold':
            self.done = True
            return self.judger.judge(self.players, self.history, self.current_player, self.pot, self.total_bets)
        
        if self.round.is_round_complete():
            if self.betting_round == 0:
                self.history.append('|')
                self._deal_public_card()
                self.betting_round = 1
                self.round.start_new_round(self.starting_player, self.betting_round)
                
            else:
                self.done = True
                return self.judger.judge(self.players, self.history, self.current_player, self.pot, self.total_bets)
        
        return None
    
    def _deal_public_card(self):
        self.public_card = self.dealer.deal_card()
        self.players[0].set_public_card(self.public_card)
        self.players[1].set_public_card(self.public_card)
    
    def get_legal_actions(self):
        return self.round.get_legal_actions(self)
    
    def get_state(self, player_id):
        return {
            'hand': self.players[player_id].private_card,
            'public_card': self.public_card,
            'history': list(self.history),
            'current_player': self.current_player,
            'betting_round': self.betting_round,
            'legal_actions': self.get_legal_actions(),
            'pot': self.pot,
            'player_bets': list(self.total_bets),
            'bet_count': self.round.bet_count,
            'passive_action_count': self.round.passive_action_count,
        }
    
    def save_state(self):
        return {
            'hand_p0': self.players[0].private_card,
            'hand_p1': self.players[1].private_card,
            'public_card': self.public_card,
            'history': list(self.history),
            'current_player': self.current_player,
            'betting_round': self.betting_round,
            'pot': self.pot,
            'total_bets': list(self.total_bets),
            'bet_count': self.round.bet_count,
            'passive_action_count': self.round.passive_action_count,
            'round_bets': list(self.round.round_bets),
            'current_bet_size': self.round.current_bet_size,
            'done': self.done,
            'dealer_deck': list(self.dealer.deck)
        }
    
    def restore_state(self, saved_state):
        self.players[0].private_card = saved_state['hand_p0']
        self.players[1].private_card = saved_state['hand_p1']
        self.public_card = saved_state['public_card']
        self.history = list(saved_state['history'])
        self.current_player = saved_state['current_player']
        self.betting_round = saved_state['betting_round']
        self.pot = saved_state['pot']
        self.total_bets = list(saved_state['total_bets'])
        self.round.bet_count = saved_state['bet_count']
        self.round.passive_action_count = saved_state['passive_action_count']
        self.round.round_bets = list(saved_state['round_bets'])
        self.round.current_bet_size = saved_state['current_bet_size']
        self.done = saved_state['done']
        self.dealer.deck = list(saved_state['dealer_deck'])
    
    def get_info_set_key(self, player_id):
        return (
            self.players[player_id].private_card,
            self.public_card if self.public_card is not None else 'None',
            tuple(self.history),
            self.current_player
        )
    
    def get_payoff(self, player_id):
        if not self.done:
            raise ValueError("Game isnt done yet")
        
        payoffs = self.judger.judge(self.players, self.history, self.current_player, self.pot, self.total_bets)
        return payoffs[player_id]
