from envs.kuhn_poker.game import KuhnPokerGame
from envs.leduc_holdem.dealer import LeducHoldemDealer
from envs.leduc_holdem.player import LeducHoldemPlayer
from envs.leduc_holdem.judger import LeducHoldemJudger
from envs.leduc_holdem.round import LeducHoldemRound

class LeducHoldemGame(KuhnPokerGame):
   
    def __init__(self):
        self.ante = 1
        self.bet_size = 2
        self.dealer = LeducHoldemDealer()
        self.players = [LeducHoldemPlayer(0), LeducHoldemPlayer(1)]
        self.judger = LeducHoldemJudger()
        self.round = LeducHoldemRound()
        self.history = []
        self.state_stack = []
        self.pot = 0
        self.player_bets = [0, 0]
        self.done = False
        self.betting_round = 0  
        self.public_card = None
        self.starting_player = 0 

    
    def reset(self, starting_player):
        super().reset(starting_player)
        self.betting_round = 0
        self.public_card = None
        self.starting_player = starting_player 
        
        self.round.start_new_round(starting_player, self.betting_round)
        self.round.pot += 2  
        self.round.total_bets = [1, 1] 
        
    def step(self, action):
        
        self.history.append(action)
    
        self.round.proceed_round(self.players, action)
        
        if action == 'fold':
            self.done = True
            return self.judger.judge(self.players, self.history, self.round.current_player)
        
        if self.round.is_round_complete():
            if self.betting_round == 0:
                self.history.append('|')
                self._deal_public_card()
                self.betting_round = 1
                self.round.start_new_round(self.starting_player, self.betting_round)
                
            else:
                self.done = True
                return self.judger.judge(self.players, self.history, self.round.current_player)
        
        return None
    
    def _deal_public_card(self):
        self.public_card = self.dealer.deal_card()
        self.players[0].set_public_card(self.public_card)
        self.players[1].set_public_card(self.public_card)
    
    def get_legal_actions(self):
        return self.round.get_legal_actions()
    
    def get_state(self, player_id):
        return {
            'hand': self.players[player_id].private_card,
            'public_card': self.public_card,
            'history': list(self.history),
            'current_player': self.round.current_player,
            'betting_round': self.betting_round,
            'legal_actions': self.get_legal_actions(),
            'pot': self.round.pot,
            'player_bets': list(self.round.total_bets),
            'is_terminal': self.is_over()
        }
    
    def is_over(self):
        return self.done
    