from envs.leduc_holdem.game import LeducHoldemGame
from envs.rhode_island.dealer import RhodeIslandDealer
from envs.rhode_island.player import RhodeIslandPlayer
from envs.rhode_island.judger import RhodeIslandJudger
from envs.rhode_island.round import RhodeIslandRound

class RhodeIslandGame(LeducHoldemGame):
   
    def __init__(self, ante=5, bet_sizes=[10, 20, 20], bet_limit=3):
        super().__init__(ante, bet_sizes, bet_limit)
        self.dealer = RhodeIslandDealer()
        self.players = [RhodeIslandPlayer(0), RhodeIslandPlayer(1)]
        self.judger = RhodeIslandJudger()
        self.round = RhodeIslandRound(bet_sizes=bet_sizes, bet_limit=bet_limit)
        self.public_cards = []
    
    def reset(self, starting_player):
        super().reset(starting_player)
        self.public_cards = []
        
    def step(self, action):
        self.state_stack.append(self.save_state())
        self.history.append(action)
    
        self.round.proceed_round(self, action)
        
        if action == 'fold':
            self.done = True
            # proceed_round flipped the player, so the one who folded is 1 - current_player
            return self.judger.judge(self.players, self.history, 1 - self.current_player, self.pot, self.total_bets)
        
        if self.round.is_round_complete():
            if self.betting_round == 0:
                self.history.append('|')
                self.deal_public_card()
                self.betting_round = 1
                self.round.start_new_round(self, self.starting_player, self.betting_round)
            elif self.betting_round == 1:
                self.history.append('|')
                self.deal_public_card()
                self.betting_round = 2
                self.round.start_new_round(self, self.starting_player, self.betting_round)
            else:
                self.done = True
                return self.judger.judge(self.players, self.history, self.current_player, self.pot, self.total_bets)
        
        return None
    
    def deal_public_card(self):
        card = self.dealer.deal_card()
        self.public_cards.append(card)
        self.players[0].public_cards.append(card)
        self.players[1].public_cards.append(card)
    
    
    def get_state(self, player_id):
        return {
            'hand': self.players[player_id].private_card,
            'public_cards': list(self.public_cards),
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
            'public_cards': list(self.public_cards),
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
        self.public_cards = list(saved_state['public_cards'])
        self.players[0].public_cards = list(saved_state['public_cards'])
        self.players[1].public_cards = list(saved_state['public_cards'])
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
            tuple(self.public_cards) if self.public_cards else (),
            tuple(self.history),
            self.current_player
        )
        
        


