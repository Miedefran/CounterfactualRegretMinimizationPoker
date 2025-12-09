from envs.royal_holdem.game import RoyalHoldemGame
from envs.limit_holdem.dealer import LimitHoldemDealer
from envs.limit_holdem.player import LimitHoldemPlayer
from envs.limit_holdem.judger import LimitHoldemJudger
from envs.limit_holdem.round import LimitHoldemRound

class LimitHoldemGame(RoyalHoldemGame):
   
    def __init__(self, small_blind=5, big_blind=10, bet_sizes=[10, 10, 20, 20], bet_limit=4):
        super().__init__(ante=0, bet_sizes=bet_sizes, bet_limit=bet_limit)
        self.dealer = LimitHoldemDealer()
        self.players = [LimitHoldemPlayer(0), LimitHoldemPlayer(1)]
        self.judger = LimitHoldemJudger()
        self.round = LimitHoldemRound(bet_sizes=bet_sizes, bet_limit=bet_limit)
        self.public_cards = []
        self.small_blind = small_blind
        self.big_blind = big_blind
    
    def reset(self, starting_player):
        super().reset(starting_player)
        self.public_cards = []
        self.total_bets = [self.small_blind, self.big_blind]
        self.pot = self.small_blind + self.big_blind
        self.round.round_bets = [self.small_blind, self.big_blind]
        self.current_player = starting_player
        self.starting_player = starting_player
        self.round.start_new_round(starting_player, self.betting_round)
    
    def get_info_set_key(self, player_id):
        return (
            tuple(sorted(self.players[player_id].private_cards)),
            tuple(self.public_cards) if self.public_cards else (),
            tuple(self.history),
            self.current_player
        )

