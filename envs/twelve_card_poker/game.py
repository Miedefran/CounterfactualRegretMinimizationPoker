from envs.rhode_island.game import RhodeIslandGame
from envs.twelve_card_poker.dealer import TwelveCardPokerDealer
from envs.twelve_card_poker.player import TwelveCardPokerPlayer
from envs.twelve_card_poker.judger import TwelveCardPokerJudger
from envs.twelve_card_poker.round import TwelveCardPokerRound

class TwelveCardPokerGame(RhodeIslandGame):
   
    def __init__(self, ante=1, bet_sizes=[2, 4, 8], bet_limit=2):
        super().__init__(ante, bet_sizes, bet_limit)
        self.dealer = TwelveCardPokerDealer()
        self.players = [TwelveCardPokerPlayer(0), TwelveCardPokerPlayer(1)]
        self.judger = TwelveCardPokerJudger()
        self.round = TwelveCardPokerRound(bet_sizes=bet_sizes, bet_limit=bet_limit)
