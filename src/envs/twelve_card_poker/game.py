from envs.rhode_island.game import RhodeIslandGame
from envs.twelve_card_poker.dealer import TwelveCardPokerDealer
from envs.twelve_card_poker.player import TwelveCardPokerPlayer
from envs.twelve_card_poker.judger import TwelveCardPokerJudger
from envs.twelve_card_poker.round import TwelveCardPokerRound

class TwelveCardPokerGame(RhodeIslandGame):
   
    def __init__(self, ante=1, bet_sizes=[2, 4, 8], bet_limit=2, abstract_suits: bool = False):
        super().__init__(ante, bet_sizes, bet_limit)
        self.abstract_suits = abstract_suits
        self.dealer = TwelveCardPokerDealer()
        self.players = [TwelveCardPokerPlayer(0), TwelveCardPokerPlayer(1)]
        self.judger = TwelveCardPokerJudger()
        self.round = TwelveCardPokerRound(bet_sizes=bet_sizes, bet_limit=bet_limit)

    def reset(self, starting_player):
        super().reset(starting_player)
        # If suit abstraction is enabled, use rank-only deck with multiplicities.
        # Twelve Card Poker: 4 ranks, 3 copies each -> 12 cards total.
        if getattr(self, 'abstract_suits', False):
            self.dealer.deck = ['J'] * 3 + ['Q'] * 3 + ['K'] * 3 + ['A'] * 3
            self.dealer.shuffle()
