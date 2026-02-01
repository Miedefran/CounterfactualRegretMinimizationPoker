from envs.rhode_island.game import RhodeIslandGame
from envs.twelve_card_poker.dealer import TwelveCardPokerDealer
from envs.twelve_card_poker.player import TwelveCardPokerPlayer
from envs.twelve_card_poker.judger import TwelveCardPokerJudger
from envs.twelve_card_poker.round import TwelveCardPokerRound
from utils.poker_utils import TwelveCardPokerCombinations, TwelveCardPokerCombinationsAbstracted


class TwelveCardPokerGame(RhodeIslandGame):
    combination_generator = TwelveCardPokerCombinations()

    @classmethod
    def game_name(cls) -> str:
        return 'twelve_card_poker'

    def __init__(self):
        super().__init__()
        self._ante = 1
        self._bet_sizes = [2, 4, 8]
        self._bet_limit = 2
        self.dealer = TwelveCardPokerDealer()
        self.players = [TwelveCardPokerPlayer(0), TwelveCardPokerPlayer(1)]
        self.judger = TwelveCardPokerJudger()
        self.round = TwelveCardPokerRound(bet_sizes=self._bet_sizes, bet_limit=self._bet_limit)

    def reset(self, starting_player):
        super().reset(starting_player)
        # If suit abstraction is enabled, use rank-only deck with multiplicities.
        # Twelve Card Poker: 4 ranks, 3 copies each -> 12 cards total.
        if self.suit_abstraction:
            self.dealer.deck = ['J'] * 3 + ['Q'] * 3 + ['K'] * 3 + ['A'] * 3
            self.dealer.shuffle()


class TwelveCardPokerAbstractedGame(TwelveCardPokerGame):
    suit_abstraction = True
    combination_generator = TwelveCardPokerCombinationsAbstracted()

