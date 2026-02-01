from envs.royal_holdem.game import RoyalHoldemGame
from envs.limit_holdem.dealer import LimitHoldemDealer
from envs.limit_holdem.player import LimitHoldemPlayer
from envs.limit_holdem.judger import LimitHoldemJudger
from envs.limit_holdem.round import LimitHoldemRound
from utils.poker_utils import LimitHoldemCombinations


class LimitHoldemGame(RoyalHoldemGame):
    combination_generator = LimitHoldemCombinations()

    @classmethod
    def game_name(cls) -> str:
        return 'limit_holdem'

    def __init__(self):
        super().__init__()
        self._bet_sizes = [10, 10, 20, 20]
        self._bet_limit = 4
        self._small_blind = 5
        self._big_blind = 10
        self._abstract_suits = False
        self.dealer = LimitHoldemDealer()
        self.players = [LimitHoldemPlayer(0), LimitHoldemPlayer(1)]
        self.judger = LimitHoldemJudger()
        self.round = LimitHoldemRound(bet_sizes=self._bet_sizes, bet_limit=self._bet_limit)

    def get_big_blind_equivalent(self) -> int:
        return self._big_blind

    def reset(self, starting_player):
        super().reset(starting_player)
        self.public_cards = []
        # In heads-up Hold'em, the button is the small blind and acts first preflop.
        # We interpret starting_player as the small blind (button).
        self.starting_player = starting_player

        sb = starting_player
        bb = 1 - starting_player

        self.total_bets = [0, 0]
        self.total_bets[sb] = self._small_blind
        self.total_bets[bb] = self._big_blind
        self.pot = self._small_blind + self._big_blind
        # Round init happens after private deal chance resolves (via _after_private_deal)

    def get_info_set_key(self, player_id):
        return (
            tuple(sorted(self.players[player_id].private_cards)),
            tuple(self.public_cards) if self.public_cards else (),
            tuple(self.history),
            self.current_player
        )
