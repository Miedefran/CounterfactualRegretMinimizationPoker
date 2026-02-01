from envs.leduc_holdem.game import LeducHoldemGame
from envs.small_island_holdem.dealer import SmallIslandHoldemDealer
from envs.small_island_holdem.player import SmallIslandHoldemPlayer
from envs.small_island_holdem.judger import SmallIslandHoldemJudger
from envs.small_island_holdem.round import SmallIslandHoldemRound
from utils.poker_utils import SmallIslandHoldemCombinations


class SmallIslandHoldemGame(LeducHoldemGame):
    combination_generator = SmallIslandHoldemCombinations()

    @classmethod
    def game_name(cls) -> str:
        return 'small_island_holdem'

    def __init__(self):
        super().__init__()
        self._ante = 5
        self._bet_sizes = [10, 20, 20]
        self._bet_limit = 2
        self.dealer = SmallIslandHoldemDealer()
        self.players = [SmallIslandHoldemPlayer(0), SmallIslandHoldemPlayer(1)]
        self.judger = SmallIslandHoldemJudger()
        self.round = SmallIslandHoldemRound(bet_sizes=self._bet_sizes, bet_limit=self._bet_limit)
        self.public_cards = []

    def reset(self, starting_player):
        super().reset(starting_player)
        self.public_cards = []

    def step(self, action):
        self.state_stack.append(self.save_state())
        if self.is_chance_node():
            self._apply_chance_action(action)
            return None

        self.history.append(action)

        self.round.proceed_round(self, action)

        if action == 'fold':
            self.done = True
            # proceed_round flipped the player, so the one who folded is 1 - current_player
            return self.judger.judge(self.players, self.history, 1 - self.current_player, self.pot, self.total_bets)

        if self.round.is_round_complete():
            if self._betting_round == 0:
                self._start_public_deal_chance(next_betting_round=1, num_cards=1)
            elif self._betting_round == 1:
                self._start_public_deal_chance(next_betting_round=2, num_cards=1)
            else:
                self.done = True
                return self.judger.judge(self.players, self.history, self.current_player, self.pot, self.total_bets)

        return None

    def _start_public_deal_chance(self, next_betting_round: int, num_cards: int):
        self._chance_targets = [('public', None)] * num_cards
        self._chance_context = {'type': 'after_public', 'next_betting_round': next_betting_round}
        self.current_player = self.CHANCE_PLAYER

    def _apply_public_card(self, card: str):
        self.public_cards.append(card)
        self.players[0].public_cards.append(card)
        self.players[1].public_cards.append(card)

    def get_state(self, player_id):
        return {
            'hand': self.players[player_id].private_card,
            'public_cards': list(self.public_cards),
            'history': list(self.history),
            'current_player': self.current_player,
            'betting_round': self._betting_round,
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
            'betting_round': self._betting_round,
            'pot': self.pot,
            'total_bets': list(self.total_bets),
            'bet_count': self.round.bet_count,
            'passive_action_count': self.round.passive_action_count,
            'round_bets': list(self.round.round_bets),
            'current_bet_size': self.round.current_bet_size,
            'done': self.done,
            'dealer_deck': list(self.dealer.deck),
            'starting_player': self.starting_player,
            'chance_targets': list(self._chance_targets),
            'chance_context': dict(self._chance_context) if self._chance_context is not None else None,
        }

    def restore_state(self, saved_state):
        self.players[0].private_card = saved_state['hand_p0']
        self.players[1].private_card = saved_state['hand_p1']
        self.public_cards = list(saved_state['public_cards'])
        self.players[0].public_cards = list(saved_state['public_cards'])
        self.players[1].public_cards = list(saved_state['public_cards'])
        self.history = list(saved_state['history'])
        self.current_player = saved_state['current_player']
        self._betting_round = saved_state['betting_round']
        self.pot = saved_state['pot']
        self.total_bets = list(saved_state['total_bets'])
        self.round.bet_count = saved_state['bet_count']
        self.round.passive_action_count = saved_state['passive_action_count']
        self.round.round_bets = list(saved_state['round_bets'])
        self.round.current_bet_size = saved_state['current_bet_size']
        self.done = saved_state['done']
        self.dealer.deck = list(saved_state['dealer_deck'])
        self.starting_player = saved_state.get('starting_player', 0)
        self._chance_targets = list(saved_state.get('chance_targets', []))
        self._chance_context = saved_state.get('chance_context', None)

    def get_info_set_key(self, player_id):
        return (
            self.players[player_id].private_card,
            tuple(self.public_cards) if self.public_cards else (),
            tuple(self.history),
            self.current_player
        )
