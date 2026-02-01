from envs.rhode_island.game import RhodeIslandGame
from envs.royal_holdem.dealer import RoyalHoldemDealer
from envs.royal_holdem.player import RoyalHoldemPlayer
from envs.royal_holdem.judger import RoyalHoldemJudger
from envs.royal_holdem.round import RoyalHoldemRound
from utils.poker_utils import RoyalHoldemCombinations


class RoyalHoldemGame(RhodeIslandGame):
    combination_generator = RoyalHoldemCombinations()

    @classmethod
    def game_name(cls) -> str:
        return 'royal_holdem'

    def __init__(self):
        super().__init__()
        self._ante = 1
        self._bet_sizes = [2, 4, 8, 8]
        self._bet_limit = 3
        self.dealer = RoyalHoldemDealer()
        self.players = [RoyalHoldemPlayer(0), RoyalHoldemPlayer(1)]
        self.judger = RoyalHoldemJudger()
        self.round = RoyalHoldemRound(bet_sizes=self._bet_sizes, bet_limit=self._bet_limit)

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
            return self.judger.judge(self.players, self.history, 1 - self.current_player, self.pot, self.total_bets)

        if self.round.is_round_complete():
            if self._betting_round == 0:
                self._start_public_deal_chance(next_betting_round=1, num_cards=3)
            elif self._betting_round == 1:
                self._start_public_deal_chance(next_betting_round=2, num_cards=1)
            elif self._betting_round == 2:
                self._start_public_deal_chance(next_betting_round=3, num_cards=1)
            else:
                self.done = True
                return self.judger.judge(self.players, self.history, self.current_player, self.pot, self.total_bets)

        return None

    def _setup_private_deal_chance(self):
        # Royal/Limit: 2 private cards per player
        self._chance_targets = [('private', 0), ('private', 0), ('private', 1), ('private', 1)]
        self._chance_context = {'type': 'after_private'}
        self.current_player = self.CHANCE_PLAYER

    def _apply_private_card_to_player(self, player_id: int, card: str):
        p = self.players[player_id]
        if not hasattr(p, 'private_cards'):
            p.private_cards = []
        p.private_cards.append(card)
        # Keep legacy attribute for any old code paths
        if getattr(p, 'private_card', None) is None:
            p.private_card = card

    def get_state(self, player_id):
        return {
            'hand': self.players[player_id].private_cards,
            'public_cards': list(self.public_cards),
            'history': list(self.history),
            'current_player': self.current_player,
            'betting_round': self._betting_round,
            'legal_actions': self.get_legal_actions(),
            'pot': self.pot,
            'done': self.done,
            'player_bets': list(self.total_bets),
            'bet_count': self.round.bet_count,
            'passive_action_count': self.round.passive_action_count,
        }

    def save_state(self):
        return {
            'hand_p0': self.players[0].private_cards,
            'hand_p1': self.players[1].private_cards,
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
        self.players[0].private_cards = list(saved_state['hand_p0'])
        self.players[1].private_cards = list(saved_state['hand_p1'])
        self.players[0].private_card = self.players[0].private_cards[0] if self.players[0].private_cards else None
        self.players[1].private_card = self.players[1].private_cards[0] if self.players[1].private_cards else None
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
            tuple(sorted(self.players[player_id].private_cards)),
            tuple(self.public_cards) if self.public_cards else (),
            tuple(self.history),
            self.current_player
        )
