from envs.kuhn_poker.game import KuhnPokerGame
from envs.leduc_holdem.dealer import LeducHoldemDealer
from envs.leduc_holdem.player import LeducHoldemPlayer
from envs.leduc_holdem.judger import LeducHoldemJudger
from envs.leduc_holdem.round import LeducHoldemRound
from utils.poker_utils import LeducHoldemCombinations, LeducHoldemCombinationsAbstracted


class LeducHoldemGame(KuhnPokerGame):
    combination_generator = LeducHoldemCombinations()

    @classmethod
    def game_name(cls) -> str:
        return 'leduc'

    def __init__(self):
        super().__init__()
        self._ante = 1
        self._bet_sizes = [2, 4]
        self._bet_limit = 2
        self.dealer = LeducHoldemDealer()
        self.players = [LeducHoldemPlayer(0), LeducHoldemPlayer(1)]
        self.judger = LeducHoldemJudger()
        self.round = LeducHoldemRound(bet_sizes=self._bet_sizes, bet_limit=self._bet_limit)
        self.public_card = None
        self.total_bets = [0, 0]

    def reset(self, starting_player):
        super().reset(starting_player)
        # If suit abstraction is enabled, use rank-only deck with multiplicities.
        # Leduc: 3 ranks, 2 copies each -> 6 cards total.
        if self.suit_abstraction:
            self.dealer.deck = ['J', 'J', 'Q', 'Q', 'K', 'K']
            self.dealer.shuffle()
        self._betting_round = 0
        self.public_card = None
        self.starting_player = starting_player
        self.total_bets = [self._ante, self._ante]
        # Round init happens after private deal chance resolves

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
                # Start explicit public-deal chance node
                self._start_public_deal_chance(next_betting_round=1, num_cards=1)

            else:
                self.done = True
                return self.judger.judge(self.players, self.history, self.current_player, self.pot, self.total_bets)

        return None

    def _start_public_deal_chance(self, next_betting_round: int, num_cards: int):
        self._chance_targets = [('public', None)] * num_cards
        self._chance_context = {'type': 'after_public', 'next_betting_round': next_betting_round}
        self.current_player = self.CHANCE_PLAYER

    def get_legal_actions(self):
        if self.is_chance_node():
            return list(self.get_chance_outcomes_with_probs().keys())
        return self.round.get_legal_actions(self)

    def _after_private_deal(self):
        # After private deal, start betting round 0
        self.round.start_new_round(self, self.starting_player, self._betting_round)

    def _after_public_deal(self, context: dict):
        # After public card is revealed, advance round and start next betting round
        self._betting_round = int(context.get('next_betting_round', self._betting_round))
        self.round.start_new_round(self, self.starting_player, self._betting_round)

    def _apply_public_card(self, card: str):
        self.public_card = card
        self.players[0].set_public_card(card)
        self.players[1].set_public_card(card)

    def get_state(self, player_id):
        return {
            'hand': self.players[player_id].private_card,
            'public_card': self.public_card,
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
            'public_card': self.public_card,
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
        self.public_card = saved_state['public_card']
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
        # Keep player public cards consistent
        self.players[0].set_public_card(self.public_card)
        self.players[1].set_public_card(self.public_card)

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

        judge_player = self.current_player
        if self.history and self.history[-1] == 'fold':
            judge_player = 1 - self.current_player

        payoffs = self.judger.judge(self.players, self.history, judge_player, self.pot, self.total_bets)
        return payoffs[player_id]


class LeducHoldemAbstractedGame(LeducHoldemGame):
    suit_abstraction = True
    combination_generator = LeducHoldemCombinationsAbstracted()

