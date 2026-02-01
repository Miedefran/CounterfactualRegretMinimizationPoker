from envs.kuhn_poker.dealer import KuhnPokerDealer
from envs.kuhn_poker.player import KuhnPokerPlayer
from envs.kuhn_poker.judger import KuhnPokerJudger
from training.registry import TrainingGame
from utils.poker_utils import KuhnPokerCombinations


class KuhnPokerGame(TrainingGame):
    combination_generator = KuhnPokerCombinations()

    @classmethod
    def game_name(cls) -> str:
        return 'kuhn_case2'

    def __init__(self):
        super().__init__()
        self._ante = 1
        self._bet_size = 1
        self.player_bets = [0, 0]
        self.dealer = KuhnPokerDealer()
        self.players = [KuhnPokerPlayer(0), KuhnPokerPlayer(1)]
        self.judger = KuhnPokerJudger()

    def get_big_blind_equivalent(self) -> int:
        return self._ante

    def reset(self, starting_player):
        self.state_stack.clear()
        self.history.clear()
        self.dealer.reset()
        self.dealer.shuffle()
        self.players[0].reset()
        self.players[1].reset()
        self.starting_player = starting_player
        self.done = False
        self.pot = 2 * self._ante
        self.player_bets = [self._ante, self._ante]
        # Start in private-deal chance node (private deals are not public)
        self._setup_private_deal_chance()

    def step(self, action):
        self.state_stack.append(self.save_state())
        # Chance node: deal without writing to history
        if self.is_chance_node():
            self._apply_chance_action(action)
            return None

        self.history.append(action)

        if action == 'bet':
            self.player_bets[self.current_player] += self._bet_size
            self.pot += self._bet_size
        elif action == 'call':
            self.player_bets[self.current_player] += self._bet_size
            self.pot += self._bet_size

        if action == 'fold' or self.history[-2:] == ['bet', 'call'] or self.history[-2:] == ['check', 'check']:
            self.done = True
            if action == 'fold':
                return self.judger.judge(self.players, self.history, 1 - self.current_player, self.pot,
                                         self.player_bets)
            else:
                return self.judger.judge(self.players, self.history, self.current_player, self.pot, self.player_bets)
        else:
            self.current_player = 1 - self.current_player
            return None

    def get_legal_actions(self):
        if self.is_chance_node():
            return list(self.get_chance_outcomes_with_probs().keys())
        if not self.history:
            return ['check', 'bet']
        last_action = self.history[-1]
        if last_action == 'check':
            return ['check', 'bet']
        elif last_action == 'bet':
            return ['call', 'fold']
        return []

    def is_chance_node(self) -> bool:
        return self.current_player == self.CHANCE_PLAYER

    def get_chance_outcomes_with_probs(self):
        """
        Returns a dict {outcome: prob} for the current chance node.
        Outcomes are card symbols; probabilities account for multiplicities in the deck.
        """
        if not self.is_chance_node():
            return {}
        deck = list(self.dealer.deck)
        if not deck:
            return {}
        counts = {}
        for c in deck:
            counts[c] = counts.get(c, 0) + 1
        total = len(deck)
        return {c: cnt / total for c, cnt in counts.items()}

    def _setup_private_deal_chance(self):
        # Default: one private card each (Kuhn/Leduc/Rhode style)
        self._chance_targets = [('private', 0), ('private', 1)]
        self._chance_context = {'type': 'after_private'}
        self.current_player = self.CHANCE_PLAYER

    def _after_private_deal(self):
        # Default: first decision after private deal
        self.current_player = self.starting_player

    def _after_public_deal(self, _context: dict):
        # No public deals in plain Kuhn
        self.current_player = self.starting_player

    def _apply_private_card_to_player(self, player_id: int, card: str):
        self.players[player_id].set_private_card(card)

    def _apply_public_card(self, card: str):
        # Plain Kuhn has no public card, but keep a default implementation
        self.players[0].set_public_card(card)
        self.players[1].set_public_card(card)

    def _apply_chance_action(self, action):
        if not self._chance_targets:
            raise ValueError("Chance node has no pending targets")
        if action not in self.dealer.deck:
            raise ValueError(f"Invalid chance action {action}: not in deck")

        # Remove one instance (supports multiplicities)
        idx = self.dealer.deck.index(action)
        self.dealer.deck.pop(idx)

        kind, player_id = self._chance_targets.pop(0)
        if kind == 'private':
            self._apply_private_card_to_player(player_id, action)
        elif kind == 'public':
            # Should not happen for Kuhn
            self._apply_public_card(action)
        else:
            raise ValueError(f"Unknown chance target kind: {kind}")

        if not self._chance_targets:
            ctx = self._chance_context or {}
            if ctx.get('type') == 'after_private':
                self._after_private_deal()
            elif ctx.get('type') == 'after_public':
                self._after_public_deal(ctx)
            else:
                # Kein Chance-Kontext: mit Startspieler fortfahren
                self.current_player = self.starting_player

    def get_state(self, player_id):
        return {
            'hand': self.players[player_id].private_card,
            'history': list(self.history),
            'current_player': self.current_player,
            'done': self.done,
            'pot': self.pot,
            'player_bets': list(self.player_bets),
            'legal_actions': self.get_legal_actions()
        }

    def save_state(self):
        return {
            'player_hands': [self.players[0].private_card, self.players[1].private_card],
            'history': self.history.copy(),
            'current_player': self.current_player,
            'done': self.done,
            'pot': self.pot,
            'player_bets': self.player_bets.copy(),
            'dealer_deck': list(self.dealer.deck),
            'starting_player': self.starting_player,
            'chance_targets': list(self._chance_targets),
            'chance_context': dict(self._chance_context) if self._chance_context is not None else None,
        }

    def restore_state(self, saved_state):
        self.players[0].private_card = saved_state['player_hands'][0]
        self.players[1].private_card = saved_state['player_hands'][1]
        self.history = saved_state['history']
        self.current_player = saved_state['current_player']
        self.done = saved_state['done']
        self.pot = saved_state['pot']
        self.player_bets = saved_state['player_bets'].copy()
        if 'dealer_deck' in saved_state:
            self.dealer.deck = list(saved_state['dealer_deck'])
        self.starting_player = saved_state.get('starting_player', 0)
        self._chance_targets = list(saved_state.get('chance_targets', []))
        self._chance_context = saved_state.get('chance_context', None)

    def step_back(self):
        if self.state_stack:
            previous_state = self.state_stack.pop()
            self.restore_state(previous_state)
        else:
            raise ValueError("No saved state available for step_back")

    def get_info_set_key(self, player_id):
        return (
            self.players[player_id].private_card,
            tuple(self.history),
            self.current_player
        )

    def get_payoff(self, player_id):
        if not self.done:
            raise ValueError("Game isnt done yet")
        # Bei Fold ist current_player der Folder; der Judger erwartet den Gewinner (den anderen).
        judge_player = self.current_player
        if self.history and self.history[-1] == 'fold':
            judge_player = 1 - self.current_player
        payoffs = self.judger.judge(self.players, self.history, judge_player, self.pot, self.player_bets)
        return payoffs[player_id]


class KuhnCase1Game(KuhnPokerGame):
    @classmethod
    def game_name(cls) -> str:
        return 'kuhn_case1'

    def __init__(self):
        super().__init__()
        self._ante = 2

class KuhnCase3Game(KuhnPokerGame):
    @classmethod
    def game_name(cls) -> str:
        return 'kuhn_case3'

    def __init__(self):
        super().__init__()
        self._bet_size = 1.5


class KuhnCase4Game(KuhnPokerGame):
    @classmethod
    def game_name(cls) -> str:
        return 'kuhn_case4'

    def __init__(self):
        super().__init__()
        self._bet_size = 2
