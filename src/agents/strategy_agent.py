import random
from utils.data_models import KeyGenerator, infoset_key_to_suit_abstracted


class StrategyAgent:

    def __init__(self, strategy, player_id, game=None):
        self.strategy = strategy
        self.player_id = player_id
        self.game = game

    def get_action(self, state):
        legal_actions = state.get('legal_actions')
        if legal_actions is None and self.game and not getattr(self.game, 'done', True):
            legal_actions = self.game.get_legal_actions()
        if not legal_actions:
            legal_actions = ['check']

        if self.game:
            try:
                info_set_key = KeyGenerator.get_info_set_key(self.game, self.player_id)
            except Exception:
                info_set_key = self.game.get_info_set_key(self.player_id)
        else:
            card = state['hand']
            history = tuple(state['history'])
            current_player = state['current_player']
            info_set_key = (card, history, current_player)

        # Lookup: zuerst exakter Key, dann suit-abstracted Key (Training oft mit abstract_suits=True, GUI mit vollen Suits)
        lookup_key = info_set_key
        if info_set_key not in self.strategy:
            lookup_key = infoset_key_to_suit_abstracted(info_set_key)
        if lookup_key not in self.strategy:
            return random.choice(legal_actions)

        action_probs = self.strategy[lookup_key]
        legal_probs = [(a, action_probs[a]) for a in legal_actions if a in action_probs and action_probs[a] > 0]
        if not legal_probs:
            return random.choice(legal_actions)
        actions = [a for a, _ in legal_probs]
        probs = [p for _, p in legal_probs]
        total = sum(probs)
        if total <= 0:
            return random.choice(legal_actions)
        probs = [p / total for p in probs]
        return random.choices(actions, weights=probs)[0]
