import random
from typing import Dict, Any, List


class RandomAgent:

    def __init__(self, player_id):
        self.player_id = player_id

    def get_action(self, state):
        legal_actions = state.get('legal_actions', [])

        if not legal_actions:
            raise ValueError(f"No legal actions available for player {self.player_id}")

        return random.choice(legal_actions)

    def reset(self):
        pass
