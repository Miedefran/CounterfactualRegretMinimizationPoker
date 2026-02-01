from envs.rhode_island.player import RhodeIslandPlayer


class RoyalHoldemPlayer(RhodeIslandPlayer):

    def __init__(self, player_id: int):
        super().__init__(player_id)
        self.private_cards = []

    def set_private_cards(self, card1, card2):
        self.private_cards = [card1, card2]

    def reset(self):
        super().reset()
        self.private_cards = []
