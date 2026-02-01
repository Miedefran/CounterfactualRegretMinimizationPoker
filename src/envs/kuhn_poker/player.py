class KuhnPokerPlayer:

    def __init__(self, player_id: int):
        self.player_id = player_id
        self.reset()

    def set_private_card(self, private_card):
        self.private_card = private_card

    def set_public_card(self, public_card):
        self.public_card = public_card

    def reset(self):
        self.set_private_card(None)
        self.set_public_card(None)
