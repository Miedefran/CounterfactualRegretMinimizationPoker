from envs.kuhn_poker.player import KuhnPokerPlayer


class LeducHoldemPlayer(KuhnPokerPlayer):

    def __init__(self, player_id: int):
        super().__init__(player_id)
        self.private_card = None
        self.public_card = None
