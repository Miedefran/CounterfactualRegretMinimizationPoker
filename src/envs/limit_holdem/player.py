from envs.royal_holdem.player import RoyalHoldemPlayer


class LimitHoldemPlayer(RoyalHoldemPlayer):

    def __init__(self, player_id: int):
        super().__init__(player_id)
