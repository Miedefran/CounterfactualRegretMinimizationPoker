from envs.kuhn_poker.player import KuhnPokerPlayer


class KuhnPokerJudger:

    def __init__(self):
        self.hand_rank = {'J': 0, 'Q': 1, 'K': 2}

    def judge(self, players, history, current_player, pot, player_bets):
        if history[-1] == 'fold':
            winner = current_player
            fold_player = 1 - winner
        else:
            if self.hand_rank[players[0].private_card] > self.hand_rank[players[1].private_card]:
                winner = 0
            else:
                winner = 1

        loser = 1 - winner

        payoffs = [0, 0]
        payoffs[winner] = pot - player_bets[winner]
        payoffs[loser] = -player_bets[loser]

        return payoffs
