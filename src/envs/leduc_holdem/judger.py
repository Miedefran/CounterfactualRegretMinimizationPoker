from envs.kuhn_poker.judger import KuhnPokerJudger


class LeducHoldemJudger(KuhnPokerJudger):

    def __init__(self):
        super().__init__()

    def judge(self, players, history, current_player, pot, player_bets):
        if history[-1] == 'fold':
            # The player passed in (current_player) is the one who acted last (folded)
            fold_player = current_player
            winner = 1 - fold_player  # The winner is the OTHER player
        else:
            # Showdown logic (unchanged)
            hand0 = self.evaluate_hand(players[0])
            hand1 = self.evaluate_hand(players[1])

            if hand0 > hand1:
                winner = 0
            elif hand1 > hand0:
                winner = 1
            else:
                return [0, 0]

        loser = 1 - winner

        payoffs = [0, 0]
        payoffs[winner] = pot - player_bets[winner]
        payoffs[loser] = -player_bets[loser]

        return payoffs

    def evaluate_hand(self, player):
        private = player.private_card
        public = player.public_card

        private_rank = private[0] if len(private) > 1 else private
        public_rank = public[0] if len(public) > 1 else public

        if private_rank == public_rank:
            return (1, self.hand_rank[private_rank], 0)

        cards = sorted([self.hand_rank[private_rank], self.hand_rank[public_rank]], reverse=True)
        return (0, cards[0], cards[1])
