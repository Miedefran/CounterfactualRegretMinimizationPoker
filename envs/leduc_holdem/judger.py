from envs.kuhn_poker.judger import KuhnPokerJudger

class LeducHoldemJudger(KuhnPokerJudger):
  
    def __init__(self):
        super().__init__()

    def judge(self, players, history, current_player, pot, player_bets):
        if history[-1] == 'fold':
            fold_player = 1 - current_player
            winner = current_player
        else:
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
        
        if private == public:
            return (1, self.hand_rank[private], 0)
        
        cards = sorted([self.hand_rank[private], self.hand_rank[public]], reverse=True)
        return (0, cards[0], cards[1])  
