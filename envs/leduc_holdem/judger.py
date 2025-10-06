from envs.kuhn_poker.judger import KuhnPokerJudger

class LeducHoldemJudger(KuhnPokerJudger):
  
    def __init__(self):
        super().__init__()

    def judge(self, players, history, current_player):
        last_action = history[-1]
        if last_action == 'fold':
            fold_player = 1 - current_player
            winner = current_player
            if winner == 0:
                return [1, -1]
            else:
                return [-1, 1]
        
        hand0 = self._evaluate_hand(players[0])
        hand1 = self._evaluate_hand(players[1])
        
        if hand0 > hand1:
            return [1, -1]  
        elif hand1 > hand0:
            return [-1, 1]  
        else:
            return [0, 0]  
    
    def _evaluate_hand(self, player):
        private = player.private_card
        public = player.public_card
        
        if private == public:
            return (1, self.hand_rank[private], 0)
        
        cards = sorted([self.hand_rank[private], self.hand_rank[public]], reverse=True)
        return (0, cards[0], cards[1])  
