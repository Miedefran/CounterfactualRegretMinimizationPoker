import pickle as pkl
import gzip

class CFRSolver:
    
    def __init__(self, game, combination_generator):
        self.game = game
        self.combination_generator = combination_generator
        self.combinations = combination_generator.get_all_combinations()
        self.regret_sum = {}
        self.strategy_sum = {}
        self.iteration_count = 0
    
    def ensure_init(self, info_set_key, legal_actions):
        if info_set_key not in self.regret_sum:
            self.regret_sum[info_set_key] = {a: 0.0 for a in legal_actions}
        if info_set_key not in self.strategy_sum:
            self.strategy_sum[info_set_key] = {a: 0.0 for a in legal_actions}
    
    def train(self, iterations):
        for i in range(iterations):
            self.cfr_iteration()
            self.iteration_count += 1
            
            if i % 1000 == 0:
                print(f"Iteration {i}")
    
    def cfr_iteration(self):
        for combination in self.combinations:
            self.combination_generator.setup_game_with_combination(self.game, combination)
            
            reach_probs = [1.0, 1.0]
            
            self.traverse_game_tree(0, reach_probs)
            self.traverse_game_tree(1, reach_probs)
    
    def traverse_game_tree(self, player_id, reach_probabilities):
        
        if self.game.done:
            return self.game.get_payoff(player_id)
        
        current_player = self.game.current_player
        
        #Opponent's node, don't update regrets
        if current_player != player_id:
            legal_actions = self.game.get_legal_actions()
            opponent = 1 - player_id
            opponent_info_set = self.game.get_info_set_key(opponent)
            self.ensure_init(opponent_info_set, legal_actions)
            opponent_strategy = self.get_current_strategy(opponent_info_set, legal_actions)
            
            state_value = 0.0
            for action in legal_actions:
                action_prob = opponent_strategy[action]
                self.game.step(action)
                
                new_reach_probs = reach_probabilities.copy()
                new_reach_probs[opponent] *= action_prob
                
                state_value += action_prob * self.traverse_game_tree(player_id, new_reach_probs)
                self.game.step_back()
            
            return state_value
        
        #Player's node, update regrets
        info_set_key = self.game.get_info_set_key(player_id)
        legal_actions = self.game.get_legal_actions()
        self.ensure_init(info_set_key, legal_actions)
        current_strategy = self.get_current_strategy(info_set_key, legal_actions)
        
        action_utilities = {}
        for action in legal_actions:
            self.game.step(action)
            
            new_reach_probs = reach_probabilities.copy()
            new_reach_probs[player_id] *= current_strategy[action]
            
            action_utilities[action] = self.traverse_game_tree(player_id, new_reach_probs)
            self.game.step_back()
        
        current_utility = sum(current_strategy[action] * action_utilities[action] for action in legal_actions)
        
        #Update regrets (Equation 7)
        counterfactual_weight = reach_probabilities[1 - player_id]
        for action in legal_actions:
            instantaneous_regret = counterfactual_weight * (action_utilities[action] - current_utility)
            self.regret_sum[info_set_key][action] += instantaneous_regret
        
        #Update strategy sum (Equation 4)
        player_reach = reach_probabilities[player_id]
        for action in legal_actions:
            self.strategy_sum[info_set_key][action] += player_reach * current_strategy[action]
        
        return current_utility
    
    def get_current_strategy(self, info_set_key, legal_actions):
        """Equation 8 from Zinkevich et al. (2007)"""
        
        regrets = {a: self.regret_sum[info_set_key][a] for a in legal_actions}
        
        # Regret Matching, only use positive regrets
        positive_regrets = {a: max(regrets[a], 0) for a in legal_actions}
        sum_pos = sum(positive_regrets.values())
        
        if sum_pos > 0:
            return {a: positive_regrets[a] / sum_pos for a in legal_actions}
        else:
            #If there are no positive regrets, play everything equally
            return {a: 1.0 / len(legal_actions) for a in legal_actions}
        
    def get_average_strategy(self):
        """Equation 4 from Zinkevich et al. (2007)"""
        
        average_strategy = {}
        
        for info_set_key in self.strategy_sum:
            total = sum(self.strategy_sum[info_set_key].values())
            if total > 0:
                average_strategy[info_set_key] = {
                    action: self.strategy_sum[info_set_key][action] / total
                    for action in self.strategy_sum[info_set_key]
                }
            else:
                num_actions = len(self.strategy_sum[info_set_key])
                average_strategy[info_set_key] = {
                    action: 1.0 / num_actions
                    for action in self.strategy_sum[info_set_key]
                }
        
        return average_strategy
    
    
    
    """Storage Methods"""
    
    def save_pickle(self, filepath):
        data = {
            'regret_sum': self.regret_sum,
            'strategy_sum': self.strategy_sum,
            'iteration_count': self.iteration_count
        }
        
        with open(filepath, 'wb') as f:
            pkl.dump(data, f)
        
        print(f"Saved to {filepath}")
    
    def load_pickle(self, filepath):
        with open(filepath, 'rb') as f:
            data = pkl.load(f)
        
        self.regret_sum = data['regret_sum']
        self.strategy_sum = data['strategy_sum']
        self.iteration_count = data['iteration_count']
        
        print(f"Loaded from {filepath}")
    
    def save_gzip(self, filepath):
        data = {
            'regret_sum': self.regret_sum,
            'strategy_sum': self.strategy_sum,
            'iteration_count': self.iteration_count
        }
        
        with gzip.open(filepath, 'wb') as f:
            pkl.dump(data, f)
        
        print(f"Saved to {filepath}")
    
    def load_gzip(self, filepath):
        with gzip.open(filepath, 'rb') as f:
            data = pkl.load(f)
        
        self.regret_sum = data['regret_sum']
        self.strategy_sum = data['strategy_sum']
        self.iteration_count = data['iteration_count']
        
        print(f"Loaded from {filepath}")
