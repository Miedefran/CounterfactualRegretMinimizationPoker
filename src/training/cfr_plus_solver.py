import gzip
import pickle as pkl
from training.cfr_solver import CFRSolver

class CFRPlusSolver(CFRSolver):
    
    def __init__(self, game, combination_generator):
        super().__init__(game, combination_generator)
        self.Q = {}
    
    def ensure_init(self, info_set_key, legal_actions):
        super().ensure_init(info_set_key, legal_actions)
        if info_set_key not in self.Q:
            self.Q[info_set_key] = {a: 0.0 for a in legal_actions}
    
    def cfr_iteration(self):
        for combination in self.combinations:
            self.combination_generator.setup_game_with_combination(self.game, combination)
            
            reach_probs = [1.0, 1.0]
            
            if self.iteration_count % 2 == 0:
                self.traverse_game_tree(0, reach_probs)
            else:
                self.traverse_game_tree(1, reach_probs)
    
    def update_regrets(self, info_set_key, legal_actions, action_utilities, current_utility, counterfactual_weight):
        for action in legal_actions:
            instantaneous_regret = counterfactual_weight * (action_utilities[action] - current_utility)
            self.Q[info_set_key][action] = max(self.Q[info_set_key][action] + instantaneous_regret, 0)
    
    def update_strategy_sum(self, info_set_key, legal_actions, current_strategy, player_reach):
        for action in legal_actions:
            self.strategy_sum[info_set_key][action] += self.iteration_count * player_reach * current_strategy[action]
    
    def get_current_strategy(self, info_set_key, legal_actions):
        Q_values = {a: self.Q[info_set_key][a] for a in legal_actions}
        sum_Q = sum(Q_values.values())
        
        if sum_Q > 0:
            return {a: Q_values[a] / sum_Q for a in legal_actions}
        else:
            return {a: 1.0 / len(legal_actions) for a in legal_actions}
    
    def save_gzip(self, filepath):
        data = {
            'Q': self.Q,
            'strategy_sum': self.strategy_sum,
            'average_strategy': self.average_strategy,
            'iteration_count': self.iteration_count,
            'training_time': self.training_time
        }
        
        with gzip.open(filepath, 'wb') as f:
            pkl.dump(data, f)
        
        print(f"Saved to {filepath}")
    
    def load_gzip(self, filepath):
        with gzip.open(filepath, 'rb') as f:
            data = pkl.load(f)
        
        self.Q = data['Q']
        self.strategy_sum = data['strategy_sum']
        self.average_strategy = data['average_strategy']
        self.iteration_count = data['iteration_count']
        self.training_time = data.get('training_time', 0)
        
        print(f"Loaded from {filepath}")
