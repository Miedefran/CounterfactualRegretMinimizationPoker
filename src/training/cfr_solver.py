import pickle as pkl
import gzip
import time

from utils.data_models import KeyGenerator

class CFRSolver:
    
    def __init__(self, game, combination_generator):
        self.game = game
        self.combination_generator = combination_generator
        self.combinations = combination_generator.get_all_combinations()
        self.regret_sum = {}
        self.strategy_sum = {}
        self.iteration_count = 0
        self.training_time = 0
    
    def ensure_init(self, info_set_key, legal_actions):
        if info_set_key not in self.regret_sum:
            self.regret_sum[info_set_key] = {a: 0.0 for a in legal_actions}
        if info_set_key not in self.strategy_sum:
            self.strategy_sum[info_set_key] = {a: 0.0 for a in legal_actions}
    
    def train(self, iterations, br_tracker=None):
        """
        Trainiert den CFR Solver.
        
        Args:
            iterations: Anzahl der Training-Iterationen
            br_tracker: Optionaler BestResponseTracker für Best Response Evaluation
        """
        start_time = time.time()
        
        for i in range(iterations):
            self.cfr_iteration()
            self.iteration_count += 1
            
            if i % 100 == 0:
                print(f"Iteration {i}")
            
            # Best Response Evaluation
            if br_tracker is not None and br_tracker.should_evaluate(i + 1):
                current_avg_strategy = self.get_average_strategy()
                # Zeit wird automatisch in evaluate_and_add berechnet wenn start_time gegeben
                br_tracker.evaluate_and_add(current_avg_strategy, i + 1, start_time=start_time)
                br_tracker.last_eval_iteration = i + 1
        
        # Finale Best Response Evaluation
        if br_tracker is not None:
            current_avg_strategy = self.get_average_strategy()
            # Zeit wird automatisch in evaluate_and_add berechnet wenn start_time gegeben
            br_tracker.evaluate_and_add(current_avg_strategy, iterations, start_time=start_time)
        
        total_time = time.time() - start_time
        
        # Ziehe Best Response Zeit von der Trainingszeit ab
        if br_tracker is not None:
            br_time = br_tracker.get_total_br_time()
            self.training_time = total_time - br_time
            if br_time > 0:
                print(f"Best Response Evaluation Zeit: {br_time:.2f}s")
        else:
            self.training_time = total_time
        
        if self.training_time >= 60:
            minutes = self.training_time / 60
            print(f"Training completed in {minutes:.2f} minutes (ohne Best Response Evaluation)")
        else:
            print(f"Training completed in {self.training_time:.2f} seconds (ohne Best Response Evaluation)")
        
        self.average_strategy = self.get_average_strategy()
    
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
            opponent_info_set = KeyGenerator.get_info_set_key(self.game, opponent)
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
        info_set_key = KeyGenerator.get_info_set_key(self.game, player_id)
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
        
        counterfactual_weight = reach_probabilities[1 - player_id]
        player_reach = reach_probabilities[player_id]
        
        self.update_regrets(info_set_key, legal_actions, action_utilities, current_utility, counterfactual_weight)
        self.update_strategy_sum(info_set_key, legal_actions, current_strategy, player_reach)
        
        return current_utility
    
    def update_regrets(self, info_set_key, legal_actions, action_utilities, current_utility, counterfactual_weight):
        for action in legal_actions:
            instantaneous_regret = counterfactual_weight * (action_utilities[action] - current_utility)
            self.regret_sum[info_set_key][action] += instantaneous_regret
    
    def update_strategy_sum(self, info_set_key, legal_actions, current_strategy, player_reach):
        for action in legal_actions:
            self.strategy_sum[info_set_key][action] += player_reach * current_strategy[action]
    
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
        
    @staticmethod
    def average_from_strategy_sum(strategy_sum):
        average_strategy = {}
        
        for info_set_key in strategy_sum:
            total = sum(strategy_sum[info_set_key].values())
            if total > 0:
                average_strategy[info_set_key] = {
                    action: strategy_sum[info_set_key][action] / total
                    for action in strategy_sum[info_set_key]
                }
            else:
                num_actions = len(strategy_sum[info_set_key])
                average_strategy[info_set_key] = {
                    action: 1.0 / num_actions
                    for action in strategy_sum[info_set_key]
                }
        
        return average_strategy
    
    def get_average_strategy(self):
        """Equation 4 from Zinkevich et al. (2007)"""
        return self.average_from_strategy_sum(self.strategy_sum)
    
    
    
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
        
        self.regret_sum = data['regret_sum']
        self.strategy_sum = data['strategy_sum']
        self.average_strategy = data['average_strategy']
        self.iteration_count = data['iteration_count']
        self.training_time = data.get('training_time', 0)
        
        print(f"Loaded from {filepath}")
