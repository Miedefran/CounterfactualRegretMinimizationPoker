import random
from training.cfr_solver import CFRSolver

class MCCFRSolver(CFRSolver):
    
    def cfr_iteration(self):
        combo = random.choice(self.combinations)
        self.combination_generator.setup_game_with_combination(self.game, combo)
        
        reach_probs = [1.0, 1.0]
        
        self.traverse_game_tree(0, reach_probs)
        self.traverse_game_tree(1, reach_probs)
