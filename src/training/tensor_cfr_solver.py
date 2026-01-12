import torch
import numpy as np
import time
import os
from utils.data_models import KeyGenerator
from training.tensor_game_tree import TensorizedGameTree, build_tensor_tree, get_tree_path

class TensorCFRSolver:
    def __init__(self, game, combination_generator, algorithm='cfr_plus', device=None, game_name=None, load_tree=True):
        self.game = game
        self.combination_generator = combination_generator
        self.algorithm = algorithm
        self.actions = ['check', 'bet', 'call', 'fold']
        self.num_actions = len(self.actions)

        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = device

        print(f"Initializing TensorCFRSolver on {self.device}...")

        self.tree = None
        
        if load_tree and game_name:
            tree_path = get_tree_path(game_name)
            if os.path.exists(tree_path):
                try:
                    self.tree = TensorizedGameTree.load(tree_path)
                    if self.tree.infoset_keys_map is None:
                        print("Tree loaded but missing infoset_keys_map, rebuilding...")
                        self.tree = None
                except Exception as e:
                    print(f"Failed to load tree: {e}")
            else:
                print(f"No existing tensor tree found at {tree_path}")
                
        if self.tree is None:
            self.tree = build_tensor_tree(game, combination_generator)
            if load_tree and game_name:
                tree_path = get_tree_path(game_name)
                self.tree.save(tree_path)

        self._move_tree_to_device()
        self._init_tensors()

    def _move_tree_to_device(self):
        print("Moving tree tensors to device...")
        t0 = time.time()
        
        self.node_types = torch.tensor(self.tree.node_types, device=self.device, dtype=torch.int8)
        self.players = torch.tensor(self.tree.players, device=self.device, dtype=torch.long)
        self.infosets = torch.tensor(self.tree.infosets, device=self.device, dtype=torch.long)
        self.children = torch.tensor(self.tree.children, device=self.device, dtype=torch.long)
        self.payoffs = torch.tensor(self.tree.payoffs, device=self.device, dtype=torch.float32)
        self.roots_tensor = torch.tensor(self.tree.roots, device=self.device, dtype=torch.long)
        
        self.max_depth = int(np.max(self.tree.depths))
        self.layer_indices = []
        for d in range(1, self.max_depth + 1):
            indices = np.where(self.tree.depths == d)[0]
            if len(indices) > 0:
                self.layer_indices.append(torch.tensor(indices, device=self.device, dtype=torch.long))
            else:
                self.layer_indices.append(torch.tensor([], device=self.device, dtype=torch.long))

        num_infosets = self.tree.infoset_counts
        self.num_infosets = num_infosets
        
        valid_actions_per_node = (self.children != -1)
        valid_float = valid_actions_per_node.float()
        
        dec_mask = (self.infosets != -1)
        dec_infosets = self.infosets[dec_mask]
        dec_valid = valid_float[dec_mask]
        
        valid_sums = torch.zeros((num_infosets, self.num_actions), device=self.device, dtype=torch.float32)
        valid_sums.index_add_(0, dec_infosets, dec_valid)
        
        self.infoset_valid_actions = (valid_sums > 0)
        self.infoset_valid_counts = self.infoset_valid_actions.sum(dim=1, keepdim=True).float()
        
        # Pre-compute uniform strategy for invalid regret cases
        self.uniform_strategy = (self.infoset_valid_actions.float() / 
                                  torch.clamp(self.infoset_valid_counts, min=1))
        
        # Pre-compute player masks for nodes
        self.player0_nodes = (self.players == 0)
        self.player1_nodes = (self.players == 1)
        
        self.root_prob = 1.0 / len(self.tree.roots)
        
        print(f"Tree moved to device in {time.time() - t0:.2f}s")

    def _init_tensors(self):
        self.regret_sum = torch.zeros((self.num_infosets, self.num_actions), 
                                       device=self.device, dtype=torch.float32)
        self.strategy_sum = torch.zeros((self.num_infosets, self.num_actions), 
                                         device=self.device, dtype=torch.float32)
        self.t = 0
        self.strategy_reconstruction_time = 0.0
        
        # Pre-allocate work tensors
        num_nodes = len(self.tree.node_types)
        self.nodes_reach = torch.zeros((num_nodes, 2), device=self.device, dtype=torch.float32)
        self.nodes_values = torch.zeros((num_nodes, 2), device=self.device, dtype=torch.float32)
        self.delta_regret = torch.zeros_like(self.regret_sum)

    def _get_current_strategy(self):
        """Compute current strategy from regrets using regret matching."""
        # Work on a copy to avoid modifying regret_sum
        positive_regrets = torch.clamp(self.regret_sum, min=0)
        
        # Zero out invalid actions
        positive_regrets = positive_regrets * self.infoset_valid_actions.float()
        
        sum_pos = positive_regrets.sum(dim=1, keepdim=True)
        has_pos = (sum_pos > 1e-12).squeeze()
        
        current_strategy = torch.zeros_like(self.regret_sum)
        
        # Where we have positive regrets, normalize
        current_strategy[has_pos] = positive_regrets[has_pos] / sum_pos[has_pos]
        
        # Where we don't, use uniform over valid actions
        current_strategy[~has_pos] = self.uniform_strategy[~has_pos]
        
        return current_strategy

    def train(self, iterations, br_tracker=None, print_interval=100):
        print(f"Starting Training for {iterations} iterations on {self.device}...")
        start_time = time.time()
        
        num_nodes = len(self.tree.node_types)
        
        with torch.no_grad():
            for i in range(1, iterations + 1):
                self.t = i
                
                if self.algorithm == 'cfr_plus':
                    # Alternating updates like Python CFR+
                    self._cfr_iteration_for_player(num_nodes, updating_player=0)
                    self.regret_sum = torch.clamp(self.regret_sum, min=0)
                    
                    self._cfr_iteration_for_player(num_nodes, updating_player=1)
                    self.regret_sum = torch.clamp(self.regret_sum, min=0)
                else:
                    # Standard CFR: simultaneous update
                    self._cfr_iteration_simultaneous(num_nodes)
                
                if i % print_interval == 0:
                    print(f"Iteration {i}")
                
                if br_tracker is not None and br_tracker.should_evaluate(i):
                    current_avg_strategy = self.get_average_strategy()
                    br_tracker.evaluate_and_add(current_avg_strategy, i, start_time=start_time)
                    br_tracker.last_eval_iteration = i
        
        if br_tracker is not None:
            current_avg_strategy = self.get_average_strategy()
            br_tracker.evaluate_and_add(current_avg_strategy, iterations, start_time=start_time)
        
        total_time = time.time() - start_time
        
        if br_tracker is not None:
            br_time = br_tracker.get_total_br_time()
            self.training_time = total_time - br_time - self.strategy_reconstruction_time
            if br_time > 0:
                print(f"Best Response Evaluation Zeit: {br_time:.2f}s")
        else:
            self.training_time = total_time - self.strategy_reconstruction_time
        
        if self.strategy_reconstruction_time > 0:
            print(f"Strategy Reconstruction Zeit: {self.strategy_reconstruction_time:.2f}s")
        
        if self.training_time >= 60:
            minutes = self.training_time / 60
            print(f"Training completed in {minutes:.2f} minutes (ohne Best Response Evaluation)")
        else:
            print(f"Training completed in {self.training_time:.2f} seconds (ohne Best Response Evaluation)")
        
        self.average_strategy = self.get_average_strategy()

    def _cfr_iteration_for_player(self, num_nodes, updating_player):
        """
        CFR+ iteration that only updates regrets for one player.
        This mimics the alternating update scheme of the Python implementation.
        """
        current_strategy = self._get_current_strategy()
        
        # Reset work tensors
        self.nodes_reach.zero_()
        self.nodes_values.zero_()
        self.delta_regret.zero_()
        
        # Initialize roots
        self.nodes_reach[self.roots_tensor, 0] = self.root_prob
        self.nodes_reach[self.roots_tensor, 1] = self.root_prob
        
        # Forward pass: compute reach probabilities
        for layer_idx in self.layer_indices:
            if len(layer_idx) == 0:
                continue
            
            node_types_layer = self.node_types[layer_idx]
            decision_mask = (node_types_layer == 1)
            decision_nodes = layer_idx[decision_mask]
            
            if len(decision_nodes) == 0:
                continue
            
            p_ids = self.players[decision_nodes]
            inf_ids = self.infosets[decision_nodes]
            child_ids = self.children[decision_nodes]
            
            node_strat = current_strategy[inf_ids]
            
            # Strategy sum accumulation (linear weighting for CFR+)
            reach_p = self.nodes_reach[decision_nodes].gather(1, p_ids.unsqueeze(1)).squeeze(1)
            contrib = node_strat * reach_p.unsqueeze(1) * self.t
            self.strategy_sum.index_add_(0, inf_ids, contrib)
            
            # Propagate reach to children
            multipliers = torch.ones((len(decision_nodes), self.num_actions, 2), device=self.device)
            
            mask0 = (p_ids == 0)
            multipliers[mask0, :, 0] = node_strat[mask0]
            
            mask1 = (p_ids == 1)
            multipliers[mask1, :, 1] = node_strat[mask1]
            
            parent_reach_expanded = self.nodes_reach[decision_nodes].unsqueeze(1)
            child_reach_vals = parent_reach_expanded * multipliers
            
            valid = (child_ids != -1)
            target_indices = child_ids[valid]
            source_vals = child_reach_vals[valid]
            self.nodes_reach.index_add_(0, target_indices, source_vals)
        
        # Backward pass: compute utilities and regrets
        for layer_idx in reversed(self.layer_indices):
            if len(layer_idx) == 0:
                continue
            
            node_types_layer = self.node_types[layer_idx]
            
            # Terminal nodes
            term_mask = (node_types_layer == 0)
            term_nodes = layer_idx[term_mask]
            if len(term_nodes) > 0:
                self.nodes_values[term_nodes] = self.payoffs[term_nodes]
            
            # Decision nodes
            dec_mask = (node_types_layer == 1)
            dec_nodes = layer_idx[dec_mask]
            
            if len(dec_nodes) == 0:
                continue
            
            child_ids = self.children[dec_nodes]
            inf_ids = self.infosets[dec_nodes]
            p_ids = self.players[dec_nodes]
            
            # Gather child values
            valid_mask = (child_ids != -1)
            safe_child_ids = child_ids.clone()
            safe_child_ids[~valid_mask] = 0
            
            c_vals = self.nodes_values[safe_child_ids]
            c_vals = c_vals * valid_mask.unsqueeze(2).float()
            
            # Expected value
            strat = current_strategy[inf_ids]
            ev = (strat.unsqueeze(2) * c_vals).sum(dim=1)
            self.nodes_values[dec_nodes] = ev
            
            # Only update regrets for the updating_player
            player_mask = (p_ids == updating_player)
            if not player_mask.any():
                continue
            
            updating_nodes = dec_nodes[player_mask]
            updating_inf_ids = inf_ids[player_mask]
            updating_p_ids = p_ids[player_mask]
            updating_child_ids = child_ids[player_mask]
            updating_valid_mask = valid_mask[player_mask]
            
            # Recompute child values for updating nodes
            safe_updating_child_ids = updating_child_ids.clone()
            safe_updating_child_ids[~updating_valid_mask] = 0
            updating_c_vals = self.nodes_values[safe_updating_child_ids]
            updating_c_vals = updating_c_vals * updating_valid_mask.unsqueeze(2).float()
            
            updating_ev = ev[player_mask]
            
            # Opponent reach (counterfactual reach)
            opp_ids = 1 - updating_p_ids
            opp_reach = self.nodes_reach[updating_nodes].gather(1, opp_ids.unsqueeze(1)).squeeze(1)
            
            # Action values for updating player
            p_idx_expanded = updating_p_ids.view(-1, 1, 1).expand(-1, self.num_actions, 1)
            q_vals = updating_c_vals.gather(2, p_idx_expanded).squeeze(2)
            v_vals = updating_ev.gather(1, updating_p_ids.unsqueeze(1)).squeeze(1)
            
            # Instantaneous regret
            inst_regret = opp_reach.unsqueeze(1) * (q_vals - v_vals.unsqueeze(1))
            
            # Accumulate
            self.delta_regret.index_add_(0, updating_inf_ids, inst_regret)
        
        # Apply regrets (clamping happens in the caller for CFR+)
        self.regret_sum += self.delta_regret

    def _cfr_iteration_simultaneous(self, num_nodes):
        """Standard CFR with simultaneous updates for both players."""
        current_strategy = self._get_current_strategy()
        
        self.nodes_reach.zero_()
        self.nodes_values.zero_()
        self.delta_regret.zero_()
        
        self.nodes_reach[self.roots_tensor, 0] = self.root_prob
        self.nodes_reach[self.roots_tensor, 1] = self.root_prob
        
        # Forward pass
        for layer_idx in self.layer_indices:
            if len(layer_idx) == 0:
                continue
            
            node_types_layer = self.node_types[layer_idx]
            decision_mask = (node_types_layer == 1)
            decision_nodes = layer_idx[decision_mask]
            
            if len(decision_nodes) == 0:
                continue
            
            p_ids = self.players[decision_nodes]
            inf_ids = self.infosets[decision_nodes]
            child_ids = self.children[decision_nodes]
            
            node_strat = current_strategy[inf_ids]
            
            reach_p = self.nodes_reach[decision_nodes].gather(1, p_ids.unsqueeze(1)).squeeze(1)
            contrib = node_strat * reach_p.unsqueeze(1)
            self.strategy_sum.index_add_(0, inf_ids, contrib)
            
            multipliers = torch.ones((len(decision_nodes), self.num_actions, 2), device=self.device)
            mask0 = (p_ids == 0)
            multipliers[mask0, :, 0] = node_strat[mask0]
            mask1 = (p_ids == 1)
            multipliers[mask1, :, 1] = node_strat[mask1]
            
            parent_reach_expanded = self.nodes_reach[decision_nodes].unsqueeze(1)
            child_reach_vals = parent_reach_expanded * multipliers
            
            valid = (child_ids != -1)
            target_indices = child_ids[valid]
            source_vals = child_reach_vals[valid]
            self.nodes_reach.index_add_(0, target_indices, source_vals)
        
        # Backward pass
        for layer_idx in reversed(self.layer_indices):
            if len(layer_idx) == 0:
                continue
            
            node_types_layer = self.node_types[layer_idx]
            
            term_mask = (node_types_layer == 0)
            term_nodes = layer_idx[term_mask]
            if len(term_nodes) > 0:
                self.nodes_values[term_nodes] = self.payoffs[term_nodes]
            
            dec_mask = (node_types_layer == 1)
            dec_nodes = layer_idx[dec_mask]
            
            if len(dec_nodes) == 0:
                continue
            
            child_ids = self.children[dec_nodes]
            inf_ids = self.infosets[dec_nodes]
            p_ids = self.players[dec_nodes]
            
            valid_mask = (child_ids != -1)
            safe_child_ids = child_ids.clone()
            safe_child_ids[~valid_mask] = 0
            
            c_vals = self.nodes_values[safe_child_ids]
            c_vals = c_vals * valid_mask.unsqueeze(2).float()
            
            strat = current_strategy[inf_ids]
            ev = (strat.unsqueeze(2) * c_vals).sum(dim=1)
            self.nodes_values[dec_nodes] = ev
            
            opp_ids = 1 - p_ids
            opp_reach = self.nodes_reach[dec_nodes].gather(1, opp_ids.unsqueeze(1)).squeeze(1)
            
            p_idx_expanded = p_ids.view(-1, 1, 1).expand(-1, self.num_actions, 1)
            q_vals = c_vals.gather(2, p_idx_expanded).squeeze(2)
            v_vals = ev.gather(1, p_ids.unsqueeze(1)).squeeze(1)
            
            inst_regret = opp_reach.unsqueeze(1) * (q_vals - v_vals.unsqueeze(1))
            self.delta_regret.index_add_(0, inf_ids, inst_regret)
        
        self.regret_sum += self.delta_regret

    def get_average_strategy(self):
        t_start = time.time()
        
        print("Reconstructing strategy dictionary...")
        strat_sum = self.strategy_sum.cpu().numpy()
        valid_mask = self.infoset_valid_actions.cpu().numpy()
        avg_strat = {}
        
        if self.tree.infoset_keys_map is not None:
            idx_to_key = {v: k for k, v in self.tree.infoset_keys_map.items()}
            
            for i in range(len(strat_sum)):
                if i not in idx_to_key:
                    continue
                
                key = idx_to_key[i]
                probs = strat_sum[i]
                
                # Only consider valid actions
                valid = valid_mask[i]
                probs = probs * valid
                total = np.sum(probs)
                
                if total > 0:
                    normalized = probs / total
                else:
                    # Uniform over valid actions
                    num_valid = np.sum(valid)
                    if num_valid > 0:
                        normalized = valid.astype(float) / num_valid
                    else:
                        normalized = np.ones_like(probs) / len(probs)
                    
                action_dict = {}
                for a_idx, prob in enumerate(normalized):
                    if valid[a_idx]:
                        action_name = self.actions[a_idx]
                        action_dict[action_name] = float(prob)
                    
                avg_strat[key] = action_dict
        else:
            print("Warning: No infoset_keys_map found!")
        
        elapsed = time.time() - t_start
        self.strategy_reconstruction_time += elapsed
        print(f"get_average_strategy took {elapsed:.3f}s")
        
        return avg_strat

    def save_gzip(self, filepath):
        import gzip
        import pickle as pkl
        data = {
            'average_strategy': self.average_strategy,
            'iteration_count': self.t,
            'training_time': getattr(self, 'training_time', 0)
        }
        with gzip.open(filepath, 'wb') as f:
            pkl.dump(data, f)
        print(f"Saved to {filepath}")