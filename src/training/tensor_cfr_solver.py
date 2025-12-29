import torch
import numpy as np
import time
import os
from collections import deque
from utils.data_models import KeyGenerator
from training.tensor_game_tree import TensorizedGameTree, build_tensor_tree, get_tree_path

class TensorCFRSolver:
    def __init__(self, game, combination_generator, algorithm='cfr_plus', device=None, game_name=None, load_tree=True):
        self.game = game
        self.combination_generator = combination_generator
        self.algorithm = algorithm
        self.actions = ['check', 'bet', 'call', 'fold']
        self.num_actions = len(self.actions)

        # Device Selection
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            #elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                #self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = device

        print(f"Initializing TensorCFRSolver on {self.device}...")

        # 1. Build or Load the Tensorized Game Tree
        self.tree = None
        
        if load_tree and game_name:
            tree_path = get_tree_path(game_name)
            if os.path.exists(tree_path):
                try:
                    self.tree = TensorizedGameTree.load(tree_path)
                except Exception as e:
                    print(f"Failed to load tree: {e}")
            else:
                print(f"No existing tensor tree found at {tree_path}")
                
        if self.tree is None:
            self.tree = build_tensor_tree(game, combination_generator)
            if load_tree and game_name:
                tree_path = get_tree_path(game_name)
                self.tree.save(tree_path)

        # 2. Move Tree to Device
        self._move_tree_to_device()

        # 3. Initialize Training Tensors
        self._init_tensors()

    def _move_tree_to_device(self):
        print("Moving tree tensors to device...")
        t0 = time.time()
        
        # Static Tree Data
        self.node_types = torch.tensor(self.tree.node_types, device=self.device, dtype=torch.int8)
        self.players = torch.tensor(self.tree.players, device=self.device, dtype=torch.long)
        self.infosets = torch.tensor(self.tree.infosets, device=self.device, dtype=torch.long)
        self.children = torch.tensor(self.tree.children, device=self.device, dtype=torch.long)
        self.payoffs = torch.tensor(self.tree.payoffs, device=self.device, dtype=torch.float32)
        self.roots_tensor = torch.tensor(self.tree.roots, device=self.device, dtype=torch.long)
        
        # Layer Indices (Group by Depth)
        self.max_depth = int(np.max(self.tree.depths))
        self.layer_indices = []
        for d in range(1, self.max_depth + 1):
            indices = np.where(self.tree.depths == d)[0]
            if len(indices) > 0:
                self.layer_indices.append(torch.tensor(indices, device=self.device, dtype=torch.long))
            else:
                self.layer_indices.append(torch.tensor([], device=self.device, dtype=torch.long))

        # Valid Actions Mask
        # (Batch, Actions) boolean
        # We recalculate this on GPU or CPU. Let's do it on GPU as in original code.
        
        num_infosets = self.tree.infoset_counts
        self.num_infosets = num_infosets
        
        # Create a mask for valid children (Where child != -1)
        valid_actions_per_node = (self.children != -1)
        valid_float = valid_actions_per_node.float()
        
        # Filter decision nodes only
        dec_mask = (self.infosets != -1)
        dec_infosets = self.infosets[dec_mask]
        dec_valid = valid_float[dec_mask]
        
        valid_sums = torch.zeros((num_infosets, self.num_actions), device=self.device, dtype=torch.float32)
        valid_sums.index_add_(0, dec_infosets, dec_valid)
        
        self.infoset_valid_actions = (valid_sums > 0)
        self.infoset_valid_counts = self.infoset_valid_actions.sum(dim=1, keepdim=True)
        
        # Root Probability
        self.root_prob = 1.0 / len(self.tree.roots)
        
        print(f"Tree moved to device in {time.time() - t0:.2f}s")

    def _init_tensors(self):
        # Regret Sum: (Infosets, Actions)
        self.regret_sum = torch.zeros((self.num_infosets, self.num_actions), device=self.device, dtype=torch.float32)
        
        # Strategy Sum: (Infosets, Actions)
        self.strategy_sum = torch.zeros((self.num_infosets, self.num_actions), device=self.device, dtype=torch.float32)
        
        # Iteration Count
        self.t = 0

    def train(self, iterations):
        print(f"Starting Training for {iterations} iterations on {self.device}...")
        start_time = time.time()
        
        # Pre-allocate reach probabilities and values for nodes
        num_nodes = len(self.tree.node_types)
        
        with torch.no_grad():
            for i in range(1, iterations + 1):
                self.t = i
                self._cfr_iteration(num_nodes)
                
                if i % 100 == 0:
                    print(f"Iteration {i}")
                
        self.training_time = time.time() - start_time
        print(f"Training completed in {self.training_time:.2f}s")
        
        self.average_strategy = self.get_average_strategy()

    def _cfr_iteration(self, num_nodes):
        # 1. Compute Current Strategy from Regrets
        
        # Mask invalid regrets first
        self.regret_sum.masked_fill_(~self.infoset_valid_actions, -1e9) 
        
        positive_regrets = torch.clamp(self.regret_sum, min=0)
        sum_pos = torch.sum(positive_regrets, dim=1, keepdim=True)
        
        current_strategy = torch.zeros_like(self.regret_sum)
        has_pos = (sum_pos > 1e-12).squeeze()
        
        # Positive regrets case
        current_strategy[has_pos] = positive_regrets[has_pos] / sum_pos[has_pos]
        
        # Uniform case (no positive regrets)
        uniform_probs = 1.0 / torch.clamp(self.infoset_valid_counts, min=1)
        no_pos_mask = ~has_pos
        uniform_contrib = uniform_probs * self.infoset_valid_actions.float()
        current_strategy[no_pos_mask] = uniform_contrib[no_pos_mask]
        
        # Ensure invalid actions are exactly 0
        current_strategy.masked_fill_(~self.infoset_valid_actions, 0.0)
 
        
        # 2. Forward Pass (Reach Probabilities)
        nodes_reach = torch.zeros((num_nodes, 2), device=self.device, dtype=torch.float32)
        
        # Initialize Roots
        nodes_reach[self.roots_tensor, 0] = self.root_prob 
        nodes_reach[self.roots_tensor, 1] = self.root_prob 
        
        for layer_idx in self.layer_indices:
            if len(layer_idx) == 0: continue
            
            # Select nodes
            node_types_layer = self.node_types[layer_idx] 
            decision_mask = (node_types_layer == 1)
            decision_nodes = layer_idx[decision_mask]
            
            if len(decision_nodes) == 0: continue
            
            p_ids = self.players[decision_nodes]
            inf_ids = self.infosets[decision_nodes]
            child_ids = self.children[decision_nodes] 
            
            # Get Strategy for these nodes
            node_strat = current_strategy[inf_ids] 
            
            # Update Strategy Sum (Weighted by reach of the playing player)
            reach_p = nodes_reach[decision_nodes].gather(1, p_ids.unsqueeze(1)).squeeze(1)
            
            contrib = node_strat * reach_p.unsqueeze(1)
            
            if self.algorithm == 'cfr_plus':
                contrib *= self.t
            
            self.strategy_sum.index_add_(0, inf_ids, contrib)
            
            # Propagate Reach to Children
            multipliers = torch.ones((len(decision_nodes), self.num_actions, 2), device=self.device)
            
            # p=0
            mask0 = (p_ids == 0)
            multipliers[mask0, :, 0] = node_strat[mask0]
            
            # p=1
            mask1 = (p_ids == 1)
            multipliers[mask1, :, 1] = node_strat[mask1]
            
            # Parent Reach
            parent_reach_expanded = nodes_reach[decision_nodes].unsqueeze(1)
            
            # Child Reach
            child_reach_vals = parent_reach_expanded * multipliers
            
            # Scatter to global array
            valid = (child_ids != -1)
            target_indices = child_ids[valid]
            source_vals = child_reach_vals[valid]
            nodes_reach.index_add_(0, target_indices, source_vals)
            
        
        # 3. Backward Pass (Utilities)
        nodes_values = torch.zeros((num_nodes, 2), device=self.device, dtype=torch.float32)
        
        # Initialize Terminal Values (Layer by Layer for bottom-up)
        for layer_idx in reversed(self.layer_indices):
            if len(layer_idx) == 0: continue
            
            # 1. Terminal Nodes
            node_types_layer = self.node_types[layer_idx]
            term_mask = (node_types_layer == 0)
            term_nodes = layer_idx[term_mask]
            
            if len(term_nodes) > 0:
                nodes_values[term_nodes] = self.payoffs[term_nodes]
            
            # 2. Decision Nodes
            dec_mask = (node_types_layer == 1)
            dec_nodes = layer_idx[dec_mask]
            
            if len(dec_nodes) == 0: continue
            
            child_ids = self.children[dec_nodes] 
            inf_ids = self.infosets[dec_nodes]
            p_ids = self.players[dec_nodes]
            
            # Gather child values
            valid_mask = (child_ids != -1)
            safe_child_ids = child_ids.clone()
            safe_child_ids[~valid_mask] = 0 
            
            c_vals = nodes_values[safe_child_ids] 
            c_vals[~valid_mask] = 0.0 
            
            # Expected Value = Sum(Prob_a * Value_a)
            strat = current_strategy[inf_ids]
            ev = (strat.unsqueeze(2) * c_vals).sum(dim=1)
            
            nodes_values[dec_nodes] = ev
            
            # 4. Regret Update
            opp_ids = 1 - p_ids
            opp_reach = nodes_reach[dec_nodes].gather(1, opp_ids.unsqueeze(1)).squeeze(1)
            
            p_idx_expanded = p_ids.view(-1, 1, 1).expand(-1, self.num_actions, 1)
            q_vals = c_vals.gather(2, p_idx_expanded).squeeze(2)
            v_vals = ev.gather(1, p_ids.unsqueeze(1)).squeeze(1)
            
            inst_regret = opp_reach.unsqueeze(1) * (q_vals - v_vals.unsqueeze(1))
            
            if not hasattr(self, 'delta_regret'):
                self.delta_regret = torch.zeros_like(self.regret_sum)
                
            self.delta_regret.index_add_(0, inf_ids, inst_regret)

        # Apply Regrets
        if self.algorithm == 'cfr_plus':
            self.regret_sum += self.delta_regret
            self.regret_sum = torch.clamp(self.regret_sum, min=0)
        else:
            self.regret_sum += self.delta_regret
        
        self.delta_regret.zero_()

    def get_average_strategy(self):
        # We need a map from infoset_id back to Key
        # The TensorizedGameTree doesn't store the keys (it's tensor only).
        # But we need them for the final output.
        # This is a trade-off. To get keys, we need to rebuild the map or save it.
        # Ideally, we should save infoset_keys in the tensor tree file (as a pickled list or string array).
        
        # For now, if we loaded the tree, we might not have the keys map.
        # We can regenerate it if necessary, OR we modify TensorizedGameTree to store keys.
        # Given "bruuuh" optimization, regenerating is slow.
        # Let's rely on the user passing `game` and `combination_generator` to regenerate keys *only* at the end.
        
        print("Reconstructing strategy dictionary...")
        strat_sum = self.strategy_sum.cpu().numpy()
        avg_strat = {}
        
        # We need to iterate the game to map keys to IDs again.
        # This seems redundant but avoids storing millions of strings in the tensor file.
        # Actually, we can just traverse the tree quickly to recover keys.
        
        # Better approach: Just like we built the tree, we can traverse and match IDs.
        # But we have `infosets` array.
        
        # Let's just do a quick traversal to map keys to infoset_ids.
        infoset_map = {}
        next_infoset_id = 0
        
        actions = ['check', 'bet', 'call', 'fold']
        action_to_idx = {a: i for i, a in enumerate(actions)}
        
        # Re-traversal helper to get keys
        def traverse(depth):
            nonlocal next_infoset_id
            if self.game.done: return
            
            player = self.game.current_player
            legal_actions = self.game.get_legal_actions()
            
            key = KeyGenerator.get_info_set_key(self.game, player)
            if key not in infoset_map:
                infoset_map[key] = next_infoset_id
                next_infoset_id += 1
            
            for action in legal_actions:
                if action not in action_to_idx: continue
                self.game.step(action)
                traverse(depth + 1)
                self.game.step_back()

        combinations = self.combination_generator.get_all_combinations()
        for combo in combinations:
            self.combination_generator.setup_game_with_combination(self.game, combo)
            traverse(1)
            
        # Now map
        # infoset_map: Key -> ID
        # strat_sum: ID -> Probs
        
        # Invert map
        idx_to_key = {v: k for k, v in infoset_map.items()}
        
        for i in range(len(strat_sum)):
            if i not in idx_to_key: continue
            
            key = idx_to_key[i]
            probs = strat_sum[i]
            total = np.sum(probs)
            
            if total > 0:
                normalized = probs / total
            else:
                normalized = np.ones_like(probs) / len(probs)
                
            action_dict = {}
            for a_idx, prob in enumerate(normalized):
                action_name = self.actions[a_idx]
                action_dict[action_name] = float(prob)
                
            avg_strat[key] = action_dict
            
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
