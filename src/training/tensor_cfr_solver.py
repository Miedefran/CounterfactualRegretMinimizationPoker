import torch
import numpy as np
import time
from collections import deque
from utils.data_models import KeyGenerator

class TensorCFRSolver:
    def __init__(self, game, combination_generator, algorithm='cfr_plus', device=None):
        self.game = game
        self.combination_generator = combination_generator
        self.algorithm = algorithm
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Initializing TensorCFRSolver on {self.device}...")
        
        # 1. Build the Game Tree (Static Graph)
        self._build_tree()
        
        # 2. Initialize Tensors
        self._init_tensors()
        
    def _build_tree(self):
        print("Building static game tree (this may take a while)...")
        start_time = time.time()
        
        # Action Mapping
        # We need a fixed global mapping for actions to tensor indices
        self.actions = ['check', 'bet', 'call', 'fold'] # Standard poker actions
        # Some games might have different bet sizes, but usually 'bet' is one logical action in step()
        # Wait, some games might have 'bet_2', 'bet_4'? 
        # Let's inspect get_legal_actions() dynamics.
        # In this repo, actions are strings like 'bet', 'check', 'call', 'fold'.
        # The bet sizing is internal to the game state.
        self.action_to_idx = {a: i for i, a in enumerate(self.actions)}
        self.num_actions = len(self.actions)
        
        # Nodes Storage
        self.nodes = [] # List of dicts temporarily
        self.infoset_map = {} # key -> int
        self.next_infoset_id = 0
        
        # We need to traverse for EVERY combination (deal)
        combinations = self.combination_generator.get_all_combinations()
        
        # We process depth by depth or just BFS.
        # Queue items: (game_state_snapshot, combination, parent_node_idx, action_from_parent)
        
        # Virtual Root (Chance Node dealing cards)
        self.virtual_root_children = []
        self.virtual_root_probs = []
        
        node_counter = 0
        
        # To avoid deep recursion, we use an iterative stack/queue
        # But we need to use the Game object's step/restore features.
        # Since we have to iterate all combinations, let's do it sequentially.
        
        # We will build a flat list of nodes.
        # Format: (depth, player, infoset_id, legal_mask, terminal_payoffs, children_indices)
        
        # Since 'Game' object handles one deal, we treat the "Tree" as a collection of disjoint trees (one per deal)
        # connected by a Virtual Root.
        
        self.flat_nodes = [] # List of NodeData
        
        # Helper to process a game trajectory
        def traverse(depth):
            current_idx = len(self.flat_nodes)
            self.flat_nodes.append(None) # Placeholder
            
            if self.game.done:
                # Terminal
                payoffs = [self.game.get_payoff(0), self.game.get_payoff(1)]
                self.flat_nodes[current_idx] = {
                    'type': 'terminal',
                    'depth': depth,
                    'payoffs': payoffs
                }
                return current_idx
            
            player = self.game.current_player
            legal_actions = self.game.get_legal_actions()
            
            # InfoSet
            key = KeyGenerator.get_info_set_key(self.game, player)
            if key not in self.infoset_map:
                self.infoset_map[key] = self.next_infoset_id
                self.next_infoset_id += 1
            infoset_id = self.infoset_map[key]
            
            children = [-1] * self.num_actions
            
            for action in legal_actions:
                if action not in self.action_to_idx:
                    # Should not happen usually
                    continue
                a_idx = self.action_to_idx[action]
                
                self.game.step(action)
                child_idx = traverse(depth + 1)
                self.game.step_back()
                
                children[a_idx] = child_idx
                
            self.flat_nodes[current_idx] = {
                'type': 'decision',
                'depth': depth,
                'player': player,
                'infoset_id': infoset_id,
                'children': children,
                'legal_mask': [1 if i != -1 else 0 for i in children]
            }
            return current_idx

        # Iterate all deals
        root_children = []
        prob_weight = 1.0 / len(combinations)
        
        for combo in combinations:
            self.combination_generator.setup_game_with_combination(self.game, combo)
            # Traverse this deal's tree
            # Start depth 1 (Depth 0 is virtual root)
            child_idx = traverse(1)
            root_children.append(child_idx)
        
        # Add Virtual Root at the end or beginning?
        # Let's add it at the end to not shift indices, or handle it separately.
        # Actually, let's treat the virtual root as implicit or just a special layer.
        # It's easier to add it as Node 0 if we pre-calculated, but we didn't.
        # So we add it now.
        root_idx = len(self.flat_nodes)
        
        # We need to map the virtual root's children to a fixed size? 
        # No, the virtual root is a Chance node with N children (N = num combinations).
        # This doesn't fit the 'MaxActions' shape if N is large.
        # STRATEGY: We don't represent the virtual root in the tensor graph.
        # Instead, we treat the 'root_children' as the "Roots" of our computation.
        # We start the Forward Pass by injecting 1.0 probability into all 'root_children'.
        
        self.roots = root_children
        self.root_prob = prob_weight
        
        build_time = time.time() - start_time
        print(f"Tree built in {build_time:.2f}s. Nodes: {len(self.flat_nodes)}. Infosets: {self.next_infoset_id}")
        
        # Convert to Tensors
        self._vectorize_tree()

    def _vectorize_tree(self):
        num_nodes = len(self.flat_nodes)
        
        # Arrays
        depths = np.zeros(num_nodes, dtype=np.int32)
        node_types = np.zeros(num_nodes, dtype=np.int8) # 0: Terminal, 1: Decision
        players = np.zeros(num_nodes, dtype=np.int8)
        infosets = np.full(num_nodes, -1, dtype=np.int32)
        children = np.full((num_nodes, self.num_actions), -1, dtype=np.int32)
        payoffs = np.zeros((num_nodes, 2), dtype=np.float32)
        
        for i, node in enumerate(self.flat_nodes):
            depths[i] = node['depth']
            if node['type'] == 'terminal':
                node_types[i] = 0
                payoffs[i] = node['payoffs']
            else:
                node_types[i] = 1
                players[i] = node['player']
                infosets[i] = node['infoset_id']
                children[i] = node['children']
                
        # Group by Depth for Layered Computation
        # We need to know which nodes are at depth D
        max_depth = np.max(depths)
        self.max_depth = int(max_depth)
        
        self.layer_indices = []
        for d in range(1, self.max_depth + 1):
            indices = np.where(depths == d)[0]
            if len(indices) > 0:
                self.layer_indices.append(torch.tensor(indices, device=self.device, dtype=torch.long))
            else:
                self.layer_indices.append(torch.tensor([], device=self.device, dtype=torch.long))
                
        # Move static data to GPU
        self.node_types = torch.tensor(node_types, device=self.device, dtype=torch.int8)
        self.players = torch.tensor(players, device=self.device, dtype=torch.long)
        self.infosets = torch.tensor(infosets, device=self.device, dtype=torch.long)
        self.children = torch.tensor(children, device=self.device, dtype=torch.long)
        self.payoffs = torch.tensor(payoffs, device=self.device, dtype=torch.float32)
        
        # Create a mask for valid children (Where child != -1)
        self.valid_children_mask = (self.children != -1)
        
        # Roots tensor
        self.roots_tensor = torch.tensor(self.roots, device=self.device, dtype=torch.long)
        
        # Compute Infoset Valid Actions Mask
        num_infosets = self.next_infoset_id
        self.infoset_valid_actions = torch.zeros((num_infosets, self.num_actions), device=self.device, dtype=torch.bool)
        
        # We can aggregate from nodes. 
        # Since all nodes in an infoset should have same valid actions, we can just scatter max or use a loop.
        # infosets tensor has mapping.
        # We can use scatter_add with type logic?
        # Or just loop over flat nodes on CPU before moving?
        # Faster: Use the tensors we just created.
        # (Batch, Actions) boolean
        valid_actions_per_node = (self.children != -1)
        
        # We want to OR-reduce this by infoset_id.
        # PyTorch doesn't have scatter_reduce('or') for boolean easily until recent versions.
        # But we can cast to float, scatter_add, then check > 0.
        valid_float = valid_actions_per_node.float()
        
        # Filter decision nodes only (-1 infoset)
        dec_mask = (self.infosets != -1)
        dec_infosets = self.infosets[dec_mask]
        dec_valid = valid_float[dec_mask]
        
        valid_sums = torch.zeros((num_infosets, self.num_actions), device=self.device, dtype=torch.float32)
        valid_sums.index_add_(0, dec_infosets, dec_valid)
        
        self.infoset_valid_actions = (valid_sums > 0)
        self.infoset_valid_counts = self.infoset_valid_actions.sum(dim=1, keepdim=True)

    def _init_tensors(self):
        num_infosets = self.next_infoset_id
        self.num_infosets = num_infosets # Store for usage
        
        # Regret Sum: (Infosets, Actions)
        self.regret_sum = torch.zeros((num_infosets, self.num_actions), device=self.device, dtype=torch.float32)
        
        # Strategy Sum: (Infosets, Actions)
        self.strategy_sum = torch.zeros((num_infosets, self.num_actions), device=self.device, dtype=torch.float32)
        
        # Iteration Count
        self.t = 0

    def train(self, iterations):
        print(f"Starting Training for {iterations} iterations on {self.device}...")
        start_time = time.time()
        
        # Pre-allocate reach probabilities and values for nodes
        num_nodes = len(self.flat_nodes)
        
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
        # Regret Matching
        # Mask invalid regrets first
        self.regret_sum.masked_fill_(~self.infoset_valid_actions, -1e9) # Negative infinity for invalid
        
        positive_regrets = torch.clamp(self.regret_sum, min=0)
        sum_pos = torch.sum(positive_regrets, dim=1, keepdim=True)
        
        current_strategy = torch.zeros_like(self.regret_sum)
        has_pos = (sum_pos > 1e-12).squeeze()
        
        # Where we have positive regrets
        current_strategy[has_pos] = positive_regrets[has_pos] / sum_pos[has_pos]
        
        # Where we don't (Uniform over VALID actions)
        # We use self.infoset_valid_counts
        # We need to set prob = 1/count for valid actions, 0 otherwise
        
        uniform_probs = 1.0 / torch.clamp(self.infoset_valid_counts, min=1) # Avoid div by 0 (though count shouldn't be 0 for reachable infosets)
        
        # Expand uniform_probs to (N, Actions) via broadcasting
        # We want to set this ONLY for rows ~has_pos AND cols valid
        
        no_pos_mask = ~has_pos
        # current_strategy[no_pos_mask] = ...
        # This requires masking logic.
        # current_strategy = current_strategy + uniform_probs * valid_mask * no_pos_mask
        
        uniform_contrib = uniform_probs * self.infoset_valid_actions.float()
        current_strategy[no_pos_mask] = uniform_contrib[no_pos_mask]
        
        # Ensure invalid actions are exactly 0 (Redundant given logic above, but safe)
        current_strategy.masked_fill_(~self.infoset_valid_actions, 0.0)
 
        
        # 2. Forward Pass (Reach Probabilities)
        # nodes_reach: (N, 2) -> Reach prob for P0 and P1
        nodes_reach = torch.zeros((num_nodes, 2), device=self.device, dtype=torch.float32)
        
        # Initialize Roots
        nodes_reach[self.roots_tensor, 0] = self.root_prob # P0 reach
        nodes_reach[self.roots_tensor, 1] = self.root_prob # P1 reach
        
        for layer_idx in self.layer_indices:
            # layer_idx contains node indices at depth D
            if len(layer_idx) == 0: continue
            
            # Select nodes
            node_types_layer = self.node_types[layer_idx] # 0 or 1
            
            # Filter decision nodes only
            decision_mask = (node_types_layer == 1)
            decision_nodes = layer_idx[decision_mask]
            
            if len(decision_nodes) == 0: continue
            
            p_ids = self.players[decision_nodes]
            inf_ids = self.infosets[decision_nodes]
            child_ids = self.children[decision_nodes] # (Batch, Actions)
            
            # Get Strategy for these nodes
            node_strat = current_strategy[inf_ids] # (Batch, Actions)
            
            # Update Strategy Sum (Weighted by reach of the playing player)
            # We need player's reach probability at this node.
            # players_reach: (Batch,)
            # gather the column corresponding to p_ids
            reach_p = nodes_reach[decision_nodes].gather(1, p_ids.unsqueeze(1)).squeeze(1)
            
            # Update Strategy Sum
            # self.strategy_sum[inf_ids] += reach_p * node_strat
            # Use scatter_add for safety if multiple nodes map to same infoset (unlikely in perfect recall but possible)
            # In Leduc, tree is tree. One node -> One infoset.
            # But wait, multiple nodes DO map to same infoset (same private, same history, different opponent private).
            # So we MUST use index_add_ or scatter_add_.
            
            # We want: strategy_sum[inf_ids[b], a] += reach_p[b] * node_strat[b, a]
            # Contribution: (Batch, Actions)
            contrib = node_strat * reach_p.unsqueeze(1)
            
            # Flatten indices for scatter
            # We iterate actions? Or just loops. PyTorch `index_add_` works on dim 0.
            self.strategy_sum.index_add_(0, inf_ids, contrib)
            
            # Propagate Reach to Children
            # For each action a:
            # child_reach[0] = parent_reach[0] * (strat[a] if p==0 else 1)
            # child_reach[1] = parent_reach[1] * (strat[a] if p==1 else 1)
            
            # Prepare Multipliers (Batch, Actions, 2)
            # multipliers = 1.0
            # if p==0: multipliers[:, :, 0] = strat
            # if p==1: multipliers[:, :, 1] = strat
            
            multipliers = torch.ones((len(decision_nodes), self.num_actions, 2), device=self.device)
            
            # p=0
            mask0 = (p_ids == 0)
            multipliers[mask0, :, 0] = node_strat[mask0]
            
            # p=1
            mask1 = (p_ids == 1)
            multipliers[mask1, :, 1] = node_strat[mask1]
            
            # Parent Reach: (Batch, 1, 2)
            parent_reach_expanded = nodes_reach[decision_nodes].unsqueeze(1)
            
            # Child Reach: (Batch, Actions, 2)
            child_reach_vals = parent_reach_expanded * multipliers
            
            # Scatter to global array
            # child_ids: (Batch, Actions)
            # valid mask
            valid = (child_ids != -1)
            
            # Flat indices
            target_indices = child_ids[valid]
            source_vals = child_reach_vals[valid]
            
            # Assign (We can assume each node has unique parent in tree, so simple assignment/add works)
            nodes_reach.index_add_(0, target_indices, source_vals)
            
        
        # 3. Backward Pass (Utilities)
        # nodes_values: (N, 2) -> Expected Value for P0 and P1
        nodes_values = torch.zeros((num_nodes, 2), device=self.device, dtype=torch.float32)
        
        # Initialize Terminal Values
        # We can do this all at once or per layer.
        # Let's do it per layer bottom-up to handle tree.
        
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
            
            child_ids = self.children[dec_nodes] # (Batch, Actions)
            inf_ids = self.infosets[dec_nodes]
            p_ids = self.players[dec_nodes]
            
            # Gather child values
            # (Batch, Actions, 2)
            # Indices where child is -1 will point to last element or error?
            # Must handle -1. Replace -1 with 0 (dummy) and mask later.
            valid_mask = (child_ids != -1)
            safe_child_ids = child_ids.clone()
            safe_child_ids[~valid_mask] = 0 # Safe dummy
            
            c_vals = nodes_values[safe_child_ids] # (Batch, Actions, 2)
            c_vals[~valid_mask] = 0.0 # Zero out invalid
            
            # Expected Value = Sum(Prob_a * Value_a)
            # Strategy: (Batch, Actions)
            strat = current_strategy[inf_ids]
            
            # Weighted Sum
            # (Batch, Actions, 1) * (Batch, Actions, 2) -> Sum over Actions -> (Batch, 2)
            ev = (strat.unsqueeze(2) * c_vals).sum(dim=1)
            
            nodes_values[dec_nodes] = ev
            
            # 4. Regret Update (Only at Decision Nodes)
            # Regret[a] = Value[a] - Value_Node (For the current player)
            # Opponent reach prob?
            # CFR: Regret += OpponentReach * (Q(a) - V)
            # CFR+: Regret = max(0, Regret + ...)
            
            # Get opponent
            opp_ids = 1 - p_ids
            
            # Opponent Reach: (Batch,)
            opp_reach = nodes_reach[dec_nodes].gather(1, opp_ids.unsqueeze(1)).squeeze(1)
            
            # Q_values for the player: (Batch, Actions)
            # Pick channel p_ids from c_vals
            # c_vals is (Batch, Actions, 2)
            # We want (Batch, Actions)
            # p_ids_expanded: (Batch, 1, 1) -> (Batch, Actions, 1)
            p_idx_expanded = p_ids.view(-1, 1, 1).expand(-1, self.num_actions, 1)
            q_vals = c_vals.gather(2, p_idx_expanded).squeeze(2)
            
            # V_values: (Batch,)
            v_vals = ev.gather(1, p_ids.unsqueeze(1)).squeeze(1)
            
            # Instant Regret: (Batch, Actions)
            inst_regret = opp_reach.unsqueeze(1) * (q_vals - v_vals.unsqueeze(1))
            
            # Update Regret Sum
            # self.regret_sum.index_add_(0, inf_ids, inst_regret)
            
            # CFR+: Update and clamp
            # To implement CFR+, we need to read current regrets, add, clamp, write back.
            # But index_add_ writes back blindly.
            # We need to act on unique infosets to read-modify-write?
            # Or use scatter_add to a temp buffer?
            # Since multiple nodes map to one infoset, we must aggregate updates first.
            
            # Aggregate updates:
            # We can use index_add_ to a zero buffer
            update_buffer = torch.zeros_like(self.regret_sum)
            update_buffer.index_add_(0, inf_ids, inst_regret)
            
            # Now apply to main storage (This applies updates from THIS layer)
            # But we are iterating layers.
            # Wait, we should apply updates continuously? Or batch per iteration?
            # Standard is batch per iteration or just accumulate.
            # For CFR+, we usually do: R += inst; R = max(R, 0).
            # If we simply add 'inst' to 'regret_sum' now, we are good.
            # But the Clamping in CFR+ happens *after* accumulation?
            # Zinkevich CFR+: "Regrets are reset to 0 if negative at each step".
            # So: R(t+1) = max(0, R(t) + inst).
            # This logic must be applied *per infoset*.
            # If an infoset appears in multiple layers (possible?), we should accumulate all 'inst' first, then update R.
            # In Poker, an infoset usually appears at a fixed depth (if structure is regular).
            # So layer-by-layer update is fine IF infosets don't span layers.
            # Even if they do, we can just accumulate to a `delta_regret` tensor for the whole tree, then apply.
            
            # Let's accumulate to `self.delta_regret`
            if not hasattr(self, 'delta_regret'):
                self.delta_regret = torch.zeros_like(self.regret_sum)
                
            self.delta_regret.index_add_(0, inf_ids, inst_regret)

        # End of Iteration Loop (After all layers)
        # Apply Regrets
        if self.algorithm == 'cfr_plus':
            self.regret_sum += self.delta_regret
            self.regret_sum = torch.clamp(self.regret_sum, min=0)
            
            # Linear averaging for CFR+?
            # "The average strategy in CFR+ is weighted by iteration t"
            # Our strategy_sum update earlier used 'reach_p'.
            # In CFR+, we usually weight by max(0, t * reach).
            # Or simplified: just weight by t.
            # The current implementation of strategy_sum accumulation is "Vanilla CFR" style (weighted by reach).
            # For CFR+, usually we weight by `t`.
            # Let's stick to Vanilla weighting for strategy for now, or use `t * reach`.
            # If `self.algorithm == 'cfr_plus'`, we might want to scale the `contrib` in the forward pass by `self.t`.
            # Let's stick to Vanilla CFR accumulation for safety unless user insists on "CFR+ Strategy Averaging".
            # Actually, standard CFR+ solver usually weights by t.
            # But let's leave strategy sum as is (Reach weighted) - it's robust.
            
        else:
            # Vanilla CFR
            self.regret_sum += self.delta_regret
        
        # Reset delta
        self.delta_regret.zero_()

    def get_average_strategy(self):
        # Convert strategy_sum to dictionary
        # Move to CPU
        strat_sum = self.strategy_sum.cpu().numpy()
        
        avg_strat = {}
        
        # Inverse Infoset Map
        idx_to_key = {v: k for k, v in self.infoset_map.items()}
        
        for i in range(len(strat_sum)):
            if i not in idx_to_key: continue
            
            key = idx_to_key[i]
            probs = strat_sum[i]
            total = np.sum(probs)
            
            if total > 0:
                normalized = probs / total
            else:
                normalized = np.ones_like(probs) / len(probs)
                
            # Convert to dict {action: prob}
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
