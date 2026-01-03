import torch
import numpy as np
import os
import time
import gzip
import pickle
from utils.data_models import KeyGenerator

class TensorizedGameTree:
    """
    Efficient, flat storage for Game Trees using Numpy/Torch arrays.
    Designed for instant loading and low memory footprint.
    """
    def __init__(self, 
                 node_types: np.ndarray,
                 players: np.ndarray,
                 infosets: np.ndarray,
                 children: np.ndarray,
                 payoffs: np.ndarray,
                 depths: np.ndarray,
                 roots: np.ndarray,
                 num_actions: int,
                 infoset_counts: int,
                 infoset_keys_map: dict = None):
        
        self.node_types = node_types # 0: Terminal, 1: Decision
        self.players = players
        self.infosets = infosets
        self.children = children
        self.payoffs = payoffs
        self.depths = depths
        self.roots = roots
        self.num_actions = num_actions
        self.infoset_counts = infoset_counts
        self.infoset_keys_map = infoset_keys_map  # {key: infoset_id}

    def save(self, filepath):
        """Saves the tree using numpy compressed format"""
        np.savez_compressed(
            filepath,
            node_types=self.node_types,
            players=self.players,
            infosets=self.infosets,
            children=self.children,
            payoffs=self.payoffs,
            depths=self.depths,
            roots=self.roots,
            meta=np.array([self.num_actions, self.infoset_counts])
        )
        
        if self.infoset_keys_map is not None:
            keys_path = filepath.replace('.npz', '_keys.pkl.gz')
            with gzip.open(keys_path, 'wb') as f:
                pickle.dump(self.infoset_keys_map, f)
            print(f"Infoset keys map saved to {keys_path}")
        
        print(f"Tensorized tree saved to {filepath}")

    @classmethod
    def load(cls, filepath):
        """Loads the tree from numpy compressed format"""
        print(f"Loading tensor tree from {filepath}...")
        t0 = time.time()
        data = np.load(filepath)
        
        keys_path = filepath.replace('.npz', '_keys.pkl.gz')
        infoset_keys_map = None
        if os.path.exists(keys_path):
            try:
                with gzip.open(keys_path, 'rb') as f:
                    infoset_keys_map = pickle.load(f)
                print(f"Infoset keys map loaded from {keys_path}")
            except Exception as e:
                print(f"Warning: Could not load infoset keys map: {e}")
        else:
            print(f"Warning: No infoset keys map found at {keys_path} (tree may be old format)")
        
        tree = cls(
            node_types=data['node_types'],
            players=data['players'],
            infosets=data['infosets'],
            children=data['children'],
            payoffs=data['payoffs'],
            depths=data['depths'],
            roots=data['roots'],
            num_actions=int(data['meta'][0]),
            infoset_counts=int(data['meta'][1]),
            infoset_keys_map=infoset_keys_map
        )
        print(f"Tree loaded in {time.time() - t0:.4f}s")
        return tree

def build_tensor_tree(game, combination_generator):
    """
    Builds the flat tensor tree from the game rules.
    This replaces the slow object-based GameTree builder.
    """
    print("Building tensor game tree...")
    start_time = time.time()
    
    actions = ['check', 'bet', 'call', 'fold']
    action_to_idx = {a: i for i, a in enumerate(actions)}
    num_actions = len(actions)
    
    infoset_map = {}
    next_infoset_id = 0
    
    # Temporary lists to hold node data (will convert to numpy at the end)
    # We use lists of lists/tuples which is faster than list of dicts
    # Structure: [type, player, infoset_id, depth, payoff0, payoff1, child0, child1, child2, child3]
    # Children will be indices.
    
    flat_nodes = [] # List of tuples
    
    # We need to traverse for EVERY combination (deal)
    combinations = combination_generator.get_all_combinations()
    
    # Helper to traverse recursively
    # We use a stack to avoid recursion limit issues and for slight speedup, 
    # but for clarity recursion is fine if depth is low (poker is usually low depth).
    # Using the existing recursive structure from TensorCFRSolver as it works.
    
    def traverse(depth):
        current_idx = len(flat_nodes)
        flat_nodes.append(None) # Placeholder
        
        if game.done:
            # Terminal
            p0 = game.get_payoff(0)
            p1 = game.get_payoff(1)
            # type=0, player=-1, infoset=-1, depth=depth, payoffs=(p0, p1), children=(-1, -1...)
            flat_nodes[current_idx] = (0, -1, -1, depth, p0, p1, -1, -1, -1, -1)
            return current_idx
        
        player = game.current_player
        legal_actions = game.get_legal_actions()
        
        # InfoSet
        nonlocal next_infoset_id
        key = KeyGenerator.get_info_set_key(game, player)
        if key not in infoset_map:
            infoset_map[key] = next_infoset_id
            next_infoset_id += 1
        infoset_id = infoset_map[key]
        
        child_indices = [-1] * num_actions
        
        for action in legal_actions:
            if action not in action_to_idx: continue
            a_idx = action_to_idx[action]
            
            game.step(action)
            child_idx = traverse(depth + 1)
            game.step_back()
            
            child_indices[a_idx] = child_idx
            
        # Decision
        # type=1, player=player, infoset=infoset, depth=depth, payoffs=(0,0), children=...
        c = child_indices
        flat_nodes[current_idx] = (1, player, infoset_id, depth, 0.0, 0.0, c[0], c[1], c[2], c[3])
        return current_idx

    root_indices = []
    
    for combo in combinations:
        combination_generator.setup_game_with_combination(game, combo)
        # Depth 1 because Root is virtual chance
        r_idx = traverse(1)
        root_indices.append(r_idx)
        
    # Convert to Numpy Arrays
    num_nodes = len(flat_nodes)
    print(f"Converting {num_nodes} nodes to tensors...")
    
    # Unzip the list of tuples
    # (type, player, infoset, depth, p0, p1, c0, c1, c2, c3)
    
    # It is faster to pre-allocate numpy arrays and fill them
    node_types = np.zeros(num_nodes, dtype=np.int8)
    players = np.zeros(num_nodes, dtype=np.int8)
    infosets = np.full(num_nodes, -1, dtype=np.int32)
    depths = np.zeros(num_nodes, dtype=np.int32)
    payoffs = np.zeros((num_nodes, 2), dtype=np.float32)
    children = np.full((num_nodes, num_actions), -1, dtype=np.int32)
    
    for i, data in enumerate(flat_nodes):
        node_types[i] = data[0]
        players[i] = data[1]
        infosets[i] = data[2]
        depths[i] = data[3]
        payoffs[i, 0] = data[4]
        payoffs[i, 1] = data[5]
        children[i, 0] = data[6]
        children[i, 1] = data[7]
        children[i, 2] = data[8]
        children[i, 3] = data[9]
        
    roots = np.array(root_indices, dtype=np.int32)
    
    tree = TensorizedGameTree(
        node_types=node_types,
        players=players,
        infosets=infosets,
        children=children,
        payoffs=payoffs,
        depths=depths,
        roots=roots,
        num_actions=num_actions,
        infoset_counts=next_infoset_id,
        infoset_keys_map=infoset_map
    )
    
    print(f"Tree built in {time.time() - start_time:.2f}s")
    return tree

def get_tree_path(game_name):
    script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    output_dir = os.path.join(script_dir, 'trees', 'tensor_trees')
    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(output_dir, f"{game_name}_tensor_tree.npz")
