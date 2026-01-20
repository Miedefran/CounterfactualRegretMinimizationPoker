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
                 chance_offsets: np.ndarray,
                 chance_outcomes: np.ndarray,
                 chance_children: np.ndarray,
                 chance_probs: np.ndarray,
                 outcome_id_to_outcome: np.ndarray,
                 payoffs: np.ndarray,
                 depths: np.ndarray,
                 roots: np.ndarray,
                 num_actions: int,
                 infoset_counts: int,
                 num_outcomes: int,
                 infoset_keys_map: dict = None):
        
        # Node types:
        # 0 = Terminal
        # 1 = Decision
        # 2 = Chance
        self.node_types = node_types
        self.players = players
        self.infosets = infosets
        self.children = children
        # Ragged chance edges stored via prefix-sum offsets into flat edge arrays
        self.chance_offsets = chance_offsets
        self.chance_outcomes = chance_outcomes
        self.chance_children = chance_children
        self.chance_probs = chance_probs
        self.outcome_id_to_outcome = outcome_id_to_outcome
        self.payoffs = payoffs
        self.depths = depths
        self.roots = roots
        self.num_actions = num_actions
        self.infoset_counts = infoset_counts
        self.num_outcomes = num_outcomes
        self.infoset_keys_map = infoset_keys_map  # {key: infoset_id}

    def save(self, filepath):
        """Saves the tree using numpy compressed format"""
        np.savez_compressed(
            filepath,
            node_types=self.node_types,
            players=self.players,
            infosets=self.infosets,
            children=self.children,
            chance_offsets=self.chance_offsets,
            chance_outcomes=self.chance_outcomes,
            chance_children=self.chance_children,
            chance_probs=self.chance_probs,
            outcome_id_to_outcome=self.outcome_id_to_outcome,
            payoffs=self.payoffs,
            depths=self.depths,
            roots=self.roots,
            meta=np.array([self.num_actions, self.infoset_counts, self.num_outcomes])
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
        
        # Legacy detection: old trees do not contain chance arrays / meta[2].
        if 'chance_offsets' not in data.files or 'chance_children' not in data.files:
            raise ValueError("Legacy tensor tree detected (no chance arrays). Rebuild required.")

        meta = data['meta']
        num_actions = int(meta[0]) if len(meta) > 0 else 4
        infoset_counts = int(meta[1]) if len(meta) > 1 else -1
        num_outcomes = int(meta[2]) if len(meta) > 2 else int(len(data['outcome_id_to_outcome']))

        tree = cls(
            node_types=data['node_types'],
            players=data['players'],
            infosets=data['infosets'],
            children=data['children'],
            chance_offsets=data['chance_offsets'],
            chance_outcomes=data['chance_outcomes'],
            chance_children=data['chance_children'],
            chance_probs=data['chance_probs'],
            outcome_id_to_outcome=data['outcome_id_to_outcome'],
            payoffs=data['payoffs'],
            depths=data['depths'],
            roots=data['roots'],
            num_actions=num_actions,
            infoset_counts=infoset_counts,
            num_outcomes=num_outcomes,
            infoset_keys_map=infoset_keys_map
        )
        print(f"Tree loaded in {time.time() - t0:.4f}s")
        return tree

def build_tensor_tree(game, combination_generator):
    """
    Builds the flat tensor tree from the game rules.
    This replaces the slow object-based GameTree builder.
    """
    print("Building tensor game tree (explicit chance nodes)...")
    start_time = time.time()
    
    actions = ['check', 'bet', 'call', 'fold']
    action_to_idx = {a: i for i, a in enumerate(actions)}
    num_actions = len(actions)
    
    infoset_map = {}
    next_infoset_id = 0

    # Global mapping for chance outcomes (cards) -> integer ids for stable storage
    # Deterministic order: sorted by string representation
    game.reset(0)
    initial_deck = list(getattr(game.dealer, 'deck', []))
    unique_outcomes = sorted(list(set(initial_deck)), key=lambda x: str(x))
    outcome_to_id = {o: i for i, o in enumerate(unique_outcomes)}
    outcome_id_to_outcome = np.array(unique_outcomes)
    
    # Temporary lists to hold node data (will convert to numpy at the end)
    # We use lists of lists/tuples which is faster than list of dicts
    # Structure: [type, player, infoset_id, depth, payoff0, payoff1, child0, child1, child2, child3]
    # Children will be indices.
    
    # We store node arrays in lists (append-only) and convert to numpy at the end.
    node_types_list = []
    players_list = []
    infosets_list = []
    depths_list = []
    payoffs_list = []
    children_list = []

    # Ragged chance edges via prefix sum offsets.
    chance_offsets_list = []  # length num_nodes+1 after final append
    chance_outcomes_list = []
    chance_children_list = []
    chance_probs_list = []
    total_chance_edges = 0
    
    # Helper to traverse recursively
    # We use a stack to avoid recursion limit issues and for slight speedup, 
    # but for clarity recursion is fine if depth is low (poker is usually low depth).
    # Using the existing recursive structure from TensorCFRSolver as it works.
    
    def traverse(depth: int) -> int:
        nonlocal next_infoset_id, total_chance_edges

        current_idx = len(node_types_list)

        # Allocate a stable node slot up-front (like the legacy placeholder approach).
        # This is crucial so that child indices created during recursion refer to the correct parent index.
        node_types_list.append(-1)
        players_list.append(-1)
        infosets_list.append(-1)
        depths_list.append(depth)
        payoffs_list.append((0.0, 0.0))
        children_list.append((-1, -1, -1, -1))
        # chance_offsets[i] is the start offset for node i in the flat chance arrays
        chance_offsets_list.append(total_chance_edges)

        # Terminal?
        if game.done:
            node_types_list[current_idx] = 0
            players_list[current_idx] = -1
            infosets_list[current_idx] = -1
            payoffs_list[current_idx] = (float(game.get_payoff(0)), float(game.get_payoff(1)))
            children_list[current_idx] = (-1, -1, -1, -1)
            return current_idx

        # Chance?
        if hasattr(game, 'is_chance_node') and game.is_chance_node():
            outcomes_with_probs = game.get_chance_outcomes_with_probs()
            outcomes = list(outcomes_with_probs.keys())
            # Ensure outcomes are in the global mapping; deterministic insertion order via sorted
            for o in sorted(outcomes, key=lambda x: str(x)):
                if o not in outcome_to_id:
                    outcome_to_id[o] = len(outcome_to_id)
            # Stable order: sort by outcome id
            outcomes_sorted = sorted(outcomes, key=lambda o: int(outcome_to_id[o]))

            node_types_list[current_idx] = 2
            players_list[current_idx] = -1
            infosets_list[current_idx] = -1
            payoffs_list[current_idx] = (0.0, 0.0)
            children_list[current_idx] = (-1, -1, -1, -1)

            # Reserve space for chance edges for this node (so chance_offsets remains prefix-sum by node index)
            start = total_chance_edges
            m = len(outcomes_sorted)
            total_chance_edges += m
            chance_outcomes_list.extend([0] * m)
            chance_children_list.extend([-1] * m)
            chance_probs_list.extend([0.0] * m)

            for j, outcome in enumerate(outcomes_sorted):
                prob = float(outcomes_with_probs.get(outcome, 0.0))
                # Traverse child
                game.step(outcome)
                child_idx = traverse(depth + 1)
                game.step_back()
                chance_outcomes_list[start + j] = int(outcome_to_id[outcome])
                chance_children_list[start + j] = int(child_idx)
                chance_probs_list[start + j] = prob

            return current_idx

        # Decision node
        player = int(game.current_player)
        legal_actions = game.get_legal_actions()

        key = KeyGenerator.get_info_set_key(game, player)
        if key not in infoset_map:
            infoset_map[key] = next_infoset_id
            next_infoset_id += 1
        infoset_id = infoset_map[key]

        child_indices = [-1] * num_actions
        for action in legal_actions:
            if action not in action_to_idx:
                continue
            a_idx = action_to_idx[action]
            game.step(action)
            child_idx = traverse(depth + 1)
            game.step_back()
            child_indices[a_idx] = child_idx

        node_types_list[current_idx] = 1
        players_list[current_idx] = player
        infosets_list[current_idx] = int(infoset_id)
        payoffs_list[current_idx] = (0.0, 0.0)
        children_list[current_idx] = tuple(int(x) for x in child_indices)
        return current_idx

    # Build from a single root state; private/public deals are explicit chance nodes in the environment.
    game.reset(0)
    root_idx = traverse(0)
    root_indices = [root_idx]

    # Finalize chance offsets
    chance_offsets_list.append(total_chance_edges)
        
    # Convert to Numpy Arrays
    num_nodes = len(node_types_list)
    print(f"Converting {num_nodes} nodes to tensors...")

    node_types = np.array(node_types_list, dtype=np.int8)
    players = np.array(players_list, dtype=np.int8)
    infosets = np.array(infosets_list, dtype=np.int32)
    depths = np.array(depths_list, dtype=np.int32)
    payoffs = np.array(payoffs_list, dtype=np.float32).reshape(num_nodes, 2)
    children = np.array(children_list, dtype=np.int32).reshape(num_nodes, num_actions)

    chance_offsets = np.array(chance_offsets_list, dtype=np.int64)
    chance_outcomes = np.array(chance_outcomes_list, dtype=np.int32)
    chance_children = np.array(chance_children_list, dtype=np.int32)
    chance_probs = np.array(chance_probs_list, dtype=np.float32)

    # Update outcome_id_to_outcome in case we discovered new outcomes
    outcome_id_to_outcome = np.array(
        [o for o, _ in sorted(outcome_to_id.items(), key=lambda kv: kv[1])],
        dtype=str,
    )
        
    roots = np.array(root_indices, dtype=np.int32)
    
    tree = TensorizedGameTree(
        node_types=node_types,
        players=players,
        infosets=infosets,
        children=children,
        chance_offsets=chance_offsets,
        chance_outcomes=chance_outcomes,
        chance_children=chance_children,
        chance_probs=chance_probs,
        outcome_id_to_outcome=outcome_id_to_outcome,
        payoffs=payoffs,
        depths=depths,
        roots=roots,
        num_actions=num_actions,
        infoset_counts=next_infoset_id,
        num_outcomes=len(outcome_id_to_outcome),
        infoset_keys_map=infoset_map
    )
    
    print(f"Tree built in {time.time() - start_time:.2f}s")
    return tree

def get_tree_path(game_name, abstract_suits=False):
    script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if abstract_suits:
        output_dir = os.path.join(script_dir, 'data', 'trees', 'game_trees', 'tensor', 'abstracted')
    else:
        output_dir = os.path.join(script_dir, 'data', 'trees', 'game_trees', 'tensor', 'normal')
    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(output_dir, f"{game_name}_tensor_tree.npz")
