"""
Module for building and saving game trees for CFR.

The tree is built once and can then be saved,
to be loaded later without having to rebuild it.
"""

import pickle
import gzip
import os
import time
from collections import defaultdict

from utils.data_models import KeyGenerator
from utils.tree_registry import record_tree_stats


class Node:
    """Represents a node in the game tree."""

    def __init__(self, node_id):
        self.node_id = node_id
        self.type = None  # 'terminal', 'decision' or 'chance'
        self.player = None  # 0 or 1 (decision nodes), -1 (chance), None (terminal)
        self.infoset_key = None  # InfoSet key (only for decision nodes)
        self.legal_actions = []  # List of legal actions
        self.children = {}  # {action: child_node_id}
        self.chance_probs = None  # {outcome: prob} (only for chance nodes)
        self.payoffs = None  # [payoff_p0, payoff_p1] (only for terminal nodes)
        self.depth = 0

    def to_dict(self):
        """Convert node to dictionary for serialization."""
        return {
            'node_id': self.node_id,
            'type': self.type,
            'player': self.player,
            'infoset_key': self.infoset_key,
            'legal_actions': self.legal_actions,
            'children': self.children,
            'chance_probs': self.chance_probs,
            'payoffs': self.payoffs,
            'depth': self.depth
        }

    @classmethod
    def from_dict(cls, data):
        """Create node from dictionary."""
        node = cls(data['node_id'])
        node.type = data['type']
        node.player = data['player']
        node.infoset_key = data['infoset_key']
        node.legal_actions = data['legal_actions']
        node.children = data['children']
        node.chance_probs = data.get('chance_probs')
        node.payoffs = data['payoffs']
        node.depth = data['depth']
        return node


class GameTree:
    """Represents a complete game tree."""

    def __init__(self):
        self.nodes = {}  # {node_id: Node}
        self.infoset_to_nodes = defaultdict(list)  # {infoset_key: [node_ids]}
        self.root_nodes = []  # List of root node IDs (one per combination)
        self.game_name = None

    def to_dict(self):
        """Convert tree to dictionary for serialization."""
        return {
            'nodes': {node_id: node.to_dict() for node_id, node in self.nodes.items()},
            'infoset_to_nodes': dict(self.infoset_to_nodes),
            'root_nodes': self.root_nodes,
            'game_name': self.game_name,
        }

    @classmethod
    def from_dict(cls, data):
        """Create tree from dictionary."""
        tree = cls()
        tree.nodes = {node_id: Node.from_dict(node_data)
                      for node_id, node_data in data['nodes'].items()}
        tree.infoset_to_nodes = defaultdict(list, data['infoset_to_nodes'])
        tree.root_nodes = data['root_nodes']
        tree.game_name = data.get('game_name')
        return tree


def build_game_tree(game, combination_generator, game_name=None, abstract_suits=False) -> GameTree:
    """
    Build a game tree for the given game.
    
    Args:
        game: The game object
        combination_generator: The CombinationGenerator for the game
        game_name: Optional name of the game (for saving)
        abstract_suits: If True, suits are removed from InfoSet keys (Suit Abstraction)
    
    Returns:
        GameTree object
    """
    abstraction_str = " (suit abstracted)" if abstract_suits else ""
    print(f"Building game tree{abstraction_str}...")
    start_time = time.time()

    tree = GameTree()
    tree.game_name = game_name
    next_node_id = 0

    # With explicit chance nodes, we build from a single root state.
    # (The old combination_generator-based enumeration is no longer needed.)

    def traverse_and_build(depth):
        """Recursive function that traverses tree and creates nodes."""
        nonlocal next_node_id

        node_id = next_node_id
        next_node_id += 1

        # Print progress every 1000 nodes
        if next_node_id % 1000 == 0:
            print(f"  Building tree: {next_node_id} nodes created...")

        node = Node(node_id)
        node.depth = depth
        tree.nodes[node_id] = node

        # Terminal node?
        if game.done:
            node.type = 'terminal'
            node.payoffs = [game.get_payoff(0), game.get_payoff(1)]
            return node_id

        # Chance node?
        if hasattr(game, 'is_chance_node') and game.is_chance_node():
            node.type = 'chance'
            node.player = getattr(game, 'CHANCE_PLAYER', -1)
            outcomes_with_probs = game.get_chance_outcomes_with_probs()
            node.chance_probs = dict(outcomes_with_probs)
            node.legal_actions = list(outcomes_with_probs.keys())

            for outcome in node.legal_actions:
                game.step(outcome)
                child_id = traverse_and_build(depth + 1)
                game.step_back()
                node.children[outcome] = child_id
            return node_id

        # Decision node
        node.type = 'decision'
        node.player = game.current_player
        node.legal_actions = game.get_legal_actions()
        node.infoset_key = KeyGenerator.get_info_set_key(game, node.player)

        # Update InfoSet mapping
        tree.infoset_to_nodes[node.infoset_key].append(node_id)

        # For each legal action: create child node
        for action in node.legal_actions:
            game.step(action)
            child_id = traverse_and_build(depth + 1)
            game.step_back()
            node.children[action] = child_id

        return node_id

    # Single root (chance at start handles private deals)
    game.reset(0)
    root_id = traverse_and_build(0)
    tree.root_nodes.append(root_id)

    build_time = time.time() - start_time
    print(f"Tree built: {len(tree.nodes)} nodes, {len(tree.infoset_to_nodes)} unique infosets")
    print(f"Tree building took {build_time:.2f}s")

    # Registry logging (tree build): only if game_name is set
    if game_name:
        type_counts = {"terminal": 0, "decision": 0, "chance": 0}
        for n in tree.nodes.values():
            t = getattr(n, "type", None)
            if t in type_counts:
                type_counts[t] += 1
        record_tree_stats(
            {
                "schema_version": 1,
                "tree_kind": "game_tree_object",
                "game": str(game_name),
                "abstract_suits": bool(abstract_suits),
                "num_nodes": int(len(tree.nodes)),
                "num_infosets": int(len(tree.infoset_to_nodes)),
                "node_type_counts": type_counts,
            }
        )

    return tree


def save_game_tree(tree, game_name, output_dir=None, abstract_suits=False):
    """
    Save a game tree to a file.
    
    Args:
        tree: GameTree object
        game_name: Name of the game (for filename)
        output_dir: Optional directory (default: data/trees/game_trees/normal or abstracted)
        abstract_suits: If True, saved in abstracted directory
    
    Returns:
        Path to saved file
    """
    if output_dir is None:
        script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        subdir = 'abstracted' if abstract_suits else 'normal'
        output_dir = os.path.join(script_dir, 'data', 'trees', 'game_trees', subdir)

    os.makedirs(output_dir, exist_ok=True)
    filename = f"{game_name}_game_tree.pkl.gz"
    filepath = os.path.join(output_dir, filename)

    print(f"Saving game tree ({len(tree.nodes)} nodes, {len(tree.infoset_to_nodes)} infosets)...")
    tree_dict = tree.to_dict()

    # Registry logging (tree save): persist sizes + node types
    type_counts = {"terminal": 0, "decision": 0, "chance": 0}
    for n in tree.nodes.values():
        t = getattr(n, "type", None)
        if t in type_counts:
            type_counts[t] += 1
    record_tree_stats(
        {
            "schema_version": 1,
            "tree_kind": "game_tree_object",
            "game": str(game_name),
            "abstract_suits": bool(abstract_suits),
            "num_nodes": int(len(tree.nodes)),
            "num_infosets": int(len(tree.infoset_to_nodes)),
            "node_type_counts": type_counts,
            "tree_path": filepath,
        }
    )

    # Print progress while saving
    print(f"  Serializing tree to dictionary...")
    with gzip.open(filepath, 'wb') as f:
        print(f"  Writing to file: {filepath}")
        pickle.dump(tree_dict, f)

    print(f"Saved game tree to: {filepath}")
    return filepath


def load_game_tree(game_name, input_dir=None, abstract_suits=False):
    """
    Load a game tree from a file.
    
    Args:
        game_name: Name of the game (for filename)
        input_dir: Optional directory (default: data/trees/game_trees/normal or abstracted)
        abstract_suits: If True, loaded from abstracted directory
    
    Returns:
        GameTree object
    """
    if input_dir is None:
        script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        subdir = 'abstracted' if abstract_suits else 'normal'
        input_dir = os.path.join(script_dir, 'data', 'trees', 'game_trees', subdir)

    filename = f"{game_name}_game_tree.pkl.gz"
    filepath = os.path.join(input_dir, filename)

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Game tree not found: {filepath}")

    with gzip.open(filepath, 'rb') as f:
        tree_dict = pickle.load(f)

    tree = GameTree.from_dict(tree_dict)
    print(f"Loaded game tree from: {filepath}")
    print(f"Tree: {len(tree.nodes)} nodes, {len(tree.infoset_to_nodes)} unique infosets")

    return tree
