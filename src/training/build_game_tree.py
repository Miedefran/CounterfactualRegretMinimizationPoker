"""
Modul zum Bauen und Speichern von Game Trees für CFR.

Der Tree wird einmal gebaut und kann dann gespeichert werden,
um später geladen zu werden ohne ihn neu bauen zu müssen.
"""

import pickle
import gzip
import os
import time
from collections import defaultdict

from utils.data_models import KeyGenerator


class Node:
    """Repräsentiert einen Node im Game Tree"""
    def __init__(self, node_id):
        self.node_id = node_id
        self.type = None  # 'terminal' oder 'decision'
        self.player = None  # 0 oder 1 (nur bei decision nodes)
        self.infoset_key = None  # InfoSet Key (nur bei decision nodes)
        self.legal_actions = []  # Liste von legalen Aktionen
        self.children = {}  # {action: child_node_id}
        self.payoffs = None  # [payoff_p0, payoff_p1] (nur bei terminal nodes)
        self.depth = 0
    
    def to_dict(self):
        """Konvertiert Node zu Dictionary für Serialisierung"""
        return {
            'node_id': self.node_id,
            'type': self.type,
            'player': self.player,
            'infoset_key': self.infoset_key,
            'legal_actions': self.legal_actions,
            'children': self.children,
            'payoffs': self.payoffs,
            'depth': self.depth
        }
    
    @classmethod
    def from_dict(cls, data):
        """Erstellt Node aus Dictionary"""
        node = cls(data['node_id'])
        node.type = data['type']
        node.player = data['player']
        node.infoset_key = data['infoset_key']
        node.legal_actions = data['legal_actions']
        node.children = data['children']
        node.payoffs = data['payoffs']
        node.depth = data['depth']
        return node


class GameTree:
    """Repräsentiert einen vollständigen Game Tree"""
    def __init__(self):
        self.nodes = {}  # {node_id: Node}
        self.infoset_to_nodes = defaultdict(list)  # {infoset_key: [node_ids]}
        self.root_nodes = []  # Liste von root node IDs (eine pro Kombination)
        self.game_name = None
        self.game_config = None
    
    def to_dict(self):
        """Konvertiert Tree zu Dictionary für Serialisierung"""
        return {
            'nodes': {node_id: node.to_dict() for node_id, node in self.nodes.items()},
            'infoset_to_nodes': dict(self.infoset_to_nodes),
            'root_nodes': self.root_nodes,
            'game_name': self.game_name,
            'game_config': self.game_config
        }
    
    @classmethod
    def from_dict(cls, data):
        """Erstellt Tree aus Dictionary"""
        tree = cls()
        tree.nodes = {node_id: Node.from_dict(node_data) 
                     for node_id, node_data in data['nodes'].items()}
        tree.infoset_to_nodes = defaultdict(list, data['infoset_to_nodes'])
        tree.root_nodes = data['root_nodes']
        tree.game_name = data.get('game_name')
        tree.game_config = data.get('game_config')
        return tree


def build_game_tree(game, combination_generator, game_name=None, game_config=None, abstract_suits=False):
    """
    Baut einen Game Tree für das gegebene Spiel.
    
    Args:
        game: Das Game-Objekt
        combination_generator: Der CombinationGenerator für das Spiel
        game_name: Optionaler Name des Spiels (für Speicherung)
        game_config: Optionale Game-Konfiguration (für Speicherung)
        abstract_suits: Wenn True, werden Suits in InfoSet Keys entfernt (Suit Abstraction)
    
    Returns:
        GameTree Objekt
    """
    abstraction_str = " (suit abstracted)" if abstract_suits else ""
    print(f"Building game tree{abstraction_str}...")
    start_time = time.time()
    
    tree = GameTree()
    tree.game_name = game_name
    tree.game_config = game_config
    next_node_id = 0
    combinations = combination_generator.get_all_combinations()
    
    def traverse_and_build(depth):
        """Rekursive Funktion die den Tree durchläuft und Nodes erstellt"""
        nonlocal next_node_id
        
        node_id = next_node_id
        next_node_id += 1
        node = Node(node_id)
        node.depth = depth
        tree.nodes[node_id] = node
        
        # Terminal Node?
        if game.done:
            node.type = 'terminal'
            node.payoffs = [game.get_payoff(0), game.get_payoff(1)]
            return node_id
        
        # Decision Node
        node.type = 'decision'
        node.player = game.current_player
        node.legal_actions = game.get_legal_actions()
        node.infoset_key = KeyGenerator.get_info_set_key(game, node.player)
        
        # InfoSet Mapping aktualisieren
        tree.infoset_to_nodes[node.infoset_key].append(node_id)
        
        # Für jede legale Aktion: Child Node erstellen
        for action in node.legal_actions:
            game.step(action)
            child_id = traverse_and_build(depth + 1)
            game.step_back()
            node.children[action] = child_id
        
        return node_id
    
    # Für jede Kombination (Deal) einen Subtree bauen
    for combo in combinations:
        combination_generator.setup_game_with_combination(game, combo)
        root_id = traverse_and_build(0)
        tree.root_nodes.append(root_id)
    
    build_time = time.time() - start_time
    print(f"Tree built: {len(tree.nodes)} nodes, {len(tree.infoset_to_nodes)} unique infosets")
    print(f"Tree building took {build_time:.2f}s")
    
    return tree


def save_game_tree(tree, game_name, output_dir=None, abstract_suits=False):
    """
    Speichert einen Game Tree in einer Datei.
    
    Args:
        tree: GameTree Objekt
        game_name: Name des Spiels (für Dateinamen)
        output_dir: Optionales Verzeichnis (default: data/trees/game_trees/normal oder abstracted)
        abstract_suits: Wenn True, wird in abstracted Verzeichnis gespeichert
    
    Returns:
        Pfad zur gespeicherten Datei
    """
    if output_dir is None:
        script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        subdir = 'abstracted' if abstract_suits else 'normal'
        output_dir = os.path.join(script_dir, 'data', 'trees', 'game_trees', subdir)
    
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{game_name}_game_tree.pkl.gz"
    filepath = os.path.join(output_dir, filename)
    
    tree_dict = tree.to_dict()
    
    with gzip.open(filepath, 'wb') as f:
        pickle.dump(tree_dict, f)
    
    print(f"Saved game tree to: {filepath}")
    return filepath


def load_game_tree(game_name, input_dir=None, abstract_suits=False):
    """
    Lädt einen Game Tree aus einer Datei.
    
    Args:
        game_name: Name des Spiels (für Dateinamen)
        input_dir: Optionales Verzeichnis (default: data/trees/game_trees/normal oder abstracted)
        abstract_suits: Wenn True, wird aus abstracted Verzeichnis geladen
    
    Returns:
        GameTree Objekt
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
