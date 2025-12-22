"""
Druckt einen Teil des Twelve Card Poker Public State Trees aus.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.twelve_card_poker.game import TwelveCardPokerGame
from utils.poker_utils import GAME_CONFIGS
from evaluation.build_public_state_tree_v2 import build_public_state_tree, print_public_state_tree


def print_tree_sample():
    """Druckt einen Teil des Trees aus."""
    print("\n" + "=" * 80)
    print("TWELVE CARD POKER PUBLIC STATE TREE - SAMPLE")
    print("=" * 80)
    
    game_class = TwelveCardPokerGame
    game_config = GAME_CONFIGS['twelve_card_poker']
    
    print("\nBauen Tree...")
    tree = build_public_state_tree(game_class, game_config)
    public_states = tree['public_states']
    
    print(f"\nTotal public states: {len(public_states)}")
    
    # Zeige den Root und einige Kinder
    print("\n" + "=" * 80)
    print("ROOT NODE UND ERSTE EBENEN:")
    print("=" * 80)
    
    root_key = ()
    if root_key in public_states:
        root_node = public_states[root_key]
        print(f"\nRoot: {root_key}")
        print(f"  Type: {root_node['type']}")
        if root_node['type'] == 'choice':
            print(f"  Player: {root_node['player']}")
        if 'children' in root_node:
            print(f"  Children: {len(root_node['children'])}")
            print(f"  Children keys: {list(root_node['children'].keys())[:10]}")
    
    # Zeige erste Chance-Nodes
    print("\n" + "=" * 80)
    print("ERSTE CHANCE-NODES (nach Round 0):")
    print("=" * 80)
    
    chance_nodes_0 = []
    for public_key, node in public_states.items():
        if node['type'] == 'chance':
            public_cards = [item for item in public_key 
                           if isinstance(item, str) and len(item) == 2 and item[1] in ['s', 'h', 'd', 'c']]
            if len(public_cards) == 0:
                chance_nodes_0.append((public_key, node))
    
    print(f"\nGefunden: {len(chance_nodes_0)} erste Chance-Nodes")
    
    for i, (public_key, node) in enumerate(chance_nodes_0[:3]):
        print(f"\n--- Chance Node {i+1} ---")
        print(f"History: {list(public_key)}")
        print(f"Type: {node['type']}")
        if 'children' in node:
            outcomes = list(node['children'].keys())
            print(f"Outcomes: {len(outcomes)}")
            print(f"Sample outcomes: {outcomes[:5]}")
            
            # Zeige einen Child
            if outcomes:
                first_outcome = outcomes[0]
                child_key = node['children'][first_outcome]
                if child_key in public_states:
                    child_node = public_states[child_key]
                    print(f"\n  Child nach Outcome '{first_outcome}':")
                    print(f"    Key: {list(child_key)[:10]}...")
                    print(f"    Type: {child_node['type']}")
                    if child_node['type'] == 'choice':
                        print(f"    Player: {child_node['player']}")
                    if 'children' in child_node:
                        print(f"    Children: {list(child_node['children'].keys())[:5]}")
    
    # Zeige zweite Chance-Nodes
    print("\n" + "=" * 80)
    print("ZWEITE CHANCE-NODES (nach Round 1):")
    print("=" * 80)
    
    chance_nodes_1 = []
    for public_key, node in public_states.items():
        if node['type'] == 'chance':
            public_cards = [item for item in public_key 
                           if isinstance(item, str) and len(item) == 2 and item[1] in ['s', 'h', 'd', 'c']]
            if len(public_cards) == 1:
                chance_nodes_1.append((public_key, node))
    
    print(f"\nGefunden: {len(chance_nodes_1)} zweite Chance-Nodes")
    
    for i, (public_key, node) in enumerate(chance_nodes_1[:3]):
        print(f"\n--- Chance Node {i+1} ---")
        public_cards = [item for item in public_key 
                       if isinstance(item, str) and len(item) == 2 and item[1] in ['s', 'h', 'd', 'c']]
        print(f"History: {list(public_key)[:15]}... (length: {len(public_key)})")
        print(f"Public Cards: {public_cards}")
        print(f"Type: {node['type']}")
        if 'children' in node:
            outcomes = list(node['children'].keys())
            print(f"Outcomes: {len(outcomes)}")
            print(f"Sample outcomes: {outcomes[:5]}")
            
            # Prüfe Deck-Reduktion
            if public_cards:
                first_card = public_cards[0]
                if first_card in outcomes:
                    print(f"  ⚠ WARNING: {first_card} ist in Outcomes!")
                else:
                    print(f"  ✓ {first_card} korrekt ausgeschlossen")
    
    # Verwende die print_public_state_tree Funktion für einen strukturierten Überblick
    print("\n" + "=" * 80)
    print("STRUKTURIERTER TREE-ÜBERBLICK (erste 3 Ebenen):")
    print("=" * 80)
    print_public_state_tree(public_states, root_key=(), indent="")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    print_tree_sample()
