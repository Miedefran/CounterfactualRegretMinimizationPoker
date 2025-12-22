"""
Zeigt einen größeren Teil des Trees mit mehr Details.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.twelve_card_poker.game import TwelveCardPokerGame
from utils.poker_utils import GAME_CONFIGS
from evaluation.build_public_state_tree_v2 import build_public_state_tree


def print_verbose_tree():
    """Zeigt einen größeren Teil des Trees."""
    print("\n" + "=" * 80)
    print("VERBOSE TREE OUTPUT")
    print("=" * 80)
    
    game_class = TwelveCardPokerGame
    game_config = GAME_CONFIGS['twelve_card_poker']
    
    print("\nBauen Tree...")
    tree = build_public_state_tree(game_class, game_config)
    public_states = tree['public_states']
    
    # Zeige alle Chance-Nodes mit ihren Histories
    print("\n" + "=" * 80)
    print("ALLE CHANCE-NODES MIT VOLLSTÄNDIGER HISTORY:")
    print("=" * 80)
    
    chance_nodes = [(k, v) for k, v in public_states.items() if v['type'] == 'chance']
    
    # Erste Chance-Nodes
    first_chance = [(k, v) for k, v in chance_nodes 
                   if len([item for item in k if isinstance(item, str) and len(item) == 2 and item[1] in ['s', 'h', 'd', 'c']]) == 0]
    
    print(f"\nERSTE CHANCE-NODES (nach Round 0): {len(first_chance)}")
    for i, (key, node) in enumerate(first_chance, 1):
        print(f"\n{i}. History: {list(key)}")
        outcomes = list(node.get('children', {}).keys())
        print(f"   Outcomes: {len(outcomes)}")
        print(f"   Alle Outcomes: {outcomes}")
    
    # Zweite Chance-Nodes (nur erste 10)
    second_chance = [(k, v) for k, v in chance_nodes 
                    if len([item for item in k if isinstance(item, str) and len(item) == 2 and item[1] in ['s', 'h', 'd', 'c']]) == 1]
    
    print(f"\nZWEITE CHANCE-NODES (nach Round 1): {len(second_chance)}")
    print(f"Zeige erste 10:")
    for i, (key, node) in enumerate(second_chance[:10], 1):
        public_cards = [item for item in key 
                       if isinstance(item, str) and len(item) == 2 and item[1] in ['s', 'h', 'd', 'c']]
        print(f"\n{i}. History: {list(key)}")
        print(f"   Public Card: {public_cards[0] if public_cards else 'None'}")
        outcomes = list(node.get('children', {}).keys())
        print(f"   Outcomes: {len(outcomes)}")
        print(f"   Alle Outcomes: {outcomes}")
        if public_cards and public_cards[0] in outcomes:
            print(f"   ⚠ FEHLER: {public_cards[0]} ist in Outcomes!")
    
    # Zeige einen kompletten Pfad von Root zu Terminal
    print("\n" + "=" * 80)
    print("KOMPLETTER PFAD: Root -> Round 0 -> Card -> Round 1 -> Card -> Terminal")
    print("=" * 80)
    
    # Pfad: check-check -> Js -> check-check -> Jh -> bet-call
    path_keys = [
        ((), "Root"),
        (('check',), "Nach check"),
        (('check', 'check'), "Nach Round 0 complete"),
        (('check', 'check', 'Js'), "Nach Js ausgeteilt"),
        (('check', 'check', 'Js', 'check'), "Nach check in Round 1"),
        (('check', 'check', 'Js', 'check', 'check'), "Nach Round 1 complete"),
        (('check', 'check', 'Js', 'check', 'check', 'Jh'), "Nach Jh ausgeteilt"),
        (('check', 'check', 'Js', 'check', 'check', 'Jh', 'bet'), "Nach bet in Round 2"),
        (('check', 'check', 'Js', 'check', 'check', 'Jh', 'bet', 'call'), "Nach call (Terminal)"),
    ]
    
    for key, description in path_keys:
        if key in public_states:
            node = public_states[key]
            print(f"\n{description}:")
            print(f"  Key: {list(key)}")
            print(f"  Type: {node['type']}")
            if node['type'] == 'choice':
                print(f"  Player: {node['player']}")
            elif node['type'] == 'chance':
                outcomes = list(node.get('children', {}).keys())
                print(f"  Outcomes: {len(outcomes)}")
            elif node['type'] == 'terminal':
                print(f"  Pot: {node['pot']}")
        else:
            print(f"\n{description}:")
            print(f"  ✗ Nicht gefunden: {list(key)}")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    print_verbose_tree()
