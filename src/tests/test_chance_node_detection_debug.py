"""
Debug-Test um zu verstehen, warum die Chance-Node-Erkennung ungewöhnliche Ergebnisse liefert.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.twelve_card_poker.game import TwelveCardPokerGame
from utils.poker_utils import GAME_CONFIGS
from evaluation.build_public_state_tree_v2 import build_public_state_tree


def analyze_chance_nodes():
    """Analysiere die Chance-Nodes im Detail."""
    print("\n" + "=" * 80)
    print("DEBUG: Chance Node Analysis")
    print("=" * 80)
    
    game_class = TwelveCardPokerGame
    game_config = GAME_CONFIGS['twelve_card_poker']
    
    print("\nBuilding tree...")
    tree = build_public_state_tree(game_class, game_config)
    public_states = tree['public_states']
    
    chance_nodes = [(k, v) for k, v in public_states.items() if v['type'] == 'chance']
    
    print(f"\nTotal chance nodes: {len(chance_nodes)}")
    
    # Analysiere erste Chance-Nodes (0 Public Cards)
    first_chance_nodes = []
    second_chance_nodes = []
    
    for public_key, node in chance_nodes:
        public_cards = [item for item in public_key 
                       if isinstance(item, str) and len(item) == 2 and item[1] in ['s', 'h', 'd', 'c']]
        if len(public_cards) == 0:
            first_chance_nodes.append((public_key, node))
        elif len(public_cards) == 1:
            second_chance_nodes.append((public_key, node))
    
    print(f"\nFirst chance nodes (0 public cards): {len(first_chance_nodes)}")
    print(f"Second chance nodes (1 public card): {len(second_chance_nodes)}")
    
    # Zeige Beispiele für erste Chance-Nodes
    print(f"\n--- Examples of first chance nodes ---")
    for i, (public_key, node) in enumerate(first_chance_nodes[:5]):
        print(f"\nExample {i+1}:")
        print(f"  History: {list(public_key)[:10]}... (length: {len(public_key)})")
        outcomes = list(node.get('children', {}).keys())
        print(f"  Outcomes: {len(outcomes)}")
        print(f"  Sample outcomes: {outcomes[:5]}")
    
    # Zeige Beispiele für zweite Chance-Nodes
    print(f"\n--- Examples of second chance nodes ---")
    for i, (public_key, node) in enumerate(second_chance_nodes[:5]):
        print(f"\nExample {i+1}:")
        public_cards = [item for item in public_key 
                       if isinstance(item, str) and len(item) == 2 and item[1] in ['s', 'h', 'd', 'c']]
        print(f"  History length: {len(public_key)}")
        print(f"  Public cards: {public_cards}")
        outcomes = list(node.get('children', {}).keys())
        print(f"  Outcomes: {len(outcomes)}")
        print(f"  Sample outcomes: {outcomes[:5]}")
        # Prüfe ob die Public Card in den Outcomes ist
        if public_cards:
            if public_cards[0] in outcomes:
                print(f"  ⚠ WARNING: {public_cards[0]} is in outcomes!")
            else:
                print(f"  ✓ {public_cards[0]} correctly excluded")
    
    # Analysiere die Histories der ersten Chance-Nodes
    print(f"\n--- Analysis of first chance node histories ---")
    history_patterns = {}
    for public_key, node in first_chance_nodes:
        # Finde die letzten Actions vor dem Chance-Node
        actions_only = [item for item in public_key if item in ['bet', 'check', 'call', 'fold']]
        if len(actions_only) >= 2:
            last_two = tuple(actions_only[-2:])
            if last_two not in history_patterns:
                history_patterns[last_two] = []
            history_patterns[last_two].append(list(public_key))
    
    print(f"Last two actions patterns for first chance nodes:")
    for pattern, histories in sorted(history_patterns.items(), key=lambda x: -len(x[1])):
        print(f"  {pattern}: {len(histories)} nodes")
        print(f"    Example histories:")
        for hist in histories[:3]:
            print(f"      {hist}")
    
    # Prüfe ob es fehlende Patterns gibt
    print(f"\n--- Checking for missing patterns ---")
    print(f"Expected ways to complete Round 0:")
    print(f"  - check, check")
    print(f"  - bet, call")
    print(f"  - check, bet, call")
    print(f"  - bet, bet, call (bet limit)")
    print(f"  - check, bet, bet, call (bet limit)")
    print(f"  - check, bet, bet, bet, call (bet limit)")
    print(f"  - ... (weitere Kombinationen mit bet limit)")
    
    all_patterns = set(history_patterns.keys())
    print(f"\nFound patterns: {all_patterns}")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    analyze_chance_nodes()
