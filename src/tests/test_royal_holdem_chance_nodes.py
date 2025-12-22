"""
Test für Royal Hold'em Chance-Node-Erkennung.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.royal_holdem.game import RoyalHoldemGame
from utils.poker_utils import GAME_CONFIGS
from evaluation.build_public_state_tree_v2 import build_public_state_tree


def test_royal_holdem_chance_nodes():
    """Test Chance-Node-Erkennung für Royal Hold'em."""
    print("\n" + "=" * 80)
    print("TEST: Royal Hold'em Chance Node Detection")
    print("=" * 80)
    
    game_class = RoyalHoldemGame
    game_config = GAME_CONFIGS['royal_holdem']
    
    print("\nBuilding public state tree for Royal Hold'em...")
    tree = build_public_state_tree(game_class, game_config)
    public_states = tree['public_states']
    
    # Count chance nodes
    chance_nodes = []
    for public_key, node in public_states.items():
        if node['type'] == 'chance':
            chance_nodes.append((public_key, node))
    
    print(f"\nResults:")
    print(f"  Total public states: {len(public_states)}")
    print(f"  Chance nodes found: {len(chance_nodes)}")
    
    if len(chance_nodes) > 0:
        print(f"\n  ✓ Chance nodes detected successfully!")
        
        # Analyze chance nodes by number of public cards
        chance_nodes_by_cards = {}
        for public_key, node in chance_nodes:
            public_cards = [item for item in public_key 
                          if isinstance(item, str) and len(item) == 2 and item[1] in ['s', 'h', 'd', 'c']]
            num_cards = len(public_cards)
            if num_cards not in chance_nodes_by_cards:
                chance_nodes_by_cards[num_cards] = []
            chance_nodes_by_cards[num_cards].append((public_key, node))
        
        print(f"\n  Chance nodes by number of public cards:")
        for num_cards in sorted(chance_nodes_by_cards.keys()):
            nodes = chance_nodes_by_cards[num_cards]
            print(f"    {num_cards} public card(s): {len(nodes)} chance node(s)")
            
            # Check deck reduction
            if len(nodes) > 0:
                sample_node = nodes[0][1]
                outcomes = list(sample_node.get('children', {}).keys())
                expected_outcomes = 20 - num_cards  # Royal Hold'em hat 20 Karten (5 Ranks × 4 Suits)
                print(f"      Sample outcomes: {len(outcomes)} (should be {expected_outcomes} for Royal Hold'em: 20 - {num_cards})")
                
                # Verify deck reduction
                public_cards_in_sample = [item for item in nodes[0][0] 
                                         if isinstance(item, str) and len(item) == 2 and item[1] in ['s', 'h', 'd', 'c']]
                if public_cards_in_sample:
                    for card in public_cards_in_sample:
                        if card in outcomes:
                            print(f"      ⚠ WARNING: {card} is in outcomes but should be excluded!")
                        else:
                            print(f"      ✓ {card} correctly excluded")
    else:
        print(f"\n  ⚠ WARNING: No chance nodes found! Detection logic may not be working.")
    
    print("\n" + "=" * 80)
    print("TEST COMPLETED!")
    print("=" * 80)


if __name__ == "__main__":
    test_royal_holdem_chance_nodes()
