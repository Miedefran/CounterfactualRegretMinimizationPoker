"""
Test to verify that chance node detection works correctly for Twelve Card Poker.

This test builds the public state tree and counts the chance nodes.
"""

import os
import sys

# Add src directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.twelve_card_poker.game import TwelveCardPokerGame
from utils.poker_utils import GAME_CONFIGS
from evaluation.build_public_state_tree_v2 import build_public_state_tree


def test_twelve_card_poker_chance_nodes():
    """Build the public state tree for Twelve Card Poker and count chance nodes."""
    print("\n" + "=" * 80)
    print("TEST: Twelve Card Poker Chance Node Detection")
    print("=" * 80)
    
    game_class = TwelveCardPokerGame
    game_config = GAME_CONFIGS['twelve_card_poker']
    
    print("\nBuilding public state tree for Twelve Card Poker...")
    tree = build_public_state_tree(game_class, game_config)
    public_states = tree['public_states']
    
    # Count chance nodes by traversing the tree
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
            
            # Check deck reduction for nodes with 1 public card
            if num_cards == 1 and len(nodes) > 0:
                sample_node = nodes[0][1]
                outcomes = list(sample_node.get('children', {}).keys())
                print(f"      Sample outcomes: {len(outcomes)} (should be 11 for Twelve Card Poker: 12 - 1)")
                
                # Verify deck reduction
                public_cards_in_sample = [item for item in nodes[0][0] 
                                         if isinstance(item, str) and len(item) == 2 and item[1] in ['s', 'h', 'd', 'c']]
                if public_cards_in_sample:
                    first_card = public_cards_in_sample[0]
                    if first_card in outcomes:
                        print(f"      ⚠ WARNING: {first_card} is in outcomes but should be excluded!")
                    else:
                        print(f"      ✓ Deck reduction working: {first_card} correctly excluded")
    else:
        print(f"\n  ⚠ WARNING: No chance nodes found! Detection logic may not be working.")
    
    print("\n" + "=" * 80)
    print("TEST COMPLETED!")
    print("=" * 80)


if __name__ == "__main__":
    test_twelve_card_poker_chance_nodes()
