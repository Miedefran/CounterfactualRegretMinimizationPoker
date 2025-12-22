"""
Robust test to verify that the public state tree correctly excludes already dealt
public cards from chance outcomes for subsequent chance nodes.

This test ensures that when building the public state tree:
1. The first chance node contains all possible cards
2. The second chance node excludes the first dealt card
3. No duplicate cards appear in chance outcomes
"""

import os
import sys
import pytest

# Add src directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.build_public_state_tree_v2 import build_public_state_tree
from envs.rhode_island.game import RhodeIslandGame
from envs.twelve_card_poker.game import TwelveCardPokerGame
from utils.poker_utils import GAME_CONFIGS


def extract_public_cards_from_history(public_hist):
    """Extract all public cards from a public history."""
    return [item for item in public_hist 
            if isinstance(item, str) and len(item) == 2 and item[1] in ['s', 'h', 'd', 'c']]


def get_all_chance_nodes(tree):
    """Extract all chance nodes from the tree with their histories."""
    chance_nodes = []
    for public_key, node in tree['public_states'].items():
        if node['type'] == 'chance':
            chance_nodes.append((public_key, node))
    return chance_nodes


def get_chance_outcomes_from_children(node):
    """Extract chance outcomes from a chance node's children.
    
    In build_public_state_tree_v2.py, the children dictionary stores:
    children[outcome] = child_key
    where outcome is the card (e.g., 'As') and child_key is the tuple history.
    So we can directly use the keys of the children dictionary as outcomes.
    """
    if 'children' not in node:
        return []
    # In v2, children dictionary keys ARE the outcomes (cards)
    outcomes = list(node['children'].keys())
    return outcomes


def test_rhode_island_deck_reduction():
    """Test that Rhode Island correctly excludes already dealt cards from second chance node."""
    print("\n" + "=" * 80)
    print("TEST: Rhode Island Deck Reduction")
    print("=" * 80)
    
    game_class = RhodeIslandGame
    game_config = GAME_CONFIGS['rhode_island']
    
    # Build the public state tree
    tree = build_public_state_tree(game_class, game_config)
    public_states = tree['public_states']
    
    # Get all chance nodes
    chance_nodes = get_all_chance_nodes(tree)
    print(f"\nFound {len(chance_nodes)} chance nodes")
    
    # Test 1: First chance node should have all 52 cards
    first_chance_nodes = []
    for public_key, node in chance_nodes:
        public_cards = extract_public_cards_from_history(list(public_key))
        if len(public_cards) == 0:  # First chance node (no cards yet)
            first_chance_nodes.append((public_key, node))
    
    assert len(first_chance_nodes) > 0, "Should have at least one first chance node"
    
    for public_key, node in first_chance_nodes:
        outcomes = get_chance_outcomes_from_children(node)
        print(f"\nFirst chance node (history: {public_key[:5]}...): {len(outcomes)} outcomes")
        
        # Should have 52 cards for Rhode Island
        assert len(outcomes) == 52, f"First chance node should have 52 cards, got {len(outcomes)}"
        assert len(set(outcomes)) == 52, "All outcomes should be unique"
    
    # Test 2: Second chance node should exclude the first dealt card
    second_chance_nodes = []
    for public_key, node in chance_nodes:
        public_cards = extract_public_cards_from_history(list(public_key))
        if len(public_cards) == 1:  # Second chance node (one card already dealt)
            second_chance_nodes.append((public_key, node, public_cards[0]))
    
    assert len(second_chance_nodes) > 0, "Should have at least one second chance node"
    
    failures = []
    for public_key, node, first_card in second_chance_nodes:
        outcomes = get_chance_outcomes_from_children(node)
        public_cards_in_hist = extract_public_cards_from_history(list(public_key))
        
        print(f"\nSecond chance node:")
        print(f"  History: {public_key[:8]}...")
        print(f"  First card: {first_card}")
        print(f"  Outcomes: {len(outcomes)} cards")
        
        # The first card should NOT be in the outcomes
        if first_card in outcomes:
            failures.append({
                'history': public_key,
                'first_card': first_card,
                'outcomes': outcomes[:10],  # First 10 for debugging
                'issue': f"First card {first_card} appears in second chance outcomes"
            })
        
        # Should have 51 cards (52 - 1 already dealt)
        if len(outcomes) != 51:
            failures.append({
                'history': public_key,
                'first_card': first_card,
                'outcomes_count': len(outcomes),
                'issue': f"Should have 51 outcomes, got {len(outcomes)}"
            })
        
        # All outcomes should be unique
        if len(set(outcomes)) != len(outcomes):
            failures.append({
                'history': public_key,
                'issue': "Duplicate outcomes found"
            })
    
    if failures:
        print("\n" + "=" * 80)
        print("FAILURES DETECTED:")
        print("=" * 80)
        for i, failure in enumerate(failures, 1):
            print(f"\nFailure {i}:")
            print(f"  Issue: {failure['issue']}")
            if 'first_card' in failure:
                print(f"  First card: {failure['first_card']}")
            if 'outcomes_count' in failure:
                print(f"  Outcomes count: {failure['outcomes_count']}")
            if 'outcomes' in failure:
                print(f"  Sample outcomes: {failure['outcomes']}")
        
        pytest.fail(f"Found {len(failures)} failures in deck reduction test")
    
    print("\n" + "=" * 80)
    print("✓ All Rhode Island deck reduction tests passed!")
    print("=" * 80)


def test_twelve_card_poker_deck_reduction():
    """Test that Twelve Card Poker correctly excludes already dealt cards from second chance node."""
    print("\n" + "=" * 80)
    print("TEST: Twelve Card Poker Deck Reduction")
    print("=" * 80)
    
    game_class = TwelveCardPokerGame
    game_config = GAME_CONFIGS['twelve_card_poker']
    
    # Build the public state tree
    tree = build_public_state_tree(game_class, game_config)
    public_states = tree['public_states']
    
    # Get all chance nodes
    chance_nodes = get_all_chance_nodes(tree)
    print(f"\nFound {len(chance_nodes)} chance nodes")
    
    # Test 1: First chance node should have all 12 cards
    first_chance_nodes = []
    for public_key, node in chance_nodes:
        public_cards = extract_public_cards_from_history(list(public_key))
        if len(public_cards) == 0:  # First chance node (no cards yet)
            first_chance_nodes.append((public_key, node))
    
    assert len(first_chance_nodes) > 0, "Should have at least one first chance node"
    
    for public_key, node in first_chance_nodes:
        outcomes = get_chance_outcomes_from_children(node)
        print(f"\nFirst chance node (history: {public_key[:5]}...): {len(outcomes)} outcomes")
        
        # Should have 12 cards for Twelve Card Poker
        assert len(outcomes) == 12, f"First chance node should have 12 cards, got {len(outcomes)}"
        assert len(set(outcomes)) == 12, "All outcomes should be unique"
    
    # Test 2: Second chance node should exclude the first dealt card
    second_chance_nodes = []
    for public_key, node in chance_nodes:
        public_cards = extract_public_cards_from_history(list(public_key))
        if len(public_cards) == 1:  # Second chance node (one card already dealt)
            second_chance_nodes.append((public_key, node, public_cards[0]))
    
    assert len(second_chance_nodes) > 0, "Should have at least one second chance node"
    
    failures = []
    for public_key, node, first_card in second_chance_nodes:
        outcomes = get_chance_outcomes_from_children(node)
        public_cards_in_hist = extract_public_cards_from_history(list(public_key))
        
        print(f"\nSecond chance node:")
        print(f"  History: {public_key[:8]}...")
        print(f"  First card: {first_card}")
        print(f"  Outcomes: {len(outcomes)} cards")
        
        # The first card should NOT be in the outcomes
        if first_card in outcomes:
            failures.append({
                'history': public_key,
                'first_card': first_card,
                'outcomes': outcomes[:10],  # First 10 for debugging
                'issue': f"First card {first_card} appears in second chance outcomes"
            })
        
        # Should have 11 cards (12 - 1 already dealt)
        if len(outcomes) != 11:
            failures.append({
                'history': public_key,
                'first_card': first_card,
                'outcomes_count': len(outcomes),
                'issue': f"Should have 11 outcomes, got {len(outcomes)}"
            })
        
        # All outcomes should be unique
        if len(set(outcomes)) != len(outcomes):
            failures.append({
                'history': public_key,
                'issue': "Duplicate outcomes found"
            })
    
    if failures:
        print("\n" + "=" * 80)
        print("FAILURES DETECTED:")
        print("=" * 80)
        for i, failure in enumerate(failures, 1):
            print(f"\nFailure {i}:")
            print(f"  Issue: {failure['issue']}")
            if 'first_card' in failure:
                print(f"  First card: {failure['first_card']}")
            if 'outcomes_count' in failure:
                print(f"  Outcomes count: {failure['outcomes_count']}")
            if 'outcomes' in failure:
                print(f"  Sample outcomes: {failure['outcomes']}")
        
        pytest.fail(f"Found {len(failures)} failures in deck reduction test")
    
    print("\n" + "=" * 80)
    print("✓ All Twelve Card Poker deck reduction tests passed!")
    print("=" * 80)


def test_chance_node_consistency():
    """Test that chance nodes are consistent across different paths to the same state."""
    print("\n" + "=" * 80)
    print("TEST: Chance Node Consistency")
    print("=" * 80)
    
    game_class = RhodeIslandGame
    game_config = GAME_CONFIGS['rhode_island']
    
    tree = build_public_state_tree(game_class, game_config)
    chance_nodes = get_all_chance_nodes(tree)
    
    # Group chance nodes by number of public cards
    nodes_by_card_count = {}
    for public_key, node in chance_nodes:
        card_count = len(extract_public_cards_from_history(list(public_key)))
        if card_count not in nodes_by_card_count:
            nodes_by_card_count[card_count] = []
        nodes_by_card_count[card_count].append((public_key, node))
    
    # For each card count, check that all nodes have the same number of outcomes
    for card_count, nodes in nodes_by_card_count.items():
        if card_count == 0:
            # First chance nodes should all have 52 outcomes
            for public_key, node in nodes:
                outcomes = get_chance_outcomes_from_children(node)
                assert len(outcomes) == 52, \
                    f"All first chance nodes should have 52 outcomes, got {len(outcomes)}"
        elif card_count == 1:
            # Second chance nodes should all have 51 outcomes
            for public_key, node in nodes:
                outcomes = get_chance_outcomes_from_children(node)
                assert len(outcomes) == 51, \
                    f"All second chance nodes should have 51 outcomes, got {len(outcomes)}"
    
    print(f"\n✓ Consistency check passed for {len(chance_nodes)} chance nodes")
    print("=" * 80)


def test_no_duplicate_cards_in_history():
    """Test that no chance node allows a card that's already in the history."""
    print("\n" + "=" * 80)
    print("TEST: No Duplicate Cards in History")
    print("=" * 80)
    
    game_class = RhodeIslandGame
    game_config = GAME_CONFIGS['rhode_island']
    
    tree = build_public_state_tree(game_class, game_config)
    chance_nodes = get_all_chance_nodes(tree)
    
    failures = []
    for public_key, node in chance_nodes:
        public_cards_in_hist = extract_public_cards_from_history(list(public_key))
        outcomes = get_chance_outcomes_from_children(node)
        
        # Check that no outcome is already in the history
        for outcome in outcomes:
            if outcome in public_cards_in_hist:
                failures.append({
                    'history': public_key,
                    'duplicate_card': outcome,
                    'public_cards_in_hist': public_cards_in_hist,
                    'issue': f"Card {outcome} appears in both history and outcomes"
                })
    
    if failures:
        print("\n" + "=" * 80)
        print("FAILURES DETECTED:")
        print("=" * 80)
        for i, failure in enumerate(failures, 1):
            print(f"\nFailure {i}:")
            print(f"  Issue: {failure['issue']}")
            print(f"  History: {failure['history'][:10]}...")
            print(f"  Public cards in history: {failure['public_cards_in_hist']}")
        
        pytest.fail(f"Found {len(failures)} cases where cards appear in both history and outcomes")
    
    print(f"\n✓ No duplicate cards found in {len(chance_nodes)} chance nodes")
    print("=" * 80)


if __name__ == "__main__":
    # Run all tests
    test_rhode_island_deck_reduction()
    test_twelve_card_poker_deck_reduction()
    test_chance_node_consistency()
    test_no_duplicate_cards_in_history()
    print("\n" + "=" * 80)
    print("ALL TESTS PASSED!")
    print("=" * 80)
