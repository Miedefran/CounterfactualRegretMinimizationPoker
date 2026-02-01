import gzip
import pickle
import argparse
import sys
import os
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TIME_STATS = {
    'compute_payoff': 0.0,
    'terminal': 0.0,
    'chance': 0.0,
    'our_choice': 0.0,
    'opponent_choice': 0.0,
    'traverse_routing': 0.0,
}


def load_public_tree(path):
    with gzip.open(path, 'rb') as f:
        return pickle.load(f)


def load_average_strategy(path):
    with gzip.open(path, 'rb') as f:
        data = pickle.load(f)
    if isinstance(data, dict) and 'average_strategy' in data:
        return data['average_strategy']
    return data


def get_action_probability(info_set_key, action, avg_strategy, legal_actions):
    """
    Return π(a|I) from avg_strategy for a specific infoset/action.
    If infoset is missing, uniform is chosen (default fallback: uniform distribution
    over legal actions). Defensively normalized before return.
    """
    strategy = avg_strategy.get(info_set_key)
    if not strategy:
        # Uniform if infoset not in avg_strategy
        if not legal_actions:
            return 0.0
        return 1.0 / len(legal_actions)

    total = 0.0
    for a in legal_actions:
        try:
            total += max(float(strategy.get(a, 0.0)), 0.0)
        except Exception:
            # Non-numeric -> treat as 0
            total += 0.0

    if total <= 0.0:
        # If sum is 0: uniform, so BR remains well-defined
        if not legal_actions:
            return 0.0
        return 1.0 / len(legal_actions)

    return max(float(strategy.get(action, 0.0)), 0.0) / total


# Global cache for judgers
JUDGER_CACHE = {}

# Global cache for hand strength evaluations
HAND_STRENGTH_CACHE = {}


class PlayerDummy:
    pass


def get_judger(game_name):
    if game_name in JUDGER_CACHE:
        return JUDGER_CACHE[game_name]

    judger = None
    if 'kuhn' in game_name.lower():
        from envs.kuhn_poker.judger import KuhnPokerJudger
        judger = KuhnPokerJudger()
    elif 'leduc' in game_name.lower():
        from envs.leduc_holdem.judger import LeducHoldemJudger
        judger = LeducHoldemJudger()
    elif 'twelve' in game_name.lower():
        from envs.twelve_card_poker.judger import TwelveCardPokerJudger
        judger = TwelveCardPokerJudger()
    elif 'rhode' in game_name.lower():
        from envs.rhode_island.judger import RhodeIslandJudger
        judger = RhodeIslandJudger()
    elif 'royal' in game_name.lower():
        from envs.royal_holdem.judger import RoyalHoldemJudger
        judger = RoyalHoldemJudger()
    elif 'small_island' in game_name.lower():
        from envs.small_island_holdem.judger import SmallIslandHoldemJudger
        judger = SmallIslandHoldemJudger()

    if judger:
        JUDGER_CACHE[game_name] = judger
    return judger


def compute_payoff(game_name, our_info, opp_info, pot, player_bets, player_id, node):
    t_start = time.time()

    # New Key Format: (private_card, public_cards_tuple, history_tuple, pid)

    # Map p0/p1 from our_info, opp_info
    if our_info[-1] == 0:
        p0_info, p1_info = our_info, opp_info
    else:
        p0_info, p1_info = opp_info, our_info

    # Unpack keys
    p0_card, p0_public, p0_hist, _ = p0_info
    p1_card, p1_public, p1_hist, _ = p1_info

    # History should be identical for both players in terms of actions
    history = list(p0_hist)

    p0 = PlayerDummy()
    p1 = PlayerDummy()

    judger = get_judger(game_name)
    if not judger:
        return 0.0

    # Set dummy player attributes for judger
    if 'kuhn' in game_name.lower():
        p0.private_card = p0_card
        p1.private_card = p1_card

    elif 'leduc' in game_name.lower():
        p0.private_card = p0_card
        p1.private_card = p1_card

        # Public cards are in the tuple
        if len(p0_public) > 0:
            p0.public_card = p0_public[0]
            p1.public_card = p1_public[0]  # Should be same
        else:
            p0.public_card = None
            p1.public_card = None

        if (p0.public_card is None) and ('fold' in history):
            # Dummy public card to prevent crash if judger checks it (though it shouldn't for fold)
            p0.public_card = 'Js'
            p1.public_card = 'Js'

    elif 'twelve' in game_name.lower():
        p0.private_card = p0_card
        p1.private_card = p1_card
        p0.public_cards = list(p0_public)
        p1.public_cards = list(p1_public)
    elif 'rhode' in game_name.lower():
        p0.private_card = p0_card
        p1.private_card = p1_card
        p0.public_cards = list(p0_public)
        p1.public_cards = list(p1_public)
    elif 'royal' in game_name.lower():
        p0.private_card = p0_card
        p1.private_card = p1_card
        p0.public_cards = list(p0_public)
        p1.public_cards = list(p1_public)
    elif 'small_island' in game_name.lower():
        p0.private_card = p0_card
        p1.private_card = p1_card
        p0.public_cards = list(p0_public)
        p1.public_cards = list(p1_public)
    else:
        return 0.0

    players = [p0, p1]

    # Use reliable actor from tree node if available
    if 'last_actor' in node:
        current_player = node['last_actor']
    else:
        # print("last actor not found") # Silence print for performance
        current_player = None

    payoffs = judger.judge(players, history, current_player, pot, player_bets)
    result = payoffs[player_id]
    TIME_STATS['compute_payoff'] += time.time() - t_start
    return result


def traverse_public_tree(game_name, player_id, tree, avg_strategy, public_hist, r_opp, our_fixed_card=None):
    t_start = time.time()

    node = tree['public_states'].get(public_hist)
    if not node:
        raise ValueError(f"Public state not found: {public_hist}")

    node_type = node['type']
    our_infosets = node[f'player{player_id}_info_sets']
    opp_infosets = node[f'player{1 - player_id}_info_sets']

    if len(r_opp) != len(opp_infosets):
        raise ValueError(f"Reach probability mismatch: {len(r_opp)} vs {len(opp_infosets)}")

    routing_time = time.time() - t_start
    TIME_STATS['traverse_routing'] += routing_time

    if node_type == 'terminal':
        return terminal_value(game_name, player_id, node, r_opp)

    elif node_type == 'chance':
        return chance_value(game_name, player_id, tree, avg_strategy, node, public_hist, r_opp, our_fixed_card)

    elif node_type == 'choice':
        if node['player'] == player_id:
            return our_choice_value(game_name, player_id, tree, avg_strategy, node, public_hist, r_opp, our_fixed_card)
        else:
            return opponent_choice_value(game_name, player_id, tree, avg_strategy, node, public_hist, r_opp,
                                         our_fixed_card)

    raise ValueError(f"Unknown node type: {node_type}")


def get_hand_strength_for_info_set(game_name, info_set):
    cache_key = (game_name, info_set)
    if cache_key in HAND_STRENGTH_CACHE:
        return HAND_STRENGTH_CACHE[cache_key]

    judger = get_judger(game_name)
    if not judger:
        HAND_STRENGTH_CACHE[cache_key] = None
        return None

    our_card = info_set[0]
    our_public = info_set[1]

    p = PlayerDummy()

    result = None
    if 'kuhn' in game_name.lower():
        p.private_card = our_card
        if hasattr(judger, 'hand_rank'):
            result = (judger.hand_rank.get(our_card, 0),)
        else:
            result = (0,)
    elif 'leduc' in game_name.lower():
        p.private_card = our_card
        if len(our_public) > 0:
            p.public_card = our_public[0]
        else:
            p.public_card = None
        if hasattr(judger, 'evaluate_hand'):
            result = judger.evaluate_hand(p)
    elif (
            'twelve' in game_name.lower()
            or 'rhode' in game_name.lower()
            or 'royal' in game_name.lower()
            or 'small_island' in game_name.lower()
    ):
        p.private_card = our_card
        p.public_cards = list(our_public)
        if hasattr(judger, 'evaluate_hand'):
            result = judger.evaluate_hand(p)

    HAND_STRENGTH_CACHE[cache_key] = result
    return result


def terminal_value(game_name, player_id, node, r_opp):
    t_start = time.time()

    our_infosets = node[f'player{player_id}_info_sets']
    opp_infosets = node[f'player{1 - player_id}_info_sets']
    pot = node['pot']
    player_bets = node['player_bets']

    if len(our_infosets) == 0 or len(opp_infosets) == 0:
        TIME_STATS['terminal'] += time.time() - t_start
        return [0.0] * len(our_infosets)

    sample_our_info = our_infosets[0]
    sample_hist = sample_our_info[2]
    is_fold = len(sample_hist) > 0 and sample_hist[-1] == 'fold'

    if is_fold:
        last_actor = node.get('last_actor', None)
        if last_actor is not None:
            # Kuhn: After fold, current_player is not flipped → last_actor = 1 - folder = winner.
            # Leduc etc.: proceed_round flips → current_player = winner → last_actor = folder.
            if 'kuhn' in game_name.lower():
                winner = last_actor
            else:
                winner = 1 - last_actor
            if winner == player_id:
                win_util = pot - player_bets[player_id]
            else:
                win_util = -player_bets[player_id]

            result = []
            for our_info in our_infosets:
                our_card = our_info[0]
                value = 0.0
                for j, opp_info in enumerate(opp_infosets):
                    if r_opp[j] == 0:
                        continue
                    opp_card = opp_info[0]
                    if our_card == opp_card:
                        continue
                    value += r_opp[j] * win_util
                result.append(value)

            TIME_STATS['terminal'] += time.time() - t_start
            return result

    our_with_strength = []
    for i, our_info in enumerate(our_infosets):
        strength = get_hand_strength_for_info_set(game_name, our_info)
        our_with_strength.append((i, our_info, strength))

    opp_with_strength = []
    for j, opp_info in enumerate(opp_infosets):
        if r_opp[j] == 0:
            continue
        strength = get_hand_strength_for_info_set(game_name, opp_info)
        opp_with_strength.append((j, opp_info, strength, r_opp[j]))

    if len(opp_with_strength) == 0:
        TIME_STATS['terminal'] += time.time() - t_start
        return [0.0] * len(our_infosets)

    our_with_strength.sort(key=lambda x: x[2] if x[2] is not None else (float('-inf'),))
    opp_with_strength.sort(key=lambda x: x[2] if x[2] is not None else (float('-inf'),))

    win_util = pot - player_bets[player_id]
    lose_util = -player_bets[player_id]
    tie_util = 0.0

    result = [0.0] * len(our_infosets)

    for our_idx, our_info, our_strength in our_with_strength:
        our_card = our_info[0]

        if our_strength is None:
            for j, opp_info, opp_strength, r_val in opp_with_strength:
                opp_card = opp_info[0]
                if our_card == opp_card:
                    continue
                util = compute_payoff(game_name, our_info, opp_info, pot, player_bets, player_id, node)
                result[our_idx] += r_val * util
            continue

        worse_prob = 0.0
        equal_prob = 0.0
        better_prob = 0.0

        for j, opp_info, opp_strength, r_val in opp_with_strength:
            opp_card = opp_info[0]
            if our_card == opp_card:
                continue

            if opp_strength is None:
                util = compute_payoff(game_name, our_info, opp_info, pot, player_bets, player_id, node)
                result[our_idx] += r_val * util
                continue

            if opp_strength < our_strength:
                worse_prob += r_val
            elif opp_strength == our_strength:
                equal_prob += r_val
            else:
                better_prob += r_val

        result[our_idx] = worse_prob * win_util + equal_prob * tie_util + better_prob * lose_util

    TIME_STATS['terminal'] += time.time() - t_start
    return result


def chance_value(game_name, player_id, tree, avg_strategy, node, public_hist, r_opp, our_fixed_card=None):
    t_start = time.time()

    our_infosets = node[f'player{player_id}_info_sets']
    opp_infosets = node[f'player{1 - player_id}_info_sets']
    children = node['children']

    result = [0.0] * len(our_infosets)
    num_chance_outcomes = len(children)

    # Public tree stores unconditional distribution; for BR condition on both private cards. Pass r_opp unnormalized.
    base_probs = node.get('chance_probs', {}) or {}
    if not base_probs:
        # No base_probs: assume outcomes uniformly distributed
        base_probs = {o: 1.0 / num_chance_outcomes for o in children.keys()} if num_chance_outcomes > 0 else {}

    # our_fixed_card is required for correct conditioning. In the current BR setup we evaluate one
    # our private card at a time in `compute_best_response_value`, so we expect it to be passed.
    if our_fixed_card is None and our_infosets:
        # Emergency: first private card from our_infosets, if our_fixed_card is missing
        our_fixed_card = our_infosets[0][0]

    parent_card_to_r_opp = {info[0]: r for info, r in zip(opp_infosets, r_opp)}

    tasks = []
    for outcome, child_hist in children.items():
        child_node = tree['public_states'].get(child_hist)
        if not child_node:
            continue

        child_opp_infosets = child_node[f'player{1 - player_id}_info_sets']
        child_our_infosets = child_node[f'player{player_id}_info_sets']

        r_child = []
        has_mass = False
        for info in child_opp_infosets:
            child_card = info[0]
            # Physical card collision: opponent cannot hold the revealed public card.
            if child_card == outcome:
                r_child.append(0.0)
                continue

            parent_r = parent_card_to_r_opp.get(child_card, 0.0)
            if parent_r == 0.0:
                r_child.append(0.0)
                continue

            # Conditional public chance probability given BOTH private cards.
            # If the public outcome equals one of the private cards, it is impossible.
            if outcome == our_fixed_card or outcome == child_card:
                r_child.append(0.0)
                continue
            p_out = float(base_probs.get(outcome, 0.0))
            if p_out <= 0.0:
                r_child.append(0.0)
                continue
            p_our = float(base_probs.get(our_fixed_card, 0.0))
            p_opp = float(base_probs.get(child_card, 0.0))
            denom = 1.0 - p_our - p_opp
            if denom <= 1e-15:
                r_child.append(0.0)
                continue
            p_cond = p_out / denom
            val = parent_r * p_cond
            r_child.append(val)
            if val > 0: has_mass = True

        if has_mass:
            tasks.append((outcome, child_hist, child_our_infosets, r_child))

    t_setup = time.time() - t_start
    TIME_STATS['chance'] += t_setup

    if not tasks:
        return result

    for outcome, child_hist, child_our_infosets, r_child in tasks:
        v_child = traverse_public_tree(
            game_name, player_id, tree, avg_strategy, child_hist, r_child, our_fixed_card=our_fixed_card
        )

        t_merge_start = time.time()
        card_to_child_value = {info[0]: val for info, val in zip(child_our_infosets, v_child)}

        for i, our_info in enumerate(our_infosets):
            our_card = our_info[0]
            if our_card == outcome:
                continue

            if our_card in card_to_child_value:
                result[i] += card_to_child_value[our_card]
        TIME_STATS['chance'] += time.time() - t_merge_start

    return result


def our_choice_value(game_name, player_id, tree, avg_strategy, node, public_hist, r_opp, our_fixed_card=None):
    t_start = time.time()

    our_infosets = node[f'player{player_id}_info_sets']
    children = node['children']

    if not children:
        return [0.0] * len(our_infosets)

    tasks = []
    for action, child_hist in children.items():
        child_node = tree['public_states'].get(child_hist)
        if child_node:
            child_our_infosets = child_node[f'player{player_id}_info_sets']
            tasks.append((action, child_hist, child_our_infosets))

    t_setup = time.time() - t_start
    TIME_STATS['our_choice'] += t_setup

    if not tasks:
        return [0.0] * len(our_infosets)

    action_values = {}

    for action, child_hist, child_our_infosets in tasks:
        v_child = traverse_public_tree(
            game_name, player_id, tree, avg_strategy, child_hist, r_opp, our_fixed_card=our_fixed_card
        )
        action_values[action] = (v_child, child_our_infosets)

    if not action_values:
        return [0.0] * len(our_infosets)

    t_max_start = time.time()
    result = []
    for our_info in our_infosets:
        our_card = our_info[0]
        max_val = float('-inf')

        for action in children.keys():
            if action in action_values:
                v_child, child_our_infosets = action_values[action]
                for j, child_info in enumerate(child_our_infosets):
                    if child_info[0] == our_card:
                        if len(v_child) > j:
                            if v_child[j] > max_val:
                                max_val = v_child[j]
                        break

        if max_val == float('-inf'):
            max_val = 0.0
        result.append(max_val)

    TIME_STATS['our_choice'] += time.time() - t_max_start
    return result


def opponent_choice_value(game_name, player_id, tree, avg_strategy, node, public_hist, r_opp, our_fixed_card=None):
    t_start = time.time()

    our_infosets = node[f'player{player_id}_info_sets']
    opp_infosets = node[f'player{1 - player_id}_info_sets']
    children = node['children']

    if not children:
        return [0.0] * len(our_infosets)

    legal_actions = list(children.keys())

    action_prob_cache = {}
    for action in legal_actions:
        action_prob_cache[action] = []
        for k, opp_info in enumerate(opp_infosets):
            prob = get_action_probability(opp_info, action, avg_strategy, legal_actions)
            action_prob_cache[action].append(prob)

    t_setup = time.time() - t_start
    TIME_STATS['opponent_choice'] += t_setup

    result = None

    for action in legal_actions:
        child_hist = children[action]

        r_child = []
        probs = action_prob_cache[action]
        for k, r_val in enumerate(r_opp):
            r_child.append(r_val * probs[k])

        v_child = traverse_public_tree(
            game_name, player_id, tree, avg_strategy, child_hist, r_child, our_fixed_card=our_fixed_card
        )

        t_merge_start = time.time()
        child_node = tree['public_states'].get(child_hist)
        if child_node:
            child_our_infosets = child_node[f'player{player_id}_info_sets']
            card_to_child_value = {child_info[0]: v_child[i] for i, child_info in enumerate(child_our_infosets)}

            if result is None:
                result = []
                for our_info in our_infosets:
                    our_card = our_info[0]
                    if our_card in card_to_child_value:
                        result.append(card_to_child_value[our_card])
                    else:
                        result.append(0.0)
            else:
                for i, our_info in enumerate(our_infosets):
                    our_card = our_info[0]
                    if our_card in card_to_child_value:
                        result[i] += card_to_child_value[our_card]
        TIME_STATS['opponent_choice'] += time.time() - t_merge_start

    if result is None:
        return [0.0] * len(our_infosets)

    return result


def compute_best_response_value(game_name, player_id, tree, avg_strategy, root_hist=()):
    global TIME_STATS, HAND_STRENGTH_CACHE
    TIME_STATS = {
        'compute_payoff': 0.0,
        'terminal': 0.0,
        'chance': 0.0,
        'our_choice': 0.0,
        'opponent_choice': 0.0,
        'traverse_routing': 0.0,
    }
    HAND_STRENGTH_CACHE.clear()

    t_total_start = time.time()

    root_node = tree['public_states'].get(root_hist)
    if not root_node:
        raise ValueError(f"Root public state not found: {root_hist}")

    our_infosets = root_node[f'player{player_id}_info_sets']
    opp_infosets = root_node[f'player{1 - player_id}_info_sets']

    # Detect suit-abstraction (rank-only cards) from infoset format.
    # In Leduc/Twelve abstracted runs, private cards are ranks like 'J'/'Q'/'K'/'A'.
    def _is_rank_only(card) -> bool:
        return isinstance(card, str) and len(card) == 1

    is_suit_abstracted = False
    if our_infosets:
        sample_card = our_infosets[0][0]
        is_suit_abstracted = _is_rank_only(sample_card)

    if is_suit_abstracted and ('leduc' in game_name.lower() or 'twelve' in game_name.lower()):
        return _compute_best_response_value_suit_abstracted(game_name, player_id, tree, avg_strategy,
                                                            root_hist=root_hist)

    total_value = 0.0

    for i, our_info in enumerate(our_infosets):
        our_card = our_info[0]

        valid_opp_indices = [j for j, opp_info in enumerate(opp_infosets) if opp_info[0] != our_card]
        num_valid = len(valid_opp_indices)

        if num_valid == 0:
            continue

        r_opp_conditional = [0.0] * len(opp_infosets)
        for j in valid_opp_indices:
            r_opp_conditional[j] = 1.0 / num_valid

        v_vector = traverse_public_tree(
            game_name, player_id, tree, avg_strategy, root_hist, r_opp_conditional, our_fixed_card=our_card
        )
        total_value += (1.0 / len(our_infosets)) * v_vector[i]

    t_total = time.time() - t_total_start

    total_measured = sum(TIME_STATS.values())

    print(f"\n[BR Time Distribution Player {player_id}] Total: {t_total:.2f}s")
    if t_total > 0:
        print(f"  Terminal: {TIME_STATS['terminal']:.2f}s ({TIME_STATS['terminal'] / t_total * 100:.1f}%)")
        print(
            f"  Compute Payoff: {TIME_STATS['compute_payoff']:.2f}s ({TIME_STATS['compute_payoff'] / t_total * 100:.1f}%)")
        print(f"  Chance: {TIME_STATS['chance']:.2f}s ({TIME_STATS['chance'] / t_total * 100:.1f}%)")
        print(f"  Our Choice: {TIME_STATS['our_choice']:.2f}s ({TIME_STATS['our_choice'] / t_total * 100:.1f}%)")
        print(
            f"  Opponent Choice: {TIME_STATS['opponent_choice']:.2f}s ({TIME_STATS['opponent_choice'] / t_total * 100:.1f}%)")
        print(
            f"  Traverse Routing: {TIME_STATS['traverse_routing']:.2f}s ({TIME_STATS['traverse_routing'] / t_total * 100:.1f}%)")
        print(f"  Sum measured: {total_measured:.2f}s ({total_measured / t_total * 100:.1f}%)")

    return total_value


def _deck_counts_for_game(game_name: str):
    g = game_name.lower()
    if 'leduc' in g:
        return {'J': 2, 'Q': 2, 'K': 2}
    if 'twelve' in g:
        return {'J': 3, 'Q': 3, 'K': 3, 'A': 3}
    return None


def _is_feasible_assignment_counts(counts: dict, our_card: str, opp_card: str, public_cards: tuple) -> bool:
    if counts is None:
        return True
    need = {}
    if our_card is not None:
        need[our_card] = need.get(our_card, 0) + 1
    if opp_card is not None:
        need[opp_card] = need.get(opp_card, 0) + 1
    for c in public_cards or ():
        need[c] = need.get(c, 0) + 1
    for k, v in need.items():
        if v > int(counts.get(k, 0)):
            return False
    return True


def _chance_prob_given_private(counts: dict, outcome: str, our_card: str, opp_card: str, public_cards: tuple) -> float:
    """
    Conditional chance probability for drawing `outcome` given private cards and already dealt public cards,
    under a deck with multiplicities described by `counts`.
    """
    if counts is None:
        return 0.0
    used = {}
    used[our_card] = used.get(our_card, 0) + 1
    used[opp_card] = used.get(opp_card, 0) + 1
    for c in public_cards or ():
        used[c] = used.get(c, 0) + 1
    remaining_total = 0
    remaining_outcome = 0
    for r, cnt in counts.items():
        rem = int(cnt) - int(used.get(r, 0))
        if rem < 0:
            rem = 0
        remaining_total += rem
        if r == outcome:
            remaining_outcome = rem
    if remaining_total <= 0:
        return 0.0
    return float(remaining_outcome) / float(remaining_total)


def _compute_best_response_value_suit_abstracted(game_name, player_id, tree, avg_strategy, root_hist=()):
    """
    BR evaluation for suit-abstracted variants with multiplicities (e.g. Leduc has 2 copies per rank).
    We cannot reuse the vectorized BR logic because chance probabilities depend on BOTH private cards.
    Instead we evaluate one our private card at a time (scalar recursion), which is correct.
    """
    root_node = tree['public_states'].get(root_hist)
    if not root_node:
        raise ValueError(f"Root public state not found: {root_hist}")

    counts = _deck_counts_for_game(game_name)
    if counts is None:
        raise ValueError(f"No abstracted deck counts configured for game {game_name}")

    our_infosets = root_node[f'player{player_id}_info_sets']
    opp_infosets = root_node[f'player{1 - player_id}_info_sets']

    # Prior over our private ranks at root: proportional to multiplicity.
    total_cards = float(sum(counts.values()))
    card_priors = {}
    for info in our_infosets:
        c = info[0]
        card_priors[c] = float(counts.get(c, 0)) / total_cards if total_cards > 0 else 0.0

    # Map opponent infosets by private card
    opp_infos_by_card = defaultdict(list)
    for info in opp_infosets:
        opp_infos_by_card[info[0]].append(info)

    def traverse_scalar(public_hist, our_card: str, r_opp_by_card: dict):
        node = tree['public_states'].get(public_hist)
        if not node:
            raise ValueError(f"Public state not found: {public_hist}")

        node_type = node['type']
        if node_type == 'terminal':
            # Find our infoset key in this node
            our_infos = node[f'player{player_id}_info_sets']
            our_info = None
            for info in our_infos:
                if info[0] == our_card:
                    our_info = info
                    break
            if our_info is None:
                return 0.0

            opp_infos = node[f'player{1 - player_id}_info_sets']
            pot = node['pot']
            player_bets = node['player_bets']
            total = 0.0
            for opp_info in opp_infos:
                opp_card = opp_info[0]
                w = float(r_opp_by_card.get(opp_card, 0.0))
                if w == 0.0:
                    continue
                public_cards = our_info[1]
                if not _is_feasible_assignment_counts(counts, our_card, opp_card, public_cards):
                    continue
                total += w * compute_payoff(game_name, our_info, opp_info, pot, player_bets, player_id, node)
            return total

        if node_type == 'chance':
            children = node.get('children', {})
            if not children:
                return 0.0
            # Current public cards can be read from any infoset in this node
            sample_infos = node[f'player{player_id}_info_sets']
            current_public = sample_infos[0][1] if sample_infos else ()

            total = 0.0
            for outcome, child_hist in children.items():
                # Update opponent reach by conditional chance prob given private cards.
                r_child = {}
                for opp_card, w in r_opp_by_card.items():
                    if w == 0.0:
                        continue
                    if not _is_feasible_assignment_counts(counts, our_card, opp_card, current_public + (outcome,)):
                        continue
                    p = _chance_prob_given_private(counts, outcome, our_card, opp_card, current_public)
                    if p == 0.0:
                        continue
                    r_child[opp_card] = r_child.get(opp_card, 0.0) + w * p
                if not r_child:
                    continue
                total += traverse_scalar(child_hist, our_card, r_child)
            return total

        if node_type == 'choice':
            children = node.get('children', {})
            if not children:
                return 0.0
            acting = node.get('player')
            if acting == player_id:
                # our choice: maximize over actions
                best = None
                for action, child_hist in children.items():
                    v = traverse_scalar(child_hist, our_card, r_opp_by_card)
                    if best is None or v > best:
                        best = v
                return 0.0 if best is None else best
            else:
                # opponent choice: expected over opponent policy per infoset
                opp_infos = node[f'player{1 - player_id}_info_sets']
                legal_actions = list(children.keys())
                total = 0.0
                for action, child_hist in children.items():
                    r_child = {}
                    for opp_info in opp_infos:
                        opp_card = opp_info[0]
                        w = float(r_opp_by_card.get(opp_card, 0.0))
                        if w == 0.0:
                            continue
                        prob = get_action_probability(opp_info, action, avg_strategy, legal_actions)
                        if prob == 0.0:
                            continue
                        r_child[opp_card] = r_child.get(opp_card, 0.0) + w * prob
                    if not r_child:
                        continue
                    total += traverse_scalar(child_hist, our_card, r_child)
                return total

        raise ValueError(f"Unknown node type: {node_type}")

    total_value = 0.0
    for our_info in our_infosets:
        our_card = our_info[0]
        p_our = float(card_priors.get(our_card, 0.0))
        if p_our == 0.0:
            continue

        # Opponent distribution conditional on our_card at root.
        # Remove our card from counts.
        denom = float(sum(counts.values()) - 1)
        r_opp = {}
        for opp_card in opp_infos_by_card.keys():
            # opponent can be same rank if at least 2 copies exist
            avail = float(counts.get(opp_card, 0))
            if opp_card == our_card:
                avail -= 1.0
            if avail <= 0:
                continue
            r_opp[opp_card] = avail / denom if denom > 0 else 0.0

        total_value += p_our * traverse_scalar(root_hist, our_card, r_opp)

    return total_value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--game', required=True)
    parser.add_argument('--player', type=int, required=True)
    parser.add_argument('--public-tree', required=True)
    parser.add_argument('--strategy', required=True)
    args = parser.parse_args()

    tree = load_public_tree(args.public_tree)
    avg_strategy = load_average_strategy(args.strategy)
    value = compute_best_response_value(args.game, args.player, tree, avg_strategy, root_hist=())
    print(f"best_response_value={value}")


if __name__ == '__main__':
    main()
