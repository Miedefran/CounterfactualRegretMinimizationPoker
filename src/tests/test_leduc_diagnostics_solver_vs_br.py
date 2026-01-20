import os
import sys
import math

import pytest

# Ensure src is on path when running via pytest from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.leduc_holdem.game import LeducHoldemGame
from utils.poker_utils import GAME_CONFIGS, LeducHoldemCombinations, LeducHoldemCombinationsAbstracted
from training.build_game_tree import build_game_tree
from evaluation.build_public_state_tree_v2 import build_public_state_tree
from evaluation.best_response_agent_v2 import compute_best_response_value


def _build_leduc_full_tree(abstract_suits: bool):
    cfg = dict(GAME_CONFIGS["leduc"])
    if abstract_suits:
        game = LeducHoldemGame(**cfg, abstract_suits=True)
        combo = LeducHoldemCombinationsAbstracted()
    else:
        game = LeducHoldemGame(**cfg, abstract_suits=False)
        combo = LeducHoldemCombinations()
    # build_game_tree ignores combo now, but keep signature stable
    tree = build_game_tree(game, combo, game_name="leduc", abstract_suits=abstract_suits)
    return tree


def _uniform_avg_strategy_from_full_tree(full_tree):
    """
    Create a complete avg_strategy mapping for BOTH players:
    - for every infoset_key in the full (private+public chance) game tree,
      assign uniform probs over the legal_actions of that infoset.
    """
    avg = {}
    for infoset_key, node_ids in full_tree.infoset_to_nodes.items():
        if not node_ids:
            continue
        node = full_tree.nodes[node_ids[0]]
        actions = list(node.legal_actions)
        assert actions, f"Empty legal_actions for infoset {infoset_key}"
        p = 1.0 / len(actions)
        avg[infoset_key] = {a: p for a in actions}
    return avg


def _eval_profile_value_full_tree(full_tree, avg_strategy, player_id: int):
    """
    Exact expected value of a *fixed strategy profile* on the full (private) game tree.
    This is independent from PST/BR and should be considered ground truth.
    """
    nodes = full_tree.nodes

    def v(node_id: int) -> float:
        node = nodes[node_id]
        if node.type == "terminal":
            return float(node.payoffs[player_id])
        if node.type == "chance":
            probs = node.chance_probs or {}
            s = 0.0
            for outcome in node.legal_actions:
                p = float(probs.get(outcome, 0.0))
                if p == 0.0:
                    continue
                s += p * v(node.children[outcome])
            return s
        # decision node
        pid = int(node.player)
        strat = avg_strategy.get(node.infoset_key)
        if strat is None:
            raise KeyError(f"Missing strategy for infoset {node.infoset_key}")
        s = 0.0
        for a in node.legal_actions:
            p = float(strat.get(a, 0.0))
            if p == 0.0:
                continue
            s += p * v(node.children[a])
        return s

    return v(full_tree.root_nodes[0])


def _best_response_value_full_tree(full_tree, avg_strategy, br_player: int) -> float:
    """
    Exact best response on the *full game tree* (private+public chance), enforcing
    the information-set constraint by aggregating counterfactual values over all
    nodes in an infoset.

    This is the diagnosis baseline for whether PST/BR is correct.
    """
    nodes = full_tree.nodes
    infoset_to_nodes = full_tree.infoset_to_nodes

    # Counterfactual weights w(h) = pi_{-i}(h) * pi_c(h) (exclude BR player's actions)
    cf_weight = {}

    def build_cf_weights(node_id: int, w: float):
        cf_weight[node_id] = w
        node = nodes[node_id]
        if node.type == "terminal":
            return
        if node.type == "chance":
            probs = node.chance_probs or {}
            for outcome in node.legal_actions:
                p = float(probs.get(outcome, 0.0))
                if p == 0.0:
                    continue
                build_cf_weights(node.children[outcome], w * p)
            return

        # decision
        pid = int(node.player)
        if pid == br_player:
            # exclude BR player's action probabilities
            for a in node.legal_actions:
                build_cf_weights(node.children[a], w)
            return

        strat = avg_strategy.get(node.infoset_key)
        if strat is None:
            raise KeyError(f"Missing strategy for infoset {node.infoset_key}")
        for a in node.legal_actions:
            p = float(strat.get(a, 0.0))
            if p == 0.0:
                continue
            build_cf_weights(node.children[a], w * p)

    build_cf_weights(full_tree.root_nodes[0], 1.0)

    # Cache of node values under BR (after BR actions decided)
    node_cache = {}
    # Selected action per infoset for BR player
    br_action = {}

    def v(node_id: int) -> float:
        if node_id in node_cache:
            return node_cache[node_id]
        node = nodes[node_id]
        if node.type == "terminal":
            out = float(node.payoffs[br_player])
            node_cache[node_id] = out
            return out
        if node.type == "chance":
            probs = node.chance_probs or {}
            out = 0.0
            for outcome in node.legal_actions:
                p = float(probs.get(outcome, 0.0))
                if p == 0.0:
                    continue
                out += p * v(node.children[outcome])
            node_cache[node_id] = out
            return out

        pid = int(node.player)
        if pid != br_player:
            strat = avg_strategy.get(node.infoset_key)
            if strat is None:
                raise KeyError(f"Missing strategy for infoset {node.infoset_key}")
            out = 0.0
            for a in node.legal_actions:
                p = float(strat.get(a, 0.0))
                if p == 0.0:
                    continue
                out += p * v(node.children[a])
            node_cache[node_id] = out
            return out

        # BR player's infoset decision
        I = node.infoset_key
        if I not in br_action:
            # Aggregate counterfactual Q-values over all nodes in this infoset.
            node_ids = infoset_to_nodes.get(I, [])
            if not node_ids:
                raise ValueError(f"Infoset has no nodes: {I}")
            # Use weights excluding BR player's actions.
            # Maximize sum_w( node ) * V(child(node,a))  (normalization cancels).
            actions = list(node.legal_actions)
            best_a = None
            best_q = -float("inf")
            for a in actions:
                q = 0.0
                for nid in node_ids:
                    w = float(cf_weight.get(nid, 0.0))
                    if w == 0.0:
                        continue
                    child = nodes[nid].children[a]
                    q += w * v(child)
                if q > best_q:
                    best_q = q
                    best_a = a
            if best_a is None:
                best_a = actions[0]
            br_action[I] = best_a

        out = v(node.children[br_action[I]])
        node_cache[node_id] = out
        return out

    return v(full_tree.root_nodes[0])


def _build_leduc_public_tree(abstract_suits: bool):
    cfg = dict(GAME_CONFIGS["leduc"])
    # build_public_state_tree will pass abstract_suits into game constructor if supported
    return build_public_state_tree(LeducHoldemGame, cfg, use_cache=False, abstract_suits=abstract_suits)


def _dump_leduc_pst_chance_nodes(pst_tree, head: int = 10):
    """
    Helper for diagnosis: prints a compact summary of PST chance nodes.
    This is used to pinpoint *where* PST diverges from the full-tree model.
    """
    states = pst_tree["public_states"]
    chance = [(k, v) for k, v in states.items() if v.get("type") == "chance"]
    chance.sort(key=lambda kv: (len(kv[0]), str(kv[0])[:50]))

    print("\n[PST chance nodes dump]")
    print(f"count={len(chance)}")
    for i, (k, node) in enumerate(chance[:head]):
        children = node.get("children", {})
        probs = node.get("chance_probs", {})
        # probs may be absent or partial; compute stats robustly
        prob_sum = sum(float(probs.get(o, 0.0)) for o in children.keys()) if probs else 0.0
        print(
            f"  idx={i} hist_len={len(k)} hist_prefix={list(k)[:8]} "
            f"children={len(children)} prob_keys={len(probs) if probs else 0} prob_sum={prob_sum:.6f}"
        )
        if i == 0:
            # Print first node's children keys (sorted) for immediate insight
            outs = sorted(list(children.keys()), key=lambda x: str(x))
            print(f"    outcomes(sample)={outs[:12]}{' ...' if len(outs) > 12 else ''}")


def test_leduc_engine_public_chance_flow_no_marker():
    """
    Verifiziert, dass Leduc nach dem Refactor:
    - mit Private-Deal als Chance startet,
    - nach Round 0 in einen Public-Chance-Node geht,
    - keine '|' Marker in history schreibt.
    """
    cfg = dict(GAME_CONFIGS["leduc"])
    g = LeducHoldemGame(**cfg, abstract_suits=False)
    g.reset(0)

    assert hasattr(g, "is_chance_node") and g.is_chance_node()
    assert g.current_player == g.CHANCE_PLAYER
    assert "|" not in g.history

    # Deal 2 private cards (2 chance actions) -> then decision node
    a0 = g.get_legal_actions()[0]
    g.step(a0)
    assert g.is_chance_node()
    a1 = g.get_legal_actions()[0]
    g.step(a1)
    assert not g.is_chance_node()
    assert g.current_player == 0

    # Round 0 ends with check-check -> should enter chance (public)
    g.step("check")
    g.step("check")
    assert g.is_chance_node()
    assert g.betting_round == 0  # advanced only after chance resolves
    assert g.public_card is None
    assert "|" not in g.history

    # Resolve public chance -> enters round 1 (betting_round=1)
    public = g.get_legal_actions()[0]
    g.step(public)
    assert not g.is_chance_node()
    assert g.betting_round == 1
    assert g.public_card is not None


def test_leduc_best_response_pst_matches_full_tree_uniform_strategy():
    """
    Diagnose-Test:
    - berechnet Best-Response auf dem FULL game tree (Ground Truth)
    - berechnet Best-Response über PST + best_response_agent_v2
    Erwartung: Beide müssen (nahezu) identisch sein.

    Wenn dieser Test FAILT, ist sehr wahrscheinlich PST/BR (oder chance_probs Modellierung)
    der Fehler – nicht der Solver.
    """
    abstract_suits = False
    full_tree = _build_leduc_full_tree(abstract_suits=abstract_suits)
    avg_strategy = _uniform_avg_strategy_from_full_tree(full_tree)

    # Ground-truth BR values on full tree
    br0_full = _best_response_value_full_tree(full_tree, avg_strategy, br_player=0)
    br1_full = _best_response_value_full_tree(full_tree, avg_strategy, br_player=1)

    # PST-based BR values
    pst = _build_leduc_public_tree(abstract_suits=abstract_suits)
    br0_pst = compute_best_response_value("leduc", 0, pst, avg_strategy, root_hist=())
    br1_pst = compute_best_response_value("leduc", 1, pst, avg_strategy, root_hist=())

    # Print for debugging in pytest output
    _dump_leduc_pst_chance_nodes(pst, head=10)
    print(f"\nBR(full)  p0={br0_full:.6f} p1={br1_full:.6f}")
    print(f"BR(pst)   p0={br0_pst:.6f} p1={br1_pst:.6f}")
    print(f"diff p0={abs(br0_full-br0_pst):.6g} p1={abs(br1_full-br1_pst):.6g}")

    assert math.isfinite(br0_full) and math.isfinite(br0_pst)
    assert math.isfinite(br1_full) and math.isfinite(br1_pst)

    # Tight tolerance: both computations are exact enumerations (no MC noise).
    assert abs(br0_full - br0_pst) < 1e-6
    assert abs(br1_full - br1_pst) < 1e-6


def test_leduc_pst_public_chance_outcomes_shape_debug():
    """
    Pinpoint-test: checks the shape of PST chance nodes for Leduc (NOT abstracted).

    For Leduc with explicit chance in the *full* game tree:
    - the public-card chance happens after 2 private cards were dealt
    - so (conditioned on a fixed private deal) the deck has 4 remaining physical cards

    In a PUBLIC tree, you typically either:
    A) enumerate the FULL deck outcomes and then mask impossible private/public collisions via reach updates, or
    B) carry private-belief-dependent chance probabilities.

    This test prints the first few chance nodes and asserts basic sanity:
    - there must be at least one chance node
    - every chance node must have children and a non-empty chance_probs map
    - the probability sum over its children must be close to 1.0

    If this fails, it strongly indicates the BR/PST chance modeling is inconsistent with the full game.
    """
    pst = _build_leduc_public_tree(abstract_suits=False)
    _dump_leduc_pst_chance_nodes(pst, head=10)

    chance_nodes = [(k, n) for k, n in pst["public_states"].items() if n.get("type") == "chance"]
    assert chance_nodes, "Expected at least one PST chance node for Leduc"

    failures = []
    for k, node in chance_nodes:
        children = node.get("children", {})
        probs = node.get("chance_probs", {})
        if not children:
            failures.append((k, "no children"))
            continue
        if not probs:
            failures.append((k, "no chance_probs"))
            continue
        prob_sum = sum(float(probs.get(o, 0.0)) for o in children.keys())
        # For a well-defined chance node distribution in PST, prob_sum should be ~1.
        # If it is not, we know the chance modeling is off.
        if abs(prob_sum - 1.0) > 1e-6:
            failures.append((k, f"prob_sum={prob_sum}"))

    if failures:
        msg = "\n".join([f"- hist={list(k)[:10]}... issue={issue}" for k, issue in failures[:10]])
        pytest.fail(f"PST chance node probability sums are inconsistent (showing up to 10):\n{msg}")


def test_leduc_br_pst_matches_full_tree_if_override_chance_prob_1_over_4():
    """
    Isolation test to pinpoint the source of the mismatch:
    We override PST chance probabilities to the *conditional* public-deal probability for Leduc:
      p(outcome | private_deal) = 1 / (6 - 2) = 1/4
    for all 6 physical outcomes, relying on the existing collision-masking in the BR code
    (skip our_card == outcome; set opponent reach 0 if opp_card == outcome) so that the
    *effective* mass per private deal is 4*(1/4)=1.

    If this makes BR(pst) match BR(full_tree), then the mismatch is caused by the PST/BR
    chance-probability modeling (currently using 1/6) rather than solver logic.
    """
    abstract_suits = False
    full_tree = _build_leduc_full_tree(abstract_suits=abstract_suits)
    avg_strategy = _uniform_avg_strategy_from_full_tree(full_tree)

    br0_full = _best_response_value_full_tree(full_tree, avg_strategy, br_player=0)
    br1_full = _best_response_value_full_tree(full_tree, avg_strategy, br_player=1)

    pst = _build_leduc_public_tree(abstract_suits=abstract_suits)
    # override chance_probs in-place
    for k, node in pst["public_states"].items():
        if node.get("type") != "chance":
            continue
        children = node.get("children", {})
        node["chance_probs"] = {o: 0.25 for o in children.keys()}

    br0_pst = compute_best_response_value("leduc", 0, pst, avg_strategy, root_hist=())
    br1_pst = compute_best_response_value("leduc", 1, pst, avg_strategy, root_hist=())

    print(f"\nBR(full)      p0={br0_full:.6f} p1={br1_full:.6f}")
    print(f"BR(pst,1/4)   p0={br0_pst:.6f} p1={br1_pst:.6f}")
    print(f"diff p0={abs(br0_full-br0_pst):.6g} p1={abs(br1_full-br1_pst):.6g}")

    assert abs(br0_full - br0_pst) < 1e-6
    assert abs(br1_full - br1_pst) < 1e-6

