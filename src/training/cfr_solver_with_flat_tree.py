"""
CFR / CFR+ Solver auf einem "Flat Tree" (Array-Repräsentation).

Motivation:
- `CFRSolverWithTree` verwendet Python-Objekte + Dicts + Rekursion.
- Dieser Solver flacht den geladenen `GameTree` einmalig in Numpy-Arrays ab und
  führt pro Update zwei DP-Pässe (Forward Reach, Backward Values/Regrets) aus.

Damit bleibt die Semantik nahe an den bestehenden Tree-Solvern, aber der Python-Overhead sinkt stark.

Operational (wie der Code "läuft"):
- Tree-Build passiert einmal direkt aus der Game-Environment-API in Arrays (oder wird aus Cache geladen).
- Pro Iteration:
  1) Current strategy sigma[infoset, action] aus regrets (Regret Matching).
  2) Forward-Pass: reach[node, player] propagieren (Chance multipliziert beide, Decision nur aktiver Spieler).
  3) Backward-Pass: values[node, player] bottom-up berechnen und Regrets/StrategySum updaten.
"""

from __future__ import annotations

import gzip
import os
import pickle as pkl
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np

from utils.data_models import KeyGenerator
from utils.tree_registry import record_tree_stats
from training.build_game_tree import GameTree


ACTION_ORDER = ("check", "bet", "call", "fold")
ACTION_TO_IDX = {a: i for i, a in enumerate(ACTION_ORDER)}
NUM_ACTIONS = len(ACTION_ORDER)

# Flat-tree cache versioning:
# - v1: float64 arrays (payoffs/chance_probs/strategy stuff), no precomputed layer lists
# - v2: float32 arrays, still reconstructs layer lists on load (so format stays stable)
FLAT_TREE_CACHE_VERSION = 2

# Performance choice:
# - float64 is unnecessary here and costs memory bandwidth (especially in large games like Twelve Card).
# - float32 matches the tensor path and is generally faster on CPU due to cache/bandwidth.
FLOAT_DTYPE = np.float32


def _get_flat_tree_cache_path(game_name: str, abstract_suits: bool) -> str:
    """
    Returns the path where the flat-tree cache is stored.

    We mirror the tensor tree layout style:
      data/trees/game_trees/flat/{abstracted|normal}/{game_name}_flat_tree.npz
    """
    script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    subdir = "abstracted" if abstract_suits else "normal"
    out_dir = os.path.join(script_dir, "data", "trees", "game_trees", "flat", subdir)
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, f"{game_name}_flat_tree.npz")


def _get_flat_tree_keys_path(npz_path: str) -> str:
    return npz_path.replace(".npz", "_keys.pkl.gz")


def _save_flat_tree_cache(ft: "FlatTree", npz_path: str) -> None:
    """
    Save FlatTree in a robust way:
    - `.npz`: only numeric/boolean arrays + small meta
    - `_keys.pkl.gz`: infoset_id_to_key (Python tuples), needed for get_average_strategy()
    """
    # Atomic write: write to temp, then replace.
    tmp_npz_path = npz_path + ".tmp.npz"
    np.savez_compressed(
        tmp_npz_path,
        node_type=ft.node_type,
        player=ft.player,
        infoset_id=ft.infoset_id,
        depth=ft.depth,
        payoffs=ft.payoffs,
        children=ft.children,
        chance_offsets=ft.chance_offsets,
        chance_children=ft.chance_children,
        chance_probs=ft.chance_probs,
        infoset_valid_actions=ft.infoset_valid_actions,
        infoset_player=ft.infoset_player,
        root=np.array([ft.root], dtype=np.int32),
        meta=np.array([FLAT_TREE_CACHE_VERSION], dtype=np.int32),
    )

    keys_path = _get_flat_tree_keys_path(npz_path)
    tmp_keys_path = keys_path + ".tmp"
    with gzip.open(tmp_keys_path, "wb") as f:
        pkl.dump(ft.infoset_id_to_key, f)

    os.replace(tmp_npz_path, npz_path)
    os.replace(tmp_keys_path, keys_path)

    # Registry-Logging (Flat-Tree Save): Größen + Node-Typen (terminal/decision/chance).
    try:
        # Infer game name from filename "<game>_flat_tree.npz"
        base = os.path.basename(npz_path)
        game_name = base.replace("_flat_tree.npz", "")
        counts = np.bincount(ft.node_type.astype(np.int64), minlength=3)
        node_type_counts = {
            "terminal": int(counts[0]),
            "decision": int(counts[1]),
            "chance": int(counts[2]),
        }
        record_tree_stats(
            {
                "schema_version": 1,
                "tree_kind": "flat_tree_npz",
                "game": game_name,
                "abstract_suits": ("abstracted" in npz_path),
                "num_nodes": int(len(ft.node_type)),
                "num_infosets": int(len(ft.infoset_id_to_key)),
                "num_chance_edges": int(len(ft.chance_children)),
                "node_type_counts": node_type_counts,
                "tree_path": npz_path,
            }
        )
    except Exception:
        # Registry ist optional; Cache-Save darf nie scheitern.
        pass


def _load_flat_tree_cache(npz_path: str) -> "FlatTree":
    """
    Loads FlatTree from `.npz` + `_keys.pkl.gz`.

    Reconstructs:
    - layer_indices from `depth`
    - infoset_key_to_id from infoset_id_to_key
    - max_depth from `depth`
    """
    data = np.load(npz_path, allow_pickle=False)

    if "meta" not in data.files:
        raise ValueError("Legacy flat-tree cache detected (missing meta).")
    meta = data["meta"]
    version = int(meta[0]) if len(meta) > 0 else -1
    if version not in (1, 2):
        raise ValueError(f"Unsupported flat-tree cache version: {version}")

    required = {
        "node_type",
        "player",
        "infoset_id",
        "depth",
        "payoffs",
        "children",
        "chance_offsets",
        "chance_children",
        "chance_probs",
        "infoset_valid_actions",
        "infoset_player",
        "root",
    }
    missing = required.difference(set(data.files))
    if missing:
        raise ValueError(f"Flat-tree cache missing arrays: {sorted(list(missing))}")

    keys_path = _get_flat_tree_keys_path(npz_path)
    if not os.path.exists(keys_path):
        raise ValueError(f"Flat-tree cache missing infoset keys file: {keys_path}")
    with gzip.open(keys_path, "rb") as f:
        infoset_id_to_key = pkl.load(f)

    node_type = data["node_type"].astype(np.int8, copy=False)
    depth = data["depth"].astype(np.int32, copy=False)
    max_depth = int(depth.max()) if len(depth) > 0 else 0
    layer_indices: List[np.ndarray] = []
    layer_terminal_nodes: List[np.ndarray] = []
    layer_chance_nodes: List[np.ndarray] = []
    layer_decision_nodes: List[np.ndarray] = []
    for d in range(max_depth + 1):
        nodes_d = np.where(depth == d)[0].astype(np.int32)
        layer_indices.append(nodes_d)
        # Precompute per-layer node subsets once (Option B).
        # This avoids filtering by node_type every iteration.
        layer_terminal_nodes.append(nodes_d[node_type[nodes_d] == 0])
        layer_chance_nodes.append(nodes_d[node_type[nodes_d] == 2])
        layer_decision_nodes.append(nodes_d[node_type[nodes_d] == 1])

    infoset_key_to_id = {k: i for i, k in enumerate(infoset_id_to_key)}

    # v1 stored floats as float64. We cast to float32 internally to match FLOAT_DTYPE.
    ft = FlatTree(
        node_type=node_type,
        player=data["player"].astype(np.int8, copy=False),
        infoset_id=data["infoset_id"].astype(np.int32, copy=False),
        depth=depth,
        payoffs=data["payoffs"].astype(FLOAT_DTYPE, copy=False),
        children=data["children"].astype(np.int32, copy=False),
        chance_offsets=data["chance_offsets"].astype(np.int64, copy=False),
        chance_children=data["chance_children"].astype(np.int32, copy=False),
        chance_probs=data["chance_probs"].astype(FLOAT_DTYPE, copy=False),
        root=int(data["root"][0]),
        layer_indices=layer_indices,
        layer_terminal_nodes=layer_terminal_nodes,
        layer_chance_nodes=layer_chance_nodes,
        layer_decision_nodes=layer_decision_nodes,
        max_depth=max_depth,
        infoset_key_to_id=infoset_key_to_id,
        infoset_id_to_key=infoset_id_to_key,
        infoset_valid_actions=data["infoset_valid_actions"].astype(bool, copy=False),
        infoset_player=data["infoset_player"].astype(np.int8, copy=False),
    )

    # Sanity check: if the root is a chance node, it must have outgoing chance edges.
    # This catches corrupted caches from earlier buggy direct-flat-tree builds.
    if int(ft.node_type[ft.root]) == 2:
        s = int(ft.chance_offsets[ft.root])
        e = int(ft.chance_offsets[ft.root + 1])
        if e <= s:
            raise ValueError(
                f"Corrupted flat-tree cache: chance root has no edges (offsets {s}..{e}). Rebuild required."
            )

    return ft


@dataclass
class FlatTree:
    """
    Flat (array-based) representation of an extensive-form game tree.

    Grundidee:
    - Jeder Node hat eine Integer-ID: 0..num_nodes-1
    - Kanten zeigen auf Child-IDs (auch Integer)
    - Chance-Nodes haben eine variable Anzahl Outcomes -> ragged arrays via offsets

    Node-Typen:
    - 0 = terminal: `payoffs[node]` ist gesetzt
    - 1 = decision: `children[node, action]` ist gesetzt (action in `ACTION_ORDER`)
    - 2 = chance: Kanten sind über `chance_offsets`/`chance_children`/`chance_probs` gesetzt

    Chance-Kanten (ragged):
    - Für Chance-Node n liegen die Edges in:
        edge_idx ∈ [chance_offsets[n], chance_offsets[n+1])
      und für jede edge_idx gilt:
        child = chance_children[edge_idx]
        prob  = chance_probs[edge_idx]
    """
    # Node arrays
    node_type: np.ndarray  # int8: 0 terminal, 1 decision, 2 chance
    player: np.ndarray  # int8: 0/1 for decision, -1 for chance/terminal
    infoset_id: np.ndarray  # int32: infoset index for decision nodes else -1
    depth: np.ndarray  # int32
    payoffs: np.ndarray  # float64 [num_nodes, 2] (terminal nodes)
    children: np.ndarray  # int32 [num_nodes, 4] (decision nodes), -1 if illegal

    # Ragged chance edges
    chance_offsets: np.ndarray  # int64 [num_nodes+1]
    chance_children: np.ndarray  # int32 [num_chance_edges]
    chance_probs: np.ndarray  # float64 [num_chance_edges]

    # Metadata
    root: int
    layer_indices: List[np.ndarray]  # list of node-id arrays per depth
    # Performance (Option B):
    # Precomputed node subsets per depth. These arrays are derived from (layer_indices, node_type)
    # and are used in the DP passes to avoid per-iteration filtering costs.
    layer_terminal_nodes: List[np.ndarray]
    layer_chance_nodes: List[np.ndarray]
    layer_decision_nodes: List[np.ndarray]
    max_depth: int

    # Infoset mapping
    infoset_key_to_id: Dict[Tuple, int]
    infoset_id_to_key: List[Tuple]
    infoset_valid_actions: np.ndarray  # bool [num_infosets, 4]
    infoset_player: np.ndarray  # int8 [num_infosets], 0 or 1


def _determine_use_suit_abstraction(game, combination_generator) -> bool:
    use_suit_abstraction = bool(getattr(game, "abstract_suits", False))
    if use_suit_abstraction:
        return True

    # Fallback: detect via combination generator types
    try:
        from utils.poker_utils import LeducHoldemCombinationsAbstracted, TwelveCardPokerCombinationsAbstracted

        return isinstance(
            combination_generator,
            (LeducHoldemCombinationsAbstracted, TwelveCardPokerCombinationsAbstracted),
        )
    except Exception:
        return False


def _flatten_game_tree(tree: GameTree) -> FlatTree:
    """
    Converts `training.build_game_tree.GameTree` into a flat array representation.

    Determinism:
    - Chance outcomes are sorted by `str(outcome)` before being written to the flat chance edge arrays.
    """
    if not tree.root_nodes or len(tree.root_nodes) != 1:
        raise ValueError(f"Expected exactly one root node, got {len(tree.root_nodes)}")

    node_ids = sorted(tree.nodes.keys())
    num_nodes = len(node_ids)
    max_id = node_ids[-1] if node_ids else -1
    if max_id != num_nodes - 1:
        # Current builder creates contiguous ids. If that ever changes, we need a remap.
        raise ValueError("GameTree node_ids are not contiguous; remapping not implemented.")

    root = int(tree.root_nodes[0])

    node_type = np.full(num_nodes, -1, dtype=np.int8)
    player = np.full(num_nodes, -1, dtype=np.int8)
    infoset_id = np.full(num_nodes, -1, dtype=np.int32)
    depth = np.zeros(num_nodes, dtype=np.int32)
    payoffs = np.zeros((num_nodes, 2), dtype=FLOAT_DTYPE)
    children = np.full((num_nodes, NUM_ACTIONS), -1, dtype=np.int32)

    # Chance ragged arrays built incrementally via offsets
    chance_offsets = np.zeros(num_nodes + 1, dtype=np.int64)
    chance_children_list: List[int] = []
    chance_probs_list: List[float] = []
    chance_edge_count = 0

    infoset_key_to_id: Dict[Tuple, int] = {}
    infoset_id_to_key: List[Tuple] = []
    infoset_player_list: List[int] = []

    # First pass: fill per-node arrays and build chance edges.
    #
    # Wichtig: Die Node-IDs des `GameTree` sind aktuell contiguous (0..N-1).
    # Das ist sehr praktisch, weil wir direkt Arrays indexieren können.
    for nid in range(num_nodes):
        n = tree.nodes[nid]
        depth[nid] = int(getattr(n, "depth", 0))

        # Maintain chance_offsets prefix-sum
        chance_offsets[nid] = chance_edge_count

        if n.type == "terminal":
            node_type[nid] = 0
            player[nid] = -1
            infoset_id[nid] = -1
            payoffs[nid, 0] = FLOAT_DTYPE(n.payoffs[0])
            payoffs[nid, 1] = FLOAT_DTYPE(n.payoffs[1])
            continue

        if n.type == "chance":
            node_type[nid] = 2
            player[nid] = -1
            infoset_id[nid] = -1

            outcomes = list(n.legal_actions or [])
            # Deterministic ordering for flattening
            outcomes = sorted(outcomes, key=lambda o: str(o))
            probs = n.chance_probs or {}

            for outcome in outcomes:
                c = int(n.children[outcome])
                p = float(probs.get(outcome, 0.0))
                chance_children_list.append(c)
                chance_probs_list.append(p)
                chance_edge_count += 1
            continue

        if n.type == "decision":
            node_type[nid] = 1
            p = int(n.player)
            if p not in (0, 1):
                raise ValueError(f"Invalid decision-node player={p} at node {nid}")
            player[nid] = p

            key = n.infoset_key
            if key not in infoset_key_to_id:
                infoset_key_to_id[key] = len(infoset_id_to_key)
                infoset_id_to_key.append(key)
                infoset_player_list.append(p)
            else:
                # Sanity: infoset belongs to a fixed player in your key format
                existing_pid = infoset_player_list[infoset_key_to_id[key]]
                if existing_pid != p:
                    raise ValueError(f"Infoset {key} appears for multiple players ({existing_pid} vs {p})")

            iid = infoset_key_to_id[key]
            infoset_id[nid] = int(iid)

            # Map children into fixed 4-action order
            for action in (n.legal_actions or []):
                if action not in ACTION_TO_IDX:
                    raise ValueError(f"Unknown action '{action}' in decision node {nid}")
                a_idx = ACTION_TO_IDX[action]
                children[nid, a_idx] = int(n.children[action])

            continue

        raise ValueError(f"Unknown node.type '{n.type}' at node {nid}")

    chance_offsets[num_nodes] = chance_edge_count

    if (node_type == -1).any():
        bad = np.where(node_type == -1)[0][:10]
        raise ValueError(f"Some nodes not classified (first: {bad})")

    chance_children_arr = np.asarray(chance_children_list, dtype=np.int32)
    chance_probs_arr = np.asarray(chance_probs_list, dtype=FLOAT_DTYPE)

    # Layers by depth (for DP passes)
    max_depth = int(depth.max()) if num_nodes > 0 else 0
    layer_indices: List[np.ndarray] = []
    layer_terminal_nodes: List[np.ndarray] = []
    layer_chance_nodes: List[np.ndarray] = []
    layer_decision_nodes: List[np.ndarray] = []
    for d in range(max_depth + 1):
        nodes_d = np.where(depth == d)[0].astype(np.int32)
        layer_indices.append(nodes_d)
        # Option B: precompute node-type subsets once (used in DP passes)
        layer_terminal_nodes.append(nodes_d[node_type[nodes_d] == 0])
        layer_chance_nodes.append(nodes_d[node_type[nodes_d] == 2])
        layer_decision_nodes.append(nodes_d[node_type[nodes_d] == 1])

    num_infosets = len(infoset_id_to_key)
    infoset_valid_actions = np.zeros((num_infosets, NUM_ACTIONS), dtype=bool)
    for nid in np.where(node_type == 1)[0]:
        iid = infoset_id[nid]
        infoset_valid_actions[iid] |= (children[nid] != -1)

    infoset_player = np.asarray(infoset_player_list, dtype=np.int8)

    return FlatTree(
        node_type=node_type,
        player=player,
        infoset_id=infoset_id,
        depth=depth,
        payoffs=payoffs,
        children=children,
        chance_offsets=chance_offsets,
        chance_children=chance_children_arr,
        chance_probs=chance_probs_arr,
        root=root,
        layer_indices=layer_indices,
        layer_terminal_nodes=layer_terminal_nodes,
        layer_chance_nodes=layer_chance_nodes,
        layer_decision_nodes=layer_decision_nodes,
        max_depth=max_depth,
        infoset_key_to_id=infoset_key_to_id,
        infoset_id_to_key=infoset_id_to_key,
        infoset_valid_actions=infoset_valid_actions,
        infoset_player=infoset_player,
    )


def _build_flat_tree_directly_from_game(game) -> FlatTree:
    """
    Build a FlatTree directly from the *game environment API* (explicit chance nodes),
    without constructing an intermediate object-based `GameTree`.

    This avoids a large RAM spike from storing thousands of Python Node objects.

    Requirements for `game` (as implemented in this repo):
    - `game.reset(starting_player)`
    - `game.done`
    - `game.current_player`
    - `game.is_chance_node()`
    - `game.get_chance_outcomes_with_probs()` -> {outcome: prob}
    - `game.get_legal_actions()` for decision nodes
    - `game.step(action_or_outcome)` and `game.step_back()`
    - `game.get_payoff(player_id)` at terminal nodes
    """
    actions = ACTION_ORDER
    action_to_idx = ACTION_TO_IDX

    node_type_list: List[int] = []
    player_list: List[int] = []
    infoset_id_list: List[int] = []
    depth_list: List[int] = []
    payoffs_list: List[Tuple[FLOAT_DTYPE, FLOAT_DTYPE]] = []
    children_list: List[Tuple[int, int, int, int]] = []

    # IMPORTANT:
    # We must NOT try to write `chance_offsets` incrementally while doing DFS.
    # Node ids are pre-order, but a chance node's outgoing edge block is only fully known
    # after *all* children were traversed. Earlier versions wrote offsets too early which
    # could make the chance root appear to have 0 edges -> reach stays 0 -> no learning.
    #
    # Instead: collect per-node edge lists and concatenate once at the end.
    chance_children_by_node: List[List[int]] = []
    chance_probs_by_node: List[List[FLOAT_DTYPE]] = []
    chance_edge_total = 0

    infoset_key_to_id: Dict[Tuple, int] = {}
    infoset_id_to_key: List[Tuple] = []
    infoset_player_list: List[int] = []

    def traverse(depth: int) -> int:
        nonlocal chance_edge_total
        # Reserve node slot (critical, so child indices point to the correct node id)
        node_id = len(node_type_list)
        node_type_list.append(-1)
        player_list.append(-1)
        infoset_id_list.append(-1)
        depth_list.append(depth)
        payoffs_list.append((FLOAT_DTYPE(0.0), FLOAT_DTYPE(0.0)))
        children_list.append((-1, -1, -1, -1))

        # Per-node chance edge lists (only used if this becomes a chance node)
        chance_children_by_node.append([])
        chance_probs_by_node.append([])

        # Progress log for very large trees (keeps overhead negligible).
        if node_id > 0 and (node_id % 100_000 == 0):
            print(
                f"[flat-tree build] touched {node_id:,} nodes "
                f"(depth={depth}, chance_edges={chance_edge_total:,}, infosets={len(infoset_id_to_key):,})"
            )

        # Terminal
        if game.done:
            node_type_list[node_id] = 0
            player_list[node_id] = -1
            infoset_id_list[node_id] = -1
            payoffs_list[node_id] = (FLOAT_DTYPE(game.get_payoff(0)), FLOAT_DTYPE(game.get_payoff(1)))
            return node_id

        # Chance node
        if hasattr(game, "is_chance_node") and game.is_chance_node():
            node_type_list[node_id] = 2
            player_list[node_id] = -1
            infoset_id_list[node_id] = -1

            outcomes_with_probs = game.get_chance_outcomes_with_probs()
            outcomes = list(outcomes_with_probs.keys())
            # Deterministic ordering for cache stability
            outcomes = sorted(outcomes, key=lambda o: str(o))

            edges_children = chance_children_by_node[node_id]
            edges_probs = chance_probs_by_node[node_id]
            for outcome in outcomes:
                prob = FLOAT_DTYPE(outcomes_with_probs.get(outcome, 0.0))
                if prob == 0:
                    continue
                game.step(outcome)
                child = traverse(depth + 1)
                game.step_back()
                edges_children.append(int(child))
                edges_probs.append(prob)
                chance_edge_total += 1

            return node_id

        # Decision node
        p = int(game.current_player)
        if p not in (0, 1):
            raise ValueError(f"Invalid decision node player={p} at depth={depth}")

        node_type_list[node_id] = 1
        player_list[node_id] = p

        key = KeyGenerator.get_info_set_key(game, p)
        if key not in infoset_key_to_id:
            infoset_key_to_id[key] = len(infoset_id_to_key)
            infoset_id_to_key.append(key)
            infoset_player_list.append(p)
        else:
            existing_pid = infoset_player_list[infoset_key_to_id[key]]
            if existing_pid != p:
                raise ValueError(f"Infoset {key} appears for multiple players ({existing_pid} vs {p})")

        iid = infoset_key_to_id[key]
        infoset_id_list[node_id] = int(iid)

        child_indices = [-1, -1, -1, -1]
        for action in game.get_legal_actions():
            if action not in action_to_idx:
                raise ValueError(f"Unknown action '{action}' at depth={depth}")
            a_idx = action_to_idx[action]
            game.step(action)
            child = traverse(depth + 1)
            game.step_back()
            child_indices[a_idx] = int(child)

        children_list[node_id] = tuple(child_indices)  # type: ignore[arg-type]
        return node_id

    # Build from single root state (explicit chance nodes handle dealing)
    game.reset(0)
    root = traverse(0)

    # Convert to numpy arrays
    node_type = np.asarray(node_type_list, dtype=np.int8)
    player = np.asarray(player_list, dtype=np.int8)
    infoset_id = np.asarray(infoset_id_list, dtype=np.int32)
    depth_arr = np.asarray(depth_list, dtype=np.int32)
    payoffs = np.asarray(payoffs_list, dtype=FLOAT_DTYPE).reshape(len(node_type_list), 2)
    children = np.asarray(children_list, dtype=np.int32).reshape(len(node_type_list), NUM_ACTIONS)

    # Finalize ragged chance edge arrays (prefix sums + concatenation)
    num_nodes = len(node_type_list)
    chance_offsets = np.zeros(num_nodes + 1, dtype=np.int64)
    for nid in range(num_nodes):
        chance_offsets[nid + 1] = chance_offsets[nid] + len(chance_children_by_node[nid])
    total_edges = int(chance_offsets[-1])
    chance_children = np.empty(total_edges, dtype=np.int32)
    chance_probs = np.empty(total_edges, dtype=FLOAT_DTYPE)
    w = 0
    for nid in range(num_nodes):
        kids = chance_children_by_node[nid]
        probs = chance_probs_by_node[nid]
        if not kids:
            continue
        k = len(kids)
        chance_children[w : w + k] = np.asarray(kids, dtype=np.int32)
        chance_probs[w : w + k] = np.asarray(probs, dtype=FLOAT_DTYPE)
        w += k

    max_depth = int(depth_arr.max()) if len(depth_arr) > 0 else 0
    layer_indices: List[np.ndarray] = []
    layer_terminal_nodes: List[np.ndarray] = []
    layer_chance_nodes: List[np.ndarray] = []
    layer_decision_nodes: List[np.ndarray] = []
    for d in range(max_depth + 1):
        nodes_d = np.where(depth_arr == d)[0].astype(np.int32)
        layer_indices.append(nodes_d)
        layer_terminal_nodes.append(nodes_d[node_type[nodes_d] == 0])
        layer_chance_nodes.append(nodes_d[node_type[nodes_d] == 2])
        layer_decision_nodes.append(nodes_d[node_type[nodes_d] == 1])

    num_infosets = len(infoset_id_to_key)
    infoset_valid_actions = np.zeros((num_infosets, NUM_ACTIONS), dtype=bool)
    for n in np.where(node_type == 1)[0]:
        iid = int(infoset_id[n])
        infoset_valid_actions[iid] |= (children[n] != -1)
    infoset_player = np.asarray(infoset_player_list, dtype=np.int8)

    # Sanity check: if the root is a chance node, it must have outgoing edges.
    if int(node_type[int(root)]) == 2:
        s = int(chance_offsets[int(root)])
        e = int(chance_offsets[int(root) + 1])
        if e <= s:
            raise ValueError("Direct flat-tree build produced a chance root without edges (bug).")

    return FlatTree(
        node_type=node_type,
        player=player,
        infoset_id=infoset_id,
        depth=depth_arr,
        payoffs=payoffs,
        children=children,
        chance_offsets=chance_offsets,
        chance_children=chance_children,
        chance_probs=chance_probs,
        root=int(root),
        layer_indices=layer_indices,
        layer_terminal_nodes=layer_terminal_nodes,
        layer_chance_nodes=layer_chance_nodes,
        layer_decision_nodes=layer_decision_nodes,
        max_depth=max_depth,
        infoset_key_to_id=infoset_key_to_id,
        infoset_id_to_key=infoset_id_to_key,
        infoset_valid_actions=infoset_valid_actions,
        infoset_player=infoset_player,
    )


class CFRSolverWithFlatTree:
    """
    CFR Solver on a flattened tree (numpy arrays).

    Matches `CFRSolverWithTree` semantics:
    - explicit chance nodes via expectation (no updates at chance nodes)
    - alternating vs simultaneous updates
    - uniform averaging for strategy_sum
    """

    def __init__(
        self,
        game,
        combination_generator,
        game_name: Optional[str] = None,
        load_tree: bool = True,
        alternating_updates: bool = True,
        partial_pruning: bool = False,
        validate_flat_tree: bool = False,
    ):
        self.game = game
        self.combination_generator = combination_generator
        self.alternating_updates = alternating_updates
        self.partial_pruning = partial_pruning

        self.iteration_count = 0
        self.training_time = 0.0
        self._current_iteration = 0  # 1-indexed

        use_suit_abstraction = _determine_use_suit_abstraction(game, combination_generator)

        # Fast path: load cached flat-tree if available.
        self.flat = None
        if load_tree and game_name:
            flat_path = _get_flat_tree_cache_path(game_name, abstract_suits=use_suit_abstraction)
            if os.path.exists(flat_path):
                try:
                    print(f"Loading flat tree cache for {game_name}...")
                    print(f"  Flat tree cache path: {flat_path}")
                    self.flat = _load_flat_tree_cache(flat_path)
                    if validate_flat_tree:
                        self._validate_flat_tree_sampled(self.flat)
                    print(
                        f"Flat tree cache loaded: {len(self.flat.node_type)} nodes, "
                        f"{len(self.flat.infoset_id_to_key)} infosets"
                    )
                except Exception as e:
                    print(f"Failed to load flat tree cache (will rebuild): {e}")
                    self.flat = None

        if self.flat is None:
            # IMPORTANT: We build the flat arrays directly from the game API to avoid
            # keeping a full Python object-tree in RAM (big spike on large games).
            print("Building flat tree directly from game environment...")
            t0 = time.time()
            self.flat = _build_flat_tree_directly_from_game(game)
            print(
                f"Flat tree ready: {len(self.flat.node_type)} nodes, "
                f"{len(self.flat.infoset_id_to_key)} infosets ({time.time() - t0:.3f}s)"
            )
            if validate_flat_tree:
                self._validate_flat_tree_sampled(self.flat)

            # If we have a game_name, also write the flat-tree cache for next time.
            if load_tree and game_name:
                flat_path = _get_flat_tree_cache_path(game_name, abstract_suits=use_suit_abstraction)
                try:
                    _save_flat_tree_cache(self.flat, flat_path)
                    print(f"Flat tree cache saved to: {flat_path}")
                except Exception as e:
                    print(f"Warning: failed to save flat tree cache: {e}")

        # CFR arrays
        self.num_infosets = len(self.flat.infoset_id_to_key)
        # Option C: use float32 internally (matches tensor path; less bandwidth/caching pressure).
        self.regret_sum = np.zeros((self.num_infosets, NUM_ACTIONS), dtype=FLOAT_DTYPE)
        self.strategy_sum = np.zeros((self.num_infosets, NUM_ACTIONS), dtype=FLOAT_DTYPE)

        # Precompute uniform strategy per infoset (over valid actions)
        valid = self.flat.infoset_valid_actions.astype(FLOAT_DTYPE)
        counts = np.maximum(valid.sum(axis=1, keepdims=True), FLOAT_DTYPE(1.0))
        self._uniform_strategy = valid / counts

        # Work buffers
        self._reach = np.zeros((len(self.flat.node_type), 2), dtype=FLOAT_DTYPE)
        self._values = np.zeros((len(self.flat.node_type), 2), dtype=FLOAT_DTYPE)

        self.average_strategy = {}

    @staticmethod
    def _validate_flat_tree_sampled(ft: FlatTree, max_nodes: int = 2000) -> None:
        """
        Optional: schnelle Sanity-Checks für Debugging/Architektur-Kapitel.

        Checks (gesampelt):
        - chance_offsets Länge/Monotonie
        - Depth-Konsistenz: child.depth == parent.depth + 1 für Decision/Chance-Edges
        """
        num_nodes = int(len(ft.node_type))
        if num_nodes == 0:
            raise ValueError("FlatTree is empty")
        if ft.root < 0 or ft.root >= num_nodes:
            raise ValueError(f"Invalid root index {ft.root}")

        if int(len(ft.chance_offsets)) != num_nodes + 1:
            raise ValueError("chance_offsets must have length num_nodes+1")
        if int(ft.chance_offsets[0]) != 0:
            raise ValueError("chance_offsets[0] must be 0")
        if np.any(ft.chance_offsets[1:] < ft.chance_offsets[:-1]):
            raise ValueError("chance_offsets must be non-decreasing")

        total_edges = int(ft.chance_offsets[-1])
        if total_edges != int(len(ft.chance_children)) or total_edges != int(len(ft.chance_probs)):
            raise ValueError("chance edge arrays length mismatch")

        # Depth check for a sample of decision edges
        decision_nodes = np.where(ft.node_type == 1)[0][:max_nodes]
        for n in decision_nodes:
            pd = int(ft.depth[n])
            for c in ft.children[n]:
                c = int(c)
                if c == -1:
                    continue
                if int(ft.depth[c]) != pd + 1:
                    raise ValueError(f"Decision edge depth mismatch: {n}->{c} ({pd} -> {int(ft.depth[c])})")

        # Depth check for a sample of chance edges
        chance_nodes = np.where(ft.node_type == 2)[0][:max_nodes]
        for n in chance_nodes:
            pd = int(ft.depth[n])
            s = int(ft.chance_offsets[n])
            e = int(ft.chance_offsets[n + 1])
            if e < s:
                raise ValueError(f"Invalid chance offset range for node {n}: {s}..{e}")
            kids = ft.chance_children[s:e]
            for c in kids[:50]:  # sample within node as well
                c = int(c)
                if int(ft.depth[c]) != pd + 1:
                    raise ValueError(f"Chance edge depth mismatch: {n}->{c} ({pd} -> {int(ft.depth[c])})")

    # --- Strategy computation ---
    def _compute_current_strategy(self) -> np.ndarray:
        """
        Regret matching over positive regrets, restricted to valid actions.
        Returns:
          sigma: float64 array [num_infosets, 4]
        """
        # Keep everything in float32 to avoid upcasting.
        pos = np.maximum(self.regret_sum, FLOAT_DTYPE(0.0)) * self.flat.infoset_valid_actions
        s = pos.sum(axis=1, keepdims=True)
        sigma = np.empty_like(pos)
        has_pos = (s[:, 0] > 1e-15)
        sigma[has_pos] = pos[has_pos] / s[has_pos]
        sigma[~has_pos] = self._uniform_strategy[~has_pos]
        return sigma

    # --- DP passes ---
    def _forward_reach(self, sigma: np.ndarray):
        """
        Forward Pass (top-down):
        - Input: sigma[infoset, action]
        - Output: reach[node, player] for both players (chance reach is included automatically).
        """
        ft = self.flat
        self._reach.fill(FLOAT_DTYPE(0.0))
        self._reach[ft.root, 0] = FLOAT_DTYPE(1.0)
        self._reach[ft.root, 1] = FLOAT_DTYPE(1.0)

        # Option B:
        # We use precomputed node-type lists per depth (ft.layer_*_nodes) to avoid
        # filtering by ft.node_type in every iteration.
        for depth in range(ft.max_depth + 1):
            if self.partial_pruning:
                # Still compute pruned subsets per type, but skip scanning all nodes by type.
                self._forward_process_chance_nodes(ft.layer_chance_nodes[depth], prune=True)
                self._forward_process_decision_nodes(ft.layer_decision_nodes[depth], sigma, prune=True)
            else:
                self._forward_process_chance_nodes(ft.layer_chance_nodes[depth], prune=False)
                self._forward_process_decision_nodes(ft.layer_decision_nodes[depth], sigma, prune=False)

    def _forward_process_chance_nodes(self, chance_nodes: np.ndarray, prune: bool) -> None:
        """
        Forward reach propagation for all chance nodes in `layer`.

        For each chance node n and each outcome edge (n -> c) with probability p:
          reach[c,0] += reach[n,0] * p
          reach[c,1] += reach[n,1] * p
        """
        ft = self.flat
        if chance_nodes.size == 0:
            return
        if prune:
            # Skip chance nodes with reach == 0 for both players.
            chance_nodes = chance_nodes[(self._reach[chance_nodes, 0] + self._reach[chance_nodes, 1]) > 0]
            if chance_nodes.size == 0:
                return
        for n in chance_nodes:
            s = int(ft.chance_offsets[n])
            e = int(ft.chance_offsets[n + 1])
            if e <= s:
                continue
            kids = ft.chance_children[s:e]
            probs = ft.chance_probs[s:e]
            r0, r1 = self._reach[n, 0], self._reach[n, 1]
            self._reach[kids, 0] += r0 * probs
            self._reach[kids, 1] += r1 * probs

    def _forward_process_decision_nodes(self, dec_nodes: np.ndarray, sigma: np.ndarray, prune: bool) -> None:
        """
        Forward reach propagation for all decision nodes in `layer`.

        Decision node semantics:
        - active player's reach is multiplied by sigma(action)
        - inactive player's reach is copied unchanged
        """
        ft = self.flat
        if dec_nodes.size == 0:
            return
        if prune:
            dec_nodes = dec_nodes[(self._reach[dec_nodes, 0] + self._reach[dec_nodes, 1]) > 0]
            if dec_nodes.size == 0:
                return

        node_players = ft.player[dec_nodes].astype(np.int8)
        node_infosets = ft.infoset_id[dec_nodes].astype(np.int32)
        parent_reach = self._reach[dec_nodes]  # shape (m,2)
        node_strategy = sigma[node_infosets]  # shape (m,4)
        kids = ft.children[dec_nodes]  # shape (m,4)

        for a in range(NUM_ACTIONS):
            child_a = kids[:, a]
            valid = (child_a != -1)
            if not np.any(valid):
                continue

            # active player 0 nodes
            m0 = valid & (node_players == 0)
            if np.any(m0):
                c = child_a[m0]
                self._reach[c, 0] += parent_reach[m0, 0] * node_strategy[m0, a]
                self._reach[c, 1] += parent_reach[m0, 1]

            # active player 1 nodes
            m1 = valid & (node_players == 1)
            if np.any(m1):
                c = child_a[m1]
                self._reach[c, 1] += parent_reach[m1, 1] * node_strategy[m1, a]
                self._reach[c, 0] += parent_reach[m1, 0]

    def _backward_values_and_updates(
        self,
        sigma: np.ndarray,
        updating_player: Optional[int],
        strategy_sum_weight: float,
    ):
        """
        Backward Pass (bottom-up):
        - Input: sigma[infoset, action], reach[node, player]
        - Computes: values[node, player]
        - Updates: regret_sum and strategy_sum.

        If `updating_player` is 0/1: updates regrets and strategy_sum only for that player (alternating pass).
        If `updating_player` is None: simultaneous update for both players (updates both).
        """
        ft = self.flat
        self._values.fill(FLOAT_DTYPE(0.0))
        delta_regret = np.zeros_like(self.regret_sum)

        # Option B: use precomputed per-depth node lists to avoid per-iteration filtering by node_type.
        for depth in range(ft.max_depth, -1, -1):
            self._backward_set_terminal_values(ft.layer_terminal_nodes[depth])
            self._backward_set_chance_values(ft.layer_chance_nodes[depth])
            self._backward_process_decision_layer(
                dec_nodes=ft.layer_decision_nodes[depth],
                sigma=sigma,
                updating_player=updating_player,
                strategy_sum_weight=strategy_sum_weight,
                delta_regret=delta_regret,
            )

        self.regret_sum += delta_regret

    def _backward_set_terminal_values(self, term_nodes: np.ndarray) -> None:
        if term_nodes.size > 0:
            self._values[term_nodes] = self.flat.payoffs[term_nodes]

    def _backward_set_chance_values(self, chance_nodes: np.ndarray) -> None:
        """
        Chance nodes are evaluated as expectation over their children:
          values[n] = Σ_o P(o) * values[child(o)]
        """
        ft = self.flat
        if chance_nodes.size == 0:
            return
        for n in chance_nodes:
            s = int(ft.chance_offsets[n])
            e = int(ft.chance_offsets[n + 1])
            if e <= s:
                continue
            kids = ft.chance_children[s:e]
            probs = ft.chance_probs[s:e]
            self._values[n] = (self._values[kids] * probs[:, None]).sum(axis=0)

    def _backward_process_decision_layer(
        self,
        dec_nodes: np.ndarray,
        sigma: np.ndarray,
        updating_player: Optional[int],
        strategy_sum_weight: float,
        delta_regret: np.ndarray,
    ) -> None:
        ft = self.flat
        if dec_nodes.size == 0:
            return

        node_players = ft.player[dec_nodes].astype(np.int8)  # (m,)
        node_infosets = ft.infoset_id[dec_nodes].astype(np.int32)  # (m,)
        node_children = ft.children[dec_nodes]  # (m,4)

        valid_actions = (node_children != -1)  # (m,4)
        safe_children = np.where(valid_actions, node_children, 0)  # (m,4)

        # child_vals has shape (m,4,2). Illegal actions are masked out to 0.
        child_vals = self._values[safe_children] * valid_actions[:, :, None]

        node_strategy = sigma[node_infosets]  # (m,4)
        node_ev = (child_vals * node_strategy[:, :, None]).sum(axis=1)  # (m,2)
        self._values[dec_nodes] = node_ev

        self._accumulate_strategy_sum_for_decisions(
            dec_nodes=dec_nodes,
            node_players=node_players,
            node_infosets=node_infosets,
            node_strategy=node_strategy,
            valid_actions=valid_actions,
            updating_player=updating_player,
            strategy_sum_weight=strategy_sum_weight,
        )

        self._accumulate_regrets_for_decisions(
            dec_nodes=dec_nodes,
            node_players=node_players,
            node_infosets=node_infosets,
            node_ev=node_ev,
            child_vals=child_vals,
            valid_actions=valid_actions,
            updating_player=updating_player,
            delta_regret=delta_regret,
        )

    def _accumulate_strategy_sum_for_decisions(
        self,
        dec_nodes: np.ndarray,
        node_players: np.ndarray,
        node_infosets: np.ndarray,
        node_strategy: np.ndarray,
        valid_actions: np.ndarray,
        updating_player: Optional[int],
        strategy_sum_weight: float,
    ) -> None:
        """
        StrategySum update (operational):
        - compute reach of the active player at each decision node
        - add reach * sigma (and optional weight) into strategy_sum[infoset]
        """
        if updating_player is None:
            reach_p = self._reach[dec_nodes, node_players]  # (m,)
            contrib = node_strategy * reach_p[:, None] * strategy_sum_weight
            contrib *= valid_actions
            np.add.at(self.strategy_sum, node_infosets, contrib)
            return

        up_mask = (node_players == updating_player)
        if not np.any(up_mask):
            return
        nodes_u = dec_nodes[up_mask]
        infosets_u = node_infosets[up_mask]
        reach_p = self._reach[nodes_u, updating_player]
        contrib = node_strategy[up_mask] * reach_p[:, None] * strategy_sum_weight
        contrib *= valid_actions[up_mask]
        np.add.at(self.strategy_sum, infosets_u, contrib)

    def _accumulate_regrets_for_decisions(
        self,
        dec_nodes: np.ndarray,
        node_players: np.ndarray,
        node_infosets: np.ndarray,
        node_ev: np.ndarray,
        child_vals: np.ndarray,
        valid_actions: np.ndarray,
        updating_player: Optional[int],
        delta_regret: np.ndarray,
    ) -> None:
        """
        Regret update (operational):
          inst_regret(a) = opp_reach * ( q(a) - v )
        where `opp_reach` is reach of the opponent at this node (chance already included in reach),
        `q(a)` is child value for the updating player, `v` is expected value at node.
        """
        if updating_player is None:
            m0 = (node_players == 0)
            if np.any(m0):
                opp_reach = self._reach[dec_nodes[m0], 1]
                v = node_ev[m0, 0]
                q = child_vals[m0, :, 0]
                inst = opp_reach[:, None] * (q - v[:, None])
                inst *= valid_actions[m0]
                np.add.at(delta_regret, node_infosets[m0], inst)

            m1 = (node_players == 1)
            if np.any(m1):
                opp_reach = self._reach[dec_nodes[m1], 0]
                v = node_ev[m1, 1]
                q = child_vals[m1, :, 1]
                inst = opp_reach[:, None] * (q - v[:, None])
                inst *= valid_actions[m1]
                np.add.at(delta_regret, node_infosets[m1], inst)
            return

        up_mask = (node_players == updating_player)
        if not np.any(up_mask):
            return
        nodes_u = dec_nodes[up_mask]
        infosets_u = node_infosets[up_mask]
        opp = 1 - updating_player
        opp_reach = self._reach[nodes_u, opp]
        v = node_ev[up_mask, updating_player]
        q = child_vals[up_mask, :, updating_player]
        inst = opp_reach[:, None] * (q - v[:, None])
        inst *= valid_actions[up_mask]
        np.add.at(delta_regret, infosets_u, inst)

    # --- Training loop ---
    def cfr_iteration(self):
        self._current_iteration = self.iteration_count + 1
        sigma = self._compute_current_strategy()

        # forward reach under current sigma
        self._forward_reach(sigma)

        if self.alternating_updates:
            # player 0 pass
            self._backward_values_and_updates(
                sigma=sigma,
                updating_player=0,
                strategy_sum_weight=self._get_strategy_sum_weight(),
            )
            self.after_player_traversal(player=0)
            sigma = self._compute_current_strategy()
            self._forward_reach(sigma)

            # player 1 pass (with updated sigma)
            self._backward_values_and_updates(
                sigma=sigma,
                updating_player=1,
                strategy_sum_weight=self._get_strategy_sum_weight(),
            )
            self.after_player_traversal(player=1)
        else:
            # simultaneous: single backward that updates both players based on same sigma
            self._backward_values_and_updates(
                sigma=sigma,
                updating_player=None,
                strategy_sum_weight=self._get_strategy_sum_weight(),
            )
            self.after_simultaneous_traversal()

        self.iteration_count += 1

    def _get_strategy_sum_weight(self) -> float:
        # CFR: uniform averaging
        return 1.0

    def after_player_traversal(self, player: int):
        return

    def after_simultaneous_traversal(self):
        return

    def train(self, iterations, br_tracker=None, print_interval=100):
        start_time = time.time()
        for i in range(iterations):
            self.cfr_iteration()

            if (i + 1) % print_interval == 0:
                print(f"Iteration {i + 1}")

            if br_tracker is not None and br_tracker.should_evaluate(i + 1):
                current_avg_strategy = self.get_average_strategy()
                br_tracker.evaluate_and_add(current_avg_strategy, i + 1, start_time=start_time)
                br_tracker.last_eval_iteration = i + 1

        if br_tracker is not None:
            current_avg_strategy = self.get_average_strategy()
            br_tracker.evaluate_and_add(current_avg_strategy, iterations, start_time=start_time)

        total_time = time.time() - start_time
        if br_tracker is not None:
            br_time = br_tracker.get_total_br_time()
            self.training_time = total_time - br_time
            if br_time > 0:
                print(f"Best Response Evaluation Zeit: {br_time:.2f}s")
        else:
            self.training_time = total_time

        if self.training_time >= 60:
            print(f"Training completed in {self.training_time / 60:.2f} minutes (ohne Best Response Evaluation)")
        else:
            print(f"Training completed in {self.training_time:.2f} seconds (ohne Best Response Evaluation)")

        self.average_strategy = self.get_average_strategy()

    # --- Strategy extraction / persistence ---
    def get_average_strategy(self):
        avg_strategy = {}
        ss = self.strategy_sum
        valid = self.flat.infoset_valid_actions

        for iid, key in enumerate(self.flat.infoset_id_to_key):
            row = ss[iid].copy()
            row *= valid[iid]
            total = float(row.sum())

            if total > 0:
                probs = row / total
            else:
                probs = self._uniform_strategy[iid]

            action_dict = {}
            for a_idx, a in enumerate(ACTION_ORDER):
                if valid[iid, a_idx]:
                    action_dict[a] = float(probs[a_idx])
            avg_strategy[key] = action_dict

        return avg_strategy

    def save_gzip(self, filepath):
        data = {
            "regret_sum": self.regret_sum,
            "strategy_sum": self.strategy_sum,
            "average_strategy": self.average_strategy,
            "iteration_count": self.iteration_count,
            "training_time": self.training_time,
        }
        with gzip.open(filepath, "wb") as f:
            pkl.dump(data, f)
        print(f"Saved to {filepath}")

    def load_gzip(self, filepath):
        with gzip.open(filepath, "rb") as f:
            data = pkl.load(f)
        self.regret_sum = data["regret_sum"]
        self.strategy_sum = data["strategy_sum"]
        self.average_strategy = data.get("average_strategy", {})
        self.iteration_count = data.get("iteration_count", 0)
        self.training_time = data.get("training_time", 0.0)
        # _current_iteration is 1-indexed
        self._current_iteration = self.iteration_count
        print(f"Loaded from {filepath}")

