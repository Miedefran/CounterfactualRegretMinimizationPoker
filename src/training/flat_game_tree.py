"""
Flat Tree Representation für CFR Solver.

Enthält:
- FlatTree Dataclass
- Funktionen zum Flatten eines GameTree
- Funktionen zum direkten Builden aus Game Environment
- Caching-Funktionen (Save/Load)
"""

from __future__ import annotations

import gzip
import os
import pickle as pkl
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from utils.data_models import KeyGenerator
from utils.tree_registry import record_tree_stats
from training.build_game_tree import GameTree

ACTION_ORDER = ("check", "bet", "call", "fold")
ACTION_TO_IDX = {a: i for i, a in enumerate(ACTION_ORDER)}
NUM_ACTIONS = len(ACTION_ORDER)

# Cache-Version: v1 float64, v2 float32 (Layer-Listen werden beim Laden rekonstruiert)
FLAT_TREE_CACHE_VERSION = 2
FLOAT_DTYPE = np.float32


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
    payoffs: np.ndarray  # float32 [num_nodes, 2] (terminal nodes)
    children: np.ndarray  # int32 [num_nodes, 4] (decision nodes), -1 if illegal

    # Ragged chance edges
    chance_offsets: np.ndarray  # int64 [num_nodes+1]
    chance_children: np.ndarray  # int32 [num_chance_edges]
    chance_probs: np.ndarray  # float32 [num_chance_edges]

    # Metadata
    root: int
    layer_indices: List[np.ndarray]  # node-IDs pro Tiefe
    # Pro Tiefe vorkomputierte Teilmengen (terminal/chance/decision) für DP-Pässe
    layer_terminal_nodes: List[np.ndarray]
    layer_chance_nodes: List[np.ndarray]
    layer_decision_nodes: List[np.ndarray]
    max_depth: int

    # Infoset mapping
    infoset_key_to_id: Dict[Tuple, int]
    infoset_id_to_key: List[Tuple]
    infoset_valid_actions: np.ndarray  # bool [num_infosets, 4]
    infoset_player: np.ndarray  # int8 [num_infosets], 0 or 1


def determine_use_suit_abstraction(game, combination_generator) -> bool:
    """Bestimmt ob Suit Abstraction verwendet wird."""
    use_suit_abstraction = bool(getattr(game, "abstract_suits", False))
    if use_suit_abstraction:
        return True

    # Alternativ: Erkennung über Combination-Generator-Typ
    try:
        from utils.poker_utils import LeducHoldemCombinationsAbstracted, TwelveCardPokerCombinationsAbstracted

        return isinstance(
            combination_generator,
            (LeducHoldemCombinationsAbstracted, TwelveCardPokerCombinationsAbstracted),
        )
    except Exception:
        return False


def get_flat_tree_cache_path(game_name: str, abstract_suits: bool) -> str:
    """
    Returns the path where the flat-tree cache is stored.

    We mirror the flat tree layout style:
      data/trees/game_trees/flat/{abstracted|normal}/{game_name}_flat_tree.npz
    """
    script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    subdir = "abstracted" if abstract_suits else "normal"
    out_dir = os.path.join(script_dir, "data", "trees", "game_trees", "flat", subdir)
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, f"{game_name}_flat_tree.npz")


def get_flat_tree_keys_path(npz_path: str) -> str:
    """Returns the path for the infoset keys pickle file."""
    return npz_path.replace(".npz", "_keys.pkl.gz")


def save_flat_tree_cache(ft: FlatTree, npz_path: str) -> None:
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

    keys_path = get_flat_tree_keys_path(npz_path)
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
        pass  # Registry-Logging optional


def load_flat_tree_cache(npz_path: str) -> FlatTree:
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

    keys_path = get_flat_tree_keys_path(npz_path)
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
        # Pro Tiefe: Teilmengen terminal/chance/decision
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

    # Validierung: Chance-Root muss Kanten haben
    if int(ft.node_type[ft.root]) == 2:
        s = int(ft.chance_offsets[ft.root])
        e = int(ft.chance_offsets[ft.root + 1])
        if e <= s:
            raise ValueError(
                f"Corrupted flat-tree cache: chance root has no edges (offsets {s}..{e}). Rebuild required."
            )

    return ft


def flatten_game_tree(tree: GameTree) -> FlatTree:
    """Konvertiert GameTree in flache Arrays. Chance-Outcomes werden deterministisch (str(outcome)) sortiert."""
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
    # Node-IDs im GameTree sind contiguous (0..N-1)
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


def build_flat_tree_directly_from_game(game) -> FlatTree:
    """
    Build a FlatTree directly from the game environment API (explicit chance nodes),
    without constructing an intermediate object-based GameTree.

    Kein intermediärer Python-Objektbaum nötig.

    Erwartete Game-API:
    - game.reset(starting_player)
    - game.done
    - game.current_player
    - game.is_chance_node()
    - game.get_chance_outcomes_with_probs() -> {outcome: prob}
    - game.get_legal_actions() for decision nodes
    - game.step(action_or_outcome) and game.step_back()
    - game.get_payoff(player_id) at terminal nodes
    """
    actions = ACTION_ORDER
    action_to_idx = ACTION_TO_IDX

    node_type_list: List[int] = []
    player_list: List[int] = []
    infoset_id_list: List[int] = []
    depth_list: List[int] = []
    payoffs_list: List[Tuple[FLOAT_DTYPE, FLOAT_DTYPE]] = []
    children_list: List[Tuple[int, int, int, int]] = []

    # chance_offsets erst am Ende setzen (pro Node sind alle Kinder erst nach Traversierung bekannt)
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
        chance_children[w: w + k] = np.asarray(kids, dtype=np.int32)
        chance_probs[w: w + k] = np.asarray(probs, dtype=FLOAT_DTYPE)
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

    # Chance-Root muss Kanten haben
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
