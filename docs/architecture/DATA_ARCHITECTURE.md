# Data Architecture & Interaction Redesign

## 1. Executive Summary

This document outlines the proposed new data architecture for the Counterfactual Regret Minimization (CFR) Poker project. The primary goal is to resolve persistent discrepancies between **Training** (CFR Solver) and **Evaluation** (Best Response/Public Tree) components, specifically regarding **Information Set Keys**.

The current system suffers from "Key Mismatch," where the dictionary keys generated during training (e.g., in `CFRSolver`) do not match those reconstructed during evaluation (e.g., `build_public_state_tree`). This is caused by inconsistent handling of history separators (`'|'`) and null values (`'None'` vs `None`).

**Core Proposal:** Centralize state representation into a single `KeyGenerator` authority and mandate **Simulation-Based Key Retrieval** instead of manual key construction.

---

## 2. Problem Analysis

### 2.1 The "Implicit vs. Explicit" Conflict
- **Training (`CFRSolver` / `Game`):** The `Game` class (e.g., `LeducHoldemGame`) automatically appends a round separator `'|'` to its internal `self.history` list during `step()`.
- **Evaluation (`PublicTreeBuilder`):** Manually attempts to reconstruct this history string. It frequently misses the exact logic for *when* `'|'` is added, or how `None` public cards are stringified.

### 2.2 Fragile Key Construction
Currently, the public tree builder manually constructs keys:
```python
# CURRENT BROKEN APPROACH in build_public_state_tree_v2.py
player0_info_sets = [(card, tuple(clean_history), 0) for card in available_cards]
```
This is brittle because it assumes the `Game` class's internal key generation logic never changes and relies on duplicating that logic imperfectly.

---

## 3. Proposed Architecture

### 3.1 Unified Information Set Key Schema

We will define a canonical structure for keys. All components must use this structure.

**Format:**
```python
(
    private_card: str,        # e.g., 'Js'
    public_cards: Tuple[str], # e.g., ('Qs', 'Kh') or ()
    history: Tuple[str],      # e.g., ('check', 'bet', 'call')
    player_id: int            # 0 or 1
)
```

**Rules:**
1.  **No Separators in History:** The `history` tuple contains **only** player actions (`'bet'`, `'check'`, `'fold'`, `'call'`). It explicitly **excludes** `'|'`, chance outcomes, or round markers. Round context is derived from `public_cards`.
2.  **Strict Public Cards:** Always a tuple. If no public cards, use empty tuple `()`. Never use string `'None'` or python `None`.
3.  **Atomic Actions:** Actions are strictly strings representing player choices.

### 3.2 The `KeyGenerator` Component

A new, stateless utility class (or module) responsible for generating keys from a `Game` state.

```python
# src/utils/key_generator.py (Pseudo-code)

class KeyGenerator:
    @staticmethod
    def get_info_set_key(game, player_id: int):
        # 1. Extract Private Card
        private_card = game.players[player_id].private_card
        
        # 2. Extract Public Cards (Normalized to tuple)
        if game.public_cards is None:
             public = ()
        elif isinstance(game.public_cards, str): # Handle single card case like Leduc
             public = (game.public_cards,)
        else:
             public = tuple(game.public_cards)
             
        # 3. Extract History (Filtered)
        # Filter out '|' or non-action strings if legacy game keeps them
        raw_history = game.history
        clean_history = tuple(a for a in raw_history if a in {'bet', 'check', 'call', 'fold'})
        
        return (private_card, public, clean_history, player_id)
```

---

## 4. Component Interaction Flow

### 4.1 Training (`CFRSolver`)
**Role:** Producer of the Strategy.
- **Change:** `CFRSolver` no longer calls `game.get_info_set_key()`. It calls `KeyGenerator.get_info_set_key(game, p)`.
- **Result:** The `average_strategy` dictionary is saved with the Canonical Key format.

### 4.2 Evaluation (`PublicTreeBuilder`)
**Role:** Constructor of the Public State Tree.
- **Change:** Stop manually constructing keys. Use **Simulation**.
- **Process:**
    1.  Traverse the public game tree (actions/chance).
    2.  At each node, spawn a `Game` instance (or use a lightweight state object).
    3.  Replay the history to reach the current state.
    4.  Iterate through all possible private cards for the player.
    5.  For each card, set it in the `Game` instance and call `KeyGenerator.get_info_set_key(game, p)`.
    6.  Store *that* key in the tree.

**Why this fixes the bug:**
The builder no longer "guesses" that the key is `(card, history, pid)`. It asks the `KeyGenerator`: "If the game was in this state, what would the key be?". Since the `CFRSolver` used the exact same `KeyGenerator`, the keys are guaranteed to match.

### 4.3 Best Response (`BestResponseAgent`)
**Role:** Consumer of Strategy and Tree.
- **Change:** Minimal. It loads the tree (which now contains correct keys) and the strategy (which now uses correct keys).
- **Lookup:** `strategy.get(key_from_tree)` will work 100% of the time without fallback logic or string patching.

---

## 5. Migration Plan

1.  **Create `src/utils/data_models.py`**: Implement `KeyGenerator`.
2.  **Update `CFRSolver`**: Inject `KeyGenerator` into the solver.
3.  **Update `PublicTreeBuilder`**: Refactor `get_info_sets_for_public_state` to instantiate a dummy Game, apply state, and query `KeyGenerator`.
4.  **Retrain Models**: Existing models (`.pkl.gz`) use the old messy keys. You must retrain them to generate new files with the canonical keys.

## 6. Diagram: Data Flow

```mermaid
graph TD
    subgraph "Training Phase"
        Game1[Game Instance] --> KG1[KeyGenerator]
        KG1 --> Solver[CFR Solver]
        Solver --> Strategy[Strategy Dict\n{Key: Probs}]
    end

    subgraph "Tree Building Phase"
        Node[Public Tree Node] --> Game2[Game Simulation]
        Game2 --> KG2[KeyGenerator]
        KG2 --> TreeKeys[Tree Node Keys]
    end

    subgraph "Evaluation Phase"
        TreeKeys --> Lookup
        Strategy --> Lookup
        Lookup{Match?} -- Yes --> Value[Compute BR]
    end
```
