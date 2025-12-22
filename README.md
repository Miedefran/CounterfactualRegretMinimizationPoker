## Introduction

Project to solve poker games using a CFR algorithm.

## Games

Unterstützte Poker-Varianten:
- `kuhn_case1`, `kuhn_case2`, `kuhn_case3`, `kuhn_case4` (Kuhn Poker)
- `leduc` (Leduc Hold'em)
- `rhode_island` (Rhode Island Hold'em)
- `twelve_card_poker` (Twelve Card Poker)
- `royal_holdem` (Royal Hold'em)
- `limit_holdem` (Limit Hold'em)

## Training

Create public state tree:
```bash
uv run src/evaluation/build_public_state_tree.py leduc
```

### Algorithmen
Standard ist **Vanilla CFR**. Optional kann **CFR+** oder **MCCFR** als drittes Argument angegeben werden:

```bash
# Vanilla CFR (default)
python training/train.py leduc 10000 cfr

# CFR+
python training/train.py leduc 10000 cfr_plus

# MCCFR
python training/train.py leduc 10000 mccfr
```

## Testing
Self-play evaluation:
```bash
python evaluation/self_play.py <game> <strategy_file> --games <num_games>
```

## Best Response / Exploitability

### 1) Public State Tree bauen
Für Leduc wird ein Tree mit **expliziten Chance-Knoten** zwischen den Runden verwendet:

```bash
python evaluation/build_public_state_tree.py leduc
```

Erzeugt u.a.:
- `evaluation/public_state_trees/leduc_public_tree_explicit_chance.pkl.gz`
- `evaluation/public_state_trees/leduc_public_tree_explicit_chance.txt`

### 2) Best Response ausführen
```bash
python evaluation/best_response_agent.py \
  --game leduc \
  --player 0 \
  --public-tree evaluation/public_state_trees/leduc_public_tree_explicit_chance.pkl.gz \
  --strategy <strategy_file>

python evaluation/best_response_agent.py \
  --game leduc \
  --player 1 \
  --public-tree evaluation/public_state_trees/leduc_public_tree_explicit_chance.pkl.gz \
  --strategy <strategy_file>
```
