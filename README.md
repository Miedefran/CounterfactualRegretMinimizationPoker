## Installation

### Core-Projekt (ohne GUI)
```bash
pip install -e .
```

Installiert nur die Core-Dependencies (`pandas`).

### Mit GUI-Unterstützung
```bash
pip install -e .[gui]
```

Installiert zusätzlich: `PyQt6`, `Flask`, `requests` für die GUI-Funktionalität.

## Games

Unterstützte Poker-Varianten:
- `kuhn_case1`, `kuhn_case2`, `kuhn_case3`, `kuhn_case4` (Kuhn Poker)
- `leduc` (Leduc Hold'em)
- `rhode_island` (Rhode Island Hold'em)
- `twelve_card_poker` (Twelve Card Poker)
- `royal_holdem` (Royal Hold'em)
- `limit_holdem` (Limit Hold'em)

## Training
```bash
python training/train.py <game> <iterations>
```
## Testing
Nash Equilibrium verification:
```bash
python tests/test_nash_equilibrium.py <strategy_file>
```
Self-play evaluation:
```bash
python self_play.py <strategy_file> --games <num_games>
```
