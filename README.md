## Installation
```bash
pip install -e .
```

Games: `kuhn_case1`, `kuhn_case2`, `kuhn_case3`, `kuhn_case4`, `leduc`

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
