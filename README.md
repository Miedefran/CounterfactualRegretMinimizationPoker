## Kurzüberblick

Wesentliche Workflows:
- **Trees bauen** (Game Trees / Public State Trees)
- **Strategie trainieren** (`src/training/train.py`)
- **Evaluieren via Best Response** (während des Trainings per `--br-eval-schedule`)

Alle Commands sind so gedacht:

```bash
uv run python ...
```

## Unterstützte Spiele

- `kuhn_case1`, `kuhn_case2`, `kuhn_case3`, `kuhn_case4`
- `leduc`
- `rhode_island`
- `twelve_card_poker`
- `royal_holdem`
- `small_island_holdem`
- `limit_holdem`

## 1) Trees bauen

### Public State Tree (für Best Response)

Wird automatisch gebaut/geladen, sobald du `--br-eval-schedule` beim Training aktivierst.
Manuell:

```bash
# Default: leduc/twelve sind suit-abstracted, sonst nicht
uv run python src/evaluation/build_public_state_tree_v2.py leduc

# explizit NOT abstracted
uv run python src/evaluation/build_public_state_tree_v2.py leduc --no-suit-abstraction
```

Ausgabe liegt unter `data/trees/public_state_trees/`.

### Game Trees (optional, für Tree-Solver)

Tree-Solver bauen/laden Game Trees automatisch. Wenn du bewusst vorab bauen willst:

```bash
uv run python src/training/build_game_trees_cli.py leduc twelve_card_poker --suit-abstraction
uv run python src/training/build_game_trees_cli.py leduc twelve_card_poker --no-suit-abstraction
```

Ausgabe liegt unter `data/trees/game_trees/{abstracted|normal}/`.

## 2) Training (Strategie lernen)

```bash
uv run python src/training/train.py <game> <iterations> <algorithm> [flags...]
```

Beispiel (Leduc, Flat-Tree CFR+):

```bash
uv run python src/training/train.py leduc 100000 cfr_plus_with_flat_tree \
  --br-eval-schedule standard \
  --early-stop-exploitability-mb 1
```

Modelle werden unter `data/models/...` gespeichert.

## 3) Evaluation via Best Response (während des Trainings)

BR/Eploitability läuft **während des Trainings**, wenn du z.B. setzt:

```bash
--br-eval-schedule custom_v2
```

Dabei werden pro Eval-Punkt gespeichert:
- `..._best_response.png` (Plot)
- `..._best_response.pkl.gz` (Rohdaten)

## Flags (kurz erklärt)

- **`--br-eval-schedule <name|int|json>`**: aktiviert BR-Evaluation während Training  
  - `<int>`: fester Abstand (z.B. `100`)  
  - `<name>`: Eintrag aus `config/br_eval_schedules.json` (z.B. `standard`, `low_density`)
- **`--early-stop-exploitability-mb <float>`**: bricht ab, sobald eine BR-Eval Exploitability < Schwelle (mb/g) ist  
  - funktioniert nur, wenn `--br-eval-schedule` aktiv ist
- **`--no-suit-abstraction`**: deaktiviert Suit-Abstraction für `leduc` und `twelve_card_poker`  
  - Default: **aktiv** für diese beiden Spiele
- **`--alternating-updates true|false`**: CFR-Update-Schema  
  - `true`: alternierend (P0 dann P1)  
  - `false`: simultan
- **`--partial-pruning true|false`**: kleines Early-Exit/Skip bei Reach=0 (Performance-Option)
- **`--tensor-algorithm cfr|cfr_plus`**: nur für `tensor_cfr`
- **`--dcfr-alpha/--dcfr-beta/--dcfr-gamma`**: Parameter für `discounted_cfr*`
