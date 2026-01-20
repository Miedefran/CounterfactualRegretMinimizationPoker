#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

RUN=(uv run python src/training/train.py)

###############################################################################
# Konfiguration
###############################################################################

# Kuhn (nicht abstracted)
KUHN_GAME="kuhn_case2"
KUHN_ITERS=1
KUHN_SCHEDULE="custom_v2"

# Leduc (suit abstracted ist Default in train.py)
LEDUC_ITERS=1
LEDUC_SCHEDULE="custom_v2"

# Twelve Card (suit abstracted ist Default in train.py)
TWELVE_ITERS=1
TWELVE_SCHEDULE="custom_v2"

# Small Island (nicht abstracted)
SMALL_ISLAND_ITERS=1
SMALL_ISLAND_SCHEDULE="very_very_large_games"

###############################################################################
# Helper
###############################################################################

train_one () {
  local game="$1"
  local iters="$2"
  local schedule="$3"
  shift 3

  echo
  echo "=== TRAIN (flat CFR): game=${game} iters=${iters} schedule=${schedule} extra=[$*] ==="

  "${RUN[@]}" "$game" "$iters" cfr_with_flat_tree \
    --br-eval-schedule "$schedule" \
    --alternating-updates true \
    --partial-pruning false \
    "$@"
}

###############################################################################
# Runs
###############################################################################

echo
echo "=== FLAT CFR RUNS ==="

# Kuhn: explizit NICHT abstracted
train_one "$KUHN_GAME" "$KUHN_ITERS" "$KUHN_SCHEDULE" --no-suit-abstraction

# Leduc: suit abstracted (Default)
train_one "leduc" "$LEDUC_ITERS" "$LEDUC_SCHEDULE"

# Twelve Card: suit abstracted (Default)
train_one "twelve_card_poker" "$TWELVE_ITERS" "$TWELVE_SCHEDULE"

# Small Island: explizit NICHT abstracted
train_one "small_island_holdem" "$SMALL_ISLAND_ITERS" "$SMALL_ISLAND_SCHEDULE" --no-suit-abstraction

echo
echo "DONE."
