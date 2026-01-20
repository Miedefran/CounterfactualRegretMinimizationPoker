#!/usr/bin/env bash
set -euo pipefail

# Batch-Training Script für Kapitel 6 (Evaluation)
#
# Prinzip:
# - Trainiert viele Konfigurationen nacheinander (gut für „starken PC“)
# - Speichert automatisch Modelle + Best-Response-Verläufe (via --br-eval-schedule)
# - Defaults: alternating=true, pruning=false (nur wenn explizit "true" übergeben wird, ist es an)
# - Suit Abstraction:
#   - Leduc/Twelve: laufen hier als suit-abstracted (Default in train.py)
#   - NOT-abstracted Varianten werden explizit mit --no-suit-abstraction gebaut/gelaufen

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

RUN=(uv run python src/training/train.py)

###############################################################################
# Konfiguration (bitte anpassen)
###############################################################################

# Leduc (Benchmark im Haupttext)
LEDUC_ITERS=50000
LEDUC_SCHEDULE="custom_v2"

# Twelve Card (Anhang; BR teuer)
TWELVE_ITERS=10000
TWELVE_SCHEDULE="very_very_large_games"

# Small Island Hold'em (Demonstrator)
SMALL_ISLAND_ITERS=1000
SMALL_ISLAND_SCHEDULE="small_island_schedule"

# Kuhn Poker
KUHN_GAME="kuhn_case2"
KUHN_ITERS=50000
KUHN_SCHEDULE="custom_v2"

###############################################################################
# Helper
###############################################################################

train_one () {
  local game="$1"
  local iters="$2"
  local algo="$3"
  local schedule="$4"
  shift 4

  echo
  echo "=== TRAIN: game=${game} iters=${iters} algo=${algo} schedule=${schedule} extra=[$*] ==="

  "${RUN[@]}" "$game" "$iters" "$algo" \
    --br-eval-schedule "$schedule" \
    --partial-pruning false \
    "$@"
}

###############################################################################
# 1) Kuhn Case 2 (50k)
###############################################################################

echo
echo "=== KUHN_CASE2 (NOT abstracted): dynamic/tree/flat — CFR alt+sim, CFR+, DCFR ==="

# Dynamisch (ohne Tree)
train_one "$KUHN_GAME" "$KUHN_ITERS" cfr "$KUHN_SCHEDULE" --no-suit-abstraction --alternating-updates true
train_one "$KUHN_GAME" "$KUHN_ITERS" cfr "$KUHN_SCHEDULE" --no-suit-abstraction --alternating-updates false
train_one "$KUHN_GAME" "$KUHN_ITERS" cfr_plus "$KUHN_SCHEDULE" --no-suit-abstraction
train_one "$KUHN_GAME" "$KUHN_ITERS" discounted_cfr "$KUHN_SCHEDULE" --no-suit-abstraction

# Normal Tree
train_one "$KUHN_GAME" "$KUHN_ITERS" cfr_with_tree "$KUHN_SCHEDULE" --no-suit-abstraction --alternating-updates true
train_one "$KUHN_GAME" "$KUHN_ITERS" cfr_with_tree "$KUHN_SCHEDULE" --no-suit-abstraction --alternating-updates false
train_one "$KUHN_GAME" "$KUHN_ITERS" cfr_plus_with_tree "$KUHN_SCHEDULE" --no-suit-abstraction
train_one "$KUHN_GAME" "$KUHN_ITERS" discounted_cfr_with_tree "$KUHN_SCHEDULE" --no-suit-abstraction

# Flat Tree
train_one "$KUHN_GAME" "$KUHN_ITERS" cfr_with_flat_tree "$KUHN_SCHEDULE" --no-suit-abstraction --alternating-updates true
train_one "$KUHN_GAME" "$KUHN_ITERS" cfr_with_flat_tree "$KUHN_SCHEDULE" --no-suit-abstraction --alternating-updates false
train_one "$KUHN_GAME" "$KUHN_ITERS" cfr_plus_with_flat_tree "$KUHN_SCHEDULE" --no-suit-abstraction
train_one "$KUHN_GAME" "$KUHN_ITERS" discounted_cfr_with_flat_tree "$KUHN_SCHEDULE" --no-suit-abstraction

###############################################################################
# 2) Leduc (abstracted, 50k)
###############################################################################

echo
echo "=== LEDUC (abstracted): dynamic/tree/flat — CFR alt+sim, CFR+, DCFR ==="

# Dynamisch (ohne Tree)
train_one "leduc" "$LEDUC_ITERS" cfr "$LEDUC_SCHEDULE" --alternating-updates true
train_one "leduc" "$LEDUC_ITERS" cfr "$LEDUC_SCHEDULE" --alternating-updates false
train_one "leduc" "$LEDUC_ITERS" cfr_plus "$LEDUC_SCHEDULE"
train_one "leduc" "$LEDUC_ITERS" discounted_cfr "$LEDUC_SCHEDULE"

# Normal Tree
train_one "leduc" "$LEDUC_ITERS" cfr_with_tree "$LEDUC_SCHEDULE" --alternating-updates true
train_one "leduc" "$LEDUC_ITERS" cfr_with_tree "$LEDUC_SCHEDULE" --alternating-updates false
train_one "leduc" "$LEDUC_ITERS" cfr_plus_with_tree "$LEDUC_SCHEDULE"
train_one "leduc" "$LEDUC_ITERS" discounted_cfr_with_tree "$LEDUC_SCHEDULE"

# Flat Tree
train_one "leduc" "$LEDUC_ITERS" cfr_with_flat_tree "$LEDUC_SCHEDULE" --alternating-updates true
train_one "leduc" "$LEDUC_ITERS" cfr_with_flat_tree "$LEDUC_SCHEDULE" --alternating-updates false
train_one "leduc" "$LEDUC_ITERS" cfr_plus_with_flat_tree "$LEDUC_SCHEDULE"
train_one "leduc" "$LEDUC_ITERS" discounted_cfr_with_flat_tree "$LEDUC_SCHEDULE"

###############################################################################
# 3) Twelve Card Poker (abstracted, 10k)
###############################################################################

echo
echo "=== TWELVE_CARD_POKER (abstracted): flat+tree — CFR alt, CFR+, DCFR ==="

# Normal Tree
train_one "twelve_card_poker" "$TWELVE_ITERS" cfr_with_tree "$TWELVE_SCHEDULE" --alternating-updates true
train_one "twelve_card_poker" "$TWELVE_ITERS" cfr_plus_with_tree "$TWELVE_SCHEDULE"
train_one "twelve_card_poker" "$TWELVE_ITERS" discounted_cfr_with_tree "$TWELVE_SCHEDULE"

# Flat Tree
train_one "twelve_card_poker" "$TWELVE_ITERS" cfr_with_flat_tree "$TWELVE_SCHEDULE" --alternating-updates true
train_one "twelve_card_poker" "$TWELVE_ITERS" cfr_plus_with_flat_tree "$TWELVE_SCHEDULE"
train_one "twelve_card_poker" "$TWELVE_ITERS" discounted_cfr_with_flat_tree "$TWELVE_SCHEDULE"

###############################################################################
# 4) Small Island Hold'em (NOT abstracted, 2k)
###############################################################################

echo
echo "=== SMALL_ISLAND_HOLDEM (NOT abstracted): flat — DCFR ==="
train_one "small_island_holdem" "$SMALL_ISLAND_ITERS" discounted_cfr_with_flat_tree "$SMALL_ISLAND_SCHEDULE" --no-suit-abstraction

echo
echo "DONE. Modelle liegen unter data/models/<game>/<algorithm>/<iters>/"

