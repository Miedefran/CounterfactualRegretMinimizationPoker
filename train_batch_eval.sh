#!/usr/bin/env bash
set -euo pipefail

# Batch-Training Script für Kapitel 6 (Evaluation)
#
# Prinzip:
# - Trainiert viele Konfigurationen nacheinander (gut für „starken PC“)
# - Speichert automatisch Modelle + Best-Response-Verläufe (via --br-eval-schedule)
# - Defaults: alternating=true, pruning=false (nur wenn explizit "true" übergeben wird, ist es an)
# - Suit Abstraction ist für die Evaluation deaktiviert: --no-suit-abstraction

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

RUN=(uv run python src/training/train.py)

###############################################################################
# Konfiguration (bitte anpassen)
###############################################################################

# Leduc (Benchmark im Haupttext)
LEDUC_ITERS=10000
LEDUC_SCHEDULE="custom_v2"

# Twelve Card (Anhang; BR teuer)
TWELVE_ITERS=1000
TWELVE_SCHEDULE="very_very_large_games"

# Small Island Hold'em (Demonstrator; BR sehr teuer)
SMALL_ISLAND_ITERS=100
SMALL_ISLAND_SCHEDULE="small_island_schedule"

# Kuhn Poker (Anhang / Sanity) — Tree-Varianten
KUHN_GAME="kuhn_case2"
KUHN_ITERS=100000
KUHN_SCHEDULE="custom_v2"

# Welche Ablationen sollen laufen?
RUN_SIMULTANEOUS=true     # alternating-updates=false
RUN_PRUNING=true          # partial-pruning=true (nur ausgewählte Runs)

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
    --no-suit-abstraction \
    --partial-pruning false \
    "$@"
}

###############################################################################
# 1) Leduc Benchmark (Haupttext)
###############################################################################

# Basisruns (Defaults: alternating=true, pruning=true)
LEDUC_DYNAMIC_ALGOS=(cfr cfr_plus discounted_cfr)
LEDUC_TREE_ALGOS=(cfr_with_tree cfr_plus_with_tree discounted_cfr_with_tree)

for algo in "${LEDUC_DYNAMIC_ALGOS[@]}"; do
  train_one "leduc" "$LEDUC_ITERS" "$algo" "$LEDUC_SCHEDULE"
done

for algo in "${LEDUC_TREE_ALGOS[@]}"; do
  train_one "leduc" "$LEDUC_ITERS" "$algo" "$LEDUC_SCHEDULE"
done

# Simultaneous Ablation
if [[ "${RUN_SIMULTANEOUS}" == "true" ]]; then
  for algo in "${LEDUC_TREE_ALGOS[@]}"; do
    train_one "leduc" "$LEDUC_ITERS" "$algo" "$LEDUC_SCHEDULE" --alternating-updates false
  done
fi

# Pruning Ablation (Early-exit aktiv) — nur für die 3 Leduc Tree-Varianten
if [[ "${RUN_PRUNING}" == "true" ]]; then
  for algo in "${LEDUC_TREE_ALGOS[@]}"; do
    train_one "leduc" "$LEDUC_ITERS" "$algo" "$LEDUC_SCHEDULE" --partial-pruning true
  done
fi

echo
echo "=== PLOT: leduc ==="
bash plot_batch_eval.sh leduc

###############################################################################
# 2) Twelve Card (Anhang – nur repräsentativ)
###############################################################################

# Twelve Card: nur DCFR (Tree)
TWELVE_TREE_ALGOS=(discounted_cfr_with_tree)
for algo in "${TWELVE_TREE_ALGOS[@]}"; do
  train_one "twelve_card_poker" "$TWELVE_ITERS" "$algo" "$TWELVE_SCHEDULE"
done

echo
echo "=== PLOT: twelve ==="
bash plot_batch_eval.sh twelve

###############################################################################
# 3) Small Island Hold'em (Demonstrator – größtes gelöstes Spiel)
###############################################################################

# Small Island: nur DCFR (ohne Tree)
SMALL_ISLAND_ALGOS=(discounted_cfr)
for algo in "${SMALL_ISLAND_ALGOS[@]}"; do
  train_one "small_island_holdem" "$SMALL_ISLAND_ITERS" "$algo" "$SMALL_ISLAND_SCHEDULE"
done

echo
echo "=== PLOT: small ==="
bash plot_batch_eval.sh small

###############################################################################
# 4) Kuhn Poker (Tree; Sanity / Anhang)
###############################################################################

# Kuhn: alle 3 Tree-Varianten (CFR / CFR+ / DCFR), alternating=true, pruning=false
KUHN_TREE_ALGOS=(cfr_with_tree cfr_plus_with_tree discounted_cfr_with_tree)
for algo in "${KUHN_TREE_ALGOS[@]}"; do
  train_one "$KUHN_GAME" "$KUHN_ITERS" "$algo" "$KUHN_SCHEDULE"
done

echo
echo "=== PLOT: kuhn ==="
bash plot_batch_eval.sh kuhn

echo
echo "DONE. Modelle liegen unter data/models/<game>/<algorithm>/<iters>/"

