#!/usr/bin/env bash
set -euo pipefail

# Batch-Plot Script für Kapitel 6 (Evaluation)
#
# Erwartung:
# - Die Trainings wurden vorher mit train_batch_eval.sh ausgeführt.
# - Pro Run existieren Dateien:
#   - data/models/<game>/<algo>/<iters>/<game>_<iters>_best_response.pkl.gz
#
# Dieses Script erzeugt Vergleichsplots passend zu `train_batch_eval.sh`:
# - kuhn_case2 (50k): dynamic/tree/flat — CFR (alt+sim), CFR+, DCFR
# - leduc (abstracted, 50k): dynamic/tree/flat — CFR (alt+sim), CFR+, DCFR
# - twelve_card_poker (abstracted, 10k): tree/flat — CFR (alt), CFR+, DCFR
# - small_island_holdem (1k): flat — DCFR

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PLOT=(uv run python src/plot_best_response.py)

OUT_DIR="data/plots/chapter6"
mkdir -p "$OUT_DIR"

# Optional: nur einen Abschnitt plotten (damit man nach jedem Trainingsblock plotten kann)
# Usage:
#   bash plot_batch_eval.sh              # all
#   bash plot_batch_eval.sh leduc
#   bash plot_batch_eval.sh twelve
#   bash plot_batch_eval.sh small
#   bash plot_batch_eval.sh kuhn
SECTION="${1:-all}"

# -----------------------------------------------------------------------------
# Konfiguration (muss zu train_batch_eval.sh passen)
# -----------------------------------------------------------------------------

LEDUC_ITERS=50000
TWELVE_ITERS=10000
SMALL_ISLAND_ITERS=1000
KUHN_GAME="kuhn_case2"
KUHN_ITERS=50000

# -----------------------------------------------------------------------------
# Helper: Best-Response-Dateipfad
# -----------------------------------------------------------------------------

br_file () {
  local game="$1"
  local algo="$2"
  local iters="$3"
  if [[ "$game" == kuhn_case* ]]; then
    local case_name="${game#kuhn_}" # e.g. case2
    echo "data/models/kuhn/${case_name}/${algo}/${iters}/${game}_${iters}_best_response.pkl.gz"
  else
    echo "data/models/${game}/${algo}/${iters}/${game}_${iters}_best_response.pkl.gz"
  fi
}

plot_iter () {
  local output="$1"
  local title="$2"
  shift 2
  "${PLOT[@]}" --output "$output" --title "$title" "$@"
}

plot_time () {
  local output="$1"
  local title="$2"
  shift 2
  "${PLOT[@]}" --time --output "$output" --title "$title" "$@"
}

should_plot () {
  local want="$1"
  [[ "$SECTION" == "all" || "$SECTION" == "$want" ]]
}

# -----------------------------------------------------------------------------
# 1) Kuhn Case 2 (NOT abstracted)
# -----------------------------------------------------------------------------

if should_plot "kuhn"; then
  echo
  echo "=== PLOT: kuhn_case2 ==="

  # CFR alternating: dynamic vs tree vs flat
  plot_iter \
    "$OUT_DIR/kuhn_case2_iter_cfr_alt_compare.png" \
    "Kuhn Case 2: CFR alternating — dynamic vs tree vs flat (pruning=false)" \
    "$(br_file "$KUHN_GAME" cfr_no_pruning "$KUHN_ITERS")" \
    "$(br_file "$KUHN_GAME" cfr_with_tree_no_pruning "$KUHN_ITERS")" \
    "$(br_file "$KUHN_GAME" cfr_with_flat_tree_no_pruning "$KUHN_ITERS")"

  plot_time \
    "$OUT_DIR/kuhn_case2_time_cfr_alt_compare.png" \
    "Kuhn Case 2: CFR alternating — dynamic vs tree vs flat (pruning=false)" \
    "$(br_file "$KUHN_GAME" cfr_no_pruning "$KUHN_ITERS")" \
    "$(br_file "$KUHN_GAME" cfr_with_tree_no_pruning "$KUHN_ITERS")" \
    "$(br_file "$KUHN_GAME" cfr_with_flat_tree_no_pruning "$KUHN_ITERS")"

  # CFR simultaneous: dynamic vs tree vs flat
  plot_iter \
    "$OUT_DIR/kuhn_case2_iter_cfr_sim_compare.png" \
    "Kuhn Case 2: CFR simultaneous — dynamic vs tree vs flat (pruning=false)" \
    "$(br_file "$KUHN_GAME" cfr_simultaneous_no_pruning "$KUHN_ITERS")" \
    "$(br_file "$KUHN_GAME" cfr_with_tree_simultaneous_no_pruning "$KUHN_ITERS")" \
    "$(br_file "$KUHN_GAME" cfr_with_flat_tree_simultaneous_no_pruning "$KUHN_ITERS")"

  # CFR+
  plot_iter \
    "$OUT_DIR/kuhn_case2_iter_cfr_plus_compare.png" \
    "Kuhn Case 2: CFR+ — dynamic vs tree vs flat (pruning=false)" \
    "$(br_file "$KUHN_GAME" cfr_plus_no_pruning "$KUHN_ITERS")" \
    "$(br_file "$KUHN_GAME" cfr_plus_with_tree_no_pruning "$KUHN_ITERS")" \
    "$(br_file "$KUHN_GAME" cfr_plus_with_flat_tree_no_pruning "$KUHN_ITERS")"

  # DCFR
  plot_iter \
    "$OUT_DIR/kuhn_case2_iter_dcfr_compare.png" \
    "Kuhn Case 2: DCFR — dynamic vs tree vs flat (pruning=false)" \
    "$(br_file "$KUHN_GAME" discounted_cfr_no_pruning "$KUHN_ITERS")" \
    "$(br_file "$KUHN_GAME" discounted_cfr_with_tree_no_pruning "$KUHN_ITERS")" \
    "$(br_file "$KUHN_GAME" discounted_cfr_with_flat_tree_no_pruning "$KUHN_ITERS")"
fi

# -----------------------------------------------------------------------------
# 2) Leduc (abstracted)
# -----------------------------------------------------------------------------

if should_plot "leduc"; then
  echo
  echo "=== PLOT: leduc (abstracted) ==="

  # CFR alternating: dynamic vs tree vs flat
  plot_iter \
    "$OUT_DIR/leduc_abs_iter_cfr_alt_compare.png" \
    "Leduc (abstracted): CFR alternating — dynamic vs tree vs flat (pruning=false)" \
    "$(br_file leduc cfr_no_pruning_abstracted "$LEDUC_ITERS")" \
    "$(br_file leduc cfr_with_tree_no_pruning_abstracted "$LEDUC_ITERS")" \
    "$(br_file leduc cfr_with_flat_tree_no_pruning_abstracted "$LEDUC_ITERS")"

  # CFR simultaneous: dynamic vs tree vs flat
  plot_iter \
    "$OUT_DIR/leduc_abs_iter_cfr_sim_compare.png" \
    "Leduc (abstracted): CFR simultaneous — dynamic vs tree vs flat (pruning=false)" \
    "$(br_file leduc cfr_simultaneous_no_pruning_abstracted "$LEDUC_ITERS")" \
    "$(br_file leduc cfr_with_tree_simultaneous_no_pruning_abstracted "$LEDUC_ITERS")" \
    "$(br_file leduc cfr_with_flat_tree_simultaneous_no_pruning_abstracted "$LEDUC_ITERS")"

  # CFR+
  plot_iter \
    "$OUT_DIR/leduc_abs_iter_cfr_plus_compare.png" \
    "Leduc (abstracted): CFR+ — dynamic vs tree vs flat (pruning=false)" \
    "$(br_file leduc cfr_plus_no_pruning_abstracted "$LEDUC_ITERS")" \
    "$(br_file leduc cfr_plus_with_tree_no_pruning_abstracted "$LEDUC_ITERS")" \
    "$(br_file leduc cfr_plus_with_flat_tree_no_pruning_abstracted "$LEDUC_ITERS")"

  # DCFR
  plot_iter \
    "$OUT_DIR/leduc_abs_iter_dcfr_compare.png" \
    "Leduc (abstracted): DCFR — dynamic vs tree vs flat (pruning=false)" \
    "$(br_file leduc discounted_cfr_no_pruning_abstracted "$LEDUC_ITERS")" \
    "$(br_file leduc discounted_cfr_with_tree_no_pruning_abstracted "$LEDUC_ITERS")" \
    "$(br_file leduc discounted_cfr_with_flat_tree_no_pruning_abstracted "$LEDUC_ITERS")"
fi

# -----------------------------------------------------------------------------
# 3) Twelve Card Poker (abstracted)
# -----------------------------------------------------------------------------

if should_plot "twelve"; then
  echo
  echo "=== PLOT: twelve_card_poker (abstracted) ==="

  plot_iter \
    "$OUT_DIR/twelve_abs_iter_cfr_tree_vs_flat.png" \
    "Twelve Card (abstracted): CFR alternating — tree vs flat (pruning=false)" \
    "$(br_file twelve_card_poker cfr_with_tree_no_pruning_abstracted "$TWELVE_ITERS")" \
    "$(br_file twelve_card_poker cfr_with_flat_tree_no_pruning_abstracted "$TWELVE_ITERS")"

  plot_iter \
    "$OUT_DIR/twelve_abs_iter_cfr_plus_tree_vs_flat.png" \
    "Twelve Card (abstracted): CFR+ — tree vs flat (pruning=false)" \
    "$(br_file twelve_card_poker cfr_plus_with_tree_no_pruning_abstracted "$TWELVE_ITERS")" \
    "$(br_file twelve_card_poker cfr_plus_with_flat_tree_no_pruning_abstracted "$TWELVE_ITERS")"

  plot_iter \
    "$OUT_DIR/twelve_abs_iter_dcfr_tree_vs_flat.png" \
    "Twelve Card (abstracted): DCFR — tree vs flat (pruning=false)" \
    "$(br_file twelve_card_poker discounted_cfr_with_tree_no_pruning_abstracted "$TWELVE_ITERS")" \
    "$(br_file twelve_card_poker discounted_cfr_with_flat_tree_no_pruning_abstracted "$TWELVE_ITERS")"
fi

# -----------------------------------------------------------------------------
# 4) Small Island Hold'em (NOT abstracted)
# -----------------------------------------------------------------------------

if should_plot "small"; then
  echo
  echo "=== PLOT: small_island_holdem ==="
  plot_iter \
    "$OUT_DIR/small_island_iter_dcfr_flat.png" \
    "Small Island: DCFR flat-tree (pruning=false)" \
    "$(br_file small_island_holdem discounted_cfr_with_flat_tree_no_pruning "$SMALL_ISLAND_ITERS")"

  plot_time \
    "$OUT_DIR/small_island_time_dcfr_flat.png" \
    "Small Island: DCFR flat-tree (pruning=false)" \
    "$(br_file small_island_holdem discounted_cfr_with_flat_tree_no_pruning "$SMALL_ISLAND_ITERS")"
fi

echo
echo "DONE. Plots liegen unter: $OUT_DIR"

