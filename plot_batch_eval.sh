#!/usr/bin/env bash
set -euo pipefail

# Batch-Plot Script für Kapitel 6 (Evaluation)
#
# Erwartung:
# - Die Trainings wurden vorher mit train_batch_eval.sh ausgeführt.
# - Pro Run existieren Dateien:
#   - data/models/<game>/<algo>/<iters>/<game>_<iters>_best_response.pkl.gz
#
# Dieses Script erzeugt Vergleichsplots:
# - Leduc: Exploitability vs Zeit (alt=true, pruning=false) über alle Runs mit/ohne Tree
# - Leduc: Exploitability vs Iteration (Tree-only vs Dynamic-only)
# - Leduc: Ablationsplots (alternating vs simultaneous) und (pruning vs no-pruning) — jeweils für Tree-Varianten
# - Twelve Card & Small Island: je Exploitability vs Iteration und vs Zeit (nur DCFR)

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

LEDUC_ITERS=10000
TWELVE_ITERS=1000
SMALL_ISLAND_ITERS=100
KUHN_GAME="kuhn_case2"
KUHN_ITERS=100000

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
# 1) Leduc
# -----------------------------------------------------------------------------

if should_plot "leduc"; then
  # 1.1 Exploitability vs Zeit: alle Leduc Trainings (mit/ohne Tree), alt=true, pruning=false
  LEDUC_TIME_ALL=(
    "$(br_file leduc cfr_no_pruning "$LEDUC_ITERS")"
    "$(br_file leduc cfr_plus_no_pruning "$LEDUC_ITERS")"
    "$(br_file leduc discounted_cfr_no_pruning "$LEDUC_ITERS")"
    "$(br_file leduc cfr_with_tree_no_pruning "$LEDUC_ITERS")"
    "$(br_file leduc cfr_plus_with_tree_no_pruning "$LEDUC_ITERS")"
    "$(br_file leduc discounted_cfr_with_tree_no_pruning "$LEDUC_ITERS")"
  )
  plot_time \
    "$OUT_DIR/leduc_time_all_alt_no_pruning.png" \
    "Leduc: Exploitability vs Zeit (alt=true, pruning=false) — mit/ohne Tree" \
    "${LEDUC_TIME_ALL[@]}"

  # 1.2 Exploitability vs Iteration: Tree-Varianten (3 Kurven)
  LEDUC_TREE_IT=(
    "$(br_file leduc cfr_with_tree_no_pruning "$LEDUC_ITERS")"
    "$(br_file leduc cfr_plus_with_tree_no_pruning "$LEDUC_ITERS")"
    "$(br_file leduc discounted_cfr_with_tree_no_pruning "$LEDUC_ITERS")"
  )
  plot_iter \
    "$OUT_DIR/leduc_iter_tree_compare.png" \
    "Leduc: Exploitability vs Iteration — Tree-Varianten (CFR/CFR+/DCFR), pruning=false" \
    "${LEDUC_TREE_IT[@]}"

  # 1.3 Exploitability vs Iteration: ohne Tree (3 Kurven)
  LEDUC_DYNAMIC_IT=(
    "$(br_file leduc cfr_no_pruning "$LEDUC_ITERS")"
    "$(br_file leduc cfr_plus_no_pruning "$LEDUC_ITERS")"
    "$(br_file leduc discounted_cfr_no_pruning "$LEDUC_ITERS")"
  )
  plot_iter \
    "$OUT_DIR/leduc_iter_dynamic_compare.png" \
    "Leduc: Exploitability vs Iteration — ohne Tree (CFR/CFR+/DCFR), pruning=false" \
    "${LEDUC_DYNAMIC_IT[@]}"

  # 1.4 Ablation: alternating vs simultaneous (Tree-Varianten)
  plot_iter \
    "$OUT_DIR/leduc_iter_alt_vs_sim_tree_cfr.png" \
    "Leduc (Tree): CFR alternating vs simultaneous (pruning=false)" \
    "$(br_file leduc cfr_with_tree_no_pruning "$LEDUC_ITERS")" \
    "$(br_file leduc cfr_with_tree_simultaneous_no_pruning "$LEDUC_ITERS")"

  plot_iter \
    "$OUT_DIR/leduc_iter_alt_vs_sim_tree_cfr_plus.png" \
    "Leduc (Tree): CFR+ alternating vs simultaneous (pruning=false)" \
    "$(br_file leduc cfr_plus_with_tree_no_pruning "$LEDUC_ITERS")" \
    "$(br_file leduc cfr_plus_with_tree_simultaneous_no_pruning "$LEDUC_ITERS")"

  plot_iter \
    "$OUT_DIR/leduc_iter_alt_vs_sim_tree_dcfr.png" \
    "Leduc (Tree): DCFR alternating vs simultaneous (pruning=false)" \
    "$(br_file leduc discounted_cfr_with_tree_no_pruning "$LEDUC_ITERS")" \
    "$(br_file leduc discounted_cfr_with_tree_simultaneous_no_pruning "$LEDUC_ITERS")"

  # 1.5 Ablation: pruning vs no-pruning (Tree-Varianten)
  plot_iter \
    "$OUT_DIR/leduc_iter_pruning_vs_no_pruning_tree_cfr.png" \
    "Leduc (Tree): CFR pruning=true vs pruning=false" \
    "$(br_file leduc cfr_with_tree "$LEDUC_ITERS")" \
    "$(br_file leduc cfr_with_tree_no_pruning "$LEDUC_ITERS")"

  plot_time \
    "$OUT_DIR/leduc_time_pruning_vs_no_pruning_tree_cfr.png" \
    "Leduc (Tree): CFR Exploitability vs Zeit — pruning=true vs pruning=false" \
    "$(br_file leduc cfr_with_tree "$LEDUC_ITERS")" \
    "$(br_file leduc cfr_with_tree_no_pruning "$LEDUC_ITERS")"

  plot_iter \
    "$OUT_DIR/leduc_iter_pruning_vs_no_pruning_tree_cfr_plus.png" \
    "Leduc (Tree): CFR+ pruning=true vs pruning=false" \
    "$(br_file leduc cfr_plus_with_tree "$LEDUC_ITERS")" \
    "$(br_file leduc cfr_plus_with_tree_no_pruning "$LEDUC_ITERS")"

  plot_time \
    "$OUT_DIR/leduc_time_pruning_vs_no_pruning_tree_cfr_plus.png" \
    "Leduc (Tree): CFR+ Exploitability vs Zeit — pruning=true vs pruning=false" \
    "$(br_file leduc cfr_plus_with_tree "$LEDUC_ITERS")" \
    "$(br_file leduc cfr_plus_with_tree_no_pruning "$LEDUC_ITERS")"

  plot_iter \
    "$OUT_DIR/leduc_iter_pruning_vs_no_pruning_tree_dcfr.png" \
    "Leduc (Tree): DCFR pruning=true vs pruning=false" \
    "$(br_file leduc discounted_cfr_with_tree "$LEDUC_ITERS")" \
    "$(br_file leduc discounted_cfr_with_tree_no_pruning "$LEDUC_ITERS")"

  plot_time \
    "$OUT_DIR/leduc_time_pruning_vs_no_pruning_tree_dcfr.png" \
    "Leduc (Tree): DCFR Exploitability vs Zeit — pruning=true vs pruning=false" \
    "$(br_file leduc discounted_cfr_with_tree "$LEDUC_ITERS")" \
    "$(br_file leduc discounted_cfr_with_tree_no_pruning "$LEDUC_ITERS")"
fi

# -----------------------------------------------------------------------------
# 2) Twelve Card (Tree-Varianten)
# -----------------------------------------------------------------------------

if should_plot "twelve"; then
  TWELVE_FILES=(
    "$(br_file twelve_card_poker discounted_cfr_with_tree_no_pruning "$TWELVE_ITERS")"
  )

  plot_iter \
    "$OUT_DIR/twelve_card_iter_tree_compare.png" \
    "Twelve Card: Exploitability vs Iteration — DCFR with tree, pruning=false" \
    "${TWELVE_FILES[@]}"

  plot_time \
    "$OUT_DIR/twelve_card_time_tree_compare.png" \
    "Twelve Card: Exploitability vs Zeit — DCFR with tree, pruning=false" \
    "${TWELVE_FILES[@]}"
fi

# -----------------------------------------------------------------------------
# 3) Small Island Hold'em (ohne Tree; CFR+ vs DCFR)
# -----------------------------------------------------------------------------

if should_plot "small"; then
  SMALL_FILES=(
    "$(br_file small_island_holdem discounted_cfr_no_pruning "$SMALL_ISLAND_ITERS")"
  )

  plot_iter \
    "$OUT_DIR/small_island_iter_compare.png" \
    "Small Island: Exploitability vs Iteration — DCFR, pruning=false" \
    "${SMALL_FILES[@]}"

  plot_time \
    "$OUT_DIR/small_island_time_compare.png" \
    "Small Island: Exploitability vs Zeit — DCFR, pruning=false" \
    "${SMALL_FILES[@]}"
fi

# -----------------------------------------------------------------------------
# 4) Kuhn Poker (Tree; 3 Varianten)
# -----------------------------------------------------------------------------

if should_plot "kuhn"; then
  KUHN_FILES=(
    "$(br_file "$KUHN_GAME" cfr_with_tree_no_pruning "$KUHN_ITERS")"
    "$(br_file "$KUHN_GAME" cfr_plus_with_tree_no_pruning "$KUHN_ITERS")"
    "$(br_file "$KUHN_GAME" discounted_cfr_with_tree_no_pruning "$KUHN_ITERS")"
  )

  plot_iter \
    "$OUT_DIR/kuhn_iter_tree_compare.png" \
    "Kuhn (Tree): Exploitability vs Iteration — CFR/CFR+/DCFR, pruning=false" \
    "${KUHN_FILES[@]}"

  plot_time \
    "$OUT_DIR/kuhn_time_tree_compare.png" \
    "Kuhn (Tree): Exploitability vs Zeit — CFR/CFR+/DCFR, pruning=false" \
    "${KUHN_FILES[@]}"
fi

echo
echo "DONE. Plots liegen unter: $OUT_DIR"

