#!/usr/bin/env bash
set -euo pipefail

# Profiling-Batch: schreibt pro Spiel eine TXT mit
# - GameTree nodes/infosets (falls vorhanden)
# - PublicTree states (+ breakdown)
# - average_strategy_infosets (aus dem Modell)
# - BR-Zeit für einen Spieler (player=0)
#
# Erwartung: Modelle + Public Trees existieren bereits.
# Ausführen z.B.:  bash profile_batch_eval.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

RUN=(uv run python src/evaluation/profile_game_metrics.py)

OUT_DIR="data/plots/chapter6/profiles"
mkdir -p "$OUT_DIR"

# Diese Pfade sollten zu train_batch_eval.sh passen (Iterationszahlen + Algorithmen).
# Du kannst hier jederzeit auf andere Algorithmen umstellen, solange die Datei existiert.

LEDUC_MODEL="data/models/leduc/cfr_plus_with_tree_no_pruning/10000/leduc_10000.pkl.gz"
TWELVE_MODEL="data/models/twelve_card_poker/cfr_plus_with_tree_no_pruning/1000/twelve_card_poker_1000.pkl.gz"
SMALL_ISLAND_MODEL="data/models/small_island_holdem/discounted_cfr_no_pruning/100/small_island_holdem_100.pkl.gz"

profile_one () {
  local game="$1"
  local model="$2"
  local out="$3"

  if [[ ! -f "$model" ]]; then
    echo "SKIP: $game (model not found: $model)"
    return 0
  fi

  echo "PROFILE: $game -> $out"
  "${RUN[@]}" --game "$game" --player 0 --strategy "$model" --output "$out"
}

profile_one "leduc" "$LEDUC_MODEL" "$OUT_DIR/leduc_profile.txt"
profile_one "twelve_card_poker" "$TWELVE_MODEL" "$OUT_DIR/twelve_card_profile.txt"
profile_one "small_island_holdem" "$SMALL_ISLAND_MODEL" "$OUT_DIR/small_island_profile.txt"

echo
echo "DONE. Reports liegen unter: $OUT_DIR"

