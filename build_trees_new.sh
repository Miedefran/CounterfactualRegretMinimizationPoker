#!/usr/bin/env bash
set -euo pipefail

# Baut nacheinander alle relevanten Trees NEU:
# - Game Trees (nur leduc & twelve_card_poker, NOT suit-abstracted)
# - Public State Trees (leduc, twelve_card_poker, small_island_holdem, NOT suit-abstracted)
#
# Default: alles neu bauen (auch wenn Dateien existieren), um alte/falsche Trees zu vermeiden.
#
# Ausführen:
#   bash build_trees_new.sh
#
# Optional:
#   SKIP_EXISTING=true bash build_trees_new.sh    # vorhandene Trees überspringen

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

RUN_PUBLIC_TREES=(uv run python src/evaluation/build_public_state_tree_v2.py)
RUN_GAME_TREES=(uv run python src/training/build_game_trees_cli.py)

# Default: alles neu bauen (auch wenn Dateien existieren), um alte/falsche Trees zu vermeiden.
# Wenn true, werden bereits vorhandene Trees übersprungen.
SKIP_EXISTING="${SKIP_EXISTING:-false}"

# Cache im Public-Tree-Builder aktiv lassen (default). Setze USE_CACHE=false um --no-cache zu nutzen.
USE_CACHE="${USE_CACHE:-true}"

EXTRA_ARGS=(--no-suit-abstraction)
if [[ "${USE_CACHE}" != "true" ]]; then
  EXTRA_ARGS+=(--no-cache)
fi

# -----------------------------------------------------------------------------
# 0) Game Trees neu bauen (nur leduc & twelve_card_poker, NOT abstracted)
# -----------------------------------------------------------------------------
echo
echo "=== BUILD GAME TREES (leduc, twelve_card_poker) ==="
if [[ "${SKIP_EXISTING}" == "true" ]]; then
  "${RUN_GAME_TREES[@]}" leduc twelve_card_poker --no-suit-abstraction
else
  "${RUN_GAME_TREES[@]}" leduc twelve_card_poker --no-suit-abstraction --force
fi

# -----------------------------------------------------------------------------
# 1) Public State Trees neu bauen
# -----------------------------------------------------------------------------

# Liste der Spiele, für die wir Public Trees bauen wollen.
# Hinweis: Kuhn-Cases würden alle in dieselbe Datei `kuhn_public_tree_v2_NOT_abstracted.pkl.gz` schreiben.
GAMES=(
  leduc
  twelve_card_poker
  small_island_holdem
  # optional / groß:
  # rhode_island
  # royal_holdem
  # limit_holdem
  # optional:
  # kuhn_case2
)

public_tree_path () {
  local game="$1"
  local save_name="$game"
  if [[ "$game" == kuhn_* ]]; then
    save_name="kuhn"
  fi
  echo "data/trees/public_state_trees/${save_name}_public_tree_v2_NOT_abstracted.pkl.gz"
}

build_one () {
  local game="$1"
  local out
  out="$(public_tree_path "$game")"

  if [[ "${SKIP_EXISTING}" == "true" && -f "$out" ]]; then
    echo "SKIP: $game (exists: $out)"
    return 0
  fi

  echo
  echo "=== BUILD PUBLIC TREE: $game ==="
  echo "Output: $out"
  echo "Args: ${EXTRA_ARGS[*]}"
  "${RUN_PUBLIC_TREES[@]}" "$game" "${EXTRA_ARGS[@]}"
}

for game in "${GAMES[@]}"; do
  build_one "$game"
done

echo
echo "DONE."
echo "- Game Trees:  data/trees/game_trees/normal/"
echo "- Public Trees: data/trees/public_state_trees/"

