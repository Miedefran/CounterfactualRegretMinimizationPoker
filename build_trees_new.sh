#!/usr/bin/env bash
set -euo pipefail

# Baut nacheinander alle relevanten Trees NEU:
# - Game Trees:
#   - kuhn_case2: NOT suit-abstracted
#   - leduc, twelve_card_poker: BOTH (suit-abstracted + NOT suit-abstracted)
#   - small_island_holdem: wird NICHT als Object-GameTree gebaut (zu groß). Flat-Tree wird lazy beim Training gebaut.
# - Public State Trees:
#   - kuhn_case2, small_island_holdem: NOT suit-abstracted
#   - leduc, twelve_card_poker: BOTH (suit-abstracted + NOT suit-abstracted)
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

EXTRA_ARGS_NOT_ABSTRACTED=(--no-suit-abstraction)
EXTRA_ARGS_ABSTRACTED=(--abstract-suits)
if [[ "${USE_CACHE}" != "true" ]]; then
  EXTRA_ARGS_NOT_ABSTRACTED+=(--no-cache)
  EXTRA_ARGS_ABSTRACTED+=(--no-cache)
fi

# -----------------------------------------------------------------------------
# 0) Game Trees neu bauen
# -----------------------------------------------------------------------------
echo
echo "=== BUILD GAME TREES ==="
if [[ "${SKIP_EXISTING}" == "true" ]]; then
  # NOT abstracted games
  "${RUN_GAME_TREES[@]}" kuhn_case2 --no-suit-abstraction
  # Leduc / Twelve Card: build BOTH variants
  "${RUN_GAME_TREES[@]}" leduc twelve_card_poker --suit-abstraction
  "${RUN_GAME_TREES[@]}" leduc twelve_card_poker --no-suit-abstraction
else
  # NOT abstracted games
  "${RUN_GAME_TREES[@]}" kuhn_case2 --no-suit-abstraction --force
  # Leduc / Twelve Card: build BOTH variants
  "${RUN_GAME_TREES[@]}" leduc twelve_card_poker --suit-abstraction --force
  "${RUN_GAME_TREES[@]}" leduc twelve_card_poker --no-suit-abstraction --force
fi

# -----------------------------------------------------------------------------
# 1) Public State Trees neu bauen
# -----------------------------------------------------------------------------

# Liste der Spiele, für die wir Public Trees bauen wollen.
# Hinweis: Kuhn-Cases würden alle in dieselbe Datei `kuhn_public_tree_v2_NOT_abstracted.pkl.gz` schreiben.
GAMES=(
  kuhn_case2
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
  local abstract="$2" # "true" | "false"
  local save_name="$game"
  if [[ "$game" == kuhn_* ]]; then
    save_name="kuhn"
  fi
  if [[ "$abstract" == "true" ]]; then
    echo "data/trees/public_state_trees/${save_name}_public_tree_v2.pkl.gz"
  else
    echo "data/trees/public_state_trees/${save_name}_public_tree_v2_NOT_abstracted.pkl.gz"
  fi
}

build_one () {
  local game="$1"
  local abstract="$2" # "true" | "false"

  local out
  out="$(public_tree_path "$game" "$abstract")"

  if [[ "${SKIP_EXISTING}" == "true" && -f "$out" ]]; then
    echo "SKIP: $game abstract=${abstract} (exists: $out)"
    return 0
  fi

  echo
  echo "=== BUILD PUBLIC TREE: $game (abstract=${abstract}) ==="
  echo "Output: $out"
  if [[ "$abstract" == "true" ]]; then
    echo "Args: ${EXTRA_ARGS_ABSTRACTED[*]}"
    "${RUN_PUBLIC_TREES[@]}" "$game" "${EXTRA_ARGS_ABSTRACTED[@]}"
  else
    echo "Args: ${EXTRA_ARGS_NOT_ABSTRACTED[*]}"
    "${RUN_PUBLIC_TREES[@]}" "$game" "${EXTRA_ARGS_NOT_ABSTRACTED[@]}"
  fi
}

for game in "${GAMES[@]}"; do
  if [[ "$game" == "leduc" || "$game" == "twelve_card_poker" ]]; then
    # Build BOTH variants
    build_one "$game" "true"
    build_one "$game" "false"
  else
    # Default: NOT abstracted
    build_one "$game" "false"
  fi
done

echo
echo "DONE."
echo "- Game Trees:  data/trees/game_trees/{normal,abstracted}/"
echo "- Public Trees: data/trees/public_state_trees/"

