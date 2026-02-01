"""
Löscht alle Plot-Artefakte unter data/plots/.

Nutzung:
  uv run python src/training/delete_all_plots.py

Optional:
  uv run python src/training/delete_all_plots.py --dry-run
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Delete all plots in data/plots/")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Nur anzeigen, was gelöscht würde (nichts wird gelöscht).",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    plots_dir = repo_root / "data" / "plots"

    if not plots_dir.exists():
        print(f"Nothing to do: {plots_dir} does not exist.")
        return

    # Sicherheitscheck: wir löschen nur exakt data/plots im Repo
    if plots_dir.name != "plots" or plots_dir.parent.name != "data":
        raise RuntimeError(f"Safety check failed, refusing to delete: {plots_dir}")

    if args.dry_run:
        print(f"[dry-run] Would delete: {plots_dir}")
        return

    shutil.rmtree(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    print(f"Deleted all plots. Re-created directory: {plots_dir}")


if __name__ == "__main__":
    main()
