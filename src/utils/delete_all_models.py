import argparse
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Delete trained model files.")
    parser.add_argument(
        "--game",
        type=str,
        default=None,
        help="If set, only delete models for this game (e.g. kuhn, leduc, rhode_island).",
    )
    parser.add_argument(
        "--keep-game",
        type=str,
        default=None,
        help="If set, delete models for all games EXCEPT this one (e.g. kuhn).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print which files would be deleted.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip confirmation prompt.",
    )
    args = parser.parse_args()

    if args.game and args.keep_game:
        raise SystemExit("Use only one of --game or --keep-game.")

    repo_root = Path(__file__).resolve().parents[2]
    models_root = repo_root / "data" / "models"

    if not models_root.exists():
        print(f"No models directory found: {models_root}")
        return

    # Sammle alle Modell-Artefakte (Model + Best-Response-Logs + Plot-Bilder)
    # - *.pkl.gz: Modelle und *_best_response.pkl.gz
    # - *.png: *_best_response.png
    def is_model_artifact(p: Path) -> bool:
        name = p.name
        return name.endswith(".pkl.gz") or name.endswith(".png")

    if args.game:
        search_root = models_root / args.game
    else:
        search_root = models_root

    if not search_root.exists():
        print(f"No model files found (directory missing): {search_root}")
        return

    model_files = [p for p in search_root.rglob("*") if p.is_file() and is_model_artifact(p)]

    if args.keep_game:
        keep_root = models_root / args.keep_game
        model_files = [p for p in model_files if not p.is_relative_to(keep_root)]

    if not model_files:
        print("No model files found.")
        return

    if args.game:
        scope = f" for game '{args.game}'"
    elif args.keep_game:
        scope = f" for all games except '{args.keep_game}'"
    else:
        scope = ""
    print(f"Found {len(model_files)} model files{scope}:")
    # Nicht unendlich viel spammen, falls extrem viele Artefakte existieren
    model_files_sorted = sorted(model_files, key=lambda p: str(p))
    max_print = 200
    for p in model_files_sorted[:max_print]:
        print(f"  - {p.relative_to(repo_root)}")
    if len(model_files_sorted) > max_print:
        print(f"  ... ({len(model_files_sorted) - max_print} more)")

    if args.dry_run:
        print("\nDry run: no files deleted.")
        return

    if not args.yes:
        answer = input("\nDelete these files? Type 'yes' to confirm: ").strip().lower()
        if answer != "yes":
            print("Aborted.")
            return

    deleted_count = 0
    for file in model_files_sorted:
        try:
            os.remove(file)
            print(f"Deleted: {Path(file).relative_to(repo_root)}")
            deleted_count += 1
        except Exception as e:
            print(f"Error deleting {file}: {e}")

    print(f"\nSuccessfully deleted {deleted_count} files.")


if __name__ == "__main__":
    main()
