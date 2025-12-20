import os
import glob
import argparse

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

    if args.game:
        pattern = f"models/{args.game}/**/*.pkl.gz"
    elif args.keep_game:
        pattern = "models/**/*.pkl.gz"
    else:
        pattern = "models/**/*.pkl.gz"

    model_files = glob.glob(pattern, recursive=True)
    if args.keep_game:
        keep_prefix = f"models/{args.keep_game}/"
        model_files = [p for p in model_files if not p.startswith(keep_prefix)]
    
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
    for file in model_files:
        print(f"  - {file}")

    if args.dry_run:
        print("\nDry run: no files deleted.")
        return

    if not args.yes:
        answer = input("\nDelete these files? Type 'yes' to confirm: ").strip().lower()
        if answer != "yes":
            print("Aborted.")
            return
    
    deleted_count = 0
    for file in model_files:
        try:
            os.remove(file)
            print(f"Deleted: {file}")
            deleted_count += 1
        except Exception as e:
            print(f"Error deleting {file}: {e}")
    
    print(f"\nSuccessfully deleted {deleted_count} files.")

if __name__ == "__main__":
    main()
