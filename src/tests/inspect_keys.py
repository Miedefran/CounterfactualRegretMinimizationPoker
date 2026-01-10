import pickle
import gzip
import sys

# Usage: python inspect_keys.py data/models/leduc/fold/leduc_1.pkl.gz
def inspect(path):
    with gzip.open(path, 'rb') as f:
        data = pickle.load(f)

    strategy = data.get('average_strategy', data)

    print(f"Total Keys: {len(strategy)}")
    print("\n--- Searching for Player 1 Keys with 'Js' ---")

    found = 0
    for key in strategy:
        # Check if key matches structure: (Card, Public, History, PlayerID)
        if isinstance(key, tuple) and len(key) == 4:
            card, public, history, pid = key

            # Filter for Player 1 and Jack of Spades
            if pid == 1 and card == 'Js':
                print(f"Found Key: {key}")
                found += 1
                if found >= 5: break # Just show first 5 matches

    if found == 0:
        print("No keys found for Player 1 with Card 'Js'.")

if __name__ == "__main__":
    inspect(sys.argv[1])