import argparse
import gzip
import pickle

def main():
    parser = argparse.ArgumentParser(description='Display model statistics')
    parser.add_argument('model_file', type=str, help='Path to model .pkl.gz file')
    args = parser.parse_args()
    
    with gzip.open(args.model_file, 'rb') as f:
        data = pickle.load(f)
    
    print("=" * 60)
    print("MODEL STATISTICS")
    print("=" * 60)
    
    print(f"\nFile: {args.model_file}")
    
    if 'iteration_count' in data:
        print(f"Iterations: {data['iteration_count']:,}")
    
    if 'training_time' in data:
        print(f"Training time: {data['training_time']:.2f} seconds")
    
    if 'average_strategy' in data:
        info_sets = len(data['average_strategy'])
        print(f"Information sets: {info_sets}")
    
    if 'regret_sum' in data:
        print(f"Regret entries: {len(data['regret_sum'])}")
    
    if 'strategy_sum' in data:
        print(f"Strategy entries: {len(data['strategy_sum'])}")
    
    print("=" * 60)

if __name__ == "__main__":
    main()

