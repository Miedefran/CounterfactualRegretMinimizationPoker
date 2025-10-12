import os
import glob

def main():
    model_files = glob.glob("models/**/*.pkl.gz", recursive=True)
    
    if not model_files:
        print("No model files found.")
        return
    
    print(f"Found {len(model_files)} model files:")
    for file in model_files:
        print(f"  - {file}")
    
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
