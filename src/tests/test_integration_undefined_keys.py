import pytest
import subprocess
import os

def run_command(command):
    # Ensure src is in pythonpath
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    
    result = subprocess.run(command, shell=True, capture_output=True, text=True, env=env)
    return result

def test_pipeline_undefined_keys():
    """
    Runs the full training/evaluation pipeline for Leduc Hold'em (Fold strategy).
    Checks specifically for 'undefined key' warnings which indicate key mismatches.
    """
    
    # 1. Train Leduc (Fold Strategy)
    cmd_train = "uv run python src/training/train.py leduc 1 fold"
    res_train = run_command(cmd_train)
    assert res_train.returncode == 0, f"Training failed: {res_train.stderr}"
    
    # 2. Build Public Tree
    cmd_tree = "uv run python src/evaluation/build_public_state_tree_v2.py leduc"
    res_tree = run_command(cmd_tree)
    assert res_tree.returncode == 0, f"Tree build failed: {res_tree.stderr}"
    
    # 3. Evaluate Best Response (Player 0)
    cmd_eval = (
        "uv run python src/evaluation/best_response_agent_v2.py "
        "--game leduc --player 0 "
        "--public-tree evaluation/public_state_trees/leduc_public_tree_v2.pkl.gz "
        "--strategy models/leduc/fold/leduc_1.pkl.gz"
    )
    res_eval = run_command(cmd_eval)
    assert res_eval.returncode == 0, f"Evaluation failed: {res_eval.stderr}"
    
    # 4. Check for undefined key warnings
    output = res_eval.stdout
    assert "WARNING: undefined key" not in output, "Found 'undefined key' warnings in output!"
