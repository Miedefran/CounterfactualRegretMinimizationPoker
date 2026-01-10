import pytest
import subprocess
import os

def run_command(command):
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    result = subprocess.run(command, shell=True, capture_output=True, text=True, env=env)
    return result

def test_pipeline_new_architecture_fold():
    """
    Runs the full pipeline with the Fold strategy.
    Verifies that the process completes without errors.
    """
    # 1. Train Leduc
    res_train = run_command("uv run python src/training/train.py leduc 1 fold")
    assert res_train.returncode == 0
    
    # 2. Build Public Tree
    res_tree = run_command("uv run python src/evaluation/build_public_state_tree_v2.py leduc")
    assert res_tree.returncode == 0
    
    # 3. Compute Best Response for Player 0
    cmd_p0 = (
        "uv run python src/evaluation/best_response_agent_v2.py "
        "--game leduc --player 0 "
        "--public-tree evaluation/public_state_trees/leduc_public_tree_v2.pkl.gz "
        "--strategy data/models/leduc/fold/leduc_1.pkl.gz"
    )
    res_p0 = run_command(cmd_p0)
    assert res_p0.returncode == 0
    
    # 4. Compute Best Response for Player 1
    cmd_p1 = (
        "uv run python src/evaluation/best_response_agent_v2.py "
        "--game leduc --player 1 "
        "--public-tree evaluation/public_state_trees/leduc_public_tree_v2.pkl.gz "
        "--strategy data/models/leduc/fold/leduc_1.pkl.gz"
    )
    res_p1 = run_command(cmd_p1)
    assert res_p1.returncode == 0
