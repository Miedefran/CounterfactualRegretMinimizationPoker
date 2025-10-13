import pandas as pd
import os
import re

CSV_PATH = 'test_data/test_results.csv'

def log_metric(algorithm, game, iterations, metric_type, metric_name, value, seed=None, config=None):
    row = {
        'algorithm': algorithm,
        'game': game,
        'iterations': iterations,
        'metric_type': metric_type,
        'metric_name': metric_name,
        'value': value,
        'seed': seed if seed else 'NA',
        'config': config if config else ''
    }
    
    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
    else:
        df = pd.DataFrame(columns=['algorithm','game','iterations','metric_type','metric_name','value','seed','config'])
    
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(CSV_PATH, index=False)
    
    print(f"Logged: {metric_type}.{metric_name} = {value}")

def log_nash(algorithm, game, iterations, **metrics):
    for metric_name, value in metrics.items():
        log_metric(algorithm, game, iterations, 'nash', metric_name, value)

def log_performance(algorithm, game, iterations, metric_name, value, seed=None, **config_params):
    config_str = ','.join([f"{k}={v}" for k, v in config_params.items()]) if config_params else ''
    log_metric(algorithm, game, iterations, 'performance', metric_name, value, seed, config_str)

def extract_iterations_from_filename(filepath):
    match = re.search(r'_(\d+)\.pkl', filepath)
    if match:
        return int(match.group(1))
    return None

def extract_game_from_filename(filepath):
    if 'kuhn_case1' in filepath:
        return 'kuhn_case1'
    elif 'kuhn_case2' in filepath:
        return 'kuhn_case2'
    elif 'kuhn_case3' in filepath:
        return 'kuhn_case3'
    elif 'kuhn_case4' in filepath:
        return 'kuhn_case4'
    elif 'leduc' in filepath:
        return 'leduc'
    return None

