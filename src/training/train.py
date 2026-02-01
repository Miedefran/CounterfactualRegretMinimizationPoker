import argparse
import os
import sys
import random; import numpy as np; random.seed(42); np.random.seed(42)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.config import TrainingConfig

# Import Solvers
from training.solvers.cfr_solver import CFRSolver
from training.solvers.cfr_plus_solver import CFRPlusSolver
from training.solvers.fold_solver import AlwaysFoldSolver
from training.solvers.cfr_solver_with_tree import CFRSolverWithTree
from training.solvers.cfr_solver_with_flat_tree import CFRSolverWithFlatTree
from training.solvers.cfr_plus_with_flat_tree import CFRPlusWithFlatTree
from training.solvers.discounted_cfr_solver_with_flat_tree import DiscountedCFRWithFlatTree
from training.solvers.chance_sampling_cfr_solver import ChanceSamplingCFRSolver
from training.solvers.external_sampling_cfr_solver import ExternalSamplingCFRSolver
from training.solvers.outcome_sampling_cfr_solver import OutcomeSamplingCFRSolver
from training.solvers.cfr_plus_with_tree import CFRPlusWithTree
from training.solvers.discounted_cfr_solver import DiscountedCFRSolver
from training.solvers.discounted_cfr_solver_with_tree import DiscountedCFRWithTreeSolver

from utils.poker_utils import get_model_path, find_game_class_for_abstraction
from training.best_response_evaluator import BestResponseTracker
from training.progress_reporter import ProgressReporter

SOLVERS = [
    CFRSolver,
    CFRPlusSolver,
    AlwaysFoldSolver,
    CFRSolverWithTree,
    ChanceSamplingCFRSolver,
    ExternalSamplingCFRSolver,
    OutcomeSamplingCFRSolver,
    CFRPlusWithTree,
    CFRSolverWithFlatTree,
    CFRPlusWithFlatTree,
    DiscountedCFRSolver,
    DiscountedCFRWithTreeSolver,
    DiscountedCFRWithFlatTree
]


def get_print_interval(iterations):
    """
    Compute print interval based on number of iterations.
    Goal: Approximately 10-20 print statements during training.
    """
    if iterations < 1000:
        return 100
    elif iterations < 10000:
        return 1000
    elif iterations < 100000:
        return 10000
    else:
        return max(10000, iterations // 10)


def run_training(config: TrainingConfig, progress_reporter: ProgressReporter = None):
    use_suit_abstraction = not config.no_suit_abstraction

    # Find Game
    game_class = find_game_class_for_abstraction(config.game, use_suit_abstraction)
    if game_class is None:
        raise ValueError(f"Unknown game: {config.game}")

    use_suit_abstraction = game_class.suit_abstraction

    abstraction_str = " (suit abstracted)" if use_suit_abstraction else ""
    print(f"Training {config.game}{abstraction_str} for {config.iterations} iterations")

    # Report task start
    if progress_reporter is not None:
        progress_reporter.start_task(
            game=config.game,
            algorithm=config.algorithm,
            target_iterations=config.iterations,
            suit_abstraction=use_suit_abstraction,
        )

    # Instantiate Game
    game = game_class()

    # Find Solver
    solver_class = None
    for s in SOLVERS:
        if s.evaluate_solver(config):
            solver_class = s
            break

    if solver_class is None:
        raise ValueError(f"Unknown algorithm: {config.algorithm}")

    # Instantiate Solver
    solver = solver_class.create_solver(config, game, game_class.combination_generator)

    # Best Response Tracker
    br_tracker = None
    if config.br_eval_schedule is not None:
        try:
            if config.br_eval_schedule.isdigit():
                schedule_config = int(config.br_eval_schedule)
                br_tracker = BestResponseTracker(config.game, game_class=game_class, schedule_config=schedule_config,
                                                 progress_reporter=progress_reporter)
                print(f"Best Response Evaluation activated (fixed interval: {schedule_config})")
            else:
                br_tracker = BestResponseTracker(config.game, game_class=game_class, schedule_config=config.br_eval_schedule,
                                                 progress_reporter=progress_reporter)
                schedule_type = br_tracker.schedule_type
                if schedule_type == "fixed":
                    print(f"Best Response Evaluation activated (fixed interval: {br_tracker.interval})")
                elif schedule_type == "logarithmic":
                    print(
                        f"Best Response Evaluation activated (logarithmic: {br_tracker.base_interval} -> {br_tracker.target_interval} at iteration {br_tracker.target_iteration})")
                elif schedule_type == "custom":
                    print(
                        f"Best Response Evaluation activated (custom schedule with {len(br_tracker.custom_schedule)} stages)")
        except Exception as e:
            print(f"WARNING: Error loading BR-eval schedule: {e}")
            print("Best Response Evaluation will be disabled")
            br_tracker = None

    print_interval = get_print_interval(config.iterations)
    if config.early_stop_exploitability_mb is not None and br_tracker is None:
        print(
            "WARNING: --early-stop-exploitability-mb set, but BR-eval is disabled -> no early stop possible.")

    # Train
    try:
        solver.train(
            config.iterations,
            br_tracker=br_tracker,
            print_interval=print_interval,
            stop_exploitability_mb=config.early_stop_exploitability_mb,
        )
        if progress_reporter is not None:
            progress_reporter.complete_task()
    except Exception as e:
        if progress_reporter is not None:
            progress_reporter.complete_task(error=str(e))
        raise

    # Actual iterations (handle early stop)
    actual_iterations = int(getattr(solver, "iteration_count", getattr(solver, "t", config.iterations)))

    # Construct Algorithm Path String
    algorithm_for_path = config.algorithm

    if not config.alternating_updates and solver_class.supports_alternating_updates():
        algorithm_for_path = f"{algorithm_for_path}_simultaneous"

    if not config.partial_pruning and solver_class.supports_partial_pruning():
        algorithm_for_path = f"{algorithm_for_path}_no_pruning"

    if use_suit_abstraction:
        algorithm_for_path = f"{algorithm_for_path}_abstracted"

    if config.squared_weight and solver_class.supports_squared_weights():
        algorithm_for_path = f"{algorithm_for_path}_squared_weight"

    # Save
    filepath = get_model_path(config.game, actual_iterations, algorithm_for_path)
    solver.save_gzip(filepath)

    # Best Response plotting and saving
    if br_tracker is not None and br_tracker.values:
        # Save plot
        plot_path = filepath.replace('.pkl.gz', '_best_response.png')
        br_tracker.plot(output_path=plot_path)

        # Save values
        br_data_path = filepath.replace('.pkl.gz', '_best_response.pkl.gz')
        br_tracker.save(br_data_path)

    avg_strategy = solver.average_strategy
    print(f"{len(avg_strategy)} information sets")

    if config.game.startswith('kuhn'):
        print("\nStrategy (Kuhn):")
        for key, actions in sorted(avg_strategy.items()):
            card, public_cards, history, player = key
            print(f"\nCard: {card}, History: {history}, Player: {player}")
            for action, prob in actions.items():
                print(f"  {action}: {prob:.4f}")

    if config.game == "leduc":
        # KeyGenerator: (private_card, public_cards, history, player_id)
        print("\nStrategy (Leduc) – Selected Infosets:")
        sorted_keys = sorted(avg_strategy.items())
        # First 15 + some with different cards/histories
        shown = set()
        for key, actions in sorted_keys[:20]:
            shown.add(key)
            priv, pub, hist, pid = key
            print(f"\n  priv={priv!r}, pub={pub!r}, hist={hist!r}, P{pid}")
            for action, prob in actions.items():
                print(f"    {action}: {prob:.4f}")
        # Additionally 5 randomly from the rest (deterministic: every 100th)
        for i, (key, actions) in enumerate(sorted_keys):
            if key in shown or i % 100 != 50:
                continue
            shown.add(key)
            priv, pub, hist, pid = key
            print(f"\n  priv={priv!r}, pub={pub!r}, hist={hist!r}, P{pid}")
            for action, prob in actions.items():
                print(f"    {action}: {prob:.4f}")
            if len(shown) >= 25:
                break
        print(f"\n  (total {len(avg_strategy)} infosets, selection above)")

    if config.game == "twelve_card_poker":
        print("\nStrategy (Twelve Card Poker) – Selected Infosets:")
        sorted_keys = sorted(avg_strategy.items())
        shown = set()
        for key, actions in sorted_keys[:20]:
            shown.add(key)
            priv, pub, hist, pid = key
            print(f"\n  priv={priv!r}, pub={pub!r}, hist={hist!r}, P{pid}")
            for action, prob in actions.items():
                print(f"    {action}: {prob:.4f}")
        for i, (key, actions) in enumerate(sorted_keys):
            if key in shown or i % 200 != 100:
                continue
            shown.add(key)
            priv, pub, hist, pid = key
            print(f"\n  priv={priv!r}, pub={pub!r}, hist={hist!r}, P{pid}")
            for action, prob in actions.items():
                print(f"    {action}: {prob:.4f}")
            if len(shown) >= 25:
                break
        print(f"\n  (total {len(avg_strategy)} infosets, selection above)")


def main():
    parser = argparse.ArgumentParser(description='Train CFR for Poker')
    parser.add_argument('game', type=str,
                        choices=['kuhn_case1', 'kuhn_case2', 'kuhn_case3', 'kuhn_case4', 'leduc', 'rhode_island',
                                 'twelve_card_poker', 'royal_holdem', 'small_island_holdem', 'limit_holdem'],
                        help='Poker variant to train on')
    parser.add_argument('iterations', type=int,
                        help='Number of CFR iterations')
    parser.add_argument('algorithm', type=str,
                        choices=[
                            'fold',
                            'cfr',
                            'cfr_plus',
                            'mccfr',
                            'cfr_with_tree',
                            'cfr_plus_with_tree',
                            'cfr_with_flat_tree',
                            'cfr_plus_with_flat_tree',
                            'chance_sampling',
                            'external_sampling',
                            'outcome_sampling',
                            'discounted_cfr',
                            'discounted_cfr_with_tree',
                            'discounted_cfr_with_flat_tree',
                        ],
                        nargs='?',
                        default='cfr',
                        help='Algorithm to use (default: cfr)')
    parser.add_argument('--br-eval-schedule', type=str, default=None,
                        help='Best Response evaluation schedule: Integer (fixed interval), JSON path, or schedule name from config/br_eval_schedules.json (None = disabled)')
    parser.add_argument('--alternating-updates', type=str, choices=['true', 'false'], default='true',
                        help='For CFR-based solvers: true for alternating updates, false for simultaneous updates. Default: true')
    parser.add_argument('--partial-pruning', type=str, choices=['true', 'false'], default='false',
                        help='Small early-exit pruning when reach probabilities are 0. true=enabled, false=disabled. Default: false')
    parser.add_argument('--no-suit-abstraction', action='store_true',
                        help='Disable suit abstraction for leduc and twelve_card_poker (default: disabled)')
    parser.add_argument('--dcfr-alpha', type=float, default=1.5,
                        help='Alpha parameter for Discounted CFR (default: 1.5)')
    parser.add_argument('--dcfr-beta', type=float, default=0.0,
                        help='Beta parameter for Discounted CFR (default: 0.0)')
    parser.add_argument('--dcfr-gamma', type=float, default=2.0,
                        help='Gamma parameter for Discounted CFR (default: 2.0)')
    parser.add_argument('--squared-weight', action='store_true',
                        help='Use t² instead of t for strategy weighting in CFR+ variants (default: disabled, uses t)')
    parser.add_argument(
        '--early-stop-exploitability-mb',
        type=float,
        default=None,
        help='Stop training early when a BR evaluation reaches exploitability < threshold (mb/g). '
             'Only works if --br-eval-schedule is active.',
    )
    args = parser.parse_args()

    config = TrainingConfig(
        game=args.game,
        iterations=args.iterations,
        algorithm=args.algorithm,
        br_eval_schedule=args.br_eval_schedule,
        alternating_updates=(args.alternating_updates.lower() == 'true'),
        partial_pruning=(args.partial_pruning.lower() == 'true'),
        no_suit_abstraction=args.no_suit_abstraction,
        dcfr_alpha=args.dcfr_alpha,
        dcfr_beta=args.dcfr_beta,
        dcfr_gamma=args.dcfr_gamma,
        squared_weight=args.squared_weight,
        early_stop_exploitability_mb=args.early_stop_exploitability_mb
    )
    run_training(config)


if __name__ == "__main__":
    main()
