import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.cfr_solver import CFRSolver
from training.cfr_plus_solver import CFRPlusSolver
from training.fold_solver import AlwaysFoldSolver
from training.mccfr_solver import MCCFRSolver
from training.tensor_cfr_solver import TensorCFRSolver
from training.cfr_solver_with_tree import CFRSolverWithTree
from training.cfr_solver_with_flat_tree import CFRSolverWithFlatTree
from training.cfr_plus_with_flat_tree import CFRPlusWithFlatTree
from training.discounted_cfr_solver_with_flat_tree import DiscountedCFRWithFlatTree
from training.chance_sampling_cfr_solver import ChanceSamplingCFRSolver
from training.external_sampling_cfr_solver import ExternalSamplingCFRSolver
from training.outcome_sampling_cfr_solver import OutcomeSamplingCFRSolver
from training.cfr_plus_with_tree import CFRPlusWithTree
from training.discounted_cfr_solver import DiscountedCFRSolver
from training.discounted_cfr_solver_with_tree import DiscountedCFRWithTreeSolver

from envs.kuhn_poker.game import KuhnPokerGame
from envs.leduc_holdem.game import LeducHoldemGame
from envs.rhode_island.game import RhodeIslandGame
from envs.twelve_card_poker.game import TwelveCardPokerGame
from envs.royal_holdem.game import RoyalHoldemGame
from envs.small_island_holdem.game import SmallIslandHoldemGame
from envs.limit_holdem.game import LimitHoldemGame

from utils.poker_utils import (
    GAME_CONFIGS,
    KuhnPokerCombinations,
    LeducHoldemCombinations,
    LeducHoldemCombinationsAbstracted,
    RhodeIslandCombinations,
    TwelveCardPokerCombinations,
    TwelveCardPokerCombinationsAbstracted,
    RoyalHoldemCombinations,
    SmallIslandHoldemCombinations,
    LimitHoldemCombinations,
    get_model_path
)
from training.best_response_evaluator import BestResponseTracker

def get_print_interval(iterations):
    """
    Berechnet den Print-Intervall basierend auf der Anzahl der Iterationen.
    
    Ziel: Etwa 10-20 Print-Statements während des Trainings.
    
    Args:
        iterations: Anzahl der Iterationen
    
    Returns:
        Print-Intervall (int)
    """
    if iterations < 1000:
        return 100
    elif iterations < 10000:
        return 1000
    elif iterations < 100000:
        return 10000
    else:
        # Für >= 100k: alle 100k oder so dass wir ~10 Prints haben
        return max(10000, iterations // 10)

def main():
    parser = argparse.ArgumentParser(description='Train CFR for Poker')
    parser.add_argument('game', type=str,
                       choices=['kuhn_case1', 'kuhn_case2', 'kuhn_case3', 'kuhn_case4', 'leduc', 'rhode_island', 'twelve_card_poker', 'royal_holdem', 'small_island_holdem', 'limit_holdem'],
                       help='Poker variant to train on')
    parser.add_argument('iterations', type=int,
                       help='Number of CFR iterations')
    parser.add_argument('algorithm', type=str,
                       choices=[
                           'fold',
                           'cfr',
                           'cfr_plus',
                           'mccfr',
                           'tensor_cfr',
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
                       help='Best Response Evaluierungs-Schedule: Integer (fester Intervall), JSON-Pfad, oder Schedule-Name aus config/br_eval_schedules.json (None = deaktiviert)')
    parser.add_argument('--tensor-algorithm', type=str, choices=['cfr', 'cfr_plus'], default='cfr_plus',
                       help='Algorithmus für Tensor CFR: cfr (ohne CFR+) oder cfr_plus (mit CFR+). Standard: cfr_plus')
    parser.add_argument('--alternating-updates', type=str, choices=['true', 'false'], default='true',
                       help='Für CFR-basierte Solver: true für alternierende Updates, false für simultane Updates. Standard: true')
    parser.add_argument('--partial-pruning', type=str, choices=['true', 'false'], default='false',
                       help='Kleines Early-Exit-Pruning wenn Reach-Probabilities 0 sind. true=aktiv, false=deaktiviert. Standard: false')
    parser.add_argument('--no-suit-abstraction', action='store_true',
                       help='Deaktiviert Suit Abstraction für leduc und twelve_card_poker (Standard: aktiviert)')
    parser.add_argument('--dcfr-alpha', type=float, default=1.5,
                       help='Alpha Parameter für Discounted CFR (Standard: 1.5)')
    parser.add_argument('--dcfr-beta', type=float, default=0.0,
                       help='Beta Parameter für Discounted CFR (Standard: 0.0)')
    parser.add_argument('--dcfr-gamma', type=float, default=2.0,
                       help='Gamma Parameter für Discounted CFR (Standard: 2.0)')
    parser.add_argument(
        '--early-stop-exploitability-mb',
        type=float,
        default=None,
        help='Bricht das Training ab, sobald eine BR-Evaluation Exploitability < Schwelle (mb/g) erreicht. '
             'Funktioniert nur, wenn --br-eval-schedule aktiv ist.',
    )
    args = parser.parse_args()
    config = GAME_CONFIGS[args.game]

    alternating = getattr(args, 'alternating_updates', 'true').lower() == 'true'
    partial_pruning = getattr(args, 'partial_pruning', 'false').lower() == 'true'
    
    # Bestimme ob Suit Abstraction verwendet werden soll
    # Standardmäßig für leduc und twelve_card_poker, außer wenn --no-suit-abstraction gesetzt ist
    use_suit_abstraction = False
    if args.game in ['leduc', 'twelve_card_poker']:
        use_suit_abstraction = not args.no_suit_abstraction
    
    abstraction_str = " (suit abstracted)" if use_suit_abstraction else ""
    print(f"Training {args.game}{abstraction_str} for {args.iterations} iterations")
    
    if args.game.startswith('kuhn'):
        game = KuhnPokerGame(ante=config['ante'], bet_size=config['bet_size'])
        combo_gen = KuhnPokerCombinations()
    elif args.game.startswith('leduc'):
        game = LeducHoldemGame(
            ante=config['ante'],
            bet_sizes=config['bet_sizes'],
            bet_limit=config['bet_limit'],
            abstract_suits=use_suit_abstraction,
        )
        if use_suit_abstraction:
            combo_gen = LeducHoldemCombinationsAbstracted()
        else:
            combo_gen = LeducHoldemCombinations()
    elif args.game.startswith('rhode'):
        game = RhodeIslandGame(ante=config['ante'], bet_sizes=config['bet_sizes'], bet_limit=config['bet_limit'])
        combo_gen = RhodeIslandCombinations()
    elif args.game == 'twelve_card_poker':
        game = TwelveCardPokerGame(
            ante=config['ante'],
            bet_sizes=config['bet_sizes'],
            bet_limit=config['bet_limit'],
            abstract_suits=use_suit_abstraction,
        )
        if use_suit_abstraction:
            combo_gen = TwelveCardPokerCombinationsAbstracted()
        else:
            combo_gen = TwelveCardPokerCombinations()
    elif args.game == 'royal_holdem':
        game = RoyalHoldemGame(ante=config['ante'], bet_sizes=config['bet_sizes'], bet_limit=config['bet_limit'])
        combo_gen = RoyalHoldemCombinations()
    elif args.game == 'small_island_holdem':
        game = SmallIslandHoldemGame(ante=config['ante'], bet_sizes=config['bet_sizes'], bet_limit=config['bet_limit'])
        combo_gen = SmallIslandHoldemCombinations()
    elif args.game == 'limit_holdem':
        game = LimitHoldemGame(small_blind=config['small_blind'], big_blind=config['big_blind'], bet_sizes=config['bet_sizes'], bet_limit=config['bet_limit'])
        combo_gen = LimitHoldemCombinations()
    
    if args.algorithm == 'cfr':
        solver = CFRSolver(game, combo_gen, alternating_updates=alternating, partial_pruning=partial_pruning)
    elif args.algorithm == 'cfr_plus':
        solver = CFRPlusSolver(game, combo_gen, alternating_updates=alternating, partial_pruning=partial_pruning)
    elif args.algorithm == 'mccfr':
        solver = MCCFRSolver(game, combo_gen)
    elif args.algorithm == 'fold':
        solver = AlwaysFoldSolver(game, combo_gen)
    elif args.algorithm == 'tensor_cfr':
        # Übergebe game_name für automatisches Laden des Trees
        # algorithm Parameter bestimmt ob CFR+ verwendet wird
        solver = TensorCFRSolver(game, combo_gen, algorithm=args.tensor_algorithm, game_name=args.game)
    elif args.algorithm == 'cfr_with_tree':
        # Übergebe game_name für automatisches Laden des Trees
        solver = CFRSolverWithTree(
            game,
            combo_gen,
            game_name=args.game,
            alternating_updates=alternating,
            partial_pruning=partial_pruning,
        )
    elif args.algorithm == 'chance_sampling':
        # Übergebe game_name für automatisches Laden des Trees
        solver = ChanceSamplingCFRSolver(game, combo_gen, game_name=args.game)
    elif args.algorithm == 'external_sampling':
        # Übergebe game_name für automatisches Laden des Trees
        solver = ExternalSamplingCFRSolver(game, combo_gen, game_name=args.game)
    elif args.algorithm == 'outcome_sampling':
        # Übergebe game_name für automatisches Laden des Trees
        solver = OutcomeSamplingCFRSolver(game, combo_gen, game_name=args.game)
    elif args.algorithm == 'cfr_plus_with_tree':
        # Übergebe game_name für automatisches Laden des Trees
        solver = CFRPlusWithTree(
            game,
            combo_gen,
            game_name=args.game,
            alternating_updates=alternating,
            partial_pruning=partial_pruning,
        )
    elif args.algorithm == 'cfr_with_flat_tree':
        solver = CFRSolverWithFlatTree(
            game,
            combo_gen,
            game_name=args.game,
            alternating_updates=alternating,
            partial_pruning=partial_pruning,
        )
    elif args.algorithm == 'cfr_plus_with_flat_tree':
        solver = CFRPlusWithFlatTree(
            game,
            combo_gen,
            game_name=args.game,
            alternating_updates=alternating,
            partial_pruning=partial_pruning,
        )
    elif args.algorithm == 'discounted_cfr':
        # Discounted CFR Solver (ohne Tree)
        solver = DiscountedCFRSolver(
            game, combo_gen,
            alternating_updates=alternating,
            partial_pruning=partial_pruning,
            alpha=args.dcfr_alpha,
            beta=args.dcfr_beta,
            gamma=args.dcfr_gamma
        )
    elif args.algorithm == 'discounted_cfr_with_tree':
        # Discounted CFR with Tree Solver
        solver = DiscountedCFRWithTreeSolver(
            game, combo_gen, 
            game_name=args.game,
            alpha=args.dcfr_alpha,
            beta=args.dcfr_beta,
            gamma=args.dcfr_gamma,
            alternating_updates=alternating,
            partial_pruning=partial_pruning,
        )
    elif args.algorithm == 'discounted_cfr_with_flat_tree':
        solver = DiscountedCFRWithFlatTree(
            game,
            combo_gen,
            game_name=args.game,
            alpha=args.dcfr_alpha,
            beta=args.dcfr_beta,
            gamma=args.dcfr_gamma,
            alternating_updates=alternating,
            partial_pruning=partial_pruning,
        )
    
    # Best Response Tracker initialisieren falls gewünscht
    br_tracker = None
    if args.br_eval_schedule is not None:
        try:
            if args.br_eval_schedule.isdigit():
                schedule_config = int(args.br_eval_schedule)
                br_tracker = BestResponseTracker(args.game, schedule_config=schedule_config, use_suit_abstraction=use_suit_abstraction)
                print(f"Best Response Evaluation aktiviert (fester Intervall: {schedule_config})")
            else:
                br_tracker = BestResponseTracker(args.game, schedule_config=args.br_eval_schedule, use_suit_abstraction=use_suit_abstraction)
                schedule_type = br_tracker.schedule_type
                if schedule_type == "fixed":
                    print(f"Best Response Evaluation aktiviert (fester Intervall: {br_tracker.interval})")
                elif schedule_type == "logarithmic":
                    print(f"Best Response Evaluation aktiviert (logarithmisch: {br_tracker.base_interval} -> {br_tracker.target_interval} bei Iteration {br_tracker.target_iteration})")
                elif schedule_type == "custom":
                    print(f"Best Response Evaluation aktiviert (custom schedule mit {len(br_tracker.custom_schedule)} Stufen)")
        except Exception as e:
            print(f"WARNING: Fehler beim Laden des BR-Eval-Schedules: {e}")
            print("Best Response Evaluation wird deaktiviert")
            br_tracker = None
    
    print_interval = get_print_interval(args.iterations)
    if args.early_stop_exploitability_mb is not None and br_tracker is None:
        print("WARNING: --early-stop-exploitability-mb gesetzt, aber BR-Eval ist deaktiviert -> kein Early-Stop möglich.")

    solver.train(
        args.iterations,
        br_tracker=br_tracker,
        print_interval=print_interval,
        stop_exploitability_mb=args.early_stop_exploitability_mb,
    )

    # Wenn Early-Stop aktiv war, können weniger Iterationen als angefordert gelaufen sein.
    actual_iterations = int(getattr(solver, "iteration_count", getattr(solver, "t", args.iterations)))
    
    # Für tensor_cfr: Unterscheide zwischen cfr und cfr_plus in Dateinamen
    # Für cfr_with_tree: Unterscheide zwischen alternierend und simultan
    algorithm_for_path = args.algorithm
    if args.algorithm == 'tensor_cfr' and args.tensor_algorithm == 'cfr_plus':
        algorithm_for_path = 'tensor_cfr_plus'

    # Codierung von alternating/simultaneous in den Output-Pfad, damit Runs nicht überschreiben
    if not alternating:
        if args.algorithm == 'cfr':
            algorithm_for_path = 'cfr_simultaneous'
        elif args.algorithm == 'cfr_plus':
            algorithm_for_path = 'cfr_plus_simultaneous'
        elif args.algorithm == 'cfr_with_tree':
            algorithm_for_path = 'cfr_with_tree_simultaneous'
        elif args.algorithm == 'cfr_plus_with_tree':
            algorithm_for_path = 'cfr_plus_with_tree_simultaneous'
        elif args.algorithm == 'cfr_with_flat_tree':
            algorithm_for_path = 'cfr_with_flat_tree_simultaneous'
        elif args.algorithm == 'cfr_plus_with_flat_tree':
            algorithm_for_path = 'cfr_plus_with_flat_tree_simultaneous'
        elif args.algorithm == 'discounted_cfr':
            algorithm_for_path = 'discounted_cfr_simultaneous'
        elif args.algorithm == 'discounted_cfr_with_tree':
            algorithm_for_path = 'discounted_cfr_with_tree_simultaneous'
        elif args.algorithm == 'discounted_cfr_with_flat_tree':
            algorithm_for_path = 'discounted_cfr_with_flat_tree_simultaneous'

    # Codierung von (partial) pruning in den Output-Pfad
    if not partial_pruning and args.algorithm in {
        'cfr',
        'cfr_plus',
        'cfr_with_tree',
        'cfr_plus_with_tree',
        'cfr_with_flat_tree',
        'cfr_plus_with_flat_tree',
        'discounted_cfr',
        'discounted_cfr_with_tree',
        'discounted_cfr_with_flat_tree',
    }:
        algorithm_for_path = f"{algorithm_for_path}_no_pruning"

    # Codierung von Suit-Abstraction in den Output-Pfad, damit Runs nicht überschreiben.
    # Standardmäßig ist Suit-Abstraction für leduc/twelve_card_poker aktiv (außer --no-suit-abstraction).
    if use_suit_abstraction and args.game in {'leduc', 'twelve_card_poker'}:
        algorithm_for_path = f"{algorithm_for_path}_abstracted"
    
    filepath = get_model_path(args.game, actual_iterations, algorithm_for_path)
    solver.save_gzip(filepath)
    
    # Best Response Plotting und Speichern
    if br_tracker is not None and br_tracker.values:
        # Plot speichern
        plot_path = filepath.replace('.pkl.gz', '_best_response.png')
        br_tracker.plot(output_path=plot_path)
        
        # Werte speichern
        br_data_path = filepath.replace('.pkl.gz', '_best_response.pkl.gz')
        br_tracker.save(br_data_path)
    
    avg_strategy = solver.average_strategy
    print(f"{len(avg_strategy)} information sets")
    
    if args.game.startswith('kuhn'):
        print("\nStrategy:")
        for key, actions in sorted(avg_strategy.items()):
            # Updated to match KeyGenerator format: (private, public, history, pid)
            card, public_cards, history, player = key
            print(f"\nCard: {card}, History: {history}, Player: {player}")
            for action, prob in actions.items():
                print(f"  {action}: {prob:.4f}")

if __name__ == "__main__":
    main()