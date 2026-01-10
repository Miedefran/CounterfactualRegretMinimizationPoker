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
from training.chance_sampling_cfr_solver import ChanceSamplingCFRSolver
from training.external_sampling_cfr_solver import ExternalSamplingCFRSolver
from envs.kuhn_poker.game import KuhnPokerGame
from envs.leduc_holdem.game import LeducHoldemGame
from envs.rhode_island.game import RhodeIslandGame
from envs.twelve_card_poker.game import TwelveCardPokerGame
from envs.royal_holdem.game import RoyalHoldemGame
from envs.limit_holdem.game import LimitHoldemGame

from utils.poker_utils import (
    GAME_CONFIGS,
    KuhnPokerCombinations,
    LeducHoldemCombinations,
    RhodeIslandCombinations,
    TwelveCardPokerCombinations,
    RoyalHoldemCombinations,
    LimitHoldemCombinations,
    get_model_path
)
from training.best_response_evaluator import BestResponseTracker

def main():
    parser = argparse.ArgumentParser(description='Train CFR for Poker')
    parser.add_argument('game', type=str,
                       choices=['kuhn_case1', 'kuhn_case2', 'kuhn_case3', 'kuhn_case4', 'leduc', 'rhode_island', 'twelve_card_poker', 'royal_holdem', 'limit_holdem'],
                       help='Poker variant to train on')
    parser.add_argument('iterations', type=int,
                       help='Number of CFR iterations')
    parser.add_argument('algorithm', type=str,
                       choices=['fold', 'cfr', 'cfr_plus', 'mccfr', 'tensor_cfr', 'cfr_with_tree', 'chance_sampling', 'external_sampling'],
                       nargs='?',
                       default='cfr',
                       help='Algorithm to use (default: cfr)')
    parser.add_argument('--br-eval-schedule', type=str, default=None,
                       help='Best Response Evaluierungs-Schedule: Integer (fester Intervall), JSON-Pfad, oder Schedule-Name aus config/br_eval_schedules.json (None = deaktiviert)')
    args = parser.parse_args()
    config = GAME_CONFIGS[args.game]
    
    print(f"Training {args.game} for {args.iterations} iterations")
    
    if args.game.startswith('kuhn'):
        game = KuhnPokerGame(ante=config['ante'], bet_size=config['bet_size'])
        combo_gen = KuhnPokerCombinations()
    elif args.game.startswith('leduc'):
        game = LeducHoldemGame(ante=config['ante'], bet_sizes=config['bet_sizes'], bet_limit=config['bet_limit'])
        combo_gen = LeducHoldemCombinations()
    elif args.game.startswith('rhode'):
        game = RhodeIslandGame(ante=config['ante'], bet_sizes=config['bet_sizes'], bet_limit=config['bet_limit'])
        combo_gen = RhodeIslandCombinations()
    elif args.game == 'twelve_card_poker':
        game = TwelveCardPokerGame(ante=config['ante'], bet_sizes=config['bet_sizes'], bet_limit=config['bet_limit'])
        combo_gen = TwelveCardPokerCombinations()
    elif args.game == 'royal_holdem':
        game = RoyalHoldemGame(ante=config['ante'], bet_sizes=config['bet_sizes'], bet_limit=config['bet_limit'])
        combo_gen = RoyalHoldemCombinations()
    elif args.game == 'limit_holdem':
        game = LimitHoldemGame(small_blind=config['small_blind'], big_blind=config['big_blind'], bet_sizes=config['bet_sizes'], bet_limit=config['bet_limit'])
        combo_gen = LimitHoldemCombinations()
    
    if args.algorithm == 'cfr':
        solver = CFRSolver(game, combo_gen)
    elif args.algorithm == 'cfr_plus':
        solver = CFRPlusSolver(game, combo_gen)
    elif args.algorithm == 'mccfr':
        solver = MCCFRSolver(game, combo_gen)
    elif args.algorithm == 'fold':
        solver = AlwaysFoldSolver(game, combo_gen)
    elif args.algorithm == 'tensor_cfr':
        # Übergebe game_name für automatisches Laden des Trees
        solver = TensorCFRSolver(game, combo_gen, game_name=args.game)
    elif args.algorithm == 'cfr_with_tree':
        # Übergebe game_name für automatisches Laden des Trees
        solver = CFRSolverWithTree(game, combo_gen, game_name=args.game)
    elif args.algorithm == 'chance_sampling':
        # Übergebe game_name für automatisches Laden des Trees
        solver = ChanceSamplingCFRSolver(game, combo_gen, game_name=args.game)
    elif args.algorithm == 'external_sampling':
        # Übergebe game_name für automatisches Laden des Trees
        solver = ExternalSamplingCFRSolver(game, combo_gen, game_name=args.game)
    
    # Best Response Tracker initialisieren falls gewünscht
    br_tracker = None
    if args.br_eval_schedule is not None:
        try:
            if args.br_eval_schedule.isdigit():
                schedule_config = int(args.br_eval_schedule)
                br_tracker = BestResponseTracker(args.game, schedule_config=schedule_config)
                print(f"Best Response Evaluation aktiviert (fester Intervall: {schedule_config})")
            else:
                br_tracker = BestResponseTracker(args.game, schedule_config=args.br_eval_schedule)
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
    
    solver.train(args.iterations, br_tracker=br_tracker)
    
    filepath = get_model_path(args.game, args.iterations, args.algorithm)
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