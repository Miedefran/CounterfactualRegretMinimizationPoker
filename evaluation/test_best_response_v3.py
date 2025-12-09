import sys
import os
import gzip
import pickle as pkl
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.kuhn_poker.game import KuhnPokerGame
from evaluation.best_response_v3 import BestResponsePolicy
from utils.poker_utils import KuhnPokerCombinations
from training.cfr_solver import CFRSolver


def load_average_strategy(strategy_file):
    with gzip.open(strategy_file, 'rb') as f:
        data = pkl.load(f)
    
    strategy_sum = data['strategy_sum']
    average_strategy = CFRSolver.average_from_strategy_sum(strategy_sum)
    
    return average_strategy


def test_best_response_against_strategy(strategy_file):
    print("=" * 60)
    print(f"Test: Best Response gegen gespeicherte Strategie")
    print(f"Strategie: {strategy_file}")
    print("=" * 60)
    
    print("\n1. Lade Strategie...")
    average_strategy = load_average_strategy(strategy_file)
    print(f"   ✓ Strategie geladen ({len(average_strategy)} InfoSets)")
    
    game_class = KuhnPokerGame
    game_config = {'ante': 1, 'bet_size': 1}
    combo_gen = KuhnPokerCombinations()
    
    print("\n2. Berechne Best Response für Player 0...")
    br_p0 = BestResponsePolicy(
        game_class=game_class,
        game_config=game_config,
        player_id=0,
        policy=average_strategy,
        combination_generator=combo_gen
    )
    print(f"   ✓ {len(br_p0.infosets)} Information Sets für Player 0 gefunden")
    
    root_game_p0 = br_p0._create_game()
    root_game_p0.reset(0)
    value_p0 = br_p0.value(root_game_p0)
    print(f"   ✓ Best Response Value für Player 0: {value_p0:.6f}")
    
    print("\n3. Berechne Best Response für Player 1...")
    br_p1 = BestResponsePolicy(
        game_class=game_class,
        game_config=game_config,
        player_id=1,
        policy=average_strategy,
        combination_generator=combo_gen
    )
    print(f"   ✓ {len(br_p1.infosets)} Information Sets für Player 1 gefunden")
    
    root_game_p1 = br_p1._create_game()
    root_game_p1.reset(0)
    value_p1 = br_p1.value(root_game_p1)
    print(f"   ✓ Best Response Value für Player 1: {value_p1:.6f}")
    
    print("\n4. Berechne Durchschnitt...")
    average_value = (value_p0 + value_p1) / 2.0
    print(f"   ✓ Durchschnittlicher Best Response Value: {average_value:.6f}")
    print(f"   (Bei Nash Equilibrium sollte dieser Wert ≈ 0 sein)")
    
    print("\n5. Zeige Best Response Actions für Player 0:")
    for infostate, states in br_p0.infosets.items():
        action = br_p0.best_response_action(infostate)
        print(f"   InfoState: {infostate}")
        print(f"   → Best Action: {action}")
        print(f"   → Anzahl States im InfoSet: {len(states)}")
        total_cf_prob = sum(cf_prob for _, cf_prob in states)
        print(f"   → Summe CF Probabilities: {total_cf_prob:.6f}")
        if infostate in average_strategy:
            print(f"     Gegner Policy: {average_strategy[infostate]}")
        print()
    
    print("\n6. Test action_probabilities()...")
    test_game = br_p0._create_game()
    test_game.reset(0)
    combo_gen.setup_game_with_combination(test_game, ('J', 'Q'))
    probs = br_p0.action_probabilities(test_game, player_id=0)
    print(f"   ✓ Action probabilities (Player 0 am Anfang): {probs}")
    
    print("\n" + "=" * 60)
    print("Test abgeschlossen!")
    print("=" * 60)


def test_info_sets(strategy_file):
    print("\n" + "=" * 60)
    print("Test: Information Sets Struktur")
    print("=" * 60)
    
    average_strategy = load_average_strategy(strategy_file)
    combo_gen = KuhnPokerCombinations()
    
    br = BestResponsePolicy(
        KuhnPokerGame,
        {'ante': 1, 'bet_size': 1},
        player_id=0,
        policy=average_strategy,
        combination_generator=combo_gen
    )
    
    print(f"\nAnzahl Information Sets: {len(br.infosets)}")
    print("\nErste 3 Information Sets:")
    for i, (infostate, states) in enumerate(list(br.infosets.items())[:3]):
        print(f"\n{i+1}. {infostate}")
        print(f"   Anzahl States in diesem InfoSet: {len(states)}")
        total_cf_prob = sum(cf_prob for _, cf_prob in states)
        print(f"   Summe CF Probabilities: {total_cf_prob:.6f}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test Best Response v3')
    parser.add_argument('--strategy', type=str, 
                       default='models/kuhn/case2/cfr/kuhn_case2_10000.pkl.gz',
                       help='Path to strategy file (default: models/kuhn/case2/cfr/kuhn_case2_10000.pkl.gz)')
    args = parser.parse_args()
    
    try:
        test_best_response_against_strategy(args.strategy)
        test_info_sets(args.strategy)
        print("\n✓ Alle Tests erfolgreich!")
    except Exception as e:
        print(f"\n✗ Fehler: {e}")
        import traceback
        traceback.print_exc()

