import sys
import os
import gzip
import pickle as pkl
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.kuhn_poker.game import KuhnPokerGame
from envs.leduc_holdem.game import LeducHoldemGame
from envs.rhode_island.game import RhodeIslandGame
from evaluation.best_response_v3 import BestResponsePolicy
from utils.poker_utils import (
    GAME_CONFIGS,
    KuhnPokerCombinations,
    LeducHoldemCombinations,
    RhodeIslandCombinations
)
from training.cfr_solver import CFRSolver


class DebugBestResponse(BestResponsePolicy):
    """Wrapper um BestResponsePolicy mit vollständigem Logging"""
    
    def __init__(self, *args, **kwargs):
        self.log_file = open('test_best_response_v3_output.txt', 'w')
        self._indent = 0
        self._verbose_logging = False
        super().__init__(*args, **kwargs)
    
    def _log(self, msg):
        if self._verbose_logging:
            indent_str = "  " * self._indent
            print(f"{indent_str}{msg}", file=self.log_file)
            self.log_file.flush()
    
    def _enter(self, func_name, **kwargs):
        if self._verbose_logging:
            self._log(f">>> {func_name}({', '.join(f'{k}={v}' for k, v in kwargs.items())})")
            self._indent += 1
    
    def _exit(self, func_name, result):
        if self._verbose_logging:
            self._indent = max(0, self._indent - 1)
            self._log(f"<<< {func_name}() -> {result}")
    
    def value(self, game):
        game_key = tuple(game.history) if hasattr(game, 'history') else "unknown"
        current_player = game.current_player if hasattr(game, 'current_player') else "?"
        done = game.done if hasattr(game, 'done') else "?"
        
        self._enter("value", 
                   history=game_key, 
                   current_player=current_player, 
                   done=done)
        
        if game.done:
            payoff = game.get_payoff(self._player_id)
            self._exit("value", payoff)
            return payoff
        
        is_init_card = self._is_initial_card_distribution_node(game)
        self._log(f"  _is_initial_card_distribution_node: {is_init_card}")
        
        if is_init_card:
            self._log("  → Initial Card Distribution Node")
            transitions = self.transitions(game)
            self._log(f"  transitions: {transitions}")
            total = 0.0
            for action, prob in transitions:
                if prob > self._cut_threshold:
                    self._log(f"    action={action}, prob={prob}")
                    q_val = self.q_value(game, action)
                    contribution = prob * q_val
                    total += contribution
                    self._log(f"      q_value={q_val}, contribution={contribution}, total={total}")
            self._exit("value", total)
            return total
        
        if game.current_player == self._player_id:
            self._log("  → Best Responder Node")
            info_state = self._get_information_state(game)
            self._log(f"  info_state: {info_state}")
            action = self.best_response_action(info_state)
            self._log(f"  best_response_action: {action}")
            if action is None:
                self._exit("value", 0.0)
                return 0.0
            q_val = self.q_value(game, action)
            self._exit("value", q_val)
            return q_val
        
        self._log("  → Opponent/Chance Node")
        transitions = self.transitions(game)
        self._log(f"  transitions: {transitions}")
        total = 0.0
        for action, prob in transitions:
            if prob > self._cut_threshold:
                self._log(f"    action={action}, prob={prob}")
                q_val = self.q_value(game, action)
                contribution = prob * q_val
                total += contribution
                self._log(f"      q_value={q_val}, contribution={contribution}, total={total}")
        self._exit("value", total)
        return total
    
    def q_value(self, game, action):
        game_key = tuple(game.history) if hasattr(game, 'history') else "unknown"
        self._enter("q_value", action=action, history=game_key)
        
        child_game = self._apply_action(game, action)
        if child_game:
            child_key = tuple(child_game.history) if hasattr(child_game, 'history') else "unknown"
            self._log(f"  child_history: {child_key}")
            result = self.value(child_game)
            self._exit("q_value", result)
            return result
        else:
            self._exit("q_value", 0.0)
            return 0.0
    
    def best_response_action(self, infostate):
        self._enter("best_response_action", infostate=infostate)
        
        if infostate not in self.infosets:
            self._exit("best_response_action", None)
            return None
        
        infoset = self.infosets[infostate]
        self._log(f"  infoset size: {len(infoset)}")
        
        if not infoset:
            self._exit("best_response_action", None)
            return None
        
        first_state = infoset[0][0]
        legal_actions = first_state.get_legal_actions()
        self._log(f"  legal_actions: {legal_actions}")
        
        if not legal_actions:
            self._exit("best_response_action", None)
            return None
        
        best_action = None
        best_value = float('-inf')
        
        for action in legal_actions:
            self._log(f"    Testing action: {action}")
            action_value = 0.0
            for state, cf_prob in infoset:
                if cf_prob > self._cut_threshold:
                    state_key = tuple(state.history) if hasattr(state, 'history') else "unknown"
                    self._log(f"      state={state_key}, cf_prob={cf_prob:.6f}")
                    q_val = self.q_value(state, action)
                    contribution = cf_prob * q_val
                    action_value += contribution
                    self._log(f"        q_value={q_val:.6f}, contribution={contribution:.6f}, action_value={action_value:.6f}")
            
            self._log(f"    action={action}, total_action_value={action_value:.6f}")
            if action_value > best_value:
                best_value = action_value
                best_action = action
                self._log(f"      → NEW BEST")
        
        self._exit("best_response_action", f"{best_action} (value={best_value:.6f})")
        return best_action
    
    def transitions(self, game):
        game_key = tuple(game.history) if hasattr(game, 'history') else "unknown"
        self._enter("transitions", history=game_key, current_player=game.current_player)
        
        result = super().transitions(game)
        self._log(f"  result: {result}")
        self._exit("transitions", f"{len(result)} transitions")
        return result
    
    def decision_nodes(self, parent_game):
        game_key = tuple(parent_game.history) if hasattr(parent_game, 'history') else "unknown"
        self._enter("decision_nodes", history=game_key, current_player=parent_game.current_player)
        
        results = []
        for result in super().decision_nodes(parent_game):
            results.append(result)
            state, cf_prob = result
            state_key = tuple(state.history) if hasattr(state, 'history') else "unknown"
            self._log(f"  yielded: state={state_key}, cf_prob={cf_prob:.6f}")
        
        self._exit("decision_nodes", f"{len(results)} nodes")
        for result in results:
            yield result
    
    def __del__(self):
        if hasattr(self, 'log_file'):
            self.log_file.close()


def detect_game_from_path(strategy_path):
    """Erkennt das Spiel aus dem Strategie-Pfad"""
    path_lower = strategy_path.lower()
    
    if 'kuhn' in path_lower:
        if 'case1' in path_lower:
            return 'kuhn_case1', KuhnPokerGame, GAME_CONFIGS['kuhn_case1'], KuhnPokerCombinations()
        elif 'case2' in path_lower:
            return 'kuhn_case2', KuhnPokerGame, GAME_CONFIGS['kuhn_case2'], KuhnPokerCombinations()
        elif 'case3' in path_lower:
            return 'kuhn_case3', KuhnPokerGame, GAME_CONFIGS['kuhn_case3'], KuhnPokerCombinations()
        elif 'case4' in path_lower:
            return 'kuhn_case4', KuhnPokerGame, GAME_CONFIGS['kuhn_case4'], KuhnPokerCombinations()
        else:
            # Default zu case2
            return 'kuhn_case2', KuhnPokerGame, GAME_CONFIGS['kuhn_case2'], KuhnPokerCombinations()
    elif 'leduc' in path_lower:
        return 'leduc', LeducHoldemGame, GAME_CONFIGS['leduc'], LeducHoldemCombinations()
    elif 'rhode' in path_lower or 'rhode_island' in path_lower:
        return 'rhode_island', RhodeIslandGame, GAME_CONFIGS['rhode_island'], RhodeIslandCombinations()
    else:
        # Default zu kuhn_case2
        return 'kuhn_case2', KuhnPokerGame, GAME_CONFIGS['kuhn_case2'], KuhnPokerCombinations()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Debug Best Response Berechnung')
    parser.add_argument('--strategy', type=str, 
                       default='models/kuhn/case2/cfr/kuhn_case2_300.pkl.gz',
                       help='Path to strategy file (default: models/kuhn/case2/cfr/kuhn_case2_300.pkl.gz)')
    args = parser.parse_args()
    
    print("Starte Debug-Test...")
    print("Output wird in test_best_response_v3_output.txt geschrieben")
    print(f"Strategie: {args.strategy}")
    
    # Erkenne Spiel aus Pfad
    game_name, game_class, game_config, combo_gen = detect_game_from_path(args.strategy)
    print(f"Erkanntes Spiel: {game_name}")
    
    print("\n1. Lade Strategie...")
    with gzip.open(args.strategy, 'rb') as f:
        data = pkl.load(f)
    strategy_sum = data['strategy_sum']
    average_strategy = CFRSolver.average_from_strategy_sum(strategy_sum)
    print(f"   ✓ Strategie geladen ({len(average_strategy)} InfoSets)")
    
    print("\n2. Berechne Best Response für Player 0...")
    br_p0 = DebugBestResponse(
        game_class,
        game_config,
        player_id=0,
        policy=average_strategy,
        combination_generator=combo_gen
    )
    print(f"   ✓ BestResponse erstellt ({len(br_p0.infosets)} InfoSets)")
    root_p0 = br_p0._create_game()
    root_p0.reset(0)
    value_p0 = br_p0.value(root_p0)
    print(f"   ✓ Best Response Value für Player 0: {value_p0:.6f}")
    
    print("\n   ALLE InfoSets für Player 0:")
    for infostate, states in sorted(br_p0.infosets.items()):
        print(f"     {infostate}: {len(states)} states")
    
    print("\n3. Berechne Best Response für Player 1...")
    br_p1 = DebugBestResponse(
        game_class,
        game_config,
        player_id=1,
        policy=average_strategy,
        combination_generator=combo_gen
    )
    print(f"   ✓ BestResponse erstellt ({len(br_p1.infosets)} InfoSets)")
    root_p1 = br_p1._create_game()
    root_p1.reset(0)
    value_p1 = br_p1.value(root_p1)
    print(f"   ✓ Best Response Value für Player 1: {value_p1:.6f}")
    
    print("\n   ALLE InfoSets für Player 1:")
    for infostate, states in sorted(br_p1.infosets.items()):
        print(f"     {infostate}: {len(states)} states")
    
    print("\n4. Berechne Durchschnitt...")
    average_value = (value_p0 + value_p1) / 2.0
    print(f"   ✓ Durchschnittlicher Best Response Value: {average_value:.6f}")
    print(f"   (Bei Nash Equilibrium sollte dieser Wert ≈ 0 sein)")
    
    print("\n5. Fertig! Siehe test_best_response_v3_output.txt")

