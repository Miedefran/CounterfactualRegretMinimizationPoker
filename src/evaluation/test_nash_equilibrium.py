import pickle as pkl
import gzip
import argparse
from utils.test_logger import log_nash, extract_iterations_from_filename, extract_game_from_filename


def main():
    parser = argparse.ArgumentParser(description='Nash Equilibrium Test for Kuhn Poker')
    parser.add_argument('strategy_file', type=str, help='Path to strategy pickle file')
    parser.add_argument('--save', action='store_true', help='Save results to CSV')
    args = parser.parse_args()

    strategy_file = args.strategy_file

    print(f"Testing Nash Equilibrium: {strategy_file}\n")

    with gzip.open(strategy_file, 'rb') as f:
        data = pkl.load(f)

    strategy_sum = data['strategy_sum']
    avg_strategy = {}

    for info_set_key in strategy_sum:
        total = sum(strategy_sum[info_set_key].values())
        if total > 0:
            avg_strategy[info_set_key] = {
                action: strategy_sum[info_set_key][action] / total
                for action in strategy_sum[info_set_key]
            }

    alpha = avg_strategy[('J', (), 0)]['bet']
    gamma = avg_strategy[('K', (), 0)]['bet']
    beta = avg_strategy[('Q', ('check', 'bet'), 0)]['call']

    xi = avg_strategy[('J', ('check',), 1)]['bet']
    eta = avg_strategy[('Q', ('bet',), 1)]['call']

    print(f"α (Jack bet): {alpha:.4f}")
    print(f"γ (King bet): {gamma:.4f}")
    print(f"β (Queen call): {beta:.4f}")
    print(f"ξ (Jack bluff): {xi:.4f}")
    print(f"η (Queen call): {eta:.4f}\n")

    alpha_expected = gamma / 3
    beta_expected = gamma / 3 + 1 / 3

    alpha_ok = abs(alpha - alpha_expected) < 0.01
    beta_ok = abs(beta - beta_expected) < 0.01
    xi_ok = abs(xi - 1 / 3) < 0.0
    eta_ok = abs(eta - 1 / 3) < 0.01

    print("Constraints:")
    print(f"  α = γ/3 → {alpha:.4f} vs {alpha_expected:.4f} ({'PASS' if alpha_ok else 'FAIL'})")
    print(f"  β = γ/3 + 1/3 → {beta:.4f} vs {beta_expected:.4f} ({'PASS' if beta_ok else 'FAIL'})")
    print(f"  ξ = 1/3 → {xi:.4f} vs 0.3333 ({'PASS' if xi_ok else 'FAIL'})")
    print(f"  η = 1/3 → {eta:.4f} vs 0.3333 ({'PASS' if eta_ok else 'FAIL'})\n")

    all_passed = alpha_ok and beta_ok and xi_ok and eta_ok

    if all_passed:
        print("Valid Nash Equilibrium")

    if args.save:
        iterations = extract_iterations_from_filename(strategy_file)
        game_name = extract_game_from_filename(strategy_file)

        if iterations and game_name:
            log_nash('vanilla_cfr', game_name, iterations,
                     alpha=alpha,
                     beta=beta,
                     gamma=gamma,
                     xi=xi,
                     eta=eta,
                     passed=1 if all_passed else 0)
        else:
            print("Warning: Could not extract iterations/game from filename, skipping save")


if __name__ == "__main__":
    main()
