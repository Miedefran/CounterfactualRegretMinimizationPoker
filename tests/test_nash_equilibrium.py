import pickle as pkl
import gzip
import sys

if len(sys.argv) < 2:
    print("Usage: python test_nash_equilibrium.py <strategy_file>")
    sys.exit(1)

strategy_file = sys.argv[1]

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
beta_expected = gamma / 3 + 1/3

alpha_ok = abs(alpha - alpha_expected) < 0.01
beta_ok = abs(beta - beta_expected) < 0.01
xi_ok = abs(xi - 1/3) < 0.01
eta_ok = abs(eta - 1/3) < 0.01

print("Constraints:")
print(f"  α = γ/3 → {alpha:.4f} vs {alpha_expected:.4f} ({'PASS' if alpha_ok else 'FAIL'})")
print(f"  β = γ/3 + 1/3 → {beta:.4f} vs {beta_expected:.4f} ({'PASS' if beta_ok else 'FAIL'})")
print(f"  ξ = 1/3 → {xi:.4f} vs 0.3333 ({'PASS' if xi_ok else 'FAIL'})")
print(f"  η = 1/3 → {eta:.4f} vs 0.3333 ({'PASS' if eta_ok else 'FAIL'})\n")

if alpha_ok and beta_ok and xi_ok and eta_ok:
    print("Valid Nash Equilibrium")
