import pickle
import gzip

path = 'evaluation/public_state_trees/leduc_public_tree.pkl.gz'
with gzip.open(path, 'rb') as f:
    tree = pickle.load(f)

states = tree['public_states']
print(f"Total public states: {len(states)}")

num_choice = sum(1 for s in states.values() if s['type'] == 'choice')
num_chance = sum(1 for s in states.values() if s['type'] == 'chance')
num_terminal = sum(1 for s in states.values() if s['type'] == 'terminal')

print(f"Choice nodes: {num_choice}")
print(f"Chance nodes: {num_chance}")
print(f"Terminal nodes: {num_terminal}")

root_node = states.get(())
if root_node:
    print(f"\nRoot state exists: ✓ (type: {root_node['type']}, player: {root_node['player']}, pot: {root_node['pot']})")

sample_terminal = None
for key, node in states.items():
    if node['type'] == 'terminal':
        sample_terminal = (key, node)
        break

if sample_terminal:
    hist_str = '|'.join(sample_terminal[0]) if sample_terminal[0] else '(root)'
    node = sample_terminal[1]
    print(f"\nSample terminal state:")
    print(f"  {hist_str}")
    print(f"  pot: {node['pot']}, bets: {node['player_bets']}")
    print(f"  player0 info sets: {len(node['player0_info_sets'])}, player1 info sets: {len(node['player1_info_sets'])}")

