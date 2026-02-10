#!/usr/bin/env python3
import subprocess
import sys
import time
import argparse
from pathlib import Path

project_root = Path(__file__).parent.parent
gui_dir = Path(__file__).parent


def open_terminal_and_run(command, title="Terminal"):
    escaped_command = command.replace('\\', '\\\\').replace('"', '\\"')
    escaped_title = title.replace('"', '\\"')
    script = f'''
    tell application "Terminal"
        activate
        set newTab to do script "{escaped_command}"
        set custom title of newTab to "{escaped_title}"
    end tell
    '''
    subprocess.run(['osascript', '-e', script])


def main():
    parser = argparse.ArgumentParser(description='Start local Human vs Human test (Server + 2 Clients)')
    parser.add_argument('--game', default='limit_holdem',
                        choices=['kuhn', 'leduc', 'twelve_card', 'rhode_island', 'royal_holdem', 'limit_holdem'],
                        help='Game type (default: limit_holdem)')
    parser.add_argument('--port', type=int, default=8888,
                        help='Server port (default: 8888)')
    parser.add_argument('--name1', default='Player1',
                        help='Name for first client (default: Player1)')
    parser.add_argument('--name2', default='Player2',
                        help='Name for second client (default: Player2)')

    args = parser.parse_args()

    python_cmd = sys.executable
    server_script = str(gui_dir / "run_server.py")
    client_script = str(gui_dir / "run_client.py")
    project_root_str = str(project_root)

    server_cmd = f'cd "{project_root_str}" && "{python_cmd}" "{server_script}" --host 0.0.0.0 --port {args.port} --game {args.game}'
    client1_cmd = f'cd "{project_root_str}" && "{python_cmd}" "{client_script}" --ip localhost --port {args.port} --name "{args.name1}"'
    client2_cmd = f'cd "{project_root_str}" && "{python_cmd}" "{client_script}" --ip localhost --port {args.port} --name "{args.name2}"'

    print("Starting local Human vs Human test...")
    print(f"Game: {args.game}")
    print(f"Server Port: {args.port}")
    print(f"\nOpening 3 terminal windows:")
    print(f"  1. Server (port {args.port})")
    print(f"  2. Client 1 ({args.name1})")
    print(f"  3. Client 2 ({args.name2})")
    print(f"\nWaiting 2 seconds for server to start before opening clients...")

    open_terminal_and_run(server_cmd, "Server")

    time.sleep(2)

    open_terminal_and_run(client1_cmd, f"Client 1 - {args.name1}")

    time.sleep(1)

    open_terminal_and_run(client2_cmd, f"Client 2 - {args.name2}")

    print("\n✅ All terminals opened!")
    print("Close the terminals or press Ctrl+C in each to stop.")


if __name__ == "__main__":
    main()
