import sys
import os
import socket
import threading
import multiprocessing
import time

# Add src directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, TabbedContent, TabPane

from tui.screens.training import TrainingScreen
from tui.screens.models import ModelBrowser
from tui.memray_pane import MemrayPane
from tui.memray_adapter import reader_process_func, RemoteReaderClient
from tui.components.current_task_pane import CurrentTaskPane

import memray
from memray import SocketDestination


def get_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


class PokerCFRApp(App):
    CSS_PATH = "app.tcss"
    TITLE = "Poker CFR Manager"

    BINDINGS = [
        ("ctrl+q", "quit", "Quit"),
    ]

    def __init__(self, reader_client):
        super().__init__()
        self.reader_client = reader_client

    def compose(self) -> ComposeResult:
        yield Header()
        with TabbedContent():
            with TabPane("Training Queue", id="tab-training"):
                yield TrainingScreen()
            with TabPane("Current Task", id="tab-current"):
                yield CurrentTaskPane()
            with TabPane("Models", id="tab-models"):
                yield ModelBrowser()
            with TabPane("Memory Profiler", id="tab-memray"):
                yield MemrayPane(self.reader_client, pid=os.getpid(), cmd_line=" ".join(sys.argv))
        yield Footer()


if __name__ == "__main__":
    port = get_free_port()
    print(f"Starting Memray setup on port {port}...")

    # Create communication queue
    queue = multiprocessing.Queue(maxsize=1)

    # Start reader process
    reader_proc = multiprocessing.Process(target=reader_process_func, args=(port, queue))
    reader_proc.start()

    # Allow reader to start polling
    time.sleep(0.5)

    try:
        # Tracker blocks here until reader connects
        print("Waiting for Memray Tracker to connect...")
        with memray.Tracker(destination=SocketDestination(server_port=port)):
            print("Memray Tracker connected.")

            # Initialize client for TUI
            client = RemoteReaderClient(queue)
            app = PokerCFRApp(client)
            app.run()

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback

        traceback.print_exc()
    finally:
        # Cleanup
        if reader_proc.is_alive():
            reader_proc.terminate()
        reader_proc.join(timeout=1.0)
