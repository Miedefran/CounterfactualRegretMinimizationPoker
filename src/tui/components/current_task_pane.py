"""
Current Task Pane for displaying live training progress.

Shows task details at the top and BR evaluation values in a table below.
"""

import time
from datetime import datetime
from textual.app import ComposeResult
from textual.containers import Vertical, Horizontal
from textual.widgets import Static, DataTable, ProgressBar, Label
from textual.reactive import reactive

from training.progress_reporter import get_global_reporter, TaskProgress


def format_time(seconds: float) -> str:
    """Format seconds into a human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def format_datetime(timestamp: float) -> str:
    """Format a timestamp into a datetime string."""
    if timestamp == 0:
        return "-"
    return datetime.fromtimestamp(timestamp).strftime("%H:%M:%S")


class TaskDetailsPanel(Static):
    """Panel showing current task details."""

    DEFAULT_CSS = """
    TaskDetailsPanel {
        height: auto;
        padding: 1;
        border: solid $primary;
        margin-bottom: 1;
    }

    TaskDetailsPanel .label {
        color: $text-muted;
    }

    TaskDetailsPanel .value {
        color: $text;
        text-style: bold;
    }

    TaskDetailsPanel .row {
        height: auto;
        margin-bottom: 0;
    }

    TaskDetailsPanel ProgressBar {
        margin-top: 1;
        width: 100%;
    }
    """

    def compose(self) -> ComposeResult:
        with Horizontal(classes="row"):
            yield Static("Game: ", classes="label")
            yield Static("-", id="game-value", classes="value")
            yield Static("  Algorithm: ", classes="label")
            yield Static("-", id="algo-value", classes="value")
            yield Static("  Suit Abstraction: ", classes="label")
            yield Static("-", id="suit-value", classes="value")
        with Horizontal(classes="row"):
            yield Static("Target: ", classes="label")
            yield Static("-", id="target-value", classes="value")
            yield Static("  Current: ", classes="label")
            yield Static("-", id="current-value", classes="value")
            yield Static("  Progress: ", classes="label")
            yield Static("-", id="progress-value", classes="value")
        with Horizontal(classes="row"):
            yield Static("Started: ", classes="label")
            yield Static("-", id="started-value", classes="value")
            yield Static("  Elapsed: ", classes="label")
            yield Static("-", id="elapsed-value", classes="value")
            yield Static("  ETA: ", classes="label")
            yield Static("-", id="eta-value", classes="value")
        with Horizontal(classes="row"):
            yield Static("Status: ", classes="label")
            yield Static("No task running", id="status-value", classes="value")
        yield ProgressBar(total=100, show_eta=False, id="progress-bar")

    def update_from_progress(self, progress: TaskProgress) -> None:
        """Update the panel with new progress data."""
        if not progress.game:
            # No task
            self.query_one("#game-value", Static).update("-")
            self.query_one("#algo-value", Static).update("-")
            self.query_one("#suit-value", Static).update("-")
            self.query_one("#target-value", Static).update("-")
            self.query_one("#current-value", Static).update("-")
            self.query_one("#progress-value", Static).update("-")
            self.query_one("#started-value", Static).update("-")
            self.query_one("#elapsed-value", Static).update("-")
            self.query_one("#eta-value", Static).update("-")
            self.query_one("#status-value", Static).update("No task running")
            self.query_one("#progress-bar", ProgressBar).update(progress=0)
            return

        # Update values
        self.query_one("#game-value", Static).update(progress.game)
        self.query_one("#algo-value", Static).update(progress.algorithm)
        self.query_one("#suit-value", Static).update("Yes" if progress.suit_abstraction else "No")
        self.query_one("#target-value", Static).update(f"{progress.target_iterations:,}")
        self.query_one("#current-value", Static).update(f"{progress.current_iteration:,}")

        # Progress percentage
        pct = progress.get_progress_fraction() * 100
        self.query_one("#progress-value", Static).update(f"{pct:.1f}%")

        # Times
        self.query_one("#started-value", Static).update(format_datetime(progress.start_time))
        self.query_one("#elapsed-value", Static).update(format_time(progress.get_elapsed_time()))

        eta = progress.get_estimated_remaining_time()
        if eta is not None:
            self.query_one("#eta-value", Static).update(format_time(eta))
        else:
            self.query_one("#eta-value", Static).update("-")

        # Status
        if progress.error:
            status = f"Error: {progress.error}"
        elif progress.is_complete:
            status = "Completed"
        elif progress.is_running:
            status = "Running"
        else:
            status = "Unknown"
        self.query_one("#status-value", Static).update(status)

        # Progress bar
        self.query_one("#progress-bar", ProgressBar).update(progress=pct)


class BRValuesTable(Vertical):
    """Table showing Best Response evaluation values."""

    DEFAULT_CSS = """
    BRValuesTable {
        height: 1fr;
    }

    BRValuesTable DataTable {
        height: 1fr;
    }

    BRValuesTable .table-header {
        height: auto;
        padding: 0 1;
        margin-bottom: 1;
    }
    """

    def compose(self) -> ComposeResult:
        yield Static("Best Response Evaluations", classes="table-header")
        yield DataTable()

    def on_mount(self) -> None:
        table = self.query_one(DataTable)
        table.add_column("Iteration", key="iteration")
        table.add_column("P0 (mb/g)", key="p0")
        table.add_column("P1 (mb/g)", key="p1")
        table.add_column("Exploitability (mb/g)", key="exploitability")
        table.add_column("Eval Time", key="eval_time")
        table.add_column("Training Time", key="train_time")
        table.cursor_type = "row"

    def update_from_progress(self, progress: TaskProgress) -> None:
        """Update the table with BR values."""
        table = self.query_one(DataTable)

        # Get current row count
        current_rows = table.row_count

        # Add new rows if needed
        for i in range(current_rows, len(progress.br_values)):
            br = progress.br_values[i]
            table.add_row(
                f"{br.iteration:,}",
                f"{br.br_value_p0_mb:.6f}",
                f"{br.br_value_p1_mb:.6f}",
                f"{br.exploitability_mb:.6f}",
                format_time(br.eval_time),
                format_time(br.cumulative_training_time),
            )

        # Scroll to bottom if new rows were added
        if len(progress.br_values) > current_rows:
            table.scroll_end()

    def clear_table(self) -> None:
        """Clear all rows from the table."""
        table = self.query_one(DataTable)
        table.clear()


class CurrentTaskPane(Vertical):
    """
    Pane showing the current training task details and BR evaluation values.

    Polls the global progress reporter for updates.
    """

    DEFAULT_CSS = """
    CurrentTaskPane {
        height: 100%;
        padding: 1;
    }
    """

    # Track the last known BR values count to detect new values
    _last_br_count = reactive(0)

    def compose(self) -> ComposeResult:
        yield TaskDetailsPanel()
        yield BRValuesTable()

    def on_mount(self) -> None:
        """Start polling for progress updates."""
        self.reporter = get_global_reporter()
        self._last_game = ""

        # Poll every 500ms
        self.set_interval(0.5, self._update_from_reporter)

    def _update_from_reporter(self) -> None:
        """Fetch progress from reporter and update UI."""
        progress = self.reporter.get_progress()

        # Check if task changed (new game started)
        if progress.game != self._last_game:
            self._last_game = progress.game
            # Clear the BR table for new task
            self.query_one(BRValuesTable).clear_table()

        # Update panels
        self.query_one(TaskDetailsPanel).update_from_progress(progress)
        self.query_one(BRValuesTable).update_from_progress(progress)
