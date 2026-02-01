"""
Progress Reporter for CFR Training.

Provides a thread-safe way to communicate training progress
from the training process to the TUI.
"""

import threading
import time
from dataclasses import dataclass, field
from typing import Optional, List, Callable


@dataclass
class BRValue:
    """Represents a single Best Response evaluation result."""
    iteration: int
    br_value_p0_mb: float
    br_value_p1_mb: float
    exploitability_mb: float
    eval_time: float  # Time taken for this evaluation
    cumulative_training_time: float  # Total training time (excluding BR evals)


@dataclass
class TaskProgress:
    """Current state of a training task."""
    # Task info
    game: str = ""
    algorithm: str = ""
    target_iterations: int = 0
    start_time: float = 0.0
    suit_abstraction: bool = False

    # Progress
    current_iteration: int = 0
    is_running: bool = False
    is_complete: bool = False
    error: Optional[str] = None

    # BR evaluation values
    br_values: List[BRValue] = field(default_factory=list)

    def get_elapsed_time(self) -> float:
        """Returns elapsed time in seconds."""
        if self.start_time == 0:
            return 0.0
        return time.time() - self.start_time

    def get_progress_fraction(self) -> float:
        """Returns progress as a fraction (0.0 to 1.0)."""
        if self.target_iterations == 0:
            return 0.0
        return min(1.0, self.current_iteration / self.target_iterations)

    def get_estimated_remaining_time(self) -> Optional[float]:
        """
        Estimates remaining time based on current progress.
        Returns None if not enough data.
        """
        if self.current_iteration == 0 or self.target_iterations == 0:
            return None

        elapsed = self.get_elapsed_time()
        if elapsed <= 0:
            return None

        iterations_remaining = self.target_iterations - self.current_iteration
        time_per_iteration = elapsed / self.current_iteration
        return iterations_remaining * time_per_iteration


class ProgressReporter:
    """
    Thread-safe progress reporter that can be shared between
    the training thread and the TUI.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._progress = TaskProgress()
        self._callbacks: List[Callable[[], None]] = []

    def add_callback(self, callback: Callable[[], None]) -> None:
        """Add a callback to be called when progress is updated."""
        with self._lock:
            self._callbacks.append(callback)

    def remove_callback(self, callback: Callable[[], None]) -> None:
        """Remove a callback."""
        with self._lock:
            if callback in self._callbacks:
                self._callbacks.remove(callback)

    def _notify_callbacks(self) -> None:
        """Notify all registered callbacks."""
        for callback in self._callbacks:
            try:
                callback()
            except Exception:
                pass  # Don't let callback errors affect training

    def start_task(self, game: str, algorithm: str, target_iterations: int,
                   suit_abstraction: bool = False) -> None:
        """Called when a new training task starts."""
        with self._lock:
            self._progress = TaskProgress(
                game=game,
                algorithm=algorithm,
                target_iterations=target_iterations,
                start_time=time.time(),
                suit_abstraction=suit_abstraction,
                is_running=True,
            )
        self._notify_callbacks()

    def update_iteration(self, iteration: int) -> None:
        """Called when an iteration completes."""
        with self._lock:
            self._progress.current_iteration = iteration
        self._notify_callbacks()

    def add_br_value(self, iteration: int, br_value_p0_mb: float,
                     br_value_p1_mb: float, exploitability_mb: float,
                     eval_time: float, cumulative_training_time: float) -> None:
        """Called when a BR evaluation completes."""
        with self._lock:
            self._progress.br_values.append(BRValue(
                iteration=iteration,
                br_value_p0_mb=br_value_p0_mb,
                br_value_p1_mb=br_value_p1_mb,
                exploitability_mb=exploitability_mb,
                eval_time=eval_time,
                cumulative_training_time=cumulative_training_time,
            ))
        self._notify_callbacks()

    def complete_task(self, error: Optional[str] = None) -> None:
        """Called when the task completes (successfully or with error)."""
        with self._lock:
            self._progress.is_running = False
            self._progress.is_complete = True
            self._progress.error = error
        self._notify_callbacks()

    def clear(self) -> None:
        """Clear all progress data."""
        with self._lock:
            self._progress = TaskProgress()
        self._notify_callbacks()

    def get_progress(self) -> TaskProgress:
        """
        Returns a copy of the current progress.
        Thread-safe.
        """
        with self._lock:
            # Return a shallow copy with a copy of the br_values list
            return TaskProgress(
                game=self._progress.game,
                algorithm=self._progress.algorithm,
                target_iterations=self._progress.target_iterations,
                start_time=self._progress.start_time,
                suit_abstraction=self._progress.suit_abstraction,
                current_iteration=self._progress.current_iteration,
                is_running=self._progress.is_running,
                is_complete=self._progress.is_complete,
                error=self._progress.error,
                br_values=list(self._progress.br_values),
            )


# Global instance for sharing between training and TUI
_global_reporter: Optional[ProgressReporter] = None
_reporter_lock = threading.Lock()


def get_global_reporter() -> ProgressReporter:
    """Get or create the global progress reporter instance."""
    global _global_reporter
    with _reporter_lock:
        if _global_reporter is None:
            _global_reporter = ProgressReporter()
        return _global_reporter
