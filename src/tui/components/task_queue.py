import asyncio
import time
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import DataTable
from textual.worker import Worker, get_current_worker
from training.train import run_training, TrainingConfig
from training.progress_reporter import get_global_reporter


class TaskQueue(Vertical):
    def compose(self) -> ComposeResult:
        yield DataTable()

    def on_mount(self) -> None:
        table = self.query_one(DataTable)
        table.add_column("ID", key="ID")
        table.add_column("Status", key="Status")
        table.add_column("Game", key="Game")
        table.add_column("Algorithm", key="Algorithm")
        table.add_column("Iterations", key="Iterations")
        table.add_column("PID", key="PID")
        self.queue = []
        self.running_task = None
        self.is_paused = False
        self.task_counter = 0

        # Start the queue processor
        self.run_worker(self._process_queue, exclusive=True, thread=True)

    def toggle_pause(self) -> bool:
        self.is_paused = not self.is_paused
        return self.is_paused

    def add_task(self, task_data: dict) -> None:
        self.task_counter += 1
        task_id = self.task_counter
        task_data["id"] = task_id
        task_data["status"] = "Queued"

        table = self.query_one(DataTable)
        table.add_row(
            str(task_id),
            "Queued",
            task_data["game"],
            task_data["algorithm"],
            str(task_data["iterations"]),
            "-"
        )

        self.queue.append(task_data)

    def _create_config(self, task: dict) -> TrainingConfig:
        return TrainingConfig(
            game=task["game"],
            iterations=int(task["iterations"]),
            algorithm=task["algorithm"],
            br_eval_schedule=task.get("br_eval_schedule"),
            alternating_updates=(str(task.get("alternating_updates", "true")).lower() == 'true'),
            partial_pruning=(str(task.get("partial_pruning", "false")).lower() == 'true'),
            no_suit_abstraction=task.get("no_suit_abstraction", False),
            dcfr_alpha=float(task.get("dcfr_alpha", 1.5)),
            dcfr_beta=float(task.get("dcfr_beta", 0.0)),
            dcfr_gamma=float(task.get("dcfr_gamma", 2.0)),
            squared_weight=task.get("squared_weight", False),
            early_stop_exploitability_mb=float(task["early_stop_exploitability_mb"]) if task.get(
                "early_stop_exploitability_mb") else None
        )

    def _update_row(self, task_id: int, status: str, pid: str = "-") -> None:
        table = self.query_one(DataTable)
        # Find row index by ID (column 0)
        # This is a bit inefficient for large tables but fine for now
        for row_key in table.rows:
            row_data = table.get_row(row_key)
            if row_data[0] == str(task_id):
                table.update_cell(row_key, "Status", status)
                table.update_cell(row_key, "PID", pid)
                break

    def _process_queue(self) -> None:
        worker = get_current_worker()
        progress_reporter = get_global_reporter()

        while not worker.is_cancelled:
            if not self.is_paused and not self.running_task and self.queue:
                # Get next task
                task = self.queue.pop(0)
                self.running_task = task

                # Clear previous progress
                progress_reporter.clear()

                # Update UI
                self.app.call_from_thread(self._update_row, task["id"], "Running", "Thread")

                try:
                    config = self._create_config(task)
                    run_training(config, progress_reporter=progress_reporter)
                    self.app.call_from_thread(self._update_row, task["id"], "Completed", "Thread")

                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    self.app.call_from_thread(self._update_row, task["id"], f"Error: {str(e)}")

                self.running_task = None

            # Sleep a bit to prevent busy loop
            time.sleep(0.1)
