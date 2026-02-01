from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Button

from tui.components.task_queue import TaskQueue
from tui.components.task_form import TaskForm


class TrainingScreen(Vertical):
    def compose(self) -> ComposeResult:
        with Horizontal(classes="toolbar"):
            yield Button("Add Task", id="btn-add-task", variant="primary")
            yield Button("Pause Queue", id="btn-pause-queue", variant="warning")
        yield TaskQueue(id="task-queue")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-add-task":
            self.app.push_screen(TaskForm(), callback=self.on_task_created)
        elif event.button.id == "btn-pause-queue":
            queue = self.query_one(TaskQueue)
            is_paused = queue.toggle_pause()
            event.button.label = "Resume Queue" if is_paused else "Pause Queue"
            event.button.variant = "success" if is_paused else "warning"

    def on_task_created(self, task_data: dict | None) -> None:
        if task_data:
            queue = self.query_one(TaskQueue)
            queue.add_task(task_data)
