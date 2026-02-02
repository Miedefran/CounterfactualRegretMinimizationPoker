import dataclasses
from datetime import datetime
from typing import Optional, Dict

# Memray imports
from memray.reporters.tui import (
    Header,
    AllocationTable,
    MemoryGraph,
    Snapshot,
    _EMPTY_SNAPSHOT
)
from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container
from textual.reactive import reactive
from textual.widgets import Label, Footer

# derived from https://github.com/bloomberg/memray/blob/main/src/memray/reporters/tui.py

class MyAllocationTable(AllocationTable):
    """Subclass to override behavior dependent on Screen actions."""

    def get_heading(self, column_idx: int) -> Text:
        sort_column = (
            self.sort_column_id if self._composed else self.default_sort_column_id
        )
        sort_column_style = self.get_component_rich_style(
            "allocationtable--sorted-column-heading",
            partial=True,
        )
        if column_idx in (0, len(self.columns) - 1):
            return Text(self.columns[column_idx], justify="center")
        elif column_idx in self.HIGHLIGHTED_COLUMNS_BY_SORT_COLUMN[sort_column]:
            return Text(
                self.columns[column_idx], justify="right", style=sort_column_style
            )
        else:
            return Text(self.columns[column_idx], justify="right").on(
                click=f"memray_sort({self.SORT_COLUMN_BY_CLICKED_COLUMN[column_idx]})"
            )


class MemrayHeader(Header):
    """Subclass of Memray Header to avoid CSS conflicts with Textual Header."""
    pass


class MemrayPane(Container):
    BINDINGS = [
        Binding("m", "toggle_merge_threads", "Disable Thread Merging"),
        Binding("left", "previous_thread", "Previous Thread"),
        Binding("<", "previous_thread", "Previous Thread"),
        Binding("right", "next_thread", "Next Thread"),
        Binding(">", "next_thread", "Next Thread"),
        Binding("t", "memray_sort(1)", "Sort by Total"),
        Binding("o", "memray_sort(3)", "Sort by Own"),
        Binding("a", "memray_sort(5)", "Sort by Allocations"),
        Binding("p", "toggle_pause", "Pause Memray"),
    ]

    _DUMMY_THREAD_LIST = [0]

    thread_idx = reactive(0)
    threads = reactive(_DUMMY_THREAD_LIST, always_update=True)
    snapshot = reactive(_EMPTY_SNAPSHOT)
    paused = reactive(True, init=False)  # Updates beim Start aus; mit "p" einschalten
    disconnected = reactive(False, init=False)

    def __init__(self, reader_client, pid: Optional[int], cmd_line: Optional[str]):
        super().__init__()
        self.reader_client = reader_client
        self.pid = pid
        self.cmd_line = cmd_line

        self._name_by_tid: Dict[int, str] = {}
        self._max_memory_seen = 0
        self._merge_threads = True
        self._latest_snapshot = _EMPTY_SNAPSHOT
        self._latest_thread_names = {}

    def on_mount(self):
        self.set_interval(0.5, self.update_from_reader)
        self.action_toggle_merge_threads()
        self.action_toggle_merge_threads()

    def update_from_reader(self):
        # Poll the remote reader client for new data
        snap, thread_names, disc = self.reader_client.poll()

        if snap is not _EMPTY_SNAPSHOT:
            self._latest_thread_names = thread_names
            self.snapshot = snap

        if disc:
            self.disconnected = True

    @property
    def current_thread(self) -> int:
        if not self.threads:
            return 0
        return self.threads[self.thread_idx]

    def compose(self) -> ComposeResult:
        yield MemrayHeader(pid=self.pid, cmd_line=self.cmd_line)
        yield MyAllocationTable()

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        """Check if an action may run."""
        if action in ("previous_thread", "next_thread"):
            return not self._merge_threads
        return True

    def action_memray_sort(self, col_number: int) -> None:
        self.update_sort_key(col_number)

    def update_sort_key(self, col_number: int) -> None:
        body = self.query_one(MyAllocationTable)
        body.sort_column_id = col_number

    def action_previous_thread(self) -> None:
        if self.threads:
            self.thread_idx = (self.thread_idx - 1) % len(self.threads)

    def action_next_thread(self) -> None:
        if self.threads:
            self.thread_idx = (self.thread_idx + 1) % len(self.threads)

    def action_toggle_merge_threads(self) -> None:
        self._merge_threads = not self._merge_threads

        self.query_one(MyAllocationTable).merge_threads = self._merge_threads
        self._populate_header_thread_labels(self.thread_idx)

        keys = self._bindings.key_to_bindings
        if "m" in keys:
            current_bindings = keys["m"]
            current_binding = current_bindings[0]
            new_desc = "Disable thread merging" if self._merge_threads else "Enable thread merging"
            if current_binding.description != new_desc:
                keys["m"][0] = dataclasses.replace(current_binding, description=new_desc)

        self.refresh_bindings()
        footer = self.app.screen.query_one(Footer)
        footer.refresh(recompose=True)

    def action_toggle_pause(self) -> None:
        if self.paused or not self.disconnected:
            self.paused = not self.paused
            if not self.paused:
                self.display_snapshot()

    def watch_thread_idx(self, thread_idx: int) -> None:
        self._populate_header_thread_labels(thread_idx)
        try:
            self.query_one(MyAllocationTable).current_thread = self.current_thread
        except Exception:
            pass

    def watch_threads(self) -> None:
        self._populate_header_thread_labels(self.thread_idx)

    def watch_disconnected(self) -> None:
        self.update_label()

    def watch_paused(self) -> None:
        self.update_label()

    def watch_snapshot(self, snapshot: Snapshot) -> None:
        self._latest_snapshot = snapshot
        self.display_snapshot()

    def _populate_header_thread_labels(self, thread_idx: int) -> None:
        if self._merge_threads:
            tid_label = "[b]TID[/]: *"
            thread_label = "[b]All threads[/]"
        else:
            tid_label = f"[b]TID[/]: {hex(self.current_thread)}"
            thread_label = f"[b]Thread[/] {thread_idx + 1} of {len(self.threads)}"
            thread_name = self._name_by_tid.get(self.current_thread)
            if thread_name:
                thread_label += f" ({thread_name})"

        self.query_one("#tid", Label).update(tid_label)
        self.query_one("#thread", Label).update(thread_label)

    def update_label(self) -> None:
        status_message = []
        if self.paused:
            status_message.append("[yellow]Table updates paused[/]")
        if self.disconnected:
            status_message.append("[red]Remote has disconnected[/]")
        if status_message:
            status_message.insert(0, "[b]Status[/]:")

        try:
            self.query_one("#status_message", Label).update(" ".join(status_message))
        except Exception:
            pass

    def display_snapshot(self) -> None:
        snapshot = self._latest_snapshot

        if snapshot is _EMPTY_SNAPSHOT:
            return

        try:
            header = self.query_one(Header)
            body = self.query_one(MyAllocationTable)
            graph = self.query_one(MemoryGraph)
        except Exception:
            return

        header.n_samples += 1
        header.last_update = datetime.now()

        graph.add_value(snapshot.heap_size)

        if self.paused:
            return

        name_by_tid = self._latest_thread_names
        new_tids = name_by_tid.keys() - self._name_by_tid.keys()
        self._name_by_tid.update(name_by_tid)

        if new_tids:
            threads = self.threads
            if threads is self._DUMMY_THREAD_LIST:
                threads = []
            for tid in sorted(new_tids):
                threads.append(tid)
            self.threads = threads

        body.current_thread = self.current_thread
        if not self.paused:
            body.snapshot = snapshot
