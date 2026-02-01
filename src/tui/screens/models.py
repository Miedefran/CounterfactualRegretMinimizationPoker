import os
import sys
from typing import List, Dict, Tuple, Set
from pathlib import Path

from textual.app import ComposeResult
from textual.widgets import DataTable, Button, Input, Select, SelectionList, Label, Static
from textual.containers import Vertical, Horizontal, Grid, Container
from textual.screen import ModalScreen
from textual.binding import Binding
from textual import on
from textual.message import Message

# Import plotting functionality
# We need to make sure src is in path, which app.py does
try:
    from plot_best_response import plot_multiple_best_responses, extract_algorithm_name
except ImportError:
    # Fallback if running directly or path issues
    sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
    from plot_best_response import plot_multiple_best_responses, extract_algorithm_name

from utils.poker_utils import find_game_class_for_abstraction
from PyQt6.QtWidgets import QApplication
from gui.agent_vs_human import AgentVsHumanGUI


class PlotCreationModal(ModalScreen):
    CSS = """
    PlotCreationModal {
        align: center middle;
    }
    
    #dialog {
        padding: 1 2;
        border: thick $primary;
        background: $surface;
        width: 60%;
        height: auto;
        max-height: 90%;
    }
    
    .dialog-title {
        text-align: center;
        text-style: bold;
        margin-bottom: 1;
    }
    
    .field-label {
        margin-top: 1;
        margin-bottom: 0; 
    }
    
    .dialog-buttons {
        align: center middle;
        height: auto;
        margin-top: 2;
    }
    
    Button {
        margin: 0 1;
    }
    """

    def __init__(self, models_data: Dict[str, Dict[str, List[Tuple[str, str]]]]):
        """
        models_data: {game: {iters: [(model_name, full_path), ...]}}
        """
        super().__init__()
        self.models_data = models_data
        self.selected_game = None
        self.selected_iters = None

    def compose(self) -> ComposeResult:
        with Container(id="dialog"):
            yield Label("Create Comparison Plot", classes="dialog-title")

            yield Label("Plot Name (Filename)", classes="field-label")
            yield Input(placeholder="e.g. my_comparison", id="input-name")

            yield Label("Game", classes="field-label")
            yield Select([], prompt="Select Game", id="select-game")

            yield Label("Iterations", classes="field-label")
            yield Select([], prompt="Select Iterations", id="select-iters", disabled=True)

            yield Label("Models to Compare", classes="field-label")
            yield SelectionList(id="list-models")

            with Horizontal(classes="dialog-buttons"):
                yield Button("Create", variant="primary", id="btn-create")
                yield Button("Cancel", variant="error", id="btn-cancel")

    def on_mount(self) -> None:
        # Populate Games
        games = sorted(list(self.models_data.keys()))
        game_options = [(g, g) for g in games]
        self.query_one("#select-game", Select).set_options(game_options)

    @on(Select.Changed, "#select-game")
    def on_game_selected(self, event: Select.Changed) -> None:
        if event.value == Select.BLANK:
            self.selected_game = None
            self.query_one("#select-iters", Select).disabled = True
            self.query_one("#select-iters", Select).set_options([])
            self.query_one("#list-models", SelectionList).clear_options()
            return

        self.selected_game = event.value
        iters_map = self.models_data[self.selected_game]

        # Iterations are strings in the dict keys
        # Sort them numerically if possible
        def try_int(s):
            try:
                return int(s)
            except:
                return 0

        sorted_iters = sorted(list(iters_map.keys()), key=try_int)

        iter_options = [(str(it), str(it)) for it in sorted_iters]

        iter_select = self.query_one("#select-iters", Select)
        iter_select.set_options(iter_options)
        iter_select.disabled = False
        iter_select.value = Select.BLANK

        # Clear models
        self.query_one("#list-models", SelectionList).clear_options()

    @on(Select.Changed, "#select-iters")
    def on_iters_selected(self, event: Select.Changed) -> None:
        if event.value == Select.BLANK or self.selected_game is None:
            self.selected_iters = None
            self.query_one("#list-models", SelectionList).clear_options()
            return

        self.selected_iters = event.value
        models = self.models_data[self.selected_game][self.selected_iters]

        # models is list of (model_name, full_path)
        options = [(name, path) for name, path in models]

        model_list = self.query_one("#list-models", SelectionList)
        model_list.clear_options()
        model_list.add_options(options)

    @on(Button.Pressed, "#btn-cancel")
    def cancel(self) -> None:
        self.dismiss()

    @on(Button.Pressed, "#btn-create")
    def create_plot(self) -> None:
        plot_name = self.query_one("#input-name", Input).value.strip()
        selected_paths = self.query_one("#list-models", SelectionList).selected

        if not plot_name:
            self.notify("Please enter a plot name.", severity="error")
            return

        if not selected_paths:
            self.notify("Please select at least one model.", severity="error")
            return

        if len(selected_paths) < 1:
            self.notify("Please select at least one model.", severity="error")
            return

        # Ensure output directory exists
        output_dir = os.path.join("data", "plots", "comparisons")
        os.makedirs(output_dir, exist_ok=True)

        if not plot_name.endswith(".png"):
            plot_name += ".png"

        output_path = os.path.join(output_dir, plot_name)

        try:
            plot_multiple_best_responses(
                selected_paths,
                output_path=output_path,
                title=f"{self.selected_game} - {self.selected_iters} Iterations",
                log_log=True
            )
            self.notify(f"Plot created at {output_path}", severity="information")
            self.dismiss()
        except Exception as e:
            self.notify(f"Error creating plot: {str(e)}", severity="error")


class ModelBrowser(Vertical):
    BINDINGS = [
        Binding("p", "open_plot_modal", "Compare Models"),
        Binding("r", "refresh_models", "Refresh"),
        Binding("g", "play_against_model", "Play vs Model"),
        Binding("enter", "play_against_model", "Play vs Model", show=False),
    ]

    def compose(self) -> ComposeResult:
        yield DataTable(cursor_type="row")

    def on_mount(self) -> None:
        table = self.query_one(DataTable)
        table.add_columns("Game", "Model", "Iterations")
        # Initialize before load_models
        self.model_files = {}
        self.row_keys_list = []
        self.load_models()

    def load_models(self) -> None:
        table = self.query_one(DataTable)
        table.clear()

        # Scanned data structure for the modal
        # {game: {iters: [(model_name, full_path)]}}
        self.scanned_data = {}
        # Store model file paths and game names for play action
        # {row_key: (game_name, model_file_path)}
        self.model_files = {}
        # Store row keys in order for index lookup
        self.row_keys_list = []

        base_dir = Path("data/models")
        if not base_dir.exists():
            return

        # Walk structure: data/models/game/algo/iters/file.pkl.gz

        # Get games
        for game_dir in base_dir.iterdir():
            if not game_dir.is_dir():
                continue
            game_name = game_dir.name

            if game_name not in self.scanned_data:
                self.scanned_data[game_name] = {}

            # Get algos (models)
            for algo_dir in game_dir.iterdir():
                if not algo_dir.is_dir():
                    continue
                algo_name = algo_dir.name

                # Get iterations
                for iter_dir in algo_dir.iterdir():
                    if not iter_dir.is_dir():
                        continue
                    try:
                        iters = iter_dir.name
                        # Check for best_response file
                        # Pattern: <game>_<iters>_best_response.pkl.gz
                        # Or just look for any *_best_response.pkl.gz

                        br_files = list(iter_dir.glob("*_best_response.pkl.gz"))
                        if br_files:
                            br_file = br_files[0]  # Take the first one if multiple (should be one)

                            # Find the model file (same name without _best_response)
                            model_file = str(br_file).replace("_best_response.pkl.gz", ".pkl.gz")
                            if not Path(model_file).exists():
                                # Try to find any .pkl.gz file that isn't a best_response file
                                model_files = [f for f in iter_dir.glob("*.pkl.gz")
                                               if "_best_response" not in f.name]
                                if model_files:
                                    model_file = str(model_files[0])
                                else:
                                    continue  # Skip if no model file found

                            row_key = str(br_file)

                            # Add to table
                            table.add_row(game_name, algo_name, iters, key=row_key)

                            # Store row key in order for index lookup
                            self.row_keys_list.append(row_key)

                            # Store model file path for play action
                            self.model_files[row_key] = (game_name, model_file)

                            # Add to scanned data
                            if iters not in self.scanned_data[game_name]:
                                self.scanned_data[game_name][iters] = []

                            self.scanned_data[game_name][iters].append((algo_name, str(br_file)))

                    except Exception as e:
                        continue

    def action_open_plot_modal(self) -> None:
        if not hasattr(self, 'scanned_data') or not self.scanned_data:
            self.notify("No models found to compare.", severity="warning")
            return

        self.app.push_screen(PlotCreationModal(self.scanned_data))

    def action_refresh_models(self) -> None:
        self.load_models()
        self.notify("Models list refreshed.")

    def action_play_against_model(self) -> None:
        """Launch the PyQt6 GUI to play against the selected model."""
        table = self.query_one(DataTable)

        # Get the selected row
        if table.cursor_row is None:
            self.notify("Please select a model first.", severity="warning")
            return

        try:
            cursor_row = table.cursor_row

            # Get row key from our ordered list
            if cursor_row >= len(self.row_keys_list):
                self.notify("Could not find model file for selected row.", severity="error")
                return

            row_key_str = self.row_keys_list[cursor_row]

            if row_key_str not in self.model_files:
                self.notify("Could not find model file for selected row.", severity="error")
                return

            game_name, model_file = self.model_files[row_key_str]

            # Determine the full game name for instantiation
            # Handle special cases like 'kuhn' -> 'kuhn_case2'
            if game_name == 'kuhn':
                # Try to extract case from the model file path
                if 'case1' in model_file:
                    full_game_name = 'kuhn_case1'
                elif 'case3' in model_file:
                    full_game_name = 'kuhn_case3'
                elif 'case4' in model_file:
                    full_game_name = 'kuhn_case4'
                else:
                    full_game_name = 'kuhn_case2'
            else:
                full_game_name = game_name

            # Check if model file exists
            if not Path(model_file).exists():
                self.notify(f"Model file not found: {model_file}", severity="error")
                return

            # Find and instantiate the game
            game_class = find_game_class_for_abstraction(full_game_name, False)
            if game_class is None:
                self.notify(f"Unknown game: {full_game_name}", severity="error")
                return

            game = game_class()

            # Launch the PyQt6 GUI
            app = QApplication.instance() or QApplication(sys.argv)
            window = AgentVsHumanGUI(game, strategy_file=model_file)
            window.show()
            app.exec()

        except Exception as e:
            self.notify(f"Error launching game: {str(e)}", severity="error")
