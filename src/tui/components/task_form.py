from textual.app import ComposeResult
from textual.containers import Vertical, Horizontal, Grid
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, Select, Checkbox, Static

GAMES = [
    "kuhn_case1", "kuhn_case2", "kuhn_case3", "kuhn_case4",
    "leduc", "rhode_island", "twelve_card_poker",
    "royal_holdem", "small_island_holdem", "limit_holdem"
]

ALGORITHMS = [
    "fold", "cfr", "cfr_plus", "mccfr",
    "cfr_with_tree", "cfr_plus_with_tree",
    "cfr_with_flat_tree", "cfr_plus_with_flat_tree",
    "chance_sampling", "external_sampling", "outcome_sampling",
    "discounted_cfr", "discounted_cfr_with_tree", "discounted_cfr_with_flat_tree"
]

SCHEDULES = ["None", "standard", "low_density"]


class TaskForm(ModalScreen):
    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Label("Create Training Task", classes="title")

            with Grid(classes="form-grid"):
                yield Label("Game:")
                yield Select([(g, g) for g in GAMES], prompt="Select Game", id="input-game")

                yield Label("Algorithm:")
                yield Select([(a, a) for a in ALGORITHMS], prompt="Select Algorithm", id="input-algo")

                yield Label("Iterations:")
                yield Input(placeholder="1000", type="integer", id="input-iters")

                yield Label("BR Eval Schedule:")
                yield Select([(s, s) for s in SCHEDULES], value="standard", id="input-schedule")

            # Dynamic / Advanced Options
            with Vertical(classes="advanced-options"):
                yield Label("Advanced Options", classes="section-title")

                with Horizontal(classes="row"):
                    yield Label("Early Stop Exploitability (mb/g):", classes="label-small")
                    yield Input(placeholder="None", id="input-early-stop", classes="input-small")

                with Horizontal(classes="row"):
                    yield Checkbox("Alternating Updates", value=True, id="check-alternating")
                    yield Checkbox("Partial Pruning", value=False, id="check-pruning")

                # Game specific
                yield Checkbox("No Suit Abstraction", value=False, id="check-no-suit", classes="hidden")

                # Algo specific
                yield Checkbox("Squared Weight (CFR+)", value=False, id="check-squared", classes="hidden")

                with Horizontal(id="container-dcfr", classes="hidden row"):
                    yield Label("A:")
                    yield Input("1.5", type="number", id="input-alpha", classes="input-mini")
                    yield Label("B:")
                    yield Input("0.0", type="number", id="input-beta", classes="input-mini")
                    yield Label("G:")
                    yield Input("2.0", type="number", id="input-gamma", classes="input-mini")

            with Horizontal(classes="dialog-buttons"):
                yield Button("Cancel", variant="error", id="btn-cancel")
                yield Button("Queue Task", variant="success", id="btn-submit")

    def on_mount(self) -> None:
        pass

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "input-game":
            game = str(event.value)
            # Show suit abstraction option for supported games
            no_suit = self.query_one("#check-no-suit")
            if game in ["leduc", "twelve_card_poker"]:
                no_suit.remove_class("hidden")
            else:
                no_suit.add_class("hidden")

        elif event.select.id == "input-algo":
            algo = str(event.value)

            # Hide/Show Schedule based on sampling
            schedule = self.query_one("#input-schedule")
            if algo in ["mccfr", "chance_sampling", "external_sampling", "outcome_sampling", "fold"]:
                schedule.disabled = True
                schedule.value = "None"
            else:
                schedule.disabled = False

            # Squared Weight
            squared = self.query_one("#check-squared")
            if "cfr_plus" in algo:
                squared.remove_class("hidden")
            else:
                squared.add_class("hidden")

            # DCFR
            dcfr_container = self.query_one("#container-dcfr")
            if "discounted_cfr" in algo:
                dcfr_container.remove_class("hidden")
            else:
                dcfr_container.add_class("hidden")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-cancel":
            self.dismiss(None)
        elif event.button.id == "btn-submit":
            self._submit_form()

    def _submit_form(self) -> None:
        # Validate inputs
        game = self.query_one("#input-game", Select).value
        algo = self.query_one("#input-algo", Select).value
        iters_str = self.query_one("#input-iters", Input).value

        if not game or game == Select.BLANK:
            self.notify("Please select a game", severity="error")
            return
        if not algo or algo == Select.BLANK:
            self.notify("Please select an algorithm", severity="error")
            return
        if not iters_str:
            self.notify("Please enter iterations", severity="error")
            return

        try:
            iterations = int(iters_str)
        except ValueError:
            self.notify("Iterations must be a number", severity="error")
            return

        # Build task data
        data = {
            "game": game,
            "algorithm": algo,
            "iterations": iterations,
            "alternating_updates": self.query_one("#check-alternating", Checkbox).value,
            "partial_pruning": self.query_one("#check-pruning", Checkbox).value,
        }

        schedule = self.query_one("#input-schedule", Select).value
        if schedule != "None":
            data["br_eval_schedule"] = schedule

        if not self.query_one("#check-no-suit").has_class("hidden"):
            data["no_suit_abstraction"] = self.query_one("#check-no-suit", Checkbox).value

        if not self.query_one("#check-squared").has_class("hidden"):
            data["squared_weight"] = self.query_one("#check-squared", Checkbox).value

        if not self.query_one("#container-dcfr").has_class("hidden"):
            data["dcfr_alpha"] = float(self.query_one("#input-alpha", Input).value)
            data["dcfr_beta"] = float(self.query_one("#input-beta", Input).value)
            data["dcfr_gamma"] = float(self.query_one("#input-gamma", Input).value)

        early_stop = self.query_one("#input-early-stop", Input).value
        if early_stop:
            try:
                data["early_stop_exploitability_mb"] = float(early_stop)
            except ValueError:
                self.notify("Early stop value must be a number", severity="error")
                return

        self.dismiss(data)
