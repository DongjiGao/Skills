from typer.main import get_command

from nemo_skills.pipeline.summarize_results import app


def test_summarize_results_supports_hyphen_and_underscore_options():
    command = get_command(app)
    if hasattr(command, "commands"):
        command = next(iter(command.commands.values()))

    option_names = {option for param in command.params for option in getattr(param, "opts", ())}
    expected = {
        "--metric_type",
        "--metric-type",
        "--save_metrics_path",
        "--save-metrics-path",
        "--wandb_name",
        "--wandb-name",
        "--wandb_group",
        "--wandb-group",
        "--wandb_project",
        "--wandb-project",
        "--metrics_kwargs",
        "--metrics-kwargs",
    }
