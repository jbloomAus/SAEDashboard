import json
import math
import os
from pathlib import Path

import requests
import torch
import typer
from rich import print
from rich.align import Align
from rich.panel import Panel
from sae_lens.analysis.neuronpedia_integration import NanAndInfReplacer
from sae_lens.sae import SAE
from sae_lens.toolkit.pretrained_saes import load_sparsity
from typing_extensions import Annotated

from sae_dashboard.neuronpedia.neuronpedia_runner import (
    DEFAULT_SPARSITY_THRESHOLD,
    NeuronpediaRunner,
    NeuronpediaRunnerConfig,
)

OUTPUT_DIR_BASE = Path("./neuronpedia_outputs")
RUN_SETTINGS_FILE = "run_settings.json"

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Tool that generates features (generate) and uploads features (upload) to Neuronpedia.",
)


@app.command()
def generate(
    ctx: typer.Context,
    sae_set: Annotated[
        str,
        typer.Option(
            help="SAE ID to generate features for (must exactly match the one used on Neuronpedia). Example: res-jb",
            prompt="""
What is the SAE ID you want to generate features for?
This was set when you did 'Add SAEs' on Neuronpedia. This must exactly match that ID (including casing).
It's in the format [abbrev hook name]-[abbrev author name], like res-jb.
Enter SAE ID""",
        ),
    ],
    sae_path: Annotated[
        Path,
        typer.Option(
            exists=True,
            dir_okay=True,
            readable=True,
            resolve_path=True,
            help="Absolute local path to the SAE directory (with cfg.json, sae_weights.safetensors, sparsity.safetensors).",
            prompt="""
What is the absolute local path to your SAE's directory (with cfg.json, sae_weights.safetensors, sparsity.safetensors)?
Enter path""",
        ),
    ],
    log_sparsity: Annotated[
        int,
        typer.Option(
            min=-10,
            max=1,
            help="Desired feature log sparsity threshold. Range -10 to 0. Use 1 to skip sparsity check.",
            prompt="""
What is your desired feature log sparsity threshold?
Enter value from -10 to 0 [1 to skip]""",
        ),
    ] = DEFAULT_SPARSITY_THRESHOLD,
    dtype: Annotated[
        str,
        typer.Option(
            help="Override DType",
            prompt="""
Override DType type?
[Enter to use SAE default]""",
        ),
    ] = "float32",
    feat_per_batch: Annotated[
        int,
        typer.Option(
            min=1,
            max=2048,
            help="Features to generate per batch. More requires more memory.",
            prompt="""
How many features do you want to generate per batch? More requires more memory.
Enter value""",
        ),
    ] = 128,
    n_prompts: Annotated[
        int,
        typer.Option(
            min=1,
            help="[Activation Text Generation] Number of prompts to select from.",
            prompt="""
[Activation Text Generation] How many prompts do you want to select from?
Enter value""",
        ),
    ] = 24576,
    n_context_tokens: Annotated[
        int,
        typer.Option(
            min=0,
            help="[Activation Text Generation] Override the context tokens length.",
            prompt="""
[Activation Text Generation] Override the context tokens length? (0 = Default)
Enter value""",
        ),
    ] = 0,
    resume_from_batch: Annotated[
        int,
        typer.Option(
            min=0,
            help="Batch number to resume from.",
            prompt="""
Do you want to resume from a specific batch number?
Enter 0 to start from the beginning. Existing batch files will not be overwritten.""",
        ),
    ] = 0,
    end_at_batch: Annotated[
        int,
        typer.Option(
            min=-1,
            help="Batch number to end at.",
            prompt="""
Do you want to end at a specific batch number?
Enter -1 to do all batches. Existing batch files will not be overwritten.""",
        ),
    ] = -1,
):
    """
    This will start a batch job that generates features for Neuronpedia for a specific SAE. To upload those features, use the 'upload' command afterwards.
    """

    print("\nRe-run command with:\n")
    command = "python neuronpedia.py generate"
    for key, value in ctx.params.items():
        command += f" --{key.replace('_', '-')}={value}"  # type: ignore
    print(command + "\n\n")

    # Check arguments
    if sae_path.is_dir() is not True:
        print("Error: SAE path must be a directory.")
        raise typer.Abort()
    if sae_path.joinpath("cfg.json").is_file() is not True:
        print("Error: cfg.json file not found in SAE directory.")
        raise typer.Abort()
    if sae_path.joinpath("sae_weights.safetensors").is_file() is not True:
        print("Error: sae_weights.safetensors file not found in SAE directory.")
        raise typer.Abort()
    # Allow skipping sparsity file
    if (
        log_sparsity != 1
        and sae_path.joinpath("sparsity.safetensors").is_file() is not True
    ):
        print("Error: sparsity.safetensors file not found in SAE directory.")
        raise typer.Abort()

    sae_path_string = sae_path.as_posix()

    # Load SAE
    device = "cpu"
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    sparse_autoencoder = SAE.load_from_pretrained(sae_path_string, device=device)
    model_id = sparse_autoencoder.cfg.model_name
    if dtype is None:
        dtype = sparse_autoencoder.cfg.dtype

    # make the outputs subdirectory if it doesn't exist, ensure it's not a file
    outputs_subdir = f"{model_id}_{sae_set}_{sparse_autoencoder.cfg.hook_name}"
    outputs_dir = OUTPUT_DIR_BASE.joinpath(outputs_subdir)
    if outputs_dir.exists() and outputs_dir.is_file():
        print(f"Error: Output directory {outputs_dir.as_posix()} exists and is a file.")
        raise typer.Abort()
    outputs_dir.mkdir(parents=True, exist_ok=True)

    # Check if output_dir has a run_settings.json file. If so, load those settings.
    run_settings_path = outputs_dir.joinpath(RUN_SETTINGS_FILE)
    print("\n")
    if run_settings_path.exists() and run_settings_path.is_file():
        # load the json file
        with open(run_settings_path, "r") as f:
            run_settings = json.load(f)
            print(
                f"[yellow]Found existing run_settings.json in {run_settings_path.as_posix()}, checking them."
            )
            if run_settings["log_sparsity"] != log_sparsity:
                print(
                    f"[red]Error: log_sparsity in {run_settings_path.as_posix()} doesn't match the current log_sparsity:\n{run_settings['log_sparsity']} vs {log_sparsity}"
                )
                raise typer.Abort()
            if run_settings["sae_set"] != sae_set:
                print(
                    f"[red]Error: sae_set in {run_settings_path.as_posix()} doesn't match the current sae_set:\n{run_settings['sae_set']} vs {sae_set}"
                )
                raise typer.Abort()
            if run_settings["dtype"] != dtype:
                print(
                    f"[red]Error: dtype in {run_settings_path.as_posix()} doesn't match the current dtype:\n{run_settings['dtype']} vs {dtype}"
                )
                raise typer.Abort()
            if run_settings["sae_path"] != sae_path_string:
                print(
                    f"[red]Error: sae_path in {run_settings_path.as_posix()} doesn't match the current sae_path:\n{run_settings['sae_path']} vs {sae_path_string}"
                )
                raise typer.Abort()
            if run_settings["n_prompts"] != n_prompts:
                print(
                    f"[red]Error: n_prompts in {run_settings_path.as_posix()} doesn't match the current n_prompts:\n{run_settings['n_prompts']} vs {n_prompts}"
                )
                raise typer.Abort()
            if run_settings["n_context_tokens"] != n_context_tokens:
                print(
                    f"[red]Error: n_context_tokens in {run_settings_path.as_posix()} doesn't match the current n_context_tokens:\n{run_settings['n_context_tokens']} vs {n_context_tokens}"
                )
                raise typer.Abort()
            if run_settings["feat_per_batch"] != feat_per_batch:
                print(
                    f"[red]Error: feat_per_batch in {run_settings_path.as_posix()} doesn't match the current feat_per_batch:\n{run_settings['feat_per_batch']} vs {feat_per_batch}"
                )
                raise typer.Abort()
            print("[green]All settings match, using existing run_settings.json.")
    else:
        print(f"[green]Creating run_settings.json in {run_settings_path.as_posix()}.")
        run_settings = {
            "sae_set": sae_set,
            "sae_path": sae_path_string,
            "log_sparsity": log_sparsity,
            "dtype": dtype,
            "n_prompts": n_prompts,
            "n_context_tokens": n_context_tokens,
            "feat_per_batch": feat_per_batch,
        }
        with open(run_settings_path, "w") as f:
            json.dump(run_settings, f, indent=4)

    if log_sparsity == 1:
        num_alive = sparse_autoencoder.cfg.d_sae
        num_dead = 0
    else:
        sparsity = load_sparsity(sae_path_string)
        # convert sparsity to logged sparsity if it's not
        # TODO: standardize the sparsity file format
        if len(sparsity) > 0 and sparsity[0] >= 0:
            sparsity = torch.log10(sparsity + 1e-10)
        sparsity = sparsity.to(device)
        alive_indexes = (sparsity > log_sparsity).nonzero(as_tuple=True)[0].tolist()
        num_alive = len(alive_indexes)
        num_dead = sparse_autoencoder.cfg.d_sae - num_alive

    num_batches = math.ceil(sparse_autoencoder.cfg.d_sae / feat_per_batch)
    if end_at_batch >= num_batches:
        print(
            f"[red]Error: end_at_batch {end_at_batch} should not be >= num_batches {num_batches}"
        )
        raise typer.Abort()

    print("\n")
    print(
        Align.center(
            Panel.fit(
                f"""
[white]SAE Path: [green]{sae_path.as_posix()}
[white]Model ID: [green]{model_id}
[white]Hook Point: [green]{sparse_autoencoder.cfg.hook_name}
[white]DType: [green]{dtype}
[white]Using Device: [green]{device}
""",
                title="SAE Info",
            )
        )
    )
    print(
        Align.center(
            Panel.fit(
                f"""
[white]Total Features: [green]{sparse_autoencoder.cfg.d_sae}
[white]Log Sparsity Threshold: [green]{log_sparsity}
[white]Alive Features: [green]{num_alive}
[white]Dead Features: [red]{num_dead}
[white]Features per Batch: [green]{feat_per_batch}
[white]Number of Batches: [green]{num_batches}
{resume_from_batch != 0 and f"[white]Resuming from Batch: [green]{resume_from_batch}" or ""}
{end_at_batch != -1 and f"[white]Ending at Batch: [green]{end_at_batch}" or ""}
""",
                title="Number of Features",
            )
        )
    )
    print(
        Align.center(
            Panel.fit(
                f"""
[white]Dataset: [green]{sparse_autoencoder.cfg.dataset_path}
[white]Prompts to Sample From: [green]{n_prompts}
[white]Context Token Length: [green]{n_context_tokens if n_context_tokens != 0 else 0}
""",
                title="Activation Text Settings",
            )
        )
    )
    print(
        Align.center(
            Panel.fit(
                f"""
[green]{outputs_dir.absolute().as_posix()}
""",
                title="Output Directory",
            )
        )
    )

    print(
        Align.center(
            "\n========== [yellow]Starting batch feature generations...[/yellow] ==========\n"
        )
    )

    print("\n\nRe-run with command:\n")
    print(command + "\n\n")

    # run the command
    cfg = NeuronpediaRunnerConfig(
        sae_set=sae_set,
        sae_path=sae_path.absolute().as_posix(),
        dtype=dtype,
        outputs_dir=outputs_dir.absolute().as_posix(),
        sparsity_threshold=log_sparsity,
        n_prompts_total=n_prompts,
        n_tokens_in_prompt=n_context_tokens if n_context_tokens != 0 else 128,
        n_features_at_a_time=feat_per_batch,
        start_batch=resume_from_batch,
        end_batch=end_at_batch if end_at_batch != -1 else num_batches,
    )

    runner = NeuronpediaRunner(cfg)
    runner.run()

    print(
        Align.center(
            Panel(
                f"""
Your Features Are In: [green]{outputs_dir.absolute().as_posix()}
Use [yellow]'neuronpedia.py upload'[/yellow] to upload your features to Neuronpedia.
""",
                title="Generation Complete",
            )
        )
    )


@app.command()
def upload(
    outputs_dir: Annotated[
        Path,
        typer.Option(
            exists=True,
            dir_okay=True,
            readable=True,
            resolve_path=True,
            prompt="What is the absolute local file path to the feature outputs directory?",
        ),
    ],
    host: Annotated[
        str,
        typer.Option(
            prompt="""Host to upload to? (Default: http://localhost:3000)""",
        ),
    ] = "http://localhost:3000",
    resume: Annotated[
        int,
        typer.Option(
            prompt="""Resume from batch? (Default: 0)""",
        ),
    ] = 0,
):
    """
    This will upload features that were generated to Neuronpedia. It currently only works if you have admin access to a Neuronpedia instance via localhost:3000.
    """

    files_to_upload = list(outputs_dir.glob("batch-*.json"))

    # filter files where batch-[number].json the number is >= resume
    files_to_upload = [
        file_path
        for file_path in files_to_upload
        if int(file_path.stem.split("-")[1]) >= resume
    ]

    # sort files by batch number
    files_to_upload.sort(key=lambda x: int(x.stem.split("-")[1]))

    print("\n")
    # Upload alive features
    for file_path in files_to_upload:
        print("===== Uploading file: " + os.path.basename(file_path))
        f = open(file_path, "r")
        data = json.load(f, parse_constant=NanAndInfReplacer)

        url = host + "/api/local/upload-features"
        requests.post(
            url,
            json=data,
        )

    print(
        Align.center(
            Panel(
                f"""
{len(files_to_upload)} batch files uploaded to Neuronpedia.
""",
                title="Uploads Complete",
            )
        )
    )


@app.command()
def upload_dead_stubs(
    outputs_dir: Annotated[
        Path,
        typer.Option(
            exists=True,
            dir_okay=True,
            readable=True,
            resolve_path=True,
            prompt="What is the absolute local file path to the feature outputs directory?",
        ),
    ],
    host: Annotated[
        str,
        typer.Option(
            prompt="""Host to upload to? (Default: http://localhost:3000)""",
        ),
    ] = "http://localhost:3000",
):
    """
    This will create "There are no activations for this feature" stubs for dead features on Neuronpedia.  It currently only works if you have admin access to a Neuronpedia instance via localhost:3000.
    """

    skipped_path = os.path.join(outputs_dir, "skipped_indexes.json")
    f = open(skipped_path, "r")
    data = json.load(f)
    url = host + "/api/local/upload-skipped-features"
    requests.post(
        url,
        json=data,
    )

    print(
        Align.center(
            Panel(
                """
Dead feature stubs created.
""",
                title="Complete",
            )
        )
    )


if __name__ == "__main__":
    app()
