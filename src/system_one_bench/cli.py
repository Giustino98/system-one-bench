"""Human-facing command line entry points."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Annotated

import typer

from system_one_bench.config import load_config
from system_one_bench.runner import dry_run_plan, run_benchmark

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Benchmark Jev and Qwen on BBH Logical Deduction.",
)


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO, format="%(levelname)s %(message)s"
    )


@app.command()
def plan(config: Annotated[Path, typer.Argument(exists=True, readable=True)]) -> None:
    """Show the exact safe plan. This command never makes network/model calls."""
    typer.echo(json.dumps(dry_run_plan(load_config(config)), indent=2))


@app.command()
def run(
    config: Annotated[Path, typer.Argument(exists=True, readable=True)],
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
    resume: Annotated[
        Path | None,
        typer.Option("--resume", file_okay=False, help="Resume an interrupted run directory."),
    ] = None,
) -> None:
    """Execute a benchmark only when the YAML explicitly sets dry_run: false."""
    _configure_logging(verbose)
    output = run_benchmark(load_config(config), resume_directory=resume)
    typer.echo(f"Saved benchmark artifacts to {output}")


if __name__ == "__main__":
    app()
