import asyncio

import typer

from prompt_runner.client import (
    LLMClient,
    MockClient,
    OpenAIClient,
    validate_temperature,
)
from prompt_runner.runner import read_prompts, run_prompts, write_results


app = typer.Typer()


@app.command()
def run(
    input_path: str = typer.Option(..., "--input"),
    output_path: str = typer.Option(..., "--output"),
    model: str = typer.Option("gpt-5-mini", "--model"),
    temperature: float | None = typer.Option(None, "--temperature"),
    max_concurrency: int = typer.Option(5, "--max-concurrency"),
    timeout: float = typer.Option(30.0, "--timeout"),
    mock: bool = typer.Option(False, "--mock"),
) -> None:
    try:
        validate_temperature(temperature)
    except ValueError as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=2)

    client: LLMClient

    if mock:
        client = MockClient()
    else:
        client = OpenAIClient()

    prompt_rows, skipped_rows = read_prompts(input_path)

    results = asyncio.run(
        run_prompts(
            client=client,
            prompt_rows=prompt_rows,
            model=model,
            temperature=temperature,
            max_concurrency=max_concurrency,
            timeout=timeout,
        )
    )

    write_results(output_path, results)

    succeeded = sum(
        1 for result in results
        if result.error is None
    )
    failed = len(results) - succeeded

    typer.echo(
        f"Processed {len(prompt_rows)} rows, "
        f"succeeded {succeeded}, "
        f"failed {failed}, "
        f"skipped {skipped_rows} invalid rows."
    )

    typer.echo(f"Results written to {output_path}")

    if failed > 0:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()