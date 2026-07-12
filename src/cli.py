import os
import warnings
import logging
import time

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

from workflow import workflow
import argparse
import uuid
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.markdown import Markdown

console = Console()

parser = argparse.ArgumentParser(description="")
parser.add_argument("--thread_id", required=False, help="Thread ID. If not provided, a random UUID will be generated.")

args = parser.parse_args()

thread_id = str(args.thread_id) if args.thread_id else str(uuid.uuid4())

if __name__ == "__main__":
    console.print(Panel.fit("[bold blue]Constitution RAG CLI[/bold blue]", border_style="blue"))
    while True:
        console.print(Panel(f"Thread ID: [cyan]{thread_id}[/cyan]", expand=False, border_style="dim"))
        human = Prompt.ask("[bold green]Human[/bold green]")
        if human.lower() == "exit":
            console.print("[bold red]Exiting...[/bold red]")
            break
        elif human.lower() == "/new":
            thread_id = str(uuid.uuid4())
            console.print(Panel.fit(f"[bold yellow]Started a new session with thread ID:[/bold yellow] [cyan]{thread_id}[/cyan]", border_style="yellow"))
            continue
        
        initial_state = {
            "user_query": human,
            "k": 2,
            "max_retry_for_groundness_checking": 1,
            "max_retry_for_answer_relevant_checking": 1,
            "input_tokens": 0,
            "output_tokens": 0,
            "retrieved_contexts":[],
            "relevant_contexts":[]
        }

        start_time = time.time()
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task("[cyan]Running workflow...", total=None)
            
            for chunk in workflow.stream(
                initial_state, {"configurable": {"thread_id": thread_id}}, stream_mode="updates"
            ):
                node_name = list(chunk.keys())[0]
                progress.update(task, description=f"[cyan]Completed {node_name}...")

        response = workflow.get_state(config={"configurable": {"thread_id": thread_id}})
        elapsed_time = time.time() - start_time
        ai_response = response.values["generated_response"]
        in_tokens = response.values.get("input_tokens", 0)
        out_tokens = response.values.get("output_tokens", 0)
        
        console.print(
            Panel(
                Markdown(ai_response),
                title="[bold magenta]AI[/bold magenta]",
                subtitle=f"[dim]Time taken: {elapsed_time:.2f}s | Input Tokens: {in_tokens} | Output Tokens: {out_tokens}[/dim]",
                border_style="magenta",
            )
        )