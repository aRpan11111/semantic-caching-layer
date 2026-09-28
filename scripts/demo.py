import time
import httpx
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console(highlight=False)

BASE_URL = "http://127.0.0.1:8000/v1/chat/completions"
STATS_URL = "http://127.0.0.1:8000/v1/cache/stats"

def run_demo():
    console.print("\n[bold cyan]=== Starting Live Semantic Caching Layer Demo ===[/bold cyan]\n")

    queries = [
        ("Query 1 (Original)", "What is Python programming language and why is it popular?"),
        ("Query 2 (Exact Duplicate)", "What is Python programming language and why is it popular?"),
        ("Query 3 (Semantically Equivalent)", "Can you explain Python programming language and why people use it?"),
        ("Query 4 (Semantically Similar Code)", "How do I reverse a string in Python?"),
        ("Query 5 (Equivalent Code Request)", "Write python code to reverse a string"),
        ("Query 6 (Time-sensitive Query)", "What is the stock price of Apple today?")
    ]

    table = Table(title="Semantic Cache Live Performance", show_header=True, header_style="bold magenta")
    table.add_column("Step", style="dim", width=30)
    table.add_column("Prompt Sample", width=40)
    table.add_column("Status", width=12)
    table.add_column("Similarity", width=12)
    table.add_column("Latency (ms)", width=15)

    with httpx.Client(timeout=30.0) as client:
        for title, prompt in queries:
            payload = {
                "model": "llama-3.1-8b-instant",
                "messages": [
                    {"role": "system", "content": "You are a helpful technical assistant."},
                    {"role": "user", "content": prompt}
                ]
            }

            t0 = time.time()
            try:
                res = client.post(BASE_URL, json=payload)
                elapsed_ms = (time.time() - t0) * 1000
                headers = res.headers
                
                is_hit = headers.get("X-Cache-Hit") == "true"
                sim_score = headers.get("X-Cache-Similarity", "N/A")
                
                status_str = "[bold green]HIT[/bold green]" if is_hit else "[bold yellow]MISS[/bold yellow]"
                lat_str = f"[green]{elapsed_ms:.1f} ms[/green]" if is_hit else f"[red]{elapsed_ms:.1f} ms[/red]"
                
                table.add_row(title, prompt[:38] + "...", status_str, sim_score, lat_str)
            except Exception as e:
                console.print(f"[bold red]Error connecting to cache proxy server: {e}[/bold red]")
                console.print("[yellow]Make sure the server is running on http://127.0.0.1:8000 (python -m uvicorn src.main:app)[/yellow]")
                return

    console.print(table)

    # Fetch stats summary
    try:
        stats_res = httpx.get(STATS_URL).json()
        stats_table = Table(title="Overall Cache Metrics & Financial Impact", show_header=False)
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", style="bold green")

        for k, v in stats_res.items():
            stats_table.add_row(k.replace("_", " ").title(), str(v))

        console.print("\n", stats_table)
    except Exception:
        pass

if __name__ == "__main__":
    run_demo()

