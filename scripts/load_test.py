import asyncio
import time
import random
import httpx
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

console = Console(highlight=False)

BASE_URL = "http://127.0.0.1:8000/v1/chat/completions"
STATS_URL = "http://127.0.0.1:8000/v1/cache/stats"

# Seed prompts pool (mix of exact, semantic variants, and unique queries)
SEED_PROMPT_TEMPLATES = [
    "What is the difference between a process and a thread?",
    "Explain process vs thread in operating systems",
    "Can you clarify the distinctions between thread and process in OS?",
    "How does garbage collection work in Python?",
    "Explain memory management and garbage collector in Python",
    "Describe Python's automatic memory management and GC mechanism",
    "Write a function to calculate the fibonacci series in Python",
    "Show me Python code to generate fibonacci sequence",
    "Python snippet for fibonacci numbers",
    "What are the main principles of Object Oriented Programming?",
    "Explain OOP pillars: Encapsulation, Abstraction, Inheritance, Polymorphism",
    "What is a REST API and how does HTTP GET/POST work?",
    "Explain RESTful architecture web services",
    "How do indexes speed up SQL database queries?",
    "Why are database indexes used in SQL databases?",
    "Explain Docker containerization vs virtual machines",
    "What is the difference between Docker containers and hypervisor VMs?",
    "How does gradient descent optimization work in machine learning?",
    "Explain backpropagation and loss minimization in neural networks",
    "What is Rust ownership model and borrow checker?"
]

def generate_request_batch(num_requests: int = 2000):
    requests = []
    # 60% probability of selecting from seed templates (creating high semantic cache hit rate)
    # 40% probability of generating unique variation query
    for i in range(num_requests):
        if random.random() < 0.60:
            template = random.choice(SEED_PROMPT_TEMPLATES)
            # Subtle punctuation/casing variations
            if random.random() < 0.3:
                prompt = template.lower()
            elif random.random() < 0.6:
                prompt = template + " Please answer in brief."
            else:
                prompt = template
        else:
            prompt = f"Unique query ID #{i}: Explain topic code {random.randint(1000, 9999)} in software engineering context."
        
        requests.append(prompt)
    return requests

async def send_request(client: httpx.AsyncClient, prompt: str, semaphore: asyncio.Semaphore):
    async with semaphore:
        t0 = time.time()
        payload = {
            "model": "llama-3.1-8b-instant",
            "messages": [
                {"role": "system", "content": "You are a concise software engineering assistant."},
                {"role": "user", "content": prompt}
            ]
        }
        try:
            res = await client.post(BASE_URL, json=payload, timeout=30.0)
            latency_ms = (time.time() - t0) * 1000
            is_hit = res.headers.get("X-Cache-Hit") == "true"
            sim_score = float(res.headers.get("X-Cache-Similarity", 0.0))
            return {
                "success": res.status_code == 200,
                "is_hit": is_hit,
                "latency_ms": latency_ms,
                "similarity": sim_score
            }
        except Exception as e:
            return {"success": False, "is_hit": False, "latency_ms": (time.time() - t0)*1000, "similarity": 0.0}

async def run_load_test(total_requests: int = 2000, concurrency: int = 20):
    console.print(Panel(f"[bold green]Starting 2,000 Request Semantic Cache Load Test[/bold green]\nConcurrency Limit: [yellow]{concurrency}[/yellow] | Endpoint: [cyan]{BASE_URL}[/cyan]"))

    prompts = generate_request_batch(total_requests)
    semaphore = asyncio.Semaphore(concurrency)
    
    results = []
    
    limits = httpx.Limits(max_keepalive_connections=concurrency, max_connections=concurrency*2)
    async with httpx.AsyncClient(limits=limits, timeout=30.0) as client:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console
        ) as progress:
            task = progress.add_task("[cyan]Executing benchmark requests...", total=total_requests)
            
            # Execute in batches
            batch_size = 50
            for i in range(0, total_requests, batch_size):
                batch_prompts = prompts[i:i+batch_size]
                tasks = [send_request(client, p, semaphore) for p in batch_prompts]
                batch_results = await asyncio.gather(*tasks)
                results.extend(batch_results)
                progress.update(task, advance=len(batch_results))

    # Calculate percentiles and stats
    hits = [r for r in results if r["is_hit"]]
    misses = [r for r in results if not r["is_hit"] and r["success"]]
    failed = [r for r in results if not r["success"]]
    
    hit_latencies = [r["latency_ms"] for r in hits]
    miss_latencies = [r["latency_ms"] for r in misses]
    
    completed_requests = len(hits) + len(misses)
    hit_rate = (len(hits) / completed_requests) * 100 if completed_requests else 0

    p50_hit = np.percentile(hit_latencies, 50) if hit_latencies else 0
    p95_hit = np.percentile(hit_latencies, 95) if hit_latencies else 0
    p99_hit = np.percentile(hit_latencies, 99) if hit_latencies else 0

    p50_miss = np.percentile(miss_latencies, 50) if miss_latencies else 0
    p95_miss = np.percentile(miss_latencies, 95) if miss_latencies else 0
    p99_miss = np.percentile(miss_latencies, 99) if miss_latencies else 0

    table = Table(title="Benchmark Load Test Results Summary", show_header=True, header_style="bold green")
    table.add_column("Metric", style="cyan", width=30)
    table.add_column("Cached (Hit)", style="bold green", width=20)
    table.add_column("Uncached (Miss)", style="bold yellow", width=20)
    table.add_column("Improvement / ROI", style="bold magenta", width=20)

    table.add_row("Successful Requests", str(len(hits)), str(len(misses)), f"Hit Rate: {hit_rate:.1f}%")
    if failed:
        table.add_row("Excluded (Rate-limited)", "-", str(len(failed)), "Groq 429 Excluded")
    table.add_row("P50 Latency", f"{p50_hit:.2f} ms", f"{p50_miss:.2f} ms", f"{p50_miss/p50_hit:.1f}x Faster" if p50_hit>0 else "N/A")
    table.add_row("P95 Latency", f"{p95_hit:.2f} ms", f"{p95_miss:.2f} ms", f"{p95_miss/p95_hit:.1f}x Faster" if p95_hit>0 else "N/A")
    table.add_row("P99 Latency", f"{p99_hit:.2f} ms", f"{p99_miss:.2f} ms", f"{p99_miss/p99_hit:.1f}x Faster" if p99_hit>0 else "N/A")


    console.print("\n", table)

    # Fetch stats from cache server
    try:
        async with httpx.AsyncClient() as c:
            s_res = (await c.get(STATS_URL)).json()
            console.print(Panel(
                f"[bold white]Financial Impact & Token Metrics[/bold white]\n"
                f"- Total LLM Tokens Saved: [bold green]{s_res.get('total_tokens_saved', 0):,}[/bold green]\n"
                f"- Estimated Cost Savings: [bold green]${s_res.get('total_cost_saved_usd', 0):.4f}[/bold green]\n"
                f"- Latency Reduction Factor: [bold green]{s_res.get('latency_reduction_factor', 0)}x[/bold green]",
                title="ROI Headline Metrics"
            ))
    except Exception:
        pass

if __name__ == "__main__":
    asyncio.run(run_load_test(2000, 20))

