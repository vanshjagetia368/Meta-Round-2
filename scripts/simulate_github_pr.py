"""
==============================================================================
Universal-Node-Resolver — Cinematic GitHub Webhook Simulator
==============================================================================

Simulates a real-world scenario where a developer pushes a broken package.json
to GitHub. This script fires a mock webhook payload to our FastAPI endpoint
and visualizes the multi-agent resolution process natively in the terminal.
"""

import json
import time
import requests
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.spinner import Spinner
from rich.live import Live
from rich.text import Text

console = Console()

# The Webhook Endpoint
API_URL = "http://localhost:8080/webhook/github/autofix"

# A completely broken, Level 3 complexity payload
BROKEN_PACKAGE_JSON = {
    "name": "enterprise-monorepo",
    "version": "1.0.0",
    "dependencies": {
        "react": "17.0.2",
        "react-dom": "17.0.2",
        "webpack": "4.44.2",
        "lodash": "4.17.20"
    },
    "scripts": {
        "build": "webpack --mode production"
    }
}

WEBHOOK_PAYLOAD = {
    "repository_name": "Meta-Hackathon/Universal-Node-Resolver",
    "pull_request_id": 999,
    "raw_package_json": json.dumps(BROKEN_PACKAGE_JSON, indent=2)
}


def run_simulation():
    console.clear()
    
    # 1. Incoming Webhook Alert
    alert = Panel(
        Text("🚨 INCOMING GITHUB WEBHOOK DETECTED: PR #999", style="bold red justify-center"),
        border_style="red",
        expand=False
    )
    console.print(alert)
    time.sleep(1)

    # 2. Display Broken Package.json
    console.print("\n[bold yellow]Commit: 'Fixing build issues (hopefully)'[/bold yellow]")
    console.print("[bold red]Diff:[/bold red] Found conflicting dependency graph in package.json.\n")
    
    broken_syntax = Syntax(
        json.dumps(BROKEN_PACKAGE_JSON, indent=2), 
        "json", 
        theme="monokai", 
        background_color="default"
    )
    console.print(Panel(broken_syntax, title="Broken package.json", border_style="yellow"))
    time.sleep(1.5)

    # 3. Trigger API with Spinner
    console.print("\n[bold cyan]Routing payload to Universal-Node-Resolver API...[/bold cyan]")
    
    with Live(Spinner("dots", text="🧠 Multi-Agent MCTS Planner resolving SemVer DAG..."), refresh_per_second=10):
        try:
            # Send the webhook
            response = requests.post(API_URL, json=WEBHOOK_PAYLOAD, timeout=60)
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.ConnectionError:
            console.print("\n[bold red]❌ Error: FastAPI server is not running on port 8080. Run `python run.py` first.[/bold red]")
            return
        except Exception as e:
            console.print(f"\n[bold red]❌ API Failure: {e}[/bold red]")
            return

    # 4. Display Result
    console.print("\n[bold green]✅ Dependency Graph Resolved![/bold green]")
    console.print(f"[bold white]Planner metrics:[/bold white] {data.get('steps_taken')} steps simulated. Reward: {data.get('total_reward')}")
    
    fixed_syntax = Syntax(
        json.dumps(data.get("resolved_package_json", {}), indent=2), 
        "json", 
        theme="monokai", 
        background_color="default"
    )
    console.print(Panel(fixed_syntax, title="Auto-Fixed package.json", border_style="green"))

    # 5. Conclusion
    console.print("\n[bold bright_green]🚀 PR #999 Auto-Fixed and Merged by Universal-Node-Resolver.[/bold bright_green]\n")


if __name__ == "__main__":
    run_simulation()
