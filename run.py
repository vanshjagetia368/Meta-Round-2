"""
==============================================================================
Universal-Node-Resolver — God Mode Bootstrapper
==============================================================================

Asynchronously launches the OpenEnv simulation server, the FastAPI webhook,
and the Gradio UI simultaneously. Gracefully handles teardown on SIGINT.
"""

import os
import signal
import subprocess
import sys
import time

processes = []

def cleanup(signum, frame):
    print("\n🛑 Caught termination signal. Gracefully shutting down all services...")
    for name, p in processes:
        print(f"Terminating {name} (PID {p.pid})...")
        p.terminate()
        
    for name, p in processes:
        try:
            p.wait(timeout=5)
            print(f"✅ {name} shut down cleanly.")
        except subprocess.TimeoutExpired:
            print(f"⚠️ {name} did not terminate in time. Killing forcefully...")
            p.kill()
            
    print("Goodbye!")
    sys.exit(0)


def main():
    print("🚀 Launching Universal-Node-Resolver (God Mode)...\n" + "="*55)
    
    # Register signal handlers for clean teardown
    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)
    
    # Ensure env variables match local orchestration
    env = os.environ.copy()
    env["OPENENV_URL"] = "http://127.0.0.1:8000"

    try:
        # 1. Launch OpenEnv Server (Port 8000)
        p_server = subprocess.Popen(
            ["openenv", "serve", "openenv.yaml", "--host", "0.0.0.0", "--port", "8000"],
            stdout=sys.stdout,
            stderr=sys.stderr,
            env=env
        )
        processes.append(("OpenEnv Server", p_server))
        print("✅ Spawned OpenEnv Server (Port 8000)")
        
        # Wait slightly for the server to bind before launching clients
        time.sleep(2)
        
        # 2. Launch FastAPI Webhook (Port 8080)
        p_api = subprocess.Popen(
            ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080"],
            stdout=sys.stdout,
            stderr=sys.stderr,
            env=env
        )
        processes.append(("FastAPI Webhook", p_api))
        print("✅ Spawned FastAPI Webhook (Port 8080)")
        
        # 3. Launch Gradio UI (Port 7860)
        p_ui = subprocess.Popen(
            ["python3", "app.py"],
            stdout=sys.stdout,
            stderr=sys.stderr,
            env=env
        )
        processes.append(("Gradio UI", p_ui))
        print("✅ Spawned Gradio UI (Port 7860)")
        
        print("="*55 + "\n✨ All systems operational. Press Ctrl+C to stop.\n")
        
        # Monitor processes
        while True:
            for name, p in processes:
                if p.poll() is not None:
                    print(f"🚨 FATAL: {name} crashed unexpectedly (Exit Code {p.returncode})!")
                    cleanup(None, None)
            time.sleep(1)
            
    except Exception as e:
        print(f"\n🚨 Exception in Bootstrapper: {e}")
        cleanup(None, None)


if __name__ == "__main__":
    main()
