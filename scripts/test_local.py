"""
==============================================================================
Universal-Node-Resolver — Local Test & Baseline
==============================================================================

Runs the NodeResolverAgent locally using a dummy heuristic function to
validate the reward shaping and environment logic. Generates a plot
for the project README.
"""

import json
import logging
import os
import random
import re
import sys

import matplotlib.pyplot as plt

# Ensure imports work from project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from client.agent import NodeResolverAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_local")

def random_heuristic_agent(prompt: str) -> str:
    """
    Dummy LLM inference function.
    Parses the prompt to find a package involved in a conflict, and
    randomly decides to 'update' or 'delete' it.
    """
    packages = []
    
    # Attempt to extract package names from the Active Conflicts log
    try:
        if "### Active Conflicts" in prompt:
            conflicts_section = prompt.split("### Active Conflicts")[1].split("## Your Task")[0]
            for line in conflicts_section.split("\n"):
                if "ERESOLVE:" in line:
                    # Match packages in "pkg-name@version"
                    matches = re.findall(r"([a-zA-Z0-9-]+)@", line)
                    packages.extend(matches)
    except Exception:
        pass
        
    # Fallback to installed packages if no errors parsed
    if not packages:
        try:
            json_match = re.search(r"```json\s+(.*?)\s+```", prompt, re.DOTALL)
            if json_match:
                state = json.loads(json_match.group(1))
                deps = state.get("dependencies", {})
                packages = list(deps.keys())
        except Exception:
            pass
            
    if not packages:
        packages = ["pkg-001"]  # Ultimate fallback
        
    # Remove duplicates
    packages = list(set(packages))
    
    # Select random target package
    target_pkg = random.choice(packages)
    
    # 80% update, 20% delete (delete is highly likely to trigger nuke penalty)
    if random.random() < 0.8:
        action_type = "update"
        # Guess a semantic version
        version = f"{random.randint(0, 3)}.{random.randint(0, 5)}.0"
    else:
        action_type = "delete"
        version = None
        
    # Return formatted JSON string as requested
    action = {
        "action_type": action_type,
        "package_name": target_pkg,
        "version_target": version
    }
    
    return json.dumps(action)


def main():
    logger.info("Initializing NodeResolverAgent for local sanity check...")
    agent = NodeResolverAgent()
    
    num_episodes = 50
    rewards = []
    steps_taken = []
    
    logger.info(f"Running {num_episodes} baseline episodes with random heuristic agent...")
    
    for i in range(num_episodes):
        # We suppress verbose logging per step to keep output clean, 
        # but print episode summaries
        reward, steps, solved = agent.run_episode(
            llm_inference_function=random_heuristic_agent, 
            verbose=False
        )
        
        rewards.append(reward)
        steps_taken.append(steps)
        
        logger.info(f"Episode {i+1:02d}: Reward = {reward:6.1f} | Steps = {steps:2d} | Solved = {solved}")

    # Plotting
    logger.info("Generating baseline performance plots...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Reward Plot
    ax1.plot(rewards, color='#4CAF50', marker='o', linestyle='-', markersize=4, alpha=0.8)
    ax1.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_title("Dummy Agent: Episode vs Total Reward", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Total Reward")
    ax1.grid(True, linestyle=':', alpha=0.7)
    
    # Steps Plot
    ax2.plot(steps_taken, color='#2196F3', marker='o', linestyle='-', markersize=4, alpha=0.8)
    ax2.set_title("Dummy Agent: Episode vs Steps Taken", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Steps Taken")
    ax2.grid(True, linestyle=':', alpha=0.7)
    
    plt.tight_layout()
    
    # Ensure assets directory exists
    assets_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "assets"))
    os.makedirs(assets_dir, exist_ok=True)
    
    plot_path = os.path.join(assets_dir, "baseline_rewards.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    
    logger.info(f"Successfully saved plot to: {plot_path}")
    logger.info("Local sanity check complete!")


if __name__ == "__main__":
    main()
