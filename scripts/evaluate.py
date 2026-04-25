"""
==============================================================================
Universal-Node-Resolver — Futuristic Evaluation Pipeline
==============================================================================

Rigorous evaluation script to benchmark the un-tuned Base LLM against
our RLHF-tuned LoRA model. 

It guarantees identical deterministic environments using fixed RNG seeds 
and evaluates exactly what the Hackathon PRD demands:
  1. Total Success Rate
  2. Average Steps to Resolution
  3. "Zero-Deletion" Successes (solving without cheating)

Generates a sophisticated dual-plot visualization and a Markdown table.
==============================================================================
"""

import json
import logging
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

# Unsloth for blazing fast inference
from unsloth import FastLanguageModel

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from client.agent import NodeResolverAgent, build_llm_prompt
from server.models import Action

# ═══════════════════════════════════════════════════════════════════════════
# Logging Configuration
# ═══════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("eval_pipeline")

# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

BASE_MODEL_NAME = "unsloth/llama-3-8b-Instruct-bnb-4bit"
TRAINED_MODEL_DIR = os.path.join("models", "trained_resolver_lora")
MAX_SEQ_LENGTH = 4096
NUM_EPISODES = 50
FIXED_EVAL_SEED = 42
MAX_STEPS = 25


def run_deterministic_evaluation(model, tokenizer, agent, num_episodes=50, base_seed=42):
    """
    Runs the model against N deterministic episodes.
    Tracks success rates, steps, and semantic strategies (deletions).
    """
    successes = 0
    zero_deletion_successes = 0
    steps_taken_list = []
    
    for ep in range(num_episodes):
        # Guarantee identical ecosystems for both models by incrementing the base seed
        ep_seed = base_seed + ep
        obs = agent.client.reset(level=2, seed=ep_seed)
        
        done = False
        step_count = 0
        used_delete_action = False
        
        while not done and step_count < MAX_STEPS:
            # 1. Build prompt
            prompt = build_llm_prompt(obs)
            
            # 2. Tokenize and generate
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            outputs = model.generate(
                **inputs, 
                max_new_tokens=100, 
                temperature=0.0,  # Greedy decoding for deterministic evaluation
                pad_token_id=tokenizer.eos_token_id
            )
            
            # Extract only the newly generated tokens
            response_text = tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:], 
                skip_special_tokens=True
            )
            
            # 3. Parse action
            try:
                action = NodeResolverAgent._parse_action(response_text)
            except Exception:
                # LLM hallucinated invalid JSON
                action = Action(
                    action_type="update", 
                    package_name="__invalid_format__", 
                    version_target="0.0.0"
                )
                
            if action.action_type == "delete":
                used_delete_action = True
                
            # 4. Step environment
            obs, reward, done, info = agent.client.step(action)
            step_count += 1
            
        # 5. Record Episode Metrics
        solved = (info.get("termination") == "all_conflicts_resolved")
        if solved:
            successes += 1
            if not used_delete_action:
                zero_deletion_successes += 1
                
        steps_taken_list.append(step_count)
        
        logger.debug(f"Episode {ep+1:02d} | Solved: {solved} | Steps: {step_count}")
        
    return {
        "success_rate": (successes / num_episodes) * 100,
        "avg_steps": sum(steps_taken_list) / len(steps_taken_list),
        "zero_delete_successes": zero_deletion_successes,
        "steps_dist": steps_taken_list
    }


def main():
    logger.info("Initializing Futuristic Evaluation Pipeline...")
    
    agent = NodeResolverAgent()
    metrics = {}

    # ──────────────────────────────────────────────────────────────────────
    # Phase 1: Evaluate Base Model
    # ──────────────────────────────────────────────────────────────────────
    logger.info(f"Loading Base Untuned Model: {BASE_MODEL_NAME}")
    base_model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(base_model)
    
    logger.info(f"Evaluating Base Model over {NUM_EPISODES} deterministic episodes...")
    metrics["Base Model"] = run_deterministic_evaluation(
        base_model, tokenizer, agent, num_episodes=NUM_EPISODES, base_seed=FIXED_EVAL_SEED
    )
    
    # Free VRAM before loading the next model
    del base_model
    torch.cuda.empty_cache()

    # ──────────────────────────────────────────────────────────────────────
    # Phase 2: Evaluate RLHF Trained Model
    # ──────────────────────────────────────────────────────────────────────
    lora_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", TRAINED_MODEL_DIR))
    
    logger.info(f"Loading RL-Trained LoRA Model from: {lora_path}")
    if not os.path.exists(lora_path):
        logger.error(f"Trained model not found at {lora_path}. Did you run train_unsloth.py?")
        sys.exit(1)
        
    trained_model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=lora_path,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(trained_model)
    
    logger.info(f"Evaluating Trained Model over {NUM_EPISODES} deterministic episodes...")
    metrics["RL-Trained Model"] = run_deterministic_evaluation(
        trained_model, tokenizer, agent, num_episodes=NUM_EPISODES, base_seed=FIXED_EVAL_SEED
    )
    
    del trained_model
    torch.cuda.empty_cache()

    # ──────────────────────────────────────────────────────────────────────
    # Phase 3: Metrics Visualization & Reporting
    # ──────────────────────────────────────────────────────────────────────
    logger.info("Generating evaluation visualizations...")
    
    models = ["Base Model", "RL-Trained Model"]
    success_rates = [metrics[m]["success_rate"] for m in models]
    avg_steps = [metrics[m]["avg_steps"] for m in models]
    zero_deletions = [metrics[m]["zero_delete_successes"] for m in models]
    
    # Setup styling
    plt.style.use('ggplot')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Success Rates (Bar Chart)
    bars = ax1.bar(models, success_rates, color=['#EF5350', '#66BB6A'], alpha=0.8)
    ax1.set_title("Resolution Success Rate (%)", fontsize=14, fontweight='bold')
    ax1.set_ylim(0, 100)
    ax1.set_ylabel("Success Rate %")
    
    # Add percentage labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 2,
                 f"{height:.1f}%", ha='center', va='bottom', fontweight='bold')

    # Plot 2: Steps Distribution (Box Plot)
    steps_data = [metrics[m]["steps_dist"] for m in models]
    bplot = ax2.boxplot(steps_data, patch_artist=True, labels=models, 
                        medianprops=dict(color='black', linewidth=1.5))
    
    colors = ['#EF5350', '#66BB6A']
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
        
    ax2.set_title("Distribution of Steps to Resolution", fontsize=14, fontweight='bold')
    ax2.set_ylabel("Steps Taken")
    
    plt.tight_layout()
    
    # Save the plot
    assets_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "assets"))
    os.makedirs(assets_dir, exist_ok=True)
    plot_path = os.path.join(assets_dir, "final_evaluation_metrics.png")
    plt.savefig(plot_path, dpi=150)
    logger.info(f"Saved visualization to: {plot_path}")
    
    # ──────────────────────────────────────────────────────────────────────
    # Output Markdown Table
    # ──────────────────────────────────────────────────────────────────────
    md_table = f"""
### 📊 Final Evaluation Metrics (50 Deterministic Episodes)

| Metric | Base Llama-3 8B | RL-Trained LoRA | Improvement |
|--------|-----------------|-----------------|-------------|
| **Success Rate** | {success_rates[0]:.1f}% | {success_rates[1]:.1f}% | **+{(success_rates[1] - success_rates[0]):.1f}%** |
| **Average Steps** | {avg_steps[0]:.1f} | {avg_steps[1]:.1f} | **{((avg_steps[0] - avg_steps[1])/avg_steps[0] * 100) if avg_steps[0] > 0 else 0:.1f}% faster** |
| **Zero-Deletion Solutions** | {zero_deletions[0]}/{NUM_EPISODES} | {zero_deletions[1]}/{NUM_EPISODES} | **+{(zero_deletions[1] - zero_deletions[0])}** |

> *Note: Both models faced the exact same randomly generated `package.json` states using a fixed seed (Seed: {FIXED_EVAL_SEED}). Zero-Deletion Solutions indicate scenarios resolved purely through semantic version math without taking the easy route of deleting packages.*
"""
    print("\n" + "="*80)
    print(md_table.strip())
    print("="*80 + "\n")

if __name__ == "__main__":
    try:
        main()
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        logger.error("Run on an instance with unsloth, torch, and matplotlib installed.")
