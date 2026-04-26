"""
==============================================================================
Universal-Node-Resolver — PPO Training Script via Unsloth & TRL
==============================================================================

This script defines the Reinforcement Learning from Human Feedback (RLHF) 
training loop using Proximal Policy Optimization (PPO). It leverages
Unsloth for high-performance 4-bit LoRA fine-tuning and Hugging Face TRL
for the PPO implementation.

The LLM Agent interacts with the OpenEnv server (via NodeResolverAgent),
receiving explicit reward signals based on its ability to resolve
SemVer dependency conflicts (e.g. +15 for progress, -100 for cheating).
These exact scalar rewards are fed directly into the PPO trainer to
update the model's policy.

Usage:
    python scripts/train_unsloth.py

Output:
    - Trained LoRA adapters saved to `models/trained_resolver_lora/`
    - Training reward curve saved to `assets/training_reward_curve.png`
==============================================================================
"""

import json
import logging
import os
import sys

import matplotlib.pyplot as plt
import torch

# Unsloth enables 2x faster, memory-efficient LoRA training
from unsloth import FastLanguageModel

# TRL (Transformer Reinforcement Learning) provides the PPO implementation
from trl import PPOConfig
import trl.trainer.ppo_trainer

# Safely extract the original interactive trainer, shedding Unsloth's wrapper if present
PPOTrainer = trl.trainer.ppo_trainer.PPOTrainer.__wrapped__ if hasattr(trl.trainer.ppo_trainer.PPOTrainer, '__wrapped__') else trl.trainer.ppo_trainer.PPOTrainer

# Add project root to python path to resolve local imports
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
logger = logging.getLogger("train_unsloth")


# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

MODEL_NAME = "unsloth/llama-3-8b-Instruct-bnb-4bit"
MAX_SEQ_LENGTH = 4096      # Sufficient to fit long package.json and error logs
MAX_EPISODES = 50          # Number of RL episodes (keep low for hackathon demo)
MAX_STEPS_PER_EPISODE = 10 # Truncate long episodes to maintain stable PPO batches


def main():
    logger.info("Starting Unsloth PPO Training Pipeline...")

    # ──────────────────────────────────────────────────────────────────────
    # REQUIREMENT 1: Unsloth Model Setup
    # ──────────────────────────────────────────────────────────────────────
    logger.info(f"Loading 4-bit base model: {MODEL_NAME}")
    
    # Load model and tokenizer in 4-bit quantization for VRAM efficiency
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,          # Auto-detect (Float16/Bfloat16)
        load_in_4bit=True,   # Enforce 4-bit quantization
    )
    
    # TRL's PPOTrainer requires a padding token to be explicitly defined
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    logger.info("Applying LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        lora_alpha=16,
        lora_dropout=0,       # 0 is recommended for Unsloth optimized training
        bias="none",
        use_gradient_checkpointing="unsloth",
        # Target all standard projection modules for high-quality adapters
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )
    
    # Enable native inference mode initially
    FastLanguageModel.for_inference(model)

    # ──────────────────────────────────────────────────────────────────────
    # REQUIREMENT 2: RLHF / PPO Loop Integration
    # ──────────────────────────────────────────────────────────────────────
    logger.info("Initializing PPO Trainer and OpenEnv Agent...")
    
    # Wrap the PEFT model with a Value Head for PPO's Actor-Critic architecture
    from trl import AutoModelForCausalLMWithValueHead
    model = AutoModelForCausalLMWithValueHead(model)
    
    # PPO configuration optimized for 16GB VRAM (T4 GPU) stability
    ppo_config = PPOConfig(
        batch_size=4,                    # Small total batch size
        mini_batch_size=1,               # Absolute minimum mini-batch to prevent OOM
        gradient_accumulation_steps=4,   # Accumulate to simulate larger batch size safely
        learning_rate=1.41e-5,           # Standard stable LR for PPO fine-tuning
    )

    # Initialize TRL's PPOTrainer
    # (Notice ref_model=None: TRL automatically disables LoRA adapters to compute KL penalty!)
    
    # --- ANTIGRAVITY BYPASS: UNSLOTH AST PATCH BUG FIX ---
    # Unsloth dynamically rewrites PPOTrainer and caches it, but its regex injection 
    # creates an UnboundLocalError with 'args' on TRL 0.7.11. 
    # We bypass this completely by reloading the module to shed Unsloth's monkey-patch
    # while preserving the parent package context for relative imports.
    import importlib
    import trl.trainer.ppo_trainer
    importlib.reload(trl.trainer.ppo_trainer)
    PurePPOTrainer = trl.trainer.ppo_trainer.PPOTrainer

    ppo_trainer = PurePPOTrainer(
        config=ppo_config,
        model=model,
        ref_model=None,
        tokenizer=tokenizer,
    )
    # -------------------------------------------------

    # Initialize our environment client (using the local shim)
    agent = NodeResolverAgent()
    
    # Tracking for visualizations
    episode_rewards = []
    
    logger.info("Entering active PPO training loop...")

    # Start RL Loop
    for episode in range(MAX_EPISODES):
        # 1. Reset Environment
        # Get a fresh broken package.json and initial conflicts
        obs = agent.client.reset(level=None) # Curriculum scaling
        
        done = False
        step_idx = 0
        total_ep_reward = 0.0
        
        while not done and step_idx < MAX_STEPS_PER_EPISODE:
            # Switch to inference mode for text generation
            FastLanguageModel.for_inference(model)

            # 2. Format the observation into a strict LLM Prompt
            prompt_text = build_llm_prompt(obs)
            
            # Encode prompt
            query_tensor = tokenizer.encode(prompt_text, return_tensors="pt").to(ppo_trainer.accelerator.device)[0]

            # 3. Generate Action (Text)
            # Use PPOTrainer's generate to automatically track gradients & probabilities
            generation_kwargs = {
                "max_new_tokens": 100,
                "temperature": 0.7, # Slight randomness for exploration
                "do_sample": True,
                "pad_token_id": tokenizer.pad_token_id,
            }
            
            response_tensor = ppo_trainer.generate(
                [query_tensor], 
                return_prompt=False, 
                **generation_kwargs
            )[0]
            
            response_text = tokenizer.decode(response_tensor, skip_special_tokens=True)
            
            # 4. Parse the generated JSON action
            try:
                # We use the agent's robust static parser that strips markdown
                action = agent._parse_action(response_text)
            except Exception as e:
                # If the LLM generated malformed JSON or ignored constraints,
                # we force an invalid action to trigger the environment's -5 penalty.
                # This directly teaches the model via RL to respect the JSON schema!
                logger.debug(f"Action parse failed: {e}. Forcing invalid action.")
                action = Action(
                    action_type="update", 
                    package_name="__invalid_format__", 
                    version_target="0.0.0"
                )

            # 5. Execute step in OpenEnv, receive scalar shaped reward
            obs, reward, terminated, truncated, info = agent.client.step(action)
            done = terminated or truncated
            total_ep_reward += reward
            
            # Switch back to training mode for the PPO update
            FastLanguageModel.for_training(model)
            
            # 6. PPO Step Buffer
            # Feed the exact state (query), action (response), and environment reward back to PPO
            reward_tensor = torch.tensor(reward, dtype=torch.float32, device=ppo_trainer.accelerator.device)
            
            # Accumulate into batch lists attached to the trainer
            if not hasattr(ppo_trainer, "_query_buffer"):
                ppo_trainer._query_buffer = []
                ppo_trainer._response_buffer = []
                ppo_trainer._reward_buffer = []
                
            ppo_trainer._query_buffer.append(query_tensor)
            ppo_trainer._response_buffer.append(response_tensor)
            ppo_trainer._reward_buffer.append(reward_tensor)
            
            # Execute PPO step only when batch is full
            if len(ppo_trainer._query_buffer) == ppo_config.batch_size:
                train_stats = ppo_trainer.step(
                    ppo_trainer._query_buffer,
                    ppo_trainer._response_buffer,
                    ppo_trainer._reward_buffer
                )
                
                # ANTIGRAVITY: Prevent Pytorch CUDA Memory Leak
                ppo_trainer._query_buffer.clear()
                ppo_trainer._response_buffer.clear()
                ppo_trainer._reward_buffer.clear()
                torch.cuda.empty_cache()
            
            step_idx += 1
            
        # Episode Complete
        episode_rewards.append(total_ep_reward)
        logger.info(
            f"Episode {episode+1:03d}/{MAX_EPISODES} | "
            f"Steps: {step_idx:02d} | "
            f"Reward: {total_ep_reward:6.1f} | "
            f"Solved: {info.get('termination') == 'all_conflicts_resolved'}"
        )

    # ──────────────────────────────────────────────────────────────────────
    # Plotting & Visualization
    # ──────────────────────────────────────────────────────────────────────
    logger.info("Training complete! Generating reward curve...")
    
    plt.figure(figsize=(10, 5))
    plt.plot(episode_rewards, color='purple', marker='.', alpha=0.8, linewidth=2)
    plt.axhline(0, color='gray', linestyle='--', alpha=0.5)
    plt.title("RLHF PPO Training: Episode vs. Total Reward", fontsize=14, fontweight='bold')
    plt.xlabel("Episode")
    plt.ylabel("Total Episodic Reward")
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    
    # Save to assets directory
    assets_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "assets"))
    os.makedirs(assets_dir, exist_ok=True)
    plot_path = os.path.join(assets_dir, "training_reward_curve.png")
    plt.savefig(plot_path, dpi=150)
    logger.info(f"Saved reward curve to: {plot_path}")

    # ──────────────────────────────────────────────────────────────────────
    # REQUIREMENT 3: Saving Weights
    # ──────────────────────────────────────────────────────────────────────
    models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "trained_resolver_lora"))
    logger.info(f"Saving trained LoRA adapters to: {models_dir}")
    
    # Create directory if it doesn't exist
    os.makedirs(models_dir, exist_ok=True)
    
    # Save using Unsloth's optimized native saver
    model.save_pretrained(models_dir)
    tokenizer.save_pretrained(models_dir)
    
    logger.info("Successfully exported weights. Project ready for deployment!")


if __name__ == "__main__":
    # Wrap in try-except for environments without GPUs (like local IDEs during checking)
    try:
        main()
    except ImportError as e:
        logger.error(f"Missing required ML dependency: {e}")
        logger.error("Please ensure unsloth, torch, and trl are installed.")
    except Exception as e:
        logger.error(f"Training loop failed: {e}")
        raise
