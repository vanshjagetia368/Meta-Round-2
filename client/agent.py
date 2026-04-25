"""
==============================================================================
Universal-Node-Resolver — Client Agent Wrapper
==============================================================================

OpenEnv-compliant client that communicates with the environment via
the standard Client API. For local development, a shim directly
instantiates the environment; in production, this connects to the
deployed Hugging Face Space via HTTP.

Usage:
    from client.agent import NodeResolverAgent

    agent = NodeResolverAgent()
    reward, steps, solved = agent.run_episode(my_llm_function)

==============================================================================
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Callable, Optional

from server.models import Action, Observation
from server.environment import UniversalNodeEnv

logger = logging.getLogger("node_resolver_agent")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(name)s] %(levelname)s — %(message)s"))
    logger.addHandler(_h)


# ═══════════════════════════════════════════════════════════════════════════
# Local OpenEnv Client Shim
# ═══════════════════════════════════════════════════════════════════════════
# In production, this would be replaced by:
#     from openenv import Client
#     client = Client("https://<space>.hf.space")
#
# For local dev & hackathon judging, we shim it so everything runs
# in-process without a network dependency.

class _LocalClient:
    """
    Local shim mimicking the OpenEnv Client API.

    Methods mirror the remote API exactly so swapping to the real
    ``openenv.Client`` requires changing only the import.
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        self._env = UniversalNodeEnv(seed=seed)

    def reset(
        self,
        level: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> Observation:
        """Reset the environment, returns initial Observation."""
        return self._env.reset(level=level, seed=seed)

    def step(
        self,
        action: Action,
    ) -> tuple[Observation, float, bool, bool, dict[str, Any]]:
        """Execute an action, returns (obs, reward, terminated, truncated, info)."""
        return self._env.step(action)

    def state(self) -> Observation:
        """Return current state without stepping."""
        return self._env.state()

    def get_valid_actions(self) -> list[dict[str, Any]]:
        """Return all valid actions from current state."""
        return self._env.get_valid_actions()

    @property
    def golden_path(self) -> dict[str, str]:
        """Expose golden path for debugging."""
        return self._env.golden_path


# ═══════════════════════════════════════════════════════════════════════════
# Prompt Builder
# ═══════════════════════════════════════════════════════════════════════════

def build_llm_prompt(obs: Observation) -> str:
    """
    Convert an Observation into a strictly formatted prompt for the LLM.

    The prompt provides:
        1. The current package.json (dependencies)
        2. All active ERESOLVE conflict errors
        3. The exact JSON schema the LLM must return
        4. Step count and complexity for context

    Returns
    -------
    str
        A self-contained prompt string.
    """
    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an expert npm dependency resolver AI.<|eot_id|><|start_header_id|>user<|end_header_id|>

## Current State (Step {obs.step_count}, Complexity Level {obs.complexity_level})

### Installed Dependencies (package.json):
```json
{obs.current_package_json}
```

### Active Conflicts ({obs.npm_error_log.count('ERESOLVE') if obs.npm_error_log else 0} errors):
```
{obs.npm_error_log if obs.npm_error_log else "No conflicts — all dependencies resolved!"}
```

## Your Task
Resolve ONE conflict by choosing an action. You must output ONLY valid JSON matching this schema:

```json
{{
  "action_type": "update" | "delete",
  "package_name": "<package-name>",
  "version_target": "<semver-version>"  // Required for "update", null for "delete"
}}
```

### Rules:
- "update": Change a package to a specific version that satisfies constraints.
- "delete": Remove a package ONLY if nothing depends on it.
- Do NOT delete packages that other packages require.
- Focus on resolving the FIRST listed conflict.

Output ONLY the JSON object, no explanation.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""


# ═══════════════════════════════════════════════════════════════════════════
# NodeResolverAgent
# ═══════════════════════════════════════════════════════════════════════════

class NodeResolverAgent:
    """
    OpenEnv client-side agent that drives the RL loop.

    Parameters
    ----------
    connection_url : str | None
        URL of the deployed OpenEnv server. If None, uses a local
        in-process shim (for development and testing).
    seed : int | None
        RNG seed for reproducible episodes.
    """

    def __init__(
        self,
        connection_url: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> None:
        if connection_url is not None:
            # Production: connect to remote OpenEnv server
            # from openenv import Client
            # self.client = Client(connection_url)
            raise NotImplementedError(
                "Remote OpenEnv client not yet configured. "
                "Use connection_url=None for local development."
            )
        else:
            # Local development: in-process shim
            self.client = _LocalClient(seed=seed)

        logger.info(
            "NodeResolverAgent initialised (url=%s)", connection_url or "local"
        )

    def run_episode(
        self,
        llm_inference_function: Callable[[str], str],
        level: Optional[int] = None,
        verbose: bool = False,
        use_lookahead_planner: bool = False,
    ) -> tuple[float, int, bool]:
        """
        Run a single episode using the provided LLM inference function.

        Parameters
        ----------
        llm_inference_function : Callable[[str], str]
            A function that accepts a prompt string and returns the
            LLM's response as a JSON string.
        level : int | None
            Complexity level (1-3). None uses curriculum.
        verbose : bool
            If True, log every step's reward and action.

        Returns
        -------
        tuple[float, int, bool]
            (total_reward, step_count, solved)
        """
        # ── Reset environment ────────────────────────────────────────
        obs = self.client.reset(level=level)
        total_reward: float = 0.0
        step_count: int = 0
        done: bool = False
        solved: bool = False

        if verbose:
            logger.info(
                "Episode started — level=%d, initial errors=%d",
                obs.complexity_level,
                obs.npm_error_log.count("ERESOLVE") if obs.npm_error_log else 0,
            )

        # ── Main loop ────────────────────────────────────────────────
        while not done:
            # 1. Build the prompt from the current observation
            prompt = build_llm_prompt(obs)

            # 2. Generate and parse action (Greedy or Planner)
            if use_lookahead_planner:
                from client.planner import HybridSemVerPlanner
                planner = HybridSemVerPlanner(self, num_samples=3)
                action = planner.plan_next_action(obs, llm_inference_function)
            else:
                try:
                    raw_response = llm_inference_function(prompt)
                    action = self._parse_action(raw_response)
                except Exception as e:
                    logger.warning("Failed to parse LLM response: %s", e)
                    try:
                        action = Action(
                            action_type="update",
                            package_name="__invalid__",
                            version_target="0.0.0",
                        )
                    except Exception:
                        break

            # 3. Step the environment
            try:
                obs, reward, terminated, truncated, info = self.client.step(action)
                done = terminated or truncated
            except RuntimeError as e:
                logger.error("Environment error: %s", e)
                break

            total_reward += reward
            step_count += 1

            if verbose:
                logger.info(
                    "  Step %d: action=%s %s@%s → reward=%.1f, "
                    "conflicts=%s",
                    step_count,
                    action.action_type,
                    action.package_name,
                    action.version_target or "N/A",
                    reward,
                    info.get("conflicts_remaining", "?"),
                )

            # Check if solved
            if done and info.get("termination") == "all_conflicts_resolved":
                solved = True

        if verbose:
            logger.info(
                "Episode finished — steps=%d, reward=%.1f, solved=%s",
                step_count,
                total_reward,
                solved,
            )

        return total_reward, step_count, solved

    @staticmethod
    def _parse_action(raw_response: str) -> Action:
        """
        Parse the LLM's raw text response into a validated Action.

        Handles common LLM output quirks:
            - Markdown code fences (```json ... ```)
            - Leading/trailing whitespace
            - Missing fields
        """
        text = raw_response.strip()

        # Attempt to extract JSON block using regex if conversational text is present
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            text = match.group(0).strip()

        # Parse JSON
        data = json.loads(text)

        # Validate via Pydantic
        return Action(**data)


# ═══════════════════════════════════════════════════════════════════════════
# Module Exports
# ═══════════════════════════════════════════════════════════════════════════

__all__: list[str] = ["NodeResolverAgent", "build_llm_prompt"]
