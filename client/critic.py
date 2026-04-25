"""
==============================================================================
Universal-Node-Resolver — Multi-Agent Critic Engine
==============================================================================

Implements the Actor-Critic Debate pattern for Large Language Models.
Before simulating actions against the environment sandbox, the Primary Agent's
proposal is violently audited by the SemVerCriticAgent. The Critic is forbidden
from generating its own actions; it exists purely to evaluate logic and catch
hallucinations (e.g. Nuke Exploit) before they consume CPU simulation cycles.
"""

import json
import logging
import re
from typing import Callable

from server.models import Action, Observation

logger = logging.getLogger("critic_agent")


class SemVerCriticAgent:
    """
    A strict, mathematically precise Node.js Dependency Auditor.
    """

    SYSTEM_PROMPT = (
        "You are a strict, mathematically precise Node.js Dependency Auditor. "
        "You do not generate fixes. You evaluate proposed fixes.\n"
        "Your only output format is a binary JSON validation:\n"
        '{"approved": true/false, "feedback": "reasoning"}'
    )

    def __init__(self):
        pass

    def evaluate_proposal(
        self,
        current_observation: Observation,
        proposed_action: Action,
        llm_inference_fn: Callable[[str], str] = None,
    ) -> dict:
        """
        Evaluate a proposed action against the current environment state.
        
        If `llm_inference_fn` is provided, it attempts to use a real LLM to audit.
        Otherwise, it falls back to a highly deterministic heuristic auditor 
        suitable for local demonstrations without multi-GPU rigs.
        """
        
        # 1. Hardware/Heuristic Nuke Detection
        # Even if the LLM hallucinates an approval, the Critic implements a strict
        # heuristic check. It identifies if the agent is deleting a package, which
        # is the most common attempt to cheat the OpenEnv simulation.
        if proposed_action.action_type == "delete":
            logger.warning(f"Critic 🛡️: Rejecting DELETE action on {proposed_action.package_name}. Nuke Exploit Detected.")
            return {
                "approved": False,
                "feedback": f"NUKE CHEAT DETECTED: Deleting '{proposed_action.package_name}' breaks transitive constraints. Reverted."
            }

        # 2. Syntax & Sandbox Escapes
        if proposed_action.action_type == "update":
            if not proposed_action.version_target:
                return {"approved": False, "feedback": "Missing version_target for update."}
            if not re.match(r"^(\d+)\.(\d+)\.(\d+)", proposed_action.version_target):
                return {"approved": False, "feedback": "Malformed SemVer format. Rejected."}
            if proposed_action.package_name in ["scripts", "engines", "bin"]:
                return {"approved": False, "feedback": "ILLEGAL TARGET: Cannot modify restricted system fields."}

        # 3. LLM Validation (if available)
        if llm_inference_fn:
            prompt = (
                f"{self.SYSTEM_PROMPT}\n\n"
                f"CURRENT STATE:\n{current_observation.current_package_json}\n\n"
                f"NPM ERRORS:\n{chr(10).join(current_observation.npm_error_log)}\n\n"
                f"PROPOSED ACTION:\n"
                f"{proposed_action.model_dump_json(indent=2)}\n\n"
                f"Is this action valid, semantically safe, and moving towards resolution?"
            )
            try:
                raw_response = llm_inference_fn(prompt)
                result = json.loads(raw_response)
                # Ensure the LLM followed the strict schema
                if "approved" in result and "feedback" in result:
                    if not result["approved"]:
                        logger.info(f"Critic 🛡️: VETO. {result['feedback']}")
                    return result
            except Exception as e:
                logger.debug(f"Critic LLM failed to return valid JSON, falling back to safe defaults: {e}")

        # If passed heuristic checks, approve the candidate so the MCTS planner can run its deepcopy simulation.
        return {
            "approved": True, 
            "feedback": "Action passed static heuristic audit."
        }
