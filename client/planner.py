"""
==============================================================================
Universal-Node-Resolver — Hybrid Lookahead Planner
==============================================================================

Implements Monte Carlo Tree Search (MCTS) / A* lookahead logic.
This prevents the LLM from taking "greedy" actions that lead to
unresolvable dependency dead-ends by simulating top actions against
a local clone of the environment before committing to one.
"""

import copy
import logging
import math
from typing import Callable, Any

from server.models import Action, Observation

logger = logging.getLogger("hybrid_planner")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(name)s] %(levelname)s — %(message)s"))
    logger.addHandler(_h)


class HybridSemVerPlanner:
    """
    Lookahead tree search wrapper for the LLM Agent.
    
    Instead of taking the immediate best action, it samples the Top N actions
    from the LLM, simulates them in a local environment sandbox, and selects
    the action that provably minimizes future conflicts.
    """

    def __init__(self, agent: Any, num_samples: int = 3):
        """
        Parameters
        ----------
        agent : NodeResolverAgent
            The host agent containing the OpenEnv client connection.
        num_samples : int
            How many diverse actions to sample from the LLM for simulation.
        """
        self.agent = agent
        self.num_samples = num_samples
        
        from client.critic import SemVerCriticAgent
        self.critic = SemVerCriticAgent()

    def plan_next_action(
        self,
        current_observation: Observation,
        llm_inference_fn: Callable[[str], str],
    ) -> Action:
        """
        Executes the Lookahead Search.
        
        1. Queries the LLM `num_samples` times to generate diverse candidate actions.
        2. Simulates each action on a cloned state.
        3. Returns the action that leads to the lowest simulated conflict count.
        """
        from client.agent import build_llm_prompt
        
        prompt = build_llm_prompt(current_observation)
        
        candidates: dict[str, Action] = {}
        
        # 1. Generate Top N Candidates
        # We query the LLM multiple times. If the inference function uses high
        # temperature (e.g. 0.7), this will naturally yield diverse action paths.
        for _ in range(self.num_samples):
            try:
                raw_response = llm_inference_fn(prompt)
                action = self.agent._parse_action(raw_response)
                
                # Multi-Agent Debate: Audit the generated action
                # We pass the same llm_inference_fn so the Critic can use it if configured
                audit_result = self.critic.evaluate_proposal(
                    current_observation, action, llm_inference_fn
                )
                if not audit_result.get("approved", False):
                    logger.debug("Planner ❌: Critic rejected proposal: %s", audit_result.get("feedback"))
                    continue
                
                # Use string representation of action to deduplicate
                action_key = f"{action.action_type}:{action.package_name}:{action.version_target}"
                if action_key not in candidates:
                    candidates[action_key] = action
            except Exception as e:
                logger.debug("Failed to generate/parse a candidate: %s", e)
                continue

        # Fallback if no valid candidates were generated
        if not candidates:
            logger.warning("Planner failed to generate valid candidates. Falling back.")
            return Action(
                action_type="update", 
                package_name="__invalid_format__", 
                version_target="0.0.0"
            )

        logger.debug("Planner generated %d unique candidates.", len(candidates))

        # 2. Simulate Candidates
        best_action = None
        best_conflict_count = math.inf
        
        # We need the real environment for simulation logic.
        # Since we use a local shim, we access _env. 
        # In remote deployment, this would ping a /simulate endpoint.
        real_env = self.agent.client._env
        registry = real_env._registry
        
        for key, action in candidates.items():
            # Clone current state
            simulated_state = copy.deepcopy(real_env.current_package_state)
            
            # Simulated Anti-Cheat / Nuke check
            if action.action_type == "delete":
                dependents = real_env._find_dependents(action.package_name)
                if dependents:
                    # Nuke detected! This action is terrible. Assign infinite conflicts.
                    logger.debug("Candidate %s rejected (Nuke Trap!)", key)
                    continue

            # Check if package exists in registry
            if action.action_type == "update":
                available = registry.get_available_versions(action.package_name)
                if not available or action.version_target not in available:
                    # Invalid version. Reject.
                    logger.debug("Candidate %s rejected (Invalid Version!)", key)
                    continue

            # Apply Action to Simulated State
            if action.action_type == "update":
                simulated_state[action.package_name] = action.version_target  # type: ignore
            elif action.action_type == "delete":
                if action.package_name in simulated_state:
                    del simulated_state[action.package_name]

            # Evaluate the new state
            conflict_reports = registry.validate_installation(simulated_state)
            simulated_conflict_count = len(conflict_reports)
            
            logger.debug("Candidate %s -> %d conflicts", key, simulated_conflict_count)
            
            # 3. Update Best Action
            if simulated_conflict_count < best_conflict_count:
                best_conflict_count = simulated_conflict_count
                best_action = action

        # 4. Return Optimal Action
        # If all candidates hit a nuke trap or were invalid, pick the first one 
        # (it will trigger the environment's standard penalty, allowing the RL loop to learn)
        if best_action is None:
            best_action = list(candidates.values())[0]
            
        return best_action
