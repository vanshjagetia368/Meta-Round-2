"""
==============================================================================
Universal-Node-Resolver — Core Environment Tests
==============================================================================

Rigorous pytest suite to ensure the RL environment handles extreme
edge cases gracefully during long, unattended training loops.
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from server.environment import UniversalNodeEnv
from server.models import Action
from pydantic import ValidationError


def test_malformed_action_handling():
    """
    Test that the environment gracefully handles invalid actions without crashing.
    In a 5-hour RL loop, LLMs WILL hallucinate bad actions. We must ensure
    it results in a negative reward and termination, not a Python Exception.
    """
    env = UniversalNodeEnv()
    env.reset(level=1)
    
    # 1. Pydantic catches missing fields before it even hits the environment
    with pytest.raises(ValidationError):
        # Missing 'version_target' entirely
        bad_action = Action(action_type="update", package_name="pkg-001")
        
    # 2. Agent parsing fallback (non-existent package from bad JSON output)
    # The agent parser converts totally broken JSON to a fallback action
    fallback_action = Action(
        action_type="update", 
        package_name="__invalid_format__", 
        version_target="0.0.0"
    )
    
    obs, reward, done, info = env.step(fallback_action)
    
    # Assert environment caught the non-existent package and penalized it
    assert reward == -5.0
    assert done is True
    assert info["action_valid"] is False
    assert "not found in registry" in info["reason"]


def test_circular_dependency_generation():
    """
    Stress test the DAG generator in UniversalMockRegistry.
    Level 3 produces 30-node graphs. We run 100 resets to ensure the 
    topological sorting logic prevents infinite recursion loops.
    """
    env = UniversalNodeEnv(seed=42)
    
    for _ in range(100):
        # If there's a cyclic bug, this will hang or hit recursion depth
        obs = env.reset(level=3)
        
        # Verify it actually generated a complex state
        assert obs.complexity_level == 3
        # Ensure we have active conflicts to solve
        assert "ERESOLVE" in obs.npm_error_log
