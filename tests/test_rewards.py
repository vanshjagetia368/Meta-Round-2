"""
==============================================================================
Universal-Node-Resolver — Reward Logic Tests
==============================================================================

Proves to Hackathon Judges that our Multi-Signal Reward system correctly
penalizes reward-hacking (The Nuke) and rewards genuine SemVer progress.
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from server.environment import UniversalNodeEnv
from server.registry import PackageDescriptor, SemVer
from server.models import Action


def test_the_nuke_penalty():
    """
    Anti-Hacking Test: Prove the agent receives a massive penalty (-100)
    and is immediately terminated if it tries to 'delete' a package that 
    is required by other installed packages.
    """
    env = UniversalNodeEnv()
    
    # ── Mocking a direct dependency relationship ──
    # Create two packages where pkg-beta DEPENDS ON pkg-alpha
    pkg_alpha = PackageDescriptor(
        name="pkg-alpha",
        available_versions=[SemVer.parse("1.0.0")]
    )
    pkg_beta = PackageDescriptor(
        name="pkg-beta",
        available_versions=[SemVer.parse("1.0.0")],
        dependencies={"1.0.0": {"pkg-alpha": "^1.0.0"}}
    )
    
    env._registry._packages = {
        "pkg-alpha": pkg_alpha,
        "pkg-beta": pkg_beta
    }
    
    # Both are installed
    env.current_package_state = {
        "pkg-alpha": "1.0.0",
        "pkg-beta": "1.0.0"
    }
    
    # Bypass standard reset, prepare for step
    env._done = False
    env.step_count = 0
    env._current_errors = []
    
    # ── The Action ──
    # The agent maliciously attempts to delete 'pkg-alpha'
    nuke_action = Action(
        action_type="delete", 
        package_name="pkg-alpha", 
        version_target=None
    )
    
    obs, reward, done, info = env.step(nuke_action)
    
    # ── The Assertions ──
    assert reward == -100.0, "Did not receive the severe -100 Nuke Penalty!"
    assert done is True, "Agent was not terminated after nuking!"
    assert info["nuke_detected"] is True
    assert "destroy the graph" in info["reason"]


def test_valid_resolution_reward():
    """
    Test that genuine SemVer conflict resolution yields the correct
    incremental (+15) and terminal (+100) rewards.
    """
    env = UniversalNodeEnv()
    
    # ── Mocking a broken state ──
    # pkg-alpha requires pkg-beta@^2.0.0, but version 1.0.0 is installed.
    pkg_alpha = PackageDescriptor(
        name="pkg-alpha",
        available_versions=[SemVer.parse("1.0.0")],
        dependencies={"1.0.0": {"pkg-beta": "^2.0.0"}}
    )
    pkg_beta = PackageDescriptor(
        name="pkg-beta",
        available_versions=[SemVer.parse("1.0.0"), SemVer.parse("2.0.0")]
    )
    
    env._registry._packages = {
        "pkg-alpha": pkg_alpha,
        "pkg-beta": pkg_beta
    }
    
    env.current_package_state = {
        "pkg-alpha": "1.0.0",
        "pkg-beta": "1.0.0" # CONFLICT: Needs ^2.0.0
    }
    
    env._done = False
    env.step_count = 0
    env._current_errors = env._evaluate_current_state()
    
    # Verify there is exactly 1 active conflict
    assert len(env._current_errors) == 1
    
    # ── The Action ──
    # The agent smartly updates pkg-beta to version 2.0.0
    fix_action = Action(
        action_type="update", 
        package_name="pkg-beta", 
        version_target="2.0.0"
    )
    
    obs, reward, done, info = env.step(fix_action)
    
    # ── The Assertions ──
    # Calculation: 
    # Step penalty (-1.0) + Progress for 1 fix (+15.0) + Terminal Success (+100.0) = 114.0
    assert reward == 114.0
    assert done is True
    assert info["termination"] == "all_conflicts_resolved"
    assert "ERESOLVE" not in obs.npm_error_log
