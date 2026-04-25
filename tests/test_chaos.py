"""
==============================================================================
Universal-Node-Resolver — Chaos Engine Tests
==============================================================================

Verifies that the AdversarialRegistryWrapper properly injects 503
timeouts and 404 left-pad yanking events, and that the environment
gracefully tracks and rewards survival (+25 Resilience Bonus).
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from server.environment import UniversalNodeEnv
from server.models import Action
from server.registry import PackageDescriptor, SemVer


def test_simulate_yanked_package():
    """
    Force a 100% probability yank event during an action.
    Assert the registry successfully deletes the version and logs the error.
    """
    env = UniversalNodeEnv(chaos_mode=True)
    
    # ── Mocking an installed package ──
    pkg_alpha = PackageDescriptor(
        name="pkg-alpha",
        available_versions=[SemVer.parse("1.0.0"), SemVer.parse("2.0.0")]
    )
    env._registry._base.packages = {"pkg-alpha": pkg_alpha}
    env.current_package_state = {"pkg-alpha": "1.0.0"}
    
    # Force the wrapper to yank 100% of the time
    env._registry.simulate_yanked_package = lambda p=1.0: ("pkg-alpha", "1.0.0") if pkg_alpha.available_versions.remove(SemVer.parse("1.0.0")) is None else None
    
    action = Action(action_type="update", package_name="pkg-alpha", version_target="2.0.0")
    
    obs, reward, done, info = env.step(action)
    
    assert env._encountered_chaos is True
    assert "Yanked pkg-alpha@1.0.0" in info.get("chaos_event", "")
    assert SemVer.parse("1.0.0") not in env._registry.packages["pkg-alpha"].available_versions


def test_simulate_network_timeout():
    """
    Force a 503 Network Timeout during an action.
    Assert the state was NOT modified and the correct penalty was applied.
    """
    env = UniversalNodeEnv(chaos_mode=True)
    env._registry._base.packages = {
        "pkg-alpha": PackageDescriptor(name="pkg-alpha", available_versions=[SemVer.parse("1.0.0"), SemVer.parse("2.0.0")])
    }
    env.current_package_state = {"pkg-alpha": "1.0.0"}
    
    # Force timeout to 100%
    env._registry.simulate_network_timeout = lambda p=1.0: True
    
    action = Action(action_type="update", package_name="pkg-alpha", version_target="2.0.0")
    obs, reward, done, info = env.step(action)
    
    # State should remain unchanged
    assert env.current_package_state["pkg-alpha"] == "1.0.0"
    assert env._encountered_chaos is True
    assert "503 Network Timeout" in info.get("chaos_event", "")
    assert any("HTTP 503" in err for err in env._current_errors)


def test_adversarial_resilience_bonus():
    """
    Prove that if an agent recovers from chaos and solves the graph,
    it is awarded the +25 bonus.
    """
    env = UniversalNodeEnv(chaos_mode=True)
    env._encountered_chaos = True # Pretend it survived a timeout previously
    
    # Setup a state that is 1 step away from solution
    env._registry._base.packages = {
        "pkg-alpha": PackageDescriptor(name="pkg-alpha", available_versions=[SemVer.parse("1.0.0"), SemVer.parse("2.0.0")])
    }
    env.current_package_state = {"pkg-alpha": "1.0.0"}
    
    # Inject a fake conflict that forces an update
    env._current_errors = ["Fake conflict"]
    
    # Prevent further chaos in this step
    env._registry.simulate_network_timeout = lambda p=0.0: False
    env._registry.simulate_yanked_package = lambda p=0.0: None
    
    action = Action(action_type="update", package_name="pkg-alpha", version_target="2.0.0")
    
    obs, reward, done, info = env.step(action)
    
    # Terminal success (100) + Progress (15) + Resilience (25) + Step Penalty (-1) = 139
    assert info.get("resilience_bonus") is True
    assert reward == 139.0
