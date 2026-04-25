"""
==============================================================================
Universal-Node-Resolver — Adversarial Chaos Engine
==============================================================================

Simulates real-world NPM failures (e.g. left-pad yanking, HTTP 503s) 
to ensure the RL Agent learns robust recovery strategies.
"""

import logging
import random
from typing import Optional

from server.registry import UniversalMockRegistry, ConflictReport, SemVer

logger = logging.getLogger("chaos_engine")


class AdversarialRegistryWrapper:
    """
    Wraps the UniversalMockRegistry to inject adversarial network 
    and package state failures.
    """

    def __init__(self, base_registry: UniversalMockRegistry):
        self._base = base_registry
        # Use a deterministic RNG seeded from the base registry
        self._rng = random.Random(self._base._rng.random())

    def __getattr__(self, name):
        """Proxy all other methods directly to the base registry."""
        return getattr(self._base, name)

    def simulate_yanked_package(self, probability: float = 0.05) -> Optional[tuple[str, str]]:
        """
        Randomly deletes a specific version of a package from the registry.
        Simulates the infamous "left-pad" incident.
        """
        if self._rng.random() > probability:
            return None

        packages = list(self._base.packages.keys())
        if not packages:
            return None

        # Pick a random package
        pkg_name = self._rng.choice(packages)
        pkg_desc = self._base.packages[pkg_name]
        
        if not pkg_desc.available_versions:
            return None

        # Pick a random version and yank it
        yanked_version = self._rng.choice(pkg_desc.available_versions)
        pkg_desc.available_versions.remove(yanked_version)
        
        logger.warning(f"CHAOS EVENT: Yanked {pkg_name}@{yanked_version} from registry!")
        return pkg_name, str(yanked_version)

    def simulate_network_timeout(self, probability: float = 0.02) -> bool:
        """
        Simulates a failed API call to the registry.
        """
        if self._rng.random() < probability:
            logger.warning("CHAOS EVENT: HTTP 503 Registry Timeout!")
            return True
        return False

    def validate_installation(self, installed_versions: dict[str, str]) -> list[ConflictReport]:
        """
        Intercepts validation to specifically flag YANKED packages as conflicts.
        """
        reports = self._base.validate_installation(installed_versions)
        
        # Verify all currently installed versions still exist in the registry
        for pkg, ver_str in installed_versions.items():
            pkg_obj = self._base.packages.get(pkg)
            if pkg_obj:
                parsed_ver = SemVer.parse(ver_str)
                if parsed_ver not in pkg_obj.available_versions:
                    # The package was yanked! Inject a critical conflict.
                    reports.append(
                        ConflictReport(
                            source_package="npm",
                            source_version="core",
                            required_package=pkg,
                            required_range="exists in registry",
                            installed_version=f"{ver_str} (YANKED)"
                        )
                    )
                    
        return reports
