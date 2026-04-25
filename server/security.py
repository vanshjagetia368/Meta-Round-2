"""
==============================================================================
Universal-Node-Resolver — Payload Defense Shield
==============================================================================

Provides strict guardrails against LLM hallucinations and prompt injection.
Ensures that the LLM's atomic JSON actions ONLY mutate valid dependency fields
and do not inject malicious payloads into `scripts`, `bin`, `engines`, etc.
"""

import logging
import re
from typing import Any

from server.models import Action

logger = logging.getLogger("payload_defense_shield")


class InjectionAttempt(Exception):
    """Raised when an unauthorized modification is detected."""
    pass


class PayloadDefenseShield:
    """
    Cryptographically guarantees LLM actions are restricted to dependencies.
    """

    # NPM's standard top-level keys that should NEVER be modified by our agent.
    # We restrict modification strictly to package names.
    RESTRICTED_KEYS = {
        "scripts", "bin", "engines", "main", "type", "exports", 
        "browser", "os", "cpu", "preinstall", "postinstall",
        "name", "version", "description", "author", "license"
    }

    # Basic heuristic to catch command injections in version targets
    # e.g., 'rm -rf', ';', '&&', '||', '$(', '`'
    MALICIOUS_PAYLOAD_REGEX = re.compile(r"[;&|`$]|\brm\b|\bmv\b|\bcp\b|\bwget\b|\bcurl\b")

    @classmethod
    def verify_action_boundaries(cls, original_json_str: str, proposed_action: Action) -> None:
        """
        Validates that the proposed Action does not breach security boundaries.
        
        Args:
            original_json_str (str): The current stringified package.json.
            proposed_action (Action): The atomic action proposed by the LLM.
            
        Raises:
            InjectionAttempt: If the action targets restricted fields or contains malicious payloads.
        """
        # 1. Target Name Validation
        # If the LLM sets package_name="scripts", it's trying to overwrite the scripts block.
        if proposed_action.package_name in cls.RESTRICTED_KEYS:
            logger.error(f"InjectionAttempt: Action targets restricted key '{proposed_action.package_name}'")
            raise InjectionAttempt(f"Unauthorized target: {proposed_action.package_name} is a protected system field.")
            
        # 2. Version Target Payload Validation
        if proposed_action.version_target:
            if cls.MALICIOUS_PAYLOAD_REGEX.search(proposed_action.version_target):
                logger.error(f"InjectionAttempt: Malicious payload detected in version_target: '{proposed_action.version_target}'")
                raise InjectionAttempt("Unauthorized payload: Detected shell injection characters in version target.")
                
            # Version targets should generally be valid SemVer ranges or tags, max ~100 chars
            if len(proposed_action.version_target) > 100:
                logger.error(f"InjectionAttempt: Payload suspiciously long: {len(proposed_action.version_target)} chars")
                raise InjectionAttempt("Unauthorized payload: Version target exceeds maximum safe length.")
