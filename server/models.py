"""
==============================================================================
Universal-Node-Resolver — Strict Pydantic Data Models
==============================================================================

This module defines the rigorous, type-safe schemas that govern all
communication between the RL agent and the environment.

Two primary contracts are established:

    Action      The agent's decision at each timestep. Supports "update"
                (set a package to a specific SemVer version) and "delete"
                (remove a package from the dependency graph entirely).

    Observation  The environment's response after each action. Provides
                the agent with a complete view of the current dependency
                graph state, all active conflict messages, the current
                step count, and the complexity tier of the generated
                ecosystem.

Design Principles:
    • Strict mode is enabled on every model — no extra fields, no
      implicit coercion. The agent MUST produce valid JSON or the
      action is rejected.
    • All fields carry explicit descriptions for automatic OpenAPI /
      JSON Schema generation.
    • Validators enforce SemVer format on version strings and logical
      consistency (e.g., version_target must be None when deleting).

Dependencies:
    pydantic >= 2.0

==============================================================================
"""

from __future__ import annotations

import re
from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Regex for a strict Semantic Versioning string (major.minor.patch).
# Pre-release and build metadata tags are intentionally supported.
_SEMVER_PATTERN: re.Pattern[str] = re.compile(
    r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)"
    r"(?:-((?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)"
    r"(?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))?"
    r"(?:\+([0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*))?$"
)


# ---------------------------------------------------------------------------
# Action Model
# ---------------------------------------------------------------------------

class Action(BaseModel):
    """
    Represents a single atomic action the RL agent can take within the
    Universal-Node-Resolver environment.

    The agent may either:
        • **update** — Set a package to a specific SemVer-compliant version.
        • **delete** — Remove a package from the dependency graph entirely.

    Examples
    --------
    Update action (valid):
        >>> Action(action_type="update", package_name="pkg-alpha", version_target="2.1.0")

    Delete action (valid):
        >>> Action(action_type="delete", package_name="pkg-beta", version_target=None)

    Invalid (delete with a version):
        >>> Action(action_type="delete", package_name="pkg-beta", version_target="1.0.0")
        ValidationError: version_target must be None when action_type is 'delete'.
    """

    model_config = {
        "strict": True,
        "extra": "forbid",
        "json_schema_extra": {
            "examples": [
                {
                    "action_type": "update",
                    "package_name": "pkg-alpha",
                    "version_target": "2.1.0",
                },
                {
                    "action_type": "delete",
                    "package_name": "pkg-beta",
                    "version_target": None,
                },
            ]
        },
    }

    action_type: Literal["update", "delete"] = Field(
        ...,
        description=(
            "The type of mutation to apply. "
            "'update' sets the package to `version_target`. "
            "'delete' removes the package from the dependency graph."
        ),
    )

    package_name: str = Field(
        ...,
        min_length=1,
        max_length=214,
        description=(
            "The canonical name of the target package in the dependency "
            "graph (e.g., 'pkg-alpha', 'lodash'). Must follow npm naming "
            "conventions: lowercase, no leading dots or underscores."
        ),
    )

    version_target: Optional[str] = Field(
        default=None,
        description=(
            "The exact SemVer version to install (e.g., '2.1.0', '3.0.0-beta.1'). "
            "Required when action_type is 'update'. "
            "Must be None when action_type is 'delete'."
        ),
    )

    # ------------------------------------------------------------------
    # Cross-field validation
    # ------------------------------------------------------------------

    @model_validator(mode="after")
    def _validate_action_consistency(self) -> "Action":
        """Ensure logical consistency between action_type and version_target."""
        if self.action_type == "update":
            if self.version_target is None:
                raise ValueError(
                    "version_target is required when action_type is 'update'. "
                    "Provide a valid SemVer string (e.g., '1.2.3')."
                )
            if not _SEMVER_PATTERN.match(self.version_target):
                raise ValueError(
                    f"version_target '{self.version_target}' is not a valid "
                    f"SemVer string. Expected format: MAJOR.MINOR.PATCH "
                    f"(e.g., '1.0.0', '2.3.1-alpha.1')."
                )

        if self.action_type == "delete" and self.version_target is not None:
            raise ValueError(
                "version_target must be None when action_type is 'delete'. "
                "A delete action removes the package entirely — no version needed."
            )

        return self


# ---------------------------------------------------------------------------
# Observation Model
# ---------------------------------------------------------------------------

class Observation(BaseModel):
    """
    Represents the full observable state returned by the environment
    after each step.

    The observation provides the agent with everything it needs to
    reason about the next action:

        • **current_package_json** — The complete, stringified JSON of the
          current dependency tree. This is dynamically generated and
          universally structured (arbitrary package names and versions).

        • **npm_error_log** — A multiline string aggregating every active
          SemVer conflict detected by the mock resolution engine. Each
          line follows the format:
              ``CONFLICT: <pkg-a>@<ver> requires <pkg-b>@<range>, but <pkg-b>@<installed> is installed.``

        • **step_count** — The number of actions the agent has taken so
          far in the current episode.

        • **complexity_level** — An integer (1–5) indicating the
          difficulty tier of the generated dependency graph.

    Examples
    --------
    >>> obs = Observation(
    ...     current_package_json='{"dependencies": {"pkg-alpha": "1.0.0"}}',
    ...     npm_error_log="CONFLICT: pkg-alpha@1.0.0 requires pkg-beta@^2.0.0, but pkg-beta@1.5.0 is installed.",
    ...     step_count=3,
    ...     complexity_level=2,
    ... )
    """

    model_config = {
        "strict": True,
        "extra": "forbid",
    }

    current_package_json: str = Field(
        ...,
        min_length=2,
        description=(
            "The complete stringified JSON representation of the current "
            "dependency graph. Contains both 'dependencies' and "
            "'peerDependencies' maps with package names as keys and "
            "SemVer version strings as values. Dynamically generated — "
            "package names are universal and arbitrary."
        ),
    )

    npm_error_log: str = Field(
        default="",
        description=(
            "Concatenated multiline string of all active SemVer conflicts "
            "generated by the mock resolution engine. An empty string "
            "indicates zero conflicts (i.e., the episode is solved). "
            "Each conflict follows the format: "
            "'CONFLICT: <pkg>@<ver> requires <dep>@<range>, but <dep>@<installed> is installed.'"
        ),
    )

    step_count: int = Field(
        ...,
        ge=0,
        description=(
            "The number of actions the agent has executed so far in the "
            "current episode. Starts at 0 and increments after each "
            "valid or invalid action."
        ),
    )

    complexity_level: int = Field(
        ...,
        ge=1,
        le=5,
        description=(
            "The difficulty tier of the current dependency graph. "
            "Ranges from 1 (simple, 3-5 packages) to 5 (adversarial, "
            "18-25 packages with cyclic peer constraints and version "
            "matrices)."
        ),
    )


# ---------------------------------------------------------------------------
# Module-level exports
# ---------------------------------------------------------------------------

__all__: list[str] = ["Action", "Observation"]
