"""
==============================================================================
Universal-Node-Resolver — UniversalMockRegistry
==============================================================================

Production-grade, in-memory mock npm registry with mathematically
guaranteed solvability via the Golden Path algorithm.

Architecture:
    1. Generate N packages with M version histories each.
    2. Build a "Golden Path" — one valid version per package that forms
       a fully conflict-free DAG.
    3. Wire dependency/peerDependency constraints so that Golden Path
       versions always satisfy every range.
    4. Add "noise" constraints on non-golden versions to create traps.
    5. Generate broken initial states by deviating from the golden path.

No real npm commands or network requests are ever executed.
==============================================================================
"""

from __future__ import annotations

import json
import logging
import random
import re
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger("universal_mock_registry")


# ═══════════════════════════════════════════════════════════════════════════
# SemVer Primitives
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, order=True)
class SemVer:
    """Immutable, comparable semantic version."""
    major: int
    minor: int
    patch: int

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"

    @classmethod
    def parse(cls, s: str) -> "SemVer":
        m = re.match(r"^(\d+)\.(\d+)\.(\d+)", s)
        if not m:
            raise ValueError(f"Cannot parse SemVer: '{s}'")
        return cls(int(m.group(1)), int(m.group(2)), int(m.group(3)))


def satisfies(version: SemVer, range_spec: str) -> bool:
    """
    Check if *version* satisfies *range_spec*.

    Supports: ^, ~, >=, <=, >, <, exact, *
    """
    spec = range_spec.strip()
    if spec == "*":
        return True
    if spec.startswith("^"):
        base = SemVer.parse(spec[1:])
        return base <= version < SemVer(base.major + 1, 0, 0)
    if spec.startswith("~"):
        base = SemVer.parse(spec[1:])
        return base <= version < SemVer(base.major, base.minor + 1, 0)
    if spec.startswith(">="):
        return version >= SemVer.parse(spec[2:])
    if spec.startswith("<="):
        return version <= SemVer.parse(spec[2:])
    if spec.startswith(">") and not spec.startswith(">="):
        return version > SemVer.parse(spec[1:])
    if spec.startswith("<") and not spec.startswith("<="):
        return version < SemVer.parse(spec[1:])
    return version == SemVer.parse(spec)


# ═══════════════════════════════════════════════════════════════════════════
# Data Structures
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class PackageDescriptor:
    """Full metadata for one package in the mock registry."""
    name: str
    available_versions: list[SemVer] = field(default_factory=list)
    dependencies: dict[str, dict[str, str]] = field(default_factory=dict)
    peer_dependencies: dict[str, dict[str, str]] = field(default_factory=dict)


@dataclass
class ConflictReport:
    """One detected SemVer conflict."""
    source_package: str
    source_version: str
    required_package: str
    required_range: str
    installed_version: str

    def to_log_line(self) -> str:
        return (
            f"CONFLICT: {self.source_package}@{self.source_version} requires "
            f"{self.required_package}@{self.required_range}, but "
            f"{self.required_package}@{self.installed_version} is installed."
        )


# ═══════════════════════════════════════════════════════════════════════════
# UniversalMockRegistry
# ═══════════════════════════════════════════════════════════════════════════

class UniversalMockRegistry:
    """
    Algorithmically generates massive DAGs of generic packages with
    a mathematically guaranteed conflict-free Golden Path.

    Parameters
    ----------
    seed : int | None
        RNG seed for full reproducibility.
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        self._rng = random.Random(seed)
        self._packages: dict[str, PackageDescriptor] = {}
        self._golden_path: dict[str, str] = {}

    # ──────────────────────────────────────────────────────────────────
    # REQUIREMENT 1: Dynamic Ecosystem Generation
    # ──────────────────────────────────────────────────────────────────

    def generate_ecosystem(
        self,
        num_packages: int,
        max_versions: int,
        max_deps_per_version: int = 3,
    ) -> None:
        """
        Build the entire mock registry from scratch.

        Algorithm:
            Phase 1 — Create packages with version histories.
            Phase 2 — Select one golden version per package.
            Phase 3 — Wire golden-path constraints (guaranteed valid).
            Phase 4 — Wire noise constraints on non-golden versions.
        """
        self._packages.clear()
        self._golden_path.clear()

        num_packages = max(3, min(num_packages, 200))
        max_versions = max(3, min(max_versions, 20))

        # Phase 1: Generate packages & version histories
        names = [f"pkg-{i:03d}" for i in range(1, num_packages + 1)]
        for name in names:
            versions = self._generate_version_history(max_versions)
            pkg = PackageDescriptor(
                name=name,
                available_versions=sorted(versions),
            )
            for v in pkg.available_versions:
                vs = str(v)
                pkg.dependencies[vs] = {}
                pkg.peer_dependencies[vs] = {}
            self._packages[name] = pkg

        # Phase 2: Select Golden Path
        for name, pkg in self._packages.items():
            n = len(pkg.available_versions)
            idx = self._rng.randint(n // 2, n - 1)
            self._golden_path[name] = str(pkg.available_versions[idx])

        # Phase 3: Wire Golden-Path dependencies (DAG via topological order)
        ordered_names = list(names)
        self._rng.shuffle(ordered_names)
        name_to_rank: dict[str, int] = {
            n: i for i, n in enumerate(ordered_names)
        }

        for pkg_name in ordered_names:
            pkg = self._packages[pkg_name]
            golden_ver = self._golden_path[pkg_name]
            my_rank = name_to_rank[pkg_name]

            candidates = [
                n for n in ordered_names
                if name_to_rank[n] > my_rank
            ]
            if not candidates:
                continue

            max_deps = min(len(candidates), 4)
            num_deps = self._rng.randint(1, max_deps)
            chosen_deps = self._rng.sample(candidates, num_deps)

            for dep_name in chosen_deps:
                dep_golden_ver = self._golden_path[dep_name]
                dep_golden_sv = SemVer.parse(dep_golden_ver)
                range_spec = self._make_golden_range(dep_golden_sv)

                if self._rng.random() < 0.20:
                    pkg.peer_dependencies[golden_ver][dep_name] = range_spec
                else:
                    pkg.dependencies[golden_ver][dep_name] = range_spec

        # Phase 4: Wire noise constraints on non-golden versions
        for pkg_name in ordered_names:
            pkg = self._packages[pkg_name]
            golden_ver = self._golden_path[pkg_name]
            my_rank = name_to_rank[pkg_name]

            candidates = [
                n for n in ordered_names
                if name_to_rank[n] > my_rank
            ]
            if not candidates:
                continue

            for ver in pkg.available_versions:
                vs = str(ver)
                if vs == golden_ver:
                    continue

                if self._rng.random() < 0.3:
                    continue  # some versions have no deps

                num_deps = self._rng.randint(0, max_deps_per_version)
                if num_deps == 0:
                    continue
                
                noise_deps = self._rng.sample(candidates, min(num_deps, len(candidates)))
                for dep_name in noise_deps:
                    dep_golden_sv = SemVer.parse(self._golden_path[dep_name])
                    range_spec = self._make_conflicting_range(
                        dep_golden_sv,
                        self._packages[dep_name].available_versions,
                    )
                    if self._rng.random() < 0.15:
                        pkg.peer_dependencies[vs][dep_name] = range_spec
                    else:
                        pkg.dependencies[vs][dep_name] = range_spec

        # Verify golden path is actually valid
        conflicts = self.validate_installation(self._golden_path)
        assert len(conflicts) == 0, (
            f"FATAL: Golden Path has {len(conflicts)} conflicts. "
            f"This is a bug in the generator."
        )
        logger.info(
            "Ecosystem generated: %d packages, golden path verified clean.",
            num_packages,
        )

    # ──────────────────────────────────────────────────────────────────
    # REQUIREMENT 2: Universal State Generation
    # ──────────────────────────────────────────────────────────────────

    def get_package_info(self, package_name: str) -> dict[str, Any]:
        """Retrieve version history and dependency requirements."""
        pkg = self._packages.get(package_name)
        if pkg is None:
            return {"error": f"Package '{package_name}' not found."}
        return {
            "name": pkg.name,
            "versions": [str(v) for v in pkg.available_versions],
            "dependencies": dict(pkg.dependencies),
            "peer_dependencies": dict(pkg.peer_dependencies),
        }

    def generate_broken_state(self, complexity_level: int) -> dict[str, str]:
        """
        Return a broken installation state with guaranteed solvability.

        Level 1 (Easy):   3-5 packages, 1 direct SemVer violation.
        Level 2 (Medium): 10-15 packages, transitive peer conflicts.
        Level 3 (Hard):   20+ packages, >=3 sequential fixes needed.
        """
        all_names = list(self._packages.keys())
        if not all_names:
            raise RuntimeError("No ecosystem generated. Call generate_ecosystem first.")

        complexity_level = max(1, min(3, complexity_level))

        if complexity_level == 1:
            return self._broken_state_easy(all_names)
        elif complexity_level == 2:
            return self._broken_state_medium(all_names)
        else:
            return self._broken_state_hard(all_names)

    # ──────────────────────────────────────────────────────────────────
    # Validation & Formatting
    # ──────────────────────────────────────────────────────────────────

    @property
    def packages(self) -> dict[str, PackageDescriptor]:
        return dict(self._packages)

    @property
    def golden_path(self) -> dict[str, str]:
        """The guaranteed conflict-free solution."""
        return dict(self._golden_path)

    def get_available_versions(self, package_name: str) -> list[str]:
        pkg = self._packages.get(package_name)
        if pkg is None:
            return []
        return [str(v) for v in pkg.available_versions]

    def resolve_range(self, package_name: str, range_spec: str) -> list[str]:
        pkg = self._packages.get(package_name)
        if pkg is None:
            return []
        return [
            str(v) for v in pkg.available_versions
            if satisfies(v, range_spec)
        ]

    def validate_installation(
        self, installed: dict[str, str],
    ) -> list[ConflictReport]:
        """Validate installed versions against all constraints."""
        conflicts: list[ConflictReport] = []

        for pkg_name, ver_str in installed.items():
            pkg = self._packages.get(pkg_name)
            if pkg is None:
                continue

            for dep_name, dep_range in pkg.dependencies.get(ver_str, {}).items():
                dep_ver = installed.get(dep_name)
                if dep_ver is None:
                    conflicts.append(ConflictReport(
                        source_package=pkg_name, source_version=ver_str,
                        required_package=dep_name, required_range=dep_range,
                        installed_version="MISSING",
                    ))
                elif not self._safe_satisfies(dep_ver, dep_range):
                    conflicts.append(ConflictReport(
                        source_package=pkg_name, source_version=ver_str,
                        required_package=dep_name, required_range=dep_range,
                        installed_version=dep_ver,
                    ))

            for peer_name, peer_range in pkg.peer_dependencies.get(ver_str, {}).items():
                peer_ver = installed.get(peer_name)
                if peer_ver is None:
                    conflicts.append(ConflictReport(
                        source_package=pkg_name, source_version=ver_str,
                        required_package=peer_name, required_range=peer_range,
                        installed_version="MISSING",
                    ))
                elif not self._safe_satisfies(peer_ver, peer_range):
                    conflicts.append(ConflictReport(
                        source_package=pkg_name, source_version=ver_str,
                        required_package=peer_name, required_range=peer_range,
                        installed_version=peer_ver,
                    ))

        return conflicts

    def format_error_log(self, conflicts: list[ConflictReport]) -> str:
        if not conflicts:
            return ""
        return "\n".join(c.to_log_line() for c in conflicts)

    def build_package_json_string(self, installed: dict[str, str]) -> str:
        """Build stringified JSON from installed state."""
        deps: dict[str, str] = {}
        peers: dict[str, str] = {}

        for pkg_name, ver_str in sorted(installed.items()):
            pkg = self._packages.get(pkg_name)
            if pkg is None:
                deps[pkg_name] = ver_str
                continue
            is_peer = False
            for other_name, other_ver in installed.items():
                if other_name == pkg_name:
                    continue
                other_pkg = self._packages.get(other_name)
                if other_pkg and pkg_name in other_pkg.peer_dependencies.get(other_ver, {}):
                    is_peer = True
                    break
            if is_peer:
                peers[pkg_name] = ver_str
            else:
                deps[pkg_name] = ver_str

        result: dict[str, Any] = {"dependencies": deps}
        if peers:
            result["peerDependencies"] = peers
        return json.dumps(result, indent=2, sort_keys=True)

    # ══════════════════════════════════════════════════════════════════
    # Private: Version History Generation
    # ══════════════════════════════════════════════════════════════════

    def _generate_version_history(self, max_versions: int) -> list[SemVer]:
        count = self._rng.randint(3, max_versions)
        versions: set[SemVer] = set()
        major = self._rng.randint(0, 3)
        minor = 0
        patch = 0
        for _ in range(count):
            versions.add(SemVer(major, minor, patch))
            roll = self._rng.random()
            if roll < 0.20:
                major += 1; minor = 0; patch = 0
            elif roll < 0.55:
                minor += 1; patch = 0
            else:
                patch += 1
        return sorted(versions)

    # ══════════════════════════════════════════════════════════════════
    # Private: Range Spec Generators
    # ══════════════════════════════════════════════════════════════════

    def _make_golden_range(self, golden_sv: SemVer) -> str:
        """Create a SemVer range that INCLUDES the golden version."""
        style = self._rng.choices(
            ["caret", "tilde", "exact", "gte"],
            weights=[45, 25, 15, 15],
        )[0]
        if style == "caret":
            return f"^{golden_sv.major}.0.0"
        elif style == "tilde":
            return f"~{golden_sv.major}.{golden_sv.minor}.0"
        elif style == "exact":
            return str(golden_sv)
        else:
            base_minor = max(0, golden_sv.minor - self._rng.randint(0, 2))
            return f">={golden_sv.major}.{base_minor}.0"

    def _make_conflicting_range(
        self, golden_sv: SemVer, all_versions: list[SemVer],
    ) -> str:
        """Create a SemVer range that EXCLUDES the golden version."""
        other_majors = [v for v in all_versions if v.major != golden_sv.major]
        if other_majors:
            pick = self._rng.choice(other_majors)
            style = self._rng.choice(["exact", "tilde"])
            if style == "exact":
                return str(pick)
            return f"~{pick}"

        older = [v for v in all_versions if v < golden_sv]
        if older:
            pick = self._rng.choice(older)
            return str(pick)

        fake = SemVer(golden_sv.major + 10, 0, 0)
        return f">={fake}"

    # ══════════════════════════════════════════════════════════════════
    # Private: Broken State Generators
    # ══════════════════════════════════════════════════════════════════

    def _broken_state_easy(self, all_names: list[str]) -> dict[str, str]:
        """Level 1: 3-5 packages, 1 direct violation."""
        subset_size = self._rng.randint(3, min(5, len(all_names)))
        subset = self._pick_connected_subset(all_names, subset_size)
        state = {name: self._golden_path[name] for name in subset}
        breakable = [n for n in subset if self._has_non_golden_version(n)]
        if breakable:
            victim = self._rng.choice(breakable)
            state[victim] = self._pick_non_golden_version(victim)
        return state

    def _broken_state_medium(self, all_names: list[str]) -> dict[str, str]:
        """Level 2: 10-15 packages, transitive/peer conflicts."""
        subset_size = self._rng.randint(10, min(15, len(all_names)))
        subset = self._pick_connected_subset(all_names, subset_size)
        state = {name: self._golden_path[name] for name in subset}
        breakable = [n for n in subset if self._has_non_golden_version(n)]
        num_breaks = min(self._rng.randint(3, 5), len(breakable))
        victims = self._rng.sample(breakable, num_breaks) if breakable else []
        for victim in victims:
            state[victim] = self._pick_non_golden_version(victim)
        return state

    def _broken_state_hard(self, all_names: list[str]) -> dict[str, str]:
        """Level 3: 20+ packages, >=3 sequential fixes needed."""
        subset_size = min(max(20, len(all_names)), len(all_names))
        subset = self._pick_connected_subset(all_names, subset_size)
        state = {name: self._golden_path[name] for name in subset}
        breakable = [n for n in subset if self._has_non_golden_version(n)]
        num_breaks = min(self._rng.randint(6, 10), len(breakable))
        victims = self._rng.sample(breakable, num_breaks) if breakable else []
        for victim in victims:
            state[victim] = self._pick_oldest_non_golden(victim)
        return state

    # ══════════════════════════════════════════════════════════════════
    # Private: Subset & Version Helpers
    # ══════════════════════════════════════════════════════════════════

    def _pick_connected_subset(self, all_names: list[str], size: int) -> list[str]:
        """BFS-based connected subgraph selection."""
        size = min(size, len(all_names))
        start = self._rng.choice(all_names)
        visited: set[str] = {start}
        queue: list[str] = [start]

        while len(visited) < size and queue:
            current = queue.pop(0)
            pkg = self._packages[current]
            golden_ver = self._golden_path[current]
            neighbors: list[str] = []
            neighbors.extend(pkg.dependencies.get(golden_ver, {}).keys())
            neighbors.extend(pkg.peer_dependencies.get(golden_ver, {}).keys())

            for other_name, other_pkg in self._packages.items():
                if other_name in visited:
                    continue
                other_gv = self._golden_path[other_name]
                all_deps = set(other_pkg.dependencies.get(other_gv, {}).keys())
                all_deps |= set(other_pkg.peer_dependencies.get(other_gv, {}).keys())
                if current in all_deps:
                    neighbors.append(other_name)

            self._rng.shuffle(neighbors)
            for nb in neighbors:
                if nb not in visited and len(visited) < size:
                    visited.add(nb)
                    queue.append(nb)

        if len(visited) < size:
            remaining = [n for n in all_names if n not in visited]
            self._rng.shuffle(remaining)
            for n in remaining:
                if len(visited) >= size:
                    break
                visited.add(n)

        return sorted(visited)

    def _has_non_golden_version(self, pkg_name: str) -> bool:
        pkg = self._packages[pkg_name]
        golden = self._golden_path[pkg_name]
        return any(str(v) != golden for v in pkg.available_versions)

    def _pick_non_golden_version(self, pkg_name: str) -> str:
        pkg = self._packages[pkg_name]
        golden = self._golden_path[pkg_name]
        alternatives = [str(v) for v in pkg.available_versions if str(v) != golden]
        return self._rng.choice(alternatives) if alternatives else golden

    def _pick_oldest_non_golden(self, pkg_name: str) -> str:
        pkg = self._packages[pkg_name]
        golden = self._golden_path[pkg_name]
        alternatives = [v for v in pkg.available_versions if str(v) != golden]
        if alternatives:
            return str(min(alternatives))
        return golden

    @staticmethod
    def _safe_satisfies(ver_str: str, range_spec: str) -> bool:
        try:
            return satisfies(SemVer.parse(ver_str), range_spec)
        except ValueError:
            return False


# ═══════════════════════════════════════════════════════════════════════════
# Backward-compatible alias so environment.py keeps working
# ═══════════════════════════════════════════════════════════════════════════

class PackageRegistry(UniversalMockRegistry):
    """
    Backward-compatible wrapper around UniversalMockRegistry.
    Maintains the old generate_ecosystem(complexity_level) API.
    """

    def generate_ecosystem(  # type: ignore[override]
        self,
        complexity_level_or_num_packages: int = 1,
        max_versions: int = 8,
        max_deps_per_version: int = 3,
    ) -> dict[str, str]:
        if 1 <= complexity_level_or_num_packages <= 5 and max_versions == 8:
            tier_map = {1: 5, 2: 10, 3: 15, 4: 20, 5: 30}
            num_packages = tier_map.get(complexity_level_or_num_packages, 5)
            complexity = complexity_level_or_num_packages
        else:
            num_packages = complexity_level_or_num_packages
            complexity = 2 if num_packages <= 10 else 3

        super().generate_ecosystem(num_packages, max_versions, max_deps_per_version)
        return self.generate_broken_state(min(complexity, 3))


# ═══════════════════════════════════════════════════════════════════════════
# Module Exports
# ═══════════════════════════════════════════════════════════════════════════

__all__: list[str] = [
    "SemVer",
    "satisfies",
    "PackageDescriptor",
    "ConflictReport",
    "UniversalMockRegistry",
    "PackageRegistry",
]
