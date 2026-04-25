"""
==============================================================================
Universal-Node-Resolver — Dynamic Curriculum Engine
==============================================================================

An auto-tuner that continuously tracks the agent's win-rate over a rolling 
window. It dynamically adjusts the generated DAG complexity parameters to 
ensure the agent is constantly challenged without suffering from 
catastrophic forgetting.
"""

import collections
import logging

logger = logging.getLogger("curriculum_engine")


class DynamicCurriculumEngine:
    """
    Tracks RL Agent success rates and scales the generation complexity.
    """

    def __init__(self, window_size: int = 100):
        # Rolling window of the last `window_size` episodes.
        # True = Success, False = Failure
        self.history = collections.deque(maxlen=window_size)
        
        # Base Complexity Parameters
        self.num_packages = 5
        self.max_versions = 5
        self.max_deps_per_version = 2

    def record_outcome(self, success: bool) -> None:
        """
        Record the outcome of the latest episode.
        """
        self.history.append(success)
        self._tune_parameters()

    def get_next_complexity(self) -> dict[str, int]:
        """
        Outputs the parameters for the next `generate_ecosystem` call.
        """
        return {
            "complexity_level_or_num_packages": self.num_packages,
            "max_versions": self.max_versions,
            "max_deps_per_version": self.max_deps_per_version
        }

    def get_win_rate(self) -> float:
        """Calculates the current rolling win rate."""
        if not self.history:
            return 0.0
        return sum(self.history) / len(self.history)

    def _tune_parameters(self) -> None:
        """
        Gracefully scales the DAG complexity up or down based on performance.
        Only adjusts if we have enough statistical significance (e.g. >= 10 eps).
        """
        if len(self.history) < 10:
            return

        win_rate = self.get_win_rate()

        if win_rate >= 0.85:
            # Agent is dominating. Make it harder!
            self.num_packages = min(100, self.num_packages + 2)
            self.max_versions = min(15, self.max_versions + 1)
            
            # Periodically increase graph density (more deps = more conflicts)
            if self.num_packages % 10 == 0:
                self.max_deps_per_version = min(5, self.max_deps_per_version + 1)
                
            logger.info(f"📈 Win rate {win_rate:.0%}. SCALING UP: packages={self.num_packages}, versions={self.max_versions}, deps={self.max_deps_per_version}")
            
            # Flush history slightly to allow stabilization at new difficulty
            for _ in range(5):
                if self.history:
                    self.history.popleft()

        elif win_rate <= 0.20:
            # Agent is failing catastrophically. Ease off to prevent forgetting.
            self.num_packages = max(5, self.num_packages - 2)
            self.max_versions = max(5, self.max_versions - 1)
            
            if self.max_deps_per_version > 2 and self.num_packages % 10 == 0:
                self.max_deps_per_version -= 1
                
            logger.info(f"📉 Win rate {win_rate:.0%}. SCALING DOWN: packages={self.num_packages}, versions={self.max_versions}, deps={self.max_deps_per_version}")
            
            # Flush history heavily so it doesn't get stuck scaling down
            for _ in range(15):
                if self.history:
                    self.history.popleft()
