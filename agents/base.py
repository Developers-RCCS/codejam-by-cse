# agents/base.py
class BaseAgent:
    """Base class for all agents."""
    def run(self, **kwargs):
        raise NotImplementedError("Each agent must implement the 'run' method.")
