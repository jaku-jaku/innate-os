#!/usr/bin/env python3
"""Minimal code-defined skill: logs a greeting and returns success."""

import time

from brain_client.skill_types import Skill, SkillResult


class HelloMars(Skill):
    """Example skill with no hardware — useful to verify discovery and execution."""

    def __init__(self, logger):
        super().__init__(logger)
        self._cancelled = False

    @property
    def name(self):
        return "hello_mars"

    def guidelines(self):
        return "Use for a quick sanity check that custom skills load and run."

    def execute(self, name: str = "MARS", pause_s: float = 0.5):
        """Say hello. Optional: name (str), pause_s (float) seconds before finishing."""
        self._cancelled = False
        self.logger.info(f"hello_mars: greeting {name!r}")
        deadline = time.monotonic() + max(0.0, pause_s)
        while time.monotonic() < deadline:
            if self._cancelled:
                return "hello_mars cancelled", SkillResult.CANCELLED
            time.sleep(0.05)
        msg = f"Hello, {name}! (custom skill from ~/skills)"
        self.logger.info(msg)
        return msg, SkillResult.SUCCESS

    def cancel(self):
        self._cancelled = True
        return "hello_mars cancel acknowledged"
