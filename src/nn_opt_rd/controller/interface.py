from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ControllerState:
    step: int = 0


class Controller:
    """Placeholder interface for v1 learned/static optimization controller."""

    def before_step(self, state: ControllerState, metrics: dict) -> None:
        # v1 hook point for schedule or meta-optimizer logic
        state.step += 1
