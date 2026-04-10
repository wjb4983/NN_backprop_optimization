from __future__ import annotations


class Probe:
    """Placeholder interface for v2 internal-state probes."""

    def capture(self, step: int, payload: dict) -> None:
        # v2 hook point for hidden-state introspection and attribution
        _ = (step, payload)
