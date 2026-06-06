"""Task-specific perception monitor for the milk tea-making task.

This is a deliberately minimal, **delay-only** perception monitor.  It runs
no detection at all — its sole job is to introduce a fixed wall-clock delay
(default 1 second) on every frame so the fast ``sense → decide → act`` loop is
paced at a realistic perception cadence.

Motivation: when the workflow is run with ground-truth intent
(``--intent-source ground_truth``), the intent step is essentially free, so
the loop spins through cycles almost instantly.  That makes it hard to watch
the ground-truth intent annotations advance in the UI.  Adding this
task-dependent perception monitor reintroduces a per-cycle cost that mimics a
real perception module's latency without needing SAM3 / a VLM.

Usage::

    monitor = MilkTeaPerceptionMonitor()
    result = await monitor.process_frame(bgr_frame)   # ~1s later
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MilkTeaPerceptionConfig:
    """Configuration for the milk-tea delay-only perception monitor."""

    # Wall-clock delay applied on every processed frame (seconds).
    delay_sec: float = 5.0
    # Run the delay every N calls (1 = every frame).
    process_every_n: int = 1


class MilkTeaPerceptionMonitor:
    """Delay-only perception monitor for the milk tea-making task.

    Does not detect anything; it simply ``await asyncio.sleep(delay_sec)`` so
    each perception cycle costs a realistic amount of wall-clock time.  This
    keeps the loop from racing ahead when intent is served from ground-truth
    annotations.
    """

    def __init__(self, config: Optional[MilkTeaPerceptionConfig] = None) -> None:
        self.config = config or MilkTeaPerceptionConfig()
        self._call_count = 0

    async def process_frame(self, frame: np.ndarray) -> Optional[Dict[str, Any]]:
        """Sleep for the configured delay, then return ``None``.

        Returns ``None`` (no perception output) so ``run_perception_node``
        treats this as a pure no-op aside from the wall-clock delay.
        """
        self._call_count += 1
        if self._call_count % self.config.process_every_n != 0:
            return None

        await asyncio.sleep(self.config.delay_sec)
        return None
