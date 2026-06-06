"""Process-wide minimum-interval rate limiter for free-tier API models.

Gemini 3.1 Flash Lite's free tier caps requests at 15 RPM. AURA issues
LLM calls from two independent code paths that share no local state —
the intent monitor (``AURAIntentMonitor``) and the decision engine
(``DecisionEngine``) — so neither can on its own guarantee the *combined*
rate stays under the cap. The intent path's per-dispatch throttle (see
``workflow.nodes.run_intent_node`` / ``predict_interval``) only spaces
intent calls; once the decision engine also calls the LLM every cycle,
the two together can blow past 15 RPM.

This module provides a single process-wide gate that *both* paths funnel
through (via ``GeminiClient.generate``), so their combined call rate is
spaced to stay within the free-tier budget.

Only free-tier models are throttled — matched by a substring of the model
name. Every other model runs on the paid tier and passes through
untouched (``throttle`` is a no-op).
"""

from __future__ import annotations

import logging
import threading
import time

logger = logging.getLogger(__name__)

# Free-tier requests-per-minute caps, keyed by a model-name substring.
# 15 RPM → one call every 4.0 s. Add more entries here if other free-tier
# models need gating; paid models are intentionally absent.
_FREE_TIER_RPM = {
    "flash-lite": 15,
}

_lock = threading.Lock()
_last_call_ts = 0.0


def min_interval_for(model: str) -> float:
    """Return the minimum seconds between calls for ``model`` (0 = unlimited)."""
    name = (model or "").lower()
    for needle, rpm in _FREE_TIER_RPM.items():
        if needle in name and rpm > 0:
            return 60.0 / rpm
    return 0.0


def throttle(model: str) -> None:
    """Block until enough time has elapsed since the last gated call.

    Shared across every caller in the process, so the intent monitor and
    the decision engine draw from one combined budget. No-op for models
    that aren't on a throttled free tier (e.g. anything on the paid tier).
    """
    interval = min_interval_for(model)
    if interval <= 0:
        return

    global _last_call_ts
    # Hold the lock across the sleep so concurrent callers queue up and
    # each is handed a slot ``interval`` seconds after the previous one,
    # rather than all waking together and bursting.
    with _lock:
        now = time.monotonic()
        wait = interval - (now - _last_call_ts)
        if wait > 0:
            logger.debug(
                "Free-tier rate limit (%s): sleeping %.2fs to honour %.1fs spacing",
                model, wait, interval,
            )
            time.sleep(wait)
            now = time.monotonic()
        _last_call_ts = now
