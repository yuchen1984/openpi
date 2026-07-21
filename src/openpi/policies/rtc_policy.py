"""Real-Time Chunking (RTC) serving wrapper.

Implements the inference-time chunk-stitching scheme of "Real-Time Execution
of Action Chunking Flow Policies" (Black, Galliker, Levine — arXiv:2506.07339)
on top of an openpi flow-matching :class:`~openpi.policies.policy.Policy`
(pi0 / pi0.5 flow heads; NOT pi0-FAST). No retraining required.

Protocol (opt-in per request, so old clients are untouched): the client may
attach two integers to the observation dict —

* ``rtc/offset``  — how many actions of the PREVIOUS returned chunk had been
  consumed at observation-capture time. The wrapper drops that many leading
  actions of its cached chunk to align it with the new chunk's time axis.
* ``rtc/delay``   — how many MORE actions are expected to execute during this
  round-trip. That prefix of the new chunk is already committed to the robot,
  so it is frozen (guidance weight 1.0) to the previous chunk's values; the
  rest of the overlap gets an exponentially decaying soft guidance weight, so
  the new chunk keeps the previous chunk's mode while still updating on the
  fresh observation.

When the keys are absent (or no previous chunk is cached — first query), the
wrapper is a transparent pass-through: sampling is bit-identical to the
unwrapped policy. The guidance itself runs inside ``Pi0.sample_actions`` via
``prev_actions``/``prefix_weights`` (see models/pi0.py) in the model's
NORMALIZED action space; the cached chunk comes from the ``raw_actions`` key
the Policy exposes, so no unnormalize/renormalize round-trip is needed.

NOTE on action-space validity: freezing/guiding toward the previous chunk's
raw rows assumes a row means the same thing regardless of which observation
the chunk was generated from. That holds for step-wise delta actions (our
LIBERO EE-delta checkpoints — deltas are anchor-invariant) and approximately
for absolute actions when the state changes little between queries. For
chunk-relative delta configs (``extra_delta_transform=True``), rows are
relative to each chunk's OWN start state; the wrapper still runs but the
frozen prefix carries an O(state drift within a chunk) anchor error. Prefer
training-time RTC conditioning for those (arXiv:2512.05964).

Concurrency: the websocket server handles one request at a time per policy,
matching the single-arm GUI client. The cache is per-process, keyed on
nothing — a client that interleaves two episodes must send ``rtc/reset``.
"""

from __future__ import annotations

import dataclasses
import logging

import numpy as np
from openpi_client import base_policy as _base_policy
from typing_extensions import override

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class RtcConfig:
    """Serve-side RTC tuning.

    ``execution_horizon``: how many actions of each chunk the guidance treats
    as the stitchable overlap (beyond it the new chunk is free). None = the
    full action horizon. ``max_guidance_weight`` scales the soft mask before
    it is clipped to [0, 1]: larger = the overlap adheres more strongly to the
    previous chunk (10 ~= the paper/LeRobot default for 10 flow steps).
    ``soft_mask_alpha`` sets how fast the exponential soft mask decays across
    the overlap (weight ~ exp(-alpha * u), u in [0, 1] across the overlap).
    """

    enabled: bool = False
    execution_horizon: int | None = None
    max_guidance_weight: float = 10.0
    soft_mask_alpha: float = 4.0


def rtc_prefix_weights(action_horizon: int, delay: int, valid: int,
                       execution_horizon: int | None,
                       max_guidance_weight: float,
                       soft_mask_alpha: float) -> np.ndarray:
    """Per-action-index guidance weights in [0, 1].

    ``delay`` leading actions -> 1.0 (frozen: they will execute regardless);
    indices up to ``min(valid, execution_horizon)`` -> clipped exponential
    decay; beyond -> 0 (the previous chunk has no opinion there). ``valid`` is
    how many aligned actions the cached chunk actually covers. Pure.
    """
    ah = int(action_horizon)
    w = np.zeros(ah, dtype=np.float32)
    end = min(int(valid), int(execution_horizon) if execution_horizon else ah, ah)
    d = max(0, min(int(delay), end))
    w[:d] = 1.0
    if end > d:
        u = (np.arange(d, end, dtype=np.float32) - d + 1.0) / float(end - d)
        w[d:end] = np.clip(
            float(max_guidance_weight) * np.exp(-float(soft_mask_alpha) * u),
            0.0, 1.0)
    return w


class RtcPolicy(_base_policy.BasePolicy):
    """Wraps a flow ``Policy`` with RTC chunk stitching (see module docstring)."""

    def __init__(self, policy, config: RtcConfig | None = None):
        self._policy = policy
        self._config = config or RtcConfig(enabled=True)
        self._prev_raw: np.ndarray | None = None  # (ah, ad) normalized

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        obs = dict(obs)
        if obs.pop("rtc/reset", False):
            self._prev_raw = None
        offset = obs.pop("rtc/offset", None)
        delay = obs.pop("rtc/delay", None)

        extra = None
        if (self._config.enabled and offset is not None
                and self._prev_raw is not None):
            offset = int(offset)
            delay = int(delay or 0)
            ah = self._prev_raw.shape[0]
            offset = max(0, min(offset, ah))
            valid = ah - offset
            if valid > 0:
                aligned = np.concatenate(
                    [self._prev_raw[offset:],
                     np.repeat(self._prev_raw[-1:], offset, axis=0)],
                    axis=0)[:ah]
                weights = rtc_prefix_weights(
                    ah, delay, valid, self._config.execution_horizon,
                    self._config.max_guidance_weight,
                    self._config.soft_mask_alpha)
                extra = {
                    "prev_actions": aligned.astype(np.float32),
                    "prefix_weights": weights,
                }

        try:
            result = self._policy.infer(obs, extra_sample_kwargs=extra)
        except TypeError:
            # Inner policy predates extra_sample_kwargs (or is a recorder
            # shim) — fail soft to plain inference.
            logger.warning("RTC: inner policy rejects extra_sample_kwargs; "
                           "serving without guidance")
            result = self._policy.infer(obs)
            extra = None

        raw = result.pop("raw_actions", None)
        if raw is not None:
            self._prev_raw = np.asarray(raw)
        result["rtc/active"] = bool(extra is not None)
        return result

    @property
    def metadata(self) -> dict:
        meta = dict(getattr(self._policy, "metadata", {}) or {})
        meta["rtc_supported"] = bool(self._config.enabled)
        return meta
