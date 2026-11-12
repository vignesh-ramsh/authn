"""
authn.throttle
-----------------
What a source address pays for FAILED logins - not for logging in.

Two things, both applied to every failed attempt in exactly the same way, whether the account
exists or not (see below for why that matters):

* a budget: after `max_failures` failed logins from one address within `window_seconds`, further
  attempts from it are refused with 429 until the window passes, before any password hashing is
  spent on them. Successful logins do not count, so 300 employees behind one office address, who
  almost all succeed, never use it up - while an attacker trying passwords or spraying usernames
  fails nearly every time and does.
* a pause of `failure_delay_ms` before a failure is reported. It costs the server an open
  connection and nothing else (the hashing slot is long released), and it slows guessing from
  every source.

The setting `authn_login_throttle` is one JSON object; a key left out keeps its default:

    {"max_failures": 60, "window_seconds": 60, "failure_delay_ms": 250}

    max_failures      failed logins per address per window before it is refused (0 turns the budget off)
    window_seconds    the length of that window
    failure_delay_ms  pause before a failure is reported (0 turns it off)

Networks listed in `gateway_rate_limit_exempt` are not held to the budget (the delay still applies):
one compromised machine in an office would otherwise use up the budget of the whole office behind its
address. They are still protected by account lockout and by the bound on concurrent hashing.

Why failures of unknown and known accounts are treated alike: if an unknown username cost more, or
less, than a known one with a wrong password - more memory, more time, a different count - the
difference would tell an attacker which accounts exist, which is the one thing the dummy hash in
PasswordHashing.verify() is there to hide.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from typing import Any

from arc.expiring import Pruner

THROTTLE_KEY = "authn_login_throttle"

DEFAULTS: dict[str, int] = {"max_failures": 60, "window_seconds": 60, "failure_delay_ms": 250}
DEFAULT_JSON = json.dumps(DEFAULTS)
#: key -> (lowest, highest)
LIMITS: dict[str, tuple[int, int]] = {
    "max_failures": (0, 1_000_000),
    "window_seconds": (1, 86_400),
    "failure_delay_ms": (0, 10_000),
}
_FALLBACK_PRUNE_THRESHOLD = 4096

_logger = logging.getLogger("authn")


@dataclass(frozen=True)
class ThrottleConfig:
    max_failures: int = 60
    window_seconds: int = 60
    failure_delay_ms: int = 250


def parse_config(text: str | None) -> ThrottleConfig:
    """The `authn_login_throttle` setting as a ThrottleConfig. Empty means all defaults. Raises
    ValueError naming the setting and the problem (runs at boot: failing there beats failing at the
    first login)."""
    if text is None or not text.strip():
        return ThrottleConfig()
    example = f"`arc settings set {THROTTLE_KEY} '{DEFAULT_JSON}'`"
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{THROTTLE_KEY} is not valid JSON ({exc.msg} at position {exc.pos}); expected an object like {example}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"{THROTTLE_KEY} must be a JSON object like {example}, not {type(raw).__name__}")
    unknown = sorted(set(raw) - set(DEFAULTS))
    if unknown:
        raise ValueError(f"{THROTTLE_KEY} has unknown key(s) {', '.join(map(repr, unknown))}; the keys are {', '.join(DEFAULTS)}")
    values = dict(DEFAULTS)
    for key, value in raw.items():
        low, high = LIMITS[key]
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{THROTTLE_KEY}: {key} must be a whole number between {low} and {high}, not {value!r}")
        if not low <= value <= high:
            raise ValueError(f"{THROTTLE_KEY}: {key} is {value}, but must be between {low} and {high}")
        values[key] = value
    return ThrottleConfig(**values)


class LoginThrottle:
    """The failed-login budget and pause. Counts in redix when installed (shared by every web
    worker and host), in this process otherwise or while Redis is unreachable - the same
    degradation authn's other rate limits make: Redis having a bad moment must not take login down."""

    def __init__(self, config: ThrottleConfig, provider: Any) -> None:
        self.config = config
        self._provider = provider  # AuthnProvider: rate_limit(), its redix client, is_trusted_network()
        self._blocked_local: dict[str, float] = {}  # address -> monotonic time the block ends
        self._blocked_pruner = Pruner(threshold=_FALLBACK_PRUNE_THRESHOLD)

    def _applies_to(self, client_ip: str | None) -> bool:
        return (
            self.config.max_failures > 0
            and bool(client_ip)
            and not self._provider.is_trusted_network(client_ip)
        )

    async def is_blocked(self, client_ip: str | None) -> bool:
        """Has this address used up its failures for the current window? Does not count as an attempt."""
        if not self._applies_to(client_ip):
            return False
        redix = self._provider._redix
        if redix is not None:
            try:
                return await redix.get(f"login-blocked:{client_ip}") is not None
            except Exception as exc:  # noqa: BLE001 - Redis failing must not break login
                _logger.warning("login throttle: redix read failed (%s) - using the in-process record", exc)
        return self._blocked_local.get(client_ip, 0.0) > time.monotonic()

    async def record_failure(self, client_ip: str | None) -> None:
        """One failed login from this address. The failure that reaches `max_failures` blocks the
        address for the rest of the window."""
        if not self._applies_to(client_ip):
            return
        # rate_limit() counts one and says whether the count is still <= limit; with limit
        # max_failures - 1 it turns False on the max_failures-th failure.
        within = await self._provider.rate_limit(
            f"login-fail:{client_ip}", limit=self.config.max_failures - 1, window_seconds=self.config.window_seconds
        )
        if not within:
            await self._block(client_ip)

    async def _block(self, client_ip: str) -> None:
        window = self.config.window_seconds
        redix = self._provider._redix
        if redix is not None:
            try:
                await redix.set(f"login-blocked:{client_ip}", "1", ex=window)
                return
            except Exception as exc:  # noqa: BLE001
                _logger.warning("login throttle: redix write failed (%s) - blocking in-process only", exc)
        now = time.monotonic()
        # keys are attacker-chosen addresses: stale blocks leave on a timer, memory stays bounded
        self._blocked_pruner.prune(self._blocked_local, lambda until: until <= now, now)
        self._blocked_local[client_ip] = now + window

    async def pause(self) -> None:
        """The wait before a failure is reported. The same for every failure."""
        if self.config.failure_delay_ms > 0:
            await asyncio.sleep(self.config.failure_delay_ms / 1000)
