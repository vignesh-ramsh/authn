"""
authn.hashing
-----------------
Argon2id password hashing for the whole server: how expensive a hash is (one JSON setting),
and how many may run at once (so a login flood cannot run the machine out of memory).

Why it exists: an Argon2 hash costs memory on purpose - 64 MiB by default - and nothing used to
limit how many ran at the same time. The 16 threads of Python's default pool each took 64 MiB:
48 concurrent wrong-password logins grew the web workers by 1.9 GB, and every other request
slowed down behind them (p99 250 ms for a bystander), whichever IPs the attempts came from.

The setting `authn_argon2` is one JSON object; any key left out keeps its default:

    {"memory_mb": 64, "time_cost": 3, "parallelism": 1, "max_concurrent": 2, "max_waiting": 16}

    memory_mb       memory one hash uses, in MiB (19 to 4096).
    time_cost       passes over that memory (1 to 20).
    parallelism     Argon2's own lanes = threads one hash uses (1 to 16). It spreads the same work over
                    that many threads; it is the memory and the passes, not this, that make a password
                    costly to crack (OWASP's Argon2id profiles all use 1).
    max_concurrent  hashes one worker process runs at once (1 to 64). Peak memory per process is
                    memory_mb x max_concurrent.
    max_waiting     how many more may wait for a slot (0 to 10000). A waiting login costs an open
                    connection and nothing else, so this is what absorbs a burst - an office all
                    signing in at nine; the wait is about max_waiting / max_concurrent hashes.
                    Anything beyond is refused at once with 503 + Retry-After rather than queued
                    without bound.

What a flood of logins can take: web workers x max_concurrent x parallelism threads, at once. Measured on a
12-core machine, one hash (64 MiB, 3 passes) takes 66 ms on 4 threads and 163 ms on 1; four at once cost 326
CPU-ms each on 4 threads and 214 on 1. The default used to be 4 threads: 2 web workers x 2 hashes x 4 = 16
threads on 12 cores, and a load test of unknown-account logins took 9-10 of them. At 1 it is 4 threads, a
third of that machine, for the same strength; a single login takes about 100 ms longer, and the most logins a
second the server can do is web workers x max_concurrent / (the 160-230 ms one takes).

On a machine short of memory, lower it - OWASP's smallest profile is
{"memory_mb": 19, "time_cost": 2, "parallelism": 1}. Only NEW hashes use the new numbers: a
stored hash carries its own, so one made at 64 MiB is still verified at 64 MiB, then replaced by
one made with the current settings the next time its owner logs in (which is what happens to every
password stored before the default became parallelism 1: it verifies as before, and that login stores a
new one).
"""

from __future__ import annotations

import asyncio
import json
import secrets
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable

from argon2 import PasswordHasher
from argon2.exceptions import VerificationError, VerifyMismatchError

HASHING_KEY = "authn_argon2"

#: argon2-cffi's memory and passes (RFC 9106's low-memory profile) on ONE thread, plus 2 concurrent hashes per
#: process. Hashes made before this setting existed, or with parallelism 4, are replaced at their owner's next
#: login. See the module docstring for why one thread.
DEFAULTS: dict[str, int] = {
    "memory_mb": 64, "time_cost": 3, "parallelism": 1, "max_concurrent": 2, "max_waiting": 16,
}
DEFAULT_JSON = json.dumps(DEFAULTS)

#: key -> (lowest, highest). memory_mb has OWASP's minimum (19 MiB) as its floor: Argon2 itself would
#: accept far less, and a typo should not be able to make every password cheap to crack.
LIMITS: dict[str, tuple[int, int]] = {
    "memory_mb": (19, 4096),
    "time_cost": (1, 20),
    "parallelism": (1, 16),
    "max_concurrent": (1, 64),
    "max_waiting": (0, 10_000),
}


@dataclass(frozen=True)
class HashingConfig:
    memory_mb: int = 64
    time_cost: int = 3
    parallelism: int = 1
    max_concurrent: int = 2
    max_waiting: int = 16

    @property
    def memory_kib(self) -> int:
        return self.memory_mb * 1024

    @property
    def weaker_than_owasp_minimum(self) -> bool:
        """OWASP's two smallest profiles are 46 MiB with t=1 and 19 MiB with t=2 (both p=1):
        less memory has to be paid for with more passes."""
        return self.memory_mb < 46 and self.time_cost < 2


def parse_config(text: str | None) -> HashingConfig:
    """The `authn_argon2` setting as a HashingConfig. Empty means all defaults. Raises ValueError
    naming the setting and the problem - a typo'd key, a value out of range, text that is not JSON -
    because this runs at boot: failing there beats failing at the first login."""
    if text is None or not text.strip():
        return HashingConfig()
    example = '`arc settings set authn_argon2 \'{"memory_mb": 19, "time_cost": 2, "parallelism": 1}\'`'
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{HASHING_KEY} is not valid JSON ({exc.msg} at position {exc.pos}); expected an object like {example}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"{HASHING_KEY} must be a JSON object like {example}, not {type(raw).__name__}")
    unknown = sorted(set(raw) - set(DEFAULTS))
    if unknown:
        raise ValueError(f"{HASHING_KEY} has unknown key(s) {', '.join(map(repr, unknown))}; the keys are {', '.join(DEFAULTS)}")
    values = dict(DEFAULTS)
    for key, value in raw.items():
        low, high = LIMITS[key]
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{HASHING_KEY}: {key} must be a whole number between {low} and {high}, not {value!r}")
        if not low <= value <= high:
            raise ValueError(f"{HASHING_KEY}: {key} is {value}, but must be between {low} and {high}")
        values[key] = value
    return HashingConfig(**values)


class HashingBusy(Exception):
    """Too many password hashes are already running or waiting in this process. Capacity, not a
    bug, so the gateway answers 503 with Retry-After (it reads `http_status` and `retry_after`
    off the exception and knows nothing about hashing)."""

    http_status = 503
    retry_after = 2


class PasswordHashing:
    """Hash and verify passwords with the configured Argon2 parameters, at most `max_concurrent`
    at a time in this process, on threads of their own."""

    def __init__(self, config: HashingConfig) -> None:
        self.config = config
        self._hasher = PasswordHasher(
            time_cost=config.time_cost,
            memory_cost=config.memory_kib,
            parallelism=config.parallelism,
        )
        # Their own threads: hashing must not use (and be starved by, or starve) the default
        # pool other code shares. Its size IS the bound on hashes running at once: work beyond
        # that waits in the pool's queue, and `_admitted` below caps how long that queue may get.
        self._executor = ThreadPoolExecutor(
            max_workers=config.max_concurrent, thread_name_prefix="authn-argon2"
        )
        self._max_admitted = config.max_concurrent + config.max_waiting
        self._admitted = 0  # running + waiting
        self._dummy: str | None = None

    # ---- the gate ------------------------------------------------------- #
    async def _run(self, fn: Callable[..., Any], *args: Any) -> Any:
        if self._admitted >= self._max_admitted:
            raise HashingBusy(
                f"{self._admitted} password hashes are already running or waiting "
                f"(max_concurrent {self.config.max_concurrent}, max_waiting {self.config.max_waiting}); "
                f"retry shortly"
            )
        self._admitted += 1
        try:
            return await asyncio.get_running_loop().run_in_executor(self._executor, fn, *args)
        finally:
            self._admitted -= 1

    # ---- what callers use ------------------------------------------------ #
    async def hash(self, password: str) -> str:
        return await self._run(self._hasher.hash, password)

    async def verify(self, stored: str | None, password: str) -> bool:
        """Does `password` match `stored`? `stored=None` (no such user) still pays for one
        verification, against a throwaway hash made with the current parameters, so a login for an
        unknown account takes as long as one for a wrong password; it is always False."""
        if stored is None:
            await self._run(self._verify, await self._dummy_hash(), password)
            return False
        return await self._run(self._verify, stored, password)

    def needs_rehash(self, stored: str) -> bool:
        """True if `stored` was made with other parameters than the current settings (or is not a
        hash at all): worth replacing while the password is at hand, at the next login."""
        try:
            return self._hasher.check_needs_rehash(stored)
        except ValueError:  # argon2's InvalidHashError: not a hash we can read, so not one to replace
            return False

    async def _dummy_hash(self) -> str:
        if self._dummy is None:
            self._dummy = await self.hash(secrets.token_urlsafe(16))
        return self._dummy

    def _verify(self, stored: str, password: str) -> bool:
        try:
            return self._hasher.verify(stored, password)
        except (VerifyMismatchError, VerificationError):
            return False
