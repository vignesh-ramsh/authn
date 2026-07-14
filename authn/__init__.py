"""authn — ARC provider plugin: session + API-key authentication.

Exports `arc.authn`: `resolve_identity(scope) -> Identity | None`, the
duck-typed contract Gateway's identity_middleware calls (docs/arc.MD §3.3,
plugins/gateway/gateway/middleware.py). The actual login/logout/session/
key management surface is whitelisted HTTP, not this capability's own
methods — see authn/api/auth_api.py.

Backed by four "system": true tables (self-declared, no soft-delete — a
hard DELETE instead, psqldb.soft_delete's own docstring — not patchable by
other plugins, a deliberate tradeoff, see authn/schemas/): _users, _roles,
_sessions, _access_keys.

redix is optional, not required (unlike an earlier draft of docs/arc.MD's
own §3.9 note, which this plugin's actual manifest overrides — see
plugin.toml): rate limiting genuinely upgrades when it's installed, and
falls back to a per-process in-memory counter otherwise (see
AuthnProvider.rate_limit). Account lockout (_users.failed_login_count/
locked_until) is DB-backed and the real backstop regardless.

Both the session cache AND the access-key cache (below) hold everything
resolve_identity needs (email/roles/status/allowed_ips) so a cache hit never
touches the DB — redix's absence just means every request re-reads
_sessions/_users (or _access_keys/_users) directly, which is already
correct, not a second fallback cache to maintain. Because the cache now
holds role/status/IP data, not just "is this token valid," it goes stale
the moment any of that changes on the user — see hooks/_users.py's
after_save hook, which is what keeps it honest.
"""

from __future__ import annotations

import asyncio
import hashlib
import ipaddress
import secrets
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal

import arc

CAPABILITY = "authn"

SESSION_TTL_HOURS_KEY = "authn_session_ttl_hours"
EXTENDED_SESSION_TTL_DAYS_KEY = "authn_extended_session_ttl_days"
LOCKOUT_THRESHOLD_KEY = "authn_lockout_threshold"
LOCKOUT_MINUTES_KEY = "authn_lockout_minutes"
MIN_PASSWORD_SCORE_KEY = "authn_min_password_score"

DEFAULT_SESSION_TTL_HOURS = 12
DEFAULT_EXTENDED_SESSION_TTL_DAYS = 30
DEFAULT_LOCKOUT_THRESHOLD = 5
DEFAULT_LOCKOUT_MINUTES = 15
DEFAULT_MIN_PASSWORD_SCORE = 3  # zxcvbn scores 0 (trivial) - 4 (very strong)

# A real row in _roles, assignable/removable via the CLI's add-role/
# remove-role like any other role name — the only thing special about it is
# what resolve_identity does when it sees it (below). Never valid inside an
# access key's scopes (auth_api.create_access_key rejects it explicitly,
# and resolve_identity's API-key path never even checks for it — see
# _resolve_api_key) — a leaked API key must never be able to carry this.
SUPERUSER_ROLE_NAME = "Superuser"

# Access keys look like "ak_<43 url-safe chars>" (secrets.token_urlsafe(32)).
# key_prefix is the first KEY_PREFIX_LEN characters — safe to display/log,
# used to look the row up before verifying the full key's hash.
KEY_PREFIX_LEN = 12

# An access key's redix cache entry is capped at this TTL even when the key
# itself never expires (expires_at is None) — bounds how long a missed
# invalidation (a bug, not the normal path — see hooks/_users.py) could stay
# stale; self-heals within this window regardless.
DEFAULT_ACCESS_KEY_CACHE_TTL_SECONDS = 3600

# How often one access key's last_used_at is allowed to actually write to
# the DB — every cache-hit-or-miss request "uses" the key, but writing on
# every single request doesn't scale (see auth_api's docstring / docs
# Review.MD P1) and last_used_at doesn't need per-request precision.
ACCESS_KEY_TOUCH_THROTTLE_SECONDS = 300


@dataclass(frozen=True)
class Identity:
    """What resolve_identity(scope) returns on success — the shape relay's
    RBAC check (relay._wire_gateway_route) reads via
    getattr(identity, "roles", None)."""

    user_id: str
    email: str
    roles: list[str]
    auth_method: Literal["session", "api_key"]


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def hash_token(raw: str) -> str:
    return hashlib.sha256(raw.encode()).hexdigest()


def _get_header(scope: dict, name: bytes) -> bytes | None:
    """authn deliberately doesn't import gateway.request.get_header here —
    resolve_identity's contract is scope-based specifically so authn stays
    decoupled from gateway's internals even though gateway is already only
    an optional_requires dependency at runtime."""
    name = name.lower()
    for k, v in scope.get("headers", []):
        if k.lower() == name:
            return v
    return None


def _ip_allowed(client_ip: str | None, allowed: list[str]) -> bool:
    if client_ip is None:
        return False
    try:
        addr = ipaddress.ip_address(client_ip)
    except ValueError:
        return client_ip in allowed
    for entry in allowed:
        try:
            if "/" in entry:
                if addr in ipaddress.ip_network(entry, strict=False):
                    return True
            elif addr == ipaddress.ip_address(entry):
                return True
        except ValueError:
            if client_ip == entry:
                return True
    return False


def has_roles_subset(candidate: list[str] | None, allowed: list[str] | None) -> bool:
    """Shared by hooks/_access_keys.py's validate hook and
    api/auth_api.py's create_access_key() — one subset-check, not two."""
    return set(candidate or []) <= set(allowed or [])


class PasswordPolicyError(ValueError):
    pass


def validate_password_strength(password: str, *, min_score: int, user_inputs: list[str] | None = None) -> None:
    """zxcvbn-scored (real crackability heuristics — dictionary words,
    keyboard walks, personal-info substitutions) rather than regex
    complexity rules ("1 uppercase + 1 digit + 1 symbol"), which push
    people toward predictable, technically-compliant-but-weak passwords
    like "P@ssw0rd1". `user_inputs` (email, name, ...) lets zxcvbn penalize
    a password that's just the user's own email with digits appended.

    Deliberately NOT a `_users` table hook: a hook only ever sees
    `ctx.payload["password_hash"]`, already one-way hashed by the time any
    write reaches it — there's no strength signal left to score at that
    point. Must be called directly by every raw-password call site instead
    (the CLI's create-user/set-password today)."""
    from zxcvbn import zxcvbn

    result = zxcvbn(password, user_inputs=user_inputs or [])
    if result["score"] < min_score:
        feedback = result.get("feedback") or {}
        warning = feedback.get("warning") or "password is too weak"
        suggestions = "; ".join(feedback.get("suggestions") or [])
        detail = warning + (f" ({suggestions})" if suggestions else "")
        raise PasswordPolicyError(f"password does not meet the minimum strength requirement: {detail}")


class AuthnProvider:
    def __init__(self, kernel: Any, *, redix: Any | None = None) -> None:
        self._kernel = kernel
        self._redix = redix
        # Rate-limit fallback, used only when redix is absent — per-process,
        # not distributed, lost on restart (register()'s kernel.advise()
        # below says so at boot). Account lockout is the real backstop
        # regardless of this — see api/auth_api.py's login().
        self._rate_local: dict[str, tuple[int, float]] = {}
        self._rate_lock = asyncio.Lock()
        # Same per-process fallback shape, for the last_used_at write
        # throttle below, when redix is absent.
        self._touch_local: dict[str, float] = {}
        # Strong references to fire-and-forget background tasks (the
        # last_used_at write) — asyncio only holds a WEAK reference to a
        # task via the event loop, so a task with nothing else referencing
        # it can be garbage-collected mid-flight and silently cancelled.
        self._background_tasks: set[asyncio.Task] = set()

    def _spawn(self, coro) -> None:
        task = asyncio.create_task(coro)
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    # ------------------------------------------------------------------ #
    # Settings
    # ------------------------------------------------------------------ #
    def _setting_int(self, key: str, default: int) -> int:
        raw = self._kernel.settings.get(key)
        return int(raw) if raw else default

    def session_ttl_seconds(self, session_type: str) -> int:
        if session_type == "Extended":
            return self._setting_int(EXTENDED_SESSION_TTL_DAYS_KEY, DEFAULT_EXTENDED_SESSION_TTL_DAYS) * 86400
        return self._setting_int(SESSION_TTL_HOURS_KEY, DEFAULT_SESSION_TTL_HOURS) * 3600

    def lockout_threshold(self) -> int:
        return self._setting_int(LOCKOUT_THRESHOLD_KEY, DEFAULT_LOCKOUT_THRESHOLD)

    def lockout_seconds(self) -> int:
        return self._setting_int(LOCKOUT_MINUTES_KEY, DEFAULT_LOCKOUT_MINUTES) * 60

    def min_password_score(self) -> int:
        return self._setting_int(MIN_PASSWORD_SCORE_KEY, DEFAULT_MIN_PASSWORD_SCORE)

    # ------------------------------------------------------------------ #
    # Rate limiting — redix-backed when present, in-process fallback
    # otherwise. Same shape either way, weaker guarantee without redix,
    # same honesty as relay's own redix-optional cache/lock (docs/arc.MD
    # §3.11).
    # ------------------------------------------------------------------ #
    async def rate_limit(self, key: str, limit: int, window_seconds: int) -> bool:
        if self._redix is not None:
            return await self._redix.rate_limit(key, limit, window_seconds)
        async with self._rate_lock:
            now = time.monotonic()
            count, window_start = self._rate_local.get(key, (0, now))
            if now - window_start >= window_seconds:
                count, window_start = 0, now
            count += 1
            self._rate_local[key] = (count, window_start)
            return count <= limit

    # ------------------------------------------------------------------ #
    # Session cache — holds everything resolve_identity needs (email/
    # roles/status/allowed_ips), so a hit never touches the DB at all.
    # _sessions/_users are always the source of truth on a miss; redix's
    # absence just means every request re-reads them directly.
    # ------------------------------------------------------------------ #
    async def _cache_get_session(self, token_hash: str) -> dict | None:
        if self._redix is None:
            return None
        raw = await self._redix.get(f"session:{token_hash}")
        return arc.codec.decode(raw) if raw else None

    async def _cache_set_session(
        self, token_hash: str, *, user_id: str, email: str, roles: list[str],
        status: str, allowed_ips: list[str] | None, ttl_seconds: int,
    ) -> None:
        if self._redix is None or ttl_seconds <= 0:
            return
        await self._redix.set(
            f"session:{token_hash}",
            arc.codec.encode({
                "user_id": user_id, "email": email, "roles": roles,
                "status": status, "allowed_ips": allowed_ips,
            }),
            ex=ttl_seconds,
        )

    async def invalidate_session_cache(self, token_hash: str) -> None:
        if self._redix is not None:
            await self._redix.delete(f"session:{token_hash}")

    # ------------------------------------------------------------------ #
    # Access-key cache — same shape/purpose as the session cache above,
    # keyed by key_prefix (parsed instantly from the raw key, no crypto
    # needed to look it up) rather than a hash of the whole secret. The
    # full secret's hash is still verified locally on every hit before the
    # cached identity is ever trusted.
    # ------------------------------------------------------------------ #
    async def _cache_get_access_key(self, prefix: str) -> dict | None:
        if self._redix is None:
            return None
        raw = await self._redix.get(f"access_key:{prefix}")
        return arc.codec.decode(raw) if raw else None

    async def _cache_set_access_key(
        self, prefix: str, *, access_key_id: str, key_hash: str, user_id: str,
        email: str, roles: list[str], status: str, allowed_ips: list[str] | None,
        expires_at: datetime | None,
    ) -> None:
        if self._redix is None:
            return
        ttl_seconds = DEFAULT_ACCESS_KEY_CACHE_TTL_SECONDS
        if expires_at is not None:
            ttl_seconds = max(0, min(int((expires_at - utcnow()).total_seconds()), ttl_seconds))
        if ttl_seconds <= 0:
            return
        await self._redix.set(
            f"access_key:{prefix}",
            arc.codec.encode({
                "id": access_key_id, "key_hash": key_hash, "user_id": user_id, "email": email,
                "roles": roles, "status": status, "allowed_ips": allowed_ips,
            }),
            ex=ttl_seconds,
        )

    async def invalidate_access_key_cache(self, prefix: str) -> None:
        if self._redix is not None:
            await self._redix.delete(f"access_key:{prefix}")

    # ------------------------------------------------------------------ #
    # Per-user cache invalidation — called from hooks/_users.py's
    # after_save hook when has_roles/status/allowed_ips actually changes.
    # The cache is keyed by token_hash / key_prefix, not by user, so this
    # has to look up the user's active sessions/keys to know which entries
    # to clear — an extra read, but only on a rare event (a role/status/
    # IP-list change), never on the hot request path.
    # ------------------------------------------------------------------ #
    async def invalidate_user_cache(self, user_id: Any) -> None:
        sessions = await arc.relay.list("_sessions", filters={"user": user_id}, fields=["token_hash"])
        for s in sessions:
            await self.invalidate_session_cache(s["token_hash"])
        keys = await arc.relay.list("_access_keys", filters={"user": user_id}, fields=["key_prefix"])
        for k in keys:
            await self.invalidate_access_key_cache(k["key_prefix"])

    # ------------------------------------------------------------------ #
    # last_used_at — throttled (see ACCESS_KEY_TOUCH_THROTTLE_SECONDS) and,
    # when it does write, a raw lightweight UPDATE (arc.relay.sql, not
    # arc.relay.save) so it skips Relay's hook/transaction machinery. Still
    # goes through the DB-level arc_audit_authn trigger either way — that's
    # attached to the table itself, not to any particular Python call path.
    # ------------------------------------------------------------------ #
    async def _should_touch_last_used(self, access_key_id: str) -> bool:
        throttle_key = f"access_key_touch:{access_key_id}"
        if self._redix is not None:
            if await self._redix.get(throttle_key):
                return False
            await self._redix.set(throttle_key, "1", ex=ACCESS_KEY_TOUCH_THROTTLE_SECONDS)
            return True
        now = time.monotonic()
        last = self._touch_local.get(access_key_id, 0.0)
        if now - last < ACCESS_KEY_TOUCH_THROTTLE_SECONDS:
            return False
        self._touch_local[access_key_id] = now
        return True

    async def _touch_last_used(self, access_key_id: str) -> None:
        # Fire-and-forget from resolve_identity's perspective — a write
        # failure here must never break authentication itself.
        try:
            await arc.relay.sql('UPDATE "_access_keys" SET last_used_at = $1 WHERE id = $2', utcnow(), access_key_id)
        except Exception as exc:
            arc.relay.log(f"authn: failed to update access key last_used_at: {exc}", level="warning")

    # ------------------------------------------------------------------ #
    # Identity resolution — the contract gateway's identity_middleware
    # calls (docs/arc.MD §3.3): async def resolve_identity(scope) -> Any | None.
    # Returns None (never raises) on any failure to authenticate — that's
    # the documented contract, identity_middleware doesn't wrap this call.
    # ------------------------------------------------------------------ #
    async def resolve_identity(self, scope: dict) -> Identity | None:
        auth_header = _get_header(scope, b"authorization")
        api_key_header = _get_header(scope, b"x-api-key")

        if auth_header is not None:
            return await self._resolve_session(auth_header, scope)
        if api_key_header is not None:
            return await self._resolve_api_key(api_key_header, scope)
        return None

    def _authorize(self, allowed_ips: list[str] | None, scope: dict) -> bool:
        if not allowed_ips:
            return True
        client_ip = scope.get("state", {}).get("arc_client_ip")
        return _ip_allowed(client_ip, allowed_ips)

    async def _resolve_session(self, auth_header: bytes, scope: dict) -> Identity | None:
        value = auth_header.decode("latin-1")
        if not value.startswith("Bearer "):
            return None
        token = value[len("Bearer "):].strip()
        if not token:
            return None
        token_hash = hash_token(token)

        cached = await self._cache_get_session(token_hash)
        if cached is not None:
            if cached["status"] != "Active" or not self._authorize(cached.get("allowed_ips"), scope):
                return None
            return Identity(user_id=cached["user_id"], email=cached["email"], roles=cached["roles"], auth_method="session")

        session = await arc.relay.get("_sessions", {"token_hash": token_hash})
        if session is None or session["revoked_at"] is not None or session["expires_at"] <= utcnow():
            return None

        user = await arc.relay.get("_users", session["user"])
        if user is None:
            return None

        roles = list(user.get("has_roles") or [])
        if SUPERUSER_ROLE_NAME in roles:
            # Injected into the resolved (and cached) roles list so relay's
            # RBAC check (the "*" in caller_roles branch, _wire_gateway_route)
            # treats this identity as satisfying any role requirement.
            # Session-only, deliberately — _resolve_api_key never does this,
            # see SUPERUSER_ROLE_NAME's docstring above.
            roles = [*roles, "*"]
        user_id = str(user["id"])
        ttl_remaining = int((session["expires_at"] - utcnow()).total_seconds())
        await self._cache_set_session(
            token_hash, user_id=user_id, email=user["email"], roles=roles,
            status=user["status"], allowed_ips=user.get("allowed_ips"), ttl_seconds=ttl_remaining,
        )

        if user["status"] != "Active" or not self._authorize(user.get("allowed_ips"), scope):
            return None
        return Identity(user_id=user_id, email=user["email"], roles=roles, auth_method="session")

    async def _resolve_api_key(self, api_key_header: bytes, scope: dict) -> Identity | None:
        raw = api_key_header.decode("latin-1").strip()
        if len(raw) <= KEY_PREFIX_LEN:
            return None
        prefix = raw[:KEY_PREFIX_LEN]

        cached = await self._cache_get_access_key(prefix)
        if cached is not None:
            if hash_token(raw) != cached["key_hash"]:
                # Prefix matched a cache entry but the secret doesn't — a
                # definitively invalid key, no need to fall through to the DB.
                return None
            if await self._should_touch_last_used(cached["id"]):
                self._spawn(self._touch_last_used(cached["id"]))
            if cached["status"] != "Active" or not self._authorize(cached.get("allowed_ips"), scope):
                return None
            return Identity(user_id=cached["user_id"], email=cached["email"], roles=cached["roles"], auth_method="api_key")

        access_key = await arc.relay.get("_access_keys", {"key_prefix": prefix})
        if access_key is None or access_key["revoked_at"] is not None:
            return None
        if access_key["expires_at"] is not None and access_key["expires_at"] <= utcnow():
            return None
        if hash_token(raw) != access_key["key_hash"]:
            return None

        user = await arc.relay.get("_users", access_key["user"])
        if user is None:
            return None

        # Live intersection, not the scopes stored at key-creation time — a
        # role revoked from the user since shouldn't still be grantable via
        # an old key's stored scopes.
        roles = [r for r in (access_key.get("scopes") or []) if r in (user.get("has_roles") or [])]
        access_key_id = str(access_key["id"])

        await self._cache_set_access_key(
            prefix, access_key_id=access_key_id, key_hash=access_key["key_hash"], user_id=str(user["id"]),
            email=user["email"], roles=roles, status=user["status"], allowed_ips=user.get("allowed_ips"),
            expires_at=access_key["expires_at"],
        )

        if await self._should_touch_last_used(access_key_id):
            self._spawn(self._touch_last_used(access_key_id))

        if user["status"] != "Active" or not self._authorize(user.get("allowed_ips"), scope):
            return None
        return Identity(user_id=str(user["id"]), email=user["email"], roles=roles, auth_method="api_key")


def register(kernel: Any) -> None:
    kernel.settings.declare(SESSION_TTL_HOURS_KEY)
    kernel.settings.declare(EXTENDED_SESSION_TTL_DAYS_KEY)
    kernel.settings.declare(LOCKOUT_THRESHOLD_KEY)
    kernel.settings.declare(LOCKOUT_MINUTES_KEY)
    kernel.settings.declare(MIN_PASSWORD_SCORE_KEY)

    psqldb = kernel.get("psqldb")
    psqldb.register_model(Path(__file__).parent / "schemas")

    redix = kernel.get("redix") if kernel.has("redix") else None
    if redix is None:
        kernel.advise(
            "authn: redix not installed — brute-force rate limiting falls back to a "
            "per-process, in-memory counter (not distributed across workers, reset on "
            "restart). Account lockout (failed_login_count/locked_until) remains the "
            "real backstop regardless. Install redix for real distributed rate limiting."
        )

    provider = AuthnProvider(kernel, redix=redix)
    # Export BEFORE registering hooks/api: auth_api.py's own whitelisted
    # functions declare roles=["*"] (a real role), and relay.whitelist()'s
    # boot-time guard hard-requires kernel.has("authn") to already be true
    # for that (§3.3's rule) — a self-reference that only resolves if authn
    # exports itself first, before its own API file gets loaded.
    kernel.export(CAPABILITY, provider, requires=["psqldb", "relay"], optional_requires=["redix", "gateway"])

    relay = kernel.get("relay")
    relay.register_hooks(Path(__file__).parent / "hooks")
    relay.register_api(Path(__file__).parent / "api")
