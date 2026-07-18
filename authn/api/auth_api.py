"""authn's HTTP-reachable surface — explicit paths (relay.whitelist's
path= override, docs/arc.MD §3.11) rather than the auto-derived
/api/method/... convention, per the proposal this plugin was built from.

Known limitation, not an oversight: whitelisted functions receive only the
JSON body as **kwargs (plus `identity`/`client_ip`, when the function's own
signature asks for them — see relay.WhitelistedFunction.wants_identity/
wants_client_ip) — never the raw Request. So login() has no way to capture
the caller's real User-Agent for _sessions.user_agent today; that column
stays nullable and unset here. Query-string-to-kwargs mapping has the same
shape of gap for GET (docs/arc.MD §3.11) — this is the same class of "not
built yet", not a new one.
"""

import asyncio
import secrets
from datetime import timedelta

import arc

from authn import (
    KEY_PREFIX_LEN,
    SUPERUSER_ROLE_NAME,
    PasswordPolicyError,
    has_roles_subset,
    hash_token,
    utcnow,
    validate_password_strength,
)

_DUMMY_PASSWORD_HASH: str | None = None


def _dummy_hash() -> str:
    """A fixed Argon2id hash of a random, never-used password, computed
    once lazily on first use — verified against on every login attempt for
    an email that doesn't exist, so "no such user" costs the same wall-clock
    time as "wrong password" and can't be distinguished by response timing
    (docs/Review.MD S3)."""
    global _DUMMY_PASSWORD_HASH
    if _DUMMY_PASSWORD_HASH is None:
        from argon2 import PasswordHasher

        _DUMMY_PASSWORD_HASH = PasswordHasher().hash(secrets.token_urlsafe(16))
    return _DUMMY_PASSWORD_HASH


def _require_identity(identity):
    if identity is None:
        arc.relay.throw("authentication required", status=401, code="unauthorized")
    return identity


@arc.relay.whitelist(methods=["POST"], roles=["Guest"], path="/login")
async def login(
    email: str | None = None,
    username: str | None = None,
    *,
    password: str,
    session_type: str = "Fixed",
    client_ip: str | None = None,
) -> dict:
    from argon2 import PasswordHasher
    from argon2.exceptions import VerificationError, VerifyMismatchError

    if session_type not in ("Fixed", "Extended"):
        arc.relay.throw("session_type must be 'Fixed' or 'Extended'", code="bad_session_type")
    if not email and not username:
        arc.relay.throw("email or username is required", code="identifier_required")

    # Two lookup fields, one login path — a real second capability (not
    # every "add a field" also means "and let it authenticate with"), added
    # because it was explicitly asked for. email stays the primary/default
    # identifier; username is a genuine alternative, not a fallback alias.
    if email:
        email = email.strip().lower()
        lookup, identifier = {"email": email}, email
    else:
        # Same normalization the _users validate hook applies on write
        # (hooks/_users.py's check_username) — a case-mismatched lookup
        # would otherwise never match the stored (lowercased) value.
        username = username.strip().lower()
        lookup, identifier = {"username": username}, username

    # Keyed on (client_ip, identifier), not identifier alone — a self-hosted
    # deployment where every legitimate user shares one public IP would
    # otherwise make the per-identifier budget the same as a single-attacker
    # budget. The account lockout below is unchanged and stays the real
    # defense against a targeted guesser; this only bounds request *volume*
    # per source.
    rate_key = f"login:{client_ip or 'unknown'}:{identifier}"
    if not await arc.authn.rate_limit(rate_key, limit=10, window_seconds=60):
        arc.relay.throw("too many login attempts, try again shortly", status=429, code="rate_limited")

    user = await arc.relay.get("_users", lookup)

    # Always pay exactly one Argon2 verify — against the real hash if the
    # user exists, a fixed dummy hash otherwise — so response timing can't
    # be used to enumerate valid emails (docs/Review.MD S3). Run in a
    # worker thread: Argon2id is ~100ms of deliberate CPU+memory cost, and
    # doing it on the event loop stalled EVERY in-flight request for the
    # duration of every login attempt (including the dummy-hash ones an
    # attacker can generate freely).
    try:
        await asyncio.to_thread(
            PasswordHasher().verify,
            user["password_hash"] if user is not None else _dummy_hash(),
            password,
        )
        password_ok = True
    except (VerifyMismatchError, VerificationError):
        password_ok = False

    if user is None:
        arc.relay.throw("invalid credentials", status=401, code="invalid_credentials")

    # A lock that has already expired clears the strike counter before this
    # attempt is evaluated — otherwise a single wrong password right after
    # expiry immediately re-locks the account (1-strike-relock, docs
    # Review.MD L2), and failed_login_count grows without bound across
    # repeated lockouts instead of resetting to a fresh N-attempt budget.
    if user["locked_until"] is not None and user["locked_until"] <= utcnow():
        await arc.relay.save("_users", {"id": user["id"], "failed_login_count": 0, "locked_until": None})
        user = {**user, "failed_login_count": 0, "locked_until": None}

    locked = user["locked_until"] is not None and user["locked_until"] > utcnow()
    inactive = user["status"] != "Active"

    if locked or inactive or not password_ok:
        # Only a genuine wrong password is new evidence of a guessing
        # attack — an already-locked account being hit again isn't (the
        # rate limiter above already bounds attempt volume), so this
        # doesn't pile strikes on top of an existing lock.
        if not password_ok and not locked:
            new_count = (user["failed_login_count"] or 0) + 1
            update = {"id": user["id"], "failed_login_count": new_count}
            if new_count >= arc.authn.lockout_threshold():
                update["locked_until"] = utcnow() + timedelta(seconds=arc.authn.lockout_seconds())
            await arc.relay.save("_users", update)
        # One generic message regardless of WHY this failed (unknown email,
        # wrong password, locked, inactive) — a distinct message per case
        # is itself an account-enumeration/state-disclosure oracle (docs
        # Review.MD S3). A genuinely locked/inactive user sees the same
        # message as a wrong password, by design.
        arc.relay.throw("invalid credentials", status=401, code="invalid_credentials")

    await arc.relay.save(
        "_users", {"id": user["id"], "failed_login_count": 0, "locked_until": None, "last_login_at": utcnow()}
    )

    # Count-then-insert is a race without this: two concurrent logins for
    # the same user could both see "under the cap" and both insert,
    # exceeding max_sessions (docs/Review.MD L4). Same lock() primitive
    # save()'s own match_on upsert already uses — real guarantee with
    # redix installed, a weaker in-process-only one otherwise, same
    # documented tradeoff as every other arc.relay.lock() use.
    async with arc.relay.lock(f"login:sessions:{user['id']}"):
        active_sessions = await arc.relay.count(
            "_sessions",
            filters={"user": user["id"], "revoked_at": {"is_null": True}, "expires_at": {"gt": utcnow()}},
        )
        if user["max_sessions"] is not None and active_sessions >= user["max_sessions"]:
            arc.relay.throw(
                "maximum active sessions reached — log out an existing session first",
                status=403, code="max_sessions_reached",
            )

        token = secrets.token_urlsafe(32)
        ttl = arc.authn.session_ttl_seconds(session_type)
        expires_at = utcnow() + timedelta(seconds=ttl)
        await arc.relay.save(
            "_sessions",
            {"user": user["id"], "token_hash": hash_token(token), "session_type": session_type, "expires_at": expires_at},
        )
    return {"token": token, "session_type": session_type, "expires_at": expires_at.isoformat()}


@arc.relay.whitelist(methods=["POST"], roles=["Guest"], path="/logout")
async def logout(token: str) -> dict:
    token_hash = hash_token(token)
    session = await arc.relay.get("_sessions", {"token_hash": token_hash})
    if session is not None and session["revoked_at"] is None:
        await arc.relay.save("_sessions", {"id": session["id"], "revoked_at": utcnow()})
        await arc.authn.invalidate_session_cache(token_hash)
    return {"ok": True}


_MAIL_NOT_CONFIGURED = (
    "password reset via email is not configured on this instance — contact your administrator"
)


@arc.relay.whitelist(methods=["POST"], roles=["Guest"], path="/forgot-password")
async def forgot_password(email: str, client_ip: str | None = None) -> dict:
    """Requires the optional `mail` plugin AND an explicitly-set
    `authn_public_url` setting (used to build the emailed reset link) — with
    neither, this throws a clean 501 the same shape as this endpoint's old
    stub, rather than emailing a broken link. `arc authn set-password`
    remains the real interim recovery path either way.

    Always returns the same generic response regardless of whether `email`
    actually has an account — same "one message regardless of why" posture
    login() already takes (docs/Review.MD S3). Deliberately NOT extended to
    exact timing-parity for a nonexistent email the way login()'s
    `_dummy_hash()` is: skipping the token/DB-write/email-send for a
    nonexistent user is a real, measurable timing difference, but "does
    this email have an account" is a much lower-severity oracle than
    login's own credential-guessing surface, and closing it fully would
    mean spending a real email send on every guess. A deliberate, flagged
    simplification, not an oversight."""
    email = email.strip().lower()

    # Keyed the same (client_ip, identifier) shape as login()'s own rate
    # limit, and for the same reason — bounds request volume per source
    # without collapsing every legitimate user behind one shared IP into a
    # single attacker's budget.
    rate_key = f"forgot-password:{client_ip or 'unknown'}:{email}"
    if not await arc.authn.rate_limit(rate_key, limit=5, window_seconds=300):
        arc.relay.throw("too many requests, try again shortly", status=429, code="rate_limited")

    if not hasattr(arc, "mail"):
        arc.relay.throw(_MAIL_NOT_CONFIGURED, status=501, code="not_implemented")

    public_url = arc.authn.public_url()
    if not public_url:
        arc.relay.throw(_MAIL_NOT_CONFIGURED, status=501, code="not_implemented")

    generic = {"ok": True, "message": "If that email has an account, a reset link has been sent."}

    user = await arc.relay.get("_users", {"email": email})
    if user is None or user["status"] != "Active":
        return generic

    raw_token = secrets.token_urlsafe(32)
    ttl_seconds = arc.authn.reset_token_ttl_seconds()
    await arc.relay.save(
        "_password_resets",
        {"user": user["id"], "token_hash": hash_token(raw_token), "expires_at": utcnow() + timedelta(seconds=ttl_seconds)},
    )

    # Imported here, not at module top — `mail` is only an optional_requires
    # dependency of authn (plugin.toml), so this module must stay importable
    # with `mail` absent; the hasattr() check above guarantees it's actually
    # installed by the time this line runs. Same narrow, precedented shape
    # as authn.cli importing psqldb.validation.ValidationError directly.
    from mail import AccountNotFoundError, TemplateNotFoundError

    reset_url = f"{public_url.rstrip('/')}/reset-password?token={raw_token}"
    try:
        await arc.mail.send(
            [email],
            template="password_reset",
            context={"email": email, "reset_url": reset_url, "ttl_minutes": ttl_seconds // 60},
        )
    except (AccountNotFoundError, TemplateNotFoundError):
        # A real operator-configuration gap (no default MailAccount, or no
        # "password_reset" MailTemplate row) — not something a caller's
        # input caused, so it's safe to surface distinctly rather than
        # folding it into the generic response above.
        arc.relay.throw(_MAIL_NOT_CONFIGURED, status=501, code="not_implemented")

    return generic


@arc.relay.whitelist(methods=["POST"], roles=["Guest"], path="/reset-password")
async def reset_password(token: str, new_password: str, client_ip: str | None = None) -> dict:
    """Consumes a token minted by forgot_password(). One generic error for
    "no such token" / "already used" / "expired" — same reasoning as
    login()'s one generic "invalid credentials", so a caller can't
    distinguish which case they hit."""
    # Light defense-in-depth only — the real defense is the token's own
    # 32-byte entropy (secrets.token_urlsafe(32)), which makes guessing it
    # computationally infeasible regardless of rate limiting. This just
    # blunts noisy automated scanning.
    rate_key = f"reset-password:{client_ip or 'unknown'}"
    if not await arc.authn.rate_limit(rate_key, limit=20, window_seconds=300):
        arc.relay.throw("too many requests, try again shortly", status=429, code="rate_limited")

    reset = await arc.relay.get("_password_resets", {"token_hash": hash_token(token)})
    if reset is None or reset["used_at"] is not None or reset["expires_at"] <= utcnow():
        arc.relay.throw("invalid or expired reset link", status=400, code="invalid_reset_token")

    user = await arc.relay.get("_users", reset["user"])
    if user is None:
        arc.relay.throw("invalid or expired reset link", status=400, code="invalid_reset_token")

    try:
        # to_thread for the same event-loop reason as login()'s verify —
        # zxcvbn scoring is real CPU on a Guest-reachable path.
        await asyncio.to_thread(
            validate_password_strength, new_password,
            min_score=arc.authn.min_password_score(), user_inputs=[user["email"]],
        )
    except PasswordPolicyError as exc:
        arc.relay.throw(str(exc), status=400, code="weak_password")

    from argon2 import PasswordHasher

    new_hash = await asyncio.to_thread(PasswordHasher().hash, new_password)
    await arc.relay.save(
        "_users",
        {
            "id": user["id"], "password_hash": new_hash,
            "failed_login_count": 0, "locked_until": None,
        },
    )
    await arc.relay.save("_password_resets", {"id": reset["id"], "used_at": utcnow()})

    # Sessions only — access keys deliberately left alone: a leaked
    # password compromises interactive sessions, not a separately-issued
    # API credential the user chose to create and scope themselves. Same
    # revoke loop as `arc authn set-password` (cli.py) — a 3rd copy of it
    # (clear-sessions is the other), not factored into a shared helper,
    # matching this feature's own "small, matching, no premature
    # abstraction" preference elsewhere.
    sessions = await arc.relay.list("_sessions", filters={"user": user["id"], "revoked_at": {"is_null": True}})
    for s in sessions:
        await arc.relay.save("_sessions", {"id": s["id"], "revoked_at": utcnow()})
        await arc.authn.invalidate_session_cache(s["token_hash"])

    return {"ok": True}


@arc.relay.whitelist(methods=["POST"], roles=["Guest"], path="/refresh")
async def refresh(token: str) -> dict:
    token_hash = hash_token(token)
    session = await arc.relay.get("_sessions", {"token_hash": token_hash})
    if session is None or session["revoked_at"] is not None or session["expires_at"] <= utcnow():
        arc.relay.throw("invalid or expired session", status=401, code="invalid_session")

    # The user must still be Active to keep the chain alive — without this,
    # a disabled/locked account could rotate a valid session indefinitely
    # (resolve_identity blocks actual use meanwhile, but a stolen token
    # chain kept warm survives until the account is re-activated). Same
    # generic error as an invalid token, deliberately — no state oracle.
    user = await arc.relay.get("_users", session["user"])
    if user is None or user["status"] != "Active":
        arc.relay.throw("invalid or expired session", status=401, code="invalid_session")

    # Rotate rather than extend in place — _sessions stays an honest log of
    # distinct sessions, and a stolen old token stops working immediately.
    await arc.relay.save("_sessions", {"id": session["id"], "revoked_at": utcnow()})
    await arc.authn.invalidate_session_cache(token_hash)

    new_token = secrets.token_urlsafe(32)
    ttl = arc.authn.session_ttl_seconds(session["session_type"])
    expires_at = utcnow() + timedelta(seconds=ttl)
    await arc.relay.save(
        "_sessions",
        {
            "user": session["user"], "token_hash": hash_token(new_token),
            "session_type": session["session_type"], "expires_at": expires_at,
        },
    )
    return {"token": new_token, "session_type": session["session_type"], "expires_at": expires_at.isoformat()}


@arc.relay.whitelist(methods=["POST"], roles=["*"], path="/access-keys")
async def create_access_key(label: str, scopes: list[str], identity=None) -> dict:
    identity = _require_identity(identity)
    # Unconditional, checked BEFORE the subset check below — a Superuser's
    # own identity.roles legitimately contains "*" (injected by
    # resolve_identity's session path), which would otherwise make
    # has_roles_subset(["*"], identity.roles) pass. An access key is a
    # weaker, more exposed credential than a session; it must never be able
    # to carry the bypass regardless of how privileged the creating user is.
    if "*" in scopes or SUPERUSER_ROLE_NAME in scopes:
        arc.relay.throw(
            "scopes cannot include the Superuser bypass — access keys may never carry it",
            code="scopes_forbidden",
        )
    if not has_roles_subset(scopes, identity.roles):
        arc.relay.throw("scopes must be a subset of your own roles", code="invalid_scopes")

    raw_key = f"ak_{secrets.token_urlsafe(32)}"
    prefix = raw_key[:KEY_PREFIX_LEN]
    row = await arc.relay.save(
        "_access_keys",
        {"user": identity.user_id, "key_prefix": prefix, "key_hash": hash_token(raw_key), "label": label, "scopes": scopes},
    )
    return {"key": raw_key, "key_prefix": prefix, "id": str(row["id"])}


@arc.relay.whitelist(methods=["POST"], roles=["*"], path="/access-keys/revoke")
async def revoke_access_key(key_id: str, identity=None) -> dict:
    identity = _require_identity(identity)
    row = await arc.relay.get("_access_keys", key_id)
    if row is None or str(row["user"]) != identity.user_id:
        arc.relay.throw("no such access key", status=404, code="not_found")
    await arc.relay.save("_access_keys", {"id": key_id, "revoked_at": utcnow()})
    await arc.authn.invalidate_access_key_cache(row["key_prefix"])
    return {"ok": True}


@arc.relay.whitelist(methods=["GET"], roles=["*"], path="/sessions")
async def list_sessions(identity=None) -> list[dict]:
    identity = _require_identity(identity)
    rows = await arc.relay.list("_sessions", filters={"user": identity.user_id}, order_by=["-expires_at"])
    return [
        {
            "id": str(r["id"]),
            "session_type": r["session_type"],
            "expires_at": r["expires_at"].isoformat(),
            "revoked_at": r["revoked_at"].isoformat() if r["revoked_at"] else None,
            "ip_address": r["ip_address"],
        }
        for r in rows
    ]


@arc.relay.whitelist(methods=["POST"], roles=["*"], path="/sessions/revoke")
async def revoke_session(session_id: str, identity=None) -> dict:
    identity = _require_identity(identity)
    row = await arc.relay.get("_sessions", session_id)
    if row is None or str(row["user"]) != identity.user_id:
        arc.relay.throw("no such session", status=404, code="not_found")
    await arc.relay.save("_sessions", {"id": session_id, "revoked_at": utcnow()})
    await arc.authn.invalidate_session_cache(row["token_hash"])
    return {"ok": True}
