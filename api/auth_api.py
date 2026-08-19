"""authn's HTTP-reachable surface — explicit paths (relay.whitelist's
path= override, docs/arc.MD §3.11) rather than the auto-derived
/api/v1/... convention, per the proposal this plugin was built from.

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
import json
import secrets
from datetime import timedelta
from typing import Any

import arc

from authn import (
    KEY_PREFIX_LEN,
    SUPERUSER_ROLE_NAME,
    PasswordPolicyError,
    _ip_allowed,
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
    (docs/review-2026-07-14.md S3)."""
    global _DUMMY_PASSWORD_HASH
    if _DUMMY_PASSWORD_HASH is None:
        from argon2 import PasswordHasher

        _DUMMY_PASSWORD_HASH = PasswordHasher().hash(secrets.token_urlsafe(16))
    return _DUMMY_PASSWORD_HASH


def _require_identity(identity):
    if identity is None:
        arc.relay.throw("authentication required", status=401, code="unauthorized")
    return identity


def _profile(user: dict) -> dict:
    """The shape login()/whoami() both return — never the raw session
    token (that only ever travels via the httpOnly Set-Cookie itself, not
    the JSON body a script or an XSS payload could read back out)."""
    return {
        "email": user["email"],
        "username": user.get("username"),
        "full_name": user.get("full_name"),
        "theme": user.get("theme") or "Daylight",
    }


def _cookie_secure() -> bool:
    """A Secure cookie is silently refused by the browser entirely over
    plain HTTP — `arc run` without a reverse proxy in front serves plain
    HTTP by default, so this has to follow whatever gateway_force_https is
    actually set to, not just default on and quietly break local dev."""
    return (arc.settings.get("gateway_force_https") or "").lower() in ("1", "true", "yes")


def _session_response(content: dict, *, token: str, ttl_seconds: int):
    """Wraps `content` in a gateway.request.Response that sets the real
    session cookie (HttpOnly) plus a paired, JS-readable CSRF cookie — only
    when gateway is actually installed; a direct arc.relay.call() has no
    HTTP response to carry a Set-Cookie over, so it just gets `content`
    back plain, same as any other whitelisted function's direct-call path.
    Deliberately no raw-token field anywhere in `content` either way —
    programmatic (non-browser) access to a live session was never this
    plugin's job; that's what API access keys (X-API-Key) are for."""
    if not hasattr(arc, "gateway"):
        return content
    from gateway.request import Cookie, Response

    secure = _cookie_secure()
    return Response(
        content=content,
        cookies=[
            Cookie("arc_session", token, max_age=ttl_seconds, http_only=True, secure=secure),
            Cookie(
                "csrf_token",
                secrets.token_urlsafe(16),
                max_age=ttl_seconds,
                http_only=False,
                secure=secure,
            ),
        ],
    )


def _cleared_session_response(content: dict):
    if not hasattr(arc, "gateway"):
        return content
    from gateway.request import Cookie, Response

    secure = _cookie_secure()
    return Response(
        content=content,
        cookies=[
            Cookie.cleared("arc_session", secure=secure),
            Cookie.cleared("csrf_token", secure=secure),
        ],
    )


async def _active_sessions_summary(user_id: str) -> list[dict]:
    """The caller's own currently-active sessions, shaped for a client to
    render a "log one of these out" picker — used only when max_sessions
    is what's blocking a login (login()'s own max_sessions_reached branch).
    Never includes token_hash — nothing here is a credential, just enough
    to tell sessions apart (docs/arc.MD §3.3's usual "return a curated
    shape, never the raw row" rule)."""
    rows = await arc.relay.list(
        "_sessions",
        filters={"user": user_id, "revoked_at": {"is_null": True}, "expires_at": {"gt": utcnow()}},
        fields=["id", "session_type", "ip_address", "user_agent", "expires_at"],
        limit=50,  # a generous backstop, not a real cap — max_sessions itself
        # is always some small number in practice; this only exists so a
        # pathological account somehow far past its own cap can't make this
        # list unbounded.
    )
    return [
        {
            "id": str(r["id"]),
            "session_type": r["session_type"],
            "ip_address": r["ip_address"],
            "user_agent": r["user_agent"],
            "expires_at": r["expires_at"].isoformat() if r["expires_at"] else None,
        }
        for r in rows
    ]


async def _authenticate_credentials(
    email: str | None, username: str | None, password: str, client_ip: str | None
) -> dict:
    """The lookup/rate-limit/password/lockout gate login() and
    terminate_login_session() both need, ahead of whatever each does
    next — factored out once rather than duplicated: two independently-
    maintained copies of security-sensitive verification logic is exactly
    how they quietly drift apart (one path checks locked status, the other
    forgets to). Returns the verified user row (every column,
    arc.relay.all_columns) on success; raises the same generic
    invalid_credentials RelayError either caller would raise anyway on any
    failure — unknown identifier, wrong password, locked, or inactive."""
    from argon2 import PasswordHasher
    from argon2.exceptions import VerificationError, VerifyMismatchError

    if not email and not username:
        arc.relay.throw("email or username is required", code="identifier_required")

    # Two lookup fields, one auth path — a real second capability (not
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
        arc.relay.throw(
            "too many login attempts, try again shortly", status=429, code="rate_limited"
        )

    # Internal use only — verified against, never returned directly
    # (see _profile()'s own curated shape below, the only thing a caller
    # ever hands back).
    user = await arc.relay.get("_users", lookup, arc.relay.all_columns("_users"))

    # Always pay exactly one Argon2 verify — against the real hash if the
    # user exists, a fixed dummy hash otherwise — so response timing can't
    # be used to enumerate valid emails (docs/review-2026-07-14.md S3). Run in a
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
    # review-2026-07-14.md L2), and failed_login_count grows without bound across
    # repeated lockouts instead of resetting to a fresh N-attempt budget.
    if user["locked_until"] is not None and user["locked_until"] <= utcnow():
        await arc.relay.save(
            "_users", {"id": user["id"], "failed_login_count": 0, "locked_until": None}
        )
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
        # review-2026-07-14.md S3). A genuinely locked/inactive user sees the same
        # message as a wrong password, by design.
        arc.relay.throw("invalid credentials", status=401, code="invalid_credentials")

    await arc.relay.save(
        "_users",
        {
            "id": user["id"],
            "failed_login_count": 0,
            "locked_until": None,
            "last_login_at": utcnow(),
        },
    )
    return user


def _check_allowed_source(user: dict, client_ip: str | None) -> None:
    """Shared by login() and terminate_login_session() — a genuinely
    correct password from a disallowed network. resolve_identity()'s own
    _authorize() already enforces allowed_ips on every request AFTER a
    session exists — the bug this closes is that neither pre-auth endpoint
    checked it themselves, so a disallowed caller got a real, working
    session cookie back (200, looks like success) that then failed on the
    very next request it tried to use, with no explanation of why. A
    distinct message here (unlike the generic invalid_credentials one) is
    deliberate, not an oversight: it only ever fires post-correct-password,
    so it can't be used to enumerate accounts the way a distinct
    locked/inactive message would."""
    if user["allowed_ips"] and not _ip_allowed(client_ip, user["allowed_ips"]):
        arc.relay.throw(
            "sign-in isn't allowed from this network — contact an administrator if this is unexpected",
            status=403,
            code="invalid_source",
        )


@arc.relay.whitelist(methods=["POST"], roles=["Guest"], path="/login")
async def login(
    email: str | None = None,
    username: str | None = None,
    *,
    password: str,
    session_type: str = "Fixed",
    client_ip: str | None = None,
    headers: dict[str, str] | None = None,
) -> dict:
    if session_type not in ("Fixed", "Extended"):
        arc.relay.throw("session_type must be 'Fixed' or 'Extended'", code="bad_session_type")

    user = await _authenticate_credentials(email, username, password, client_ip)
    _check_allowed_source(user, client_ip)

    # Count-then-insert is a race without this: two concurrent logins for
    # the same user could both see "under the cap" and both insert,
    # exceeding max_sessions (docs/review-2026-07-14.md L4). Same lock() primitive
    # save()'s own match_on upsert already uses — real guarantee with
    # redix installed, a weaker in-process-only one otherwise, same
    # documented tradeoff as every other arc.relay.lock() use.
    async with arc.relay.lock(f"login:sessions:{user['id']}"):
        active_sessions = await arc.relay.count(
            "_sessions",
            filters={
                "user": user["id"],
                "revoked_at": {"is_null": True},
                "expires_at": {"gt": utcnow()},
            },
        )
        if user["max_sessions"] is not None and active_sessions >= user["max_sessions"]:
            # extra.sessions rides along on the SAME error a plain client
            # already handles (parseError's existing {error, code} path
            # still works unchanged) — a client that knows to look for it
            # can offer "log one of these out and retry" instead of a dead-
            # end message; one that doesn't just shows the message as
            # before. See RelayError.extra's own docstring for why this is
            # a response field, not a second round trip.
            arc.relay.throw(
                "maximum active sessions reached — log out an existing session first",
                status=403,
                code="max_sessions_reached",
                extra={"sessions": await _active_sessions_summary(user["id"])},
            )

        token = secrets.token_urlsafe(32)
        ttl = arc.authn.session_ttl_seconds(session_type)
        expires_at = utcnow() + timedelta(seconds=ttl)
        await arc.relay.save(
            "_sessions",
            {
                "user": user["id"],
                "token_hash": hash_token(token),
                "session_type": session_type,
                "expires_at": expires_at,
                "ip_address": client_ip,
                "user_agent": (headers or {}).get("user-agent"),
            },
        )
    return _session_response(
        _profile(user),
        token=token,
        ttl_seconds=ttl,
    )


@arc.relay.whitelist(methods=["POST"], roles=["Guest"], path="/login/terminate-session")
async def terminate_login_session(
    session_id: str,
    email: str | None = None,
    username: str | None = None,
    *,
    password: str,
    client_ip: str | None = None,
) -> dict:
    """Lets a caller who's hit max_sessions on /login pick one of their OWN
    active sessions (the list login()'s max_sessions_reached error hands
    back in extra.sessions) to revoke, then retry /login — the "you're
    already signed in on N devices, log one out" flow.

    Re-verifies email/password itself, through the exact same gate login()
    uses — there's no session to trust for "who is this" here at all, the
    caller isn't authenticated yet. Only ever revokes a session confirmed
    to belong to THIS credential-verified user (the {"id":..., "user":...}
    filter below, not id alone) — a session_id for someone else's session
    is indistinguishable from one that doesn't exist, same non-enumeration
    posture login() itself already takes on a bad identifier."""
    user = await _authenticate_credentials(email, username, password, client_ip)
    _check_allowed_source(user, client_ip)

    session = await arc.relay.get(
        "_sessions", {"id": session_id, "user": user["id"]}, ["id", "revoked_at"]
    )
    if session is None or session["revoked_at"] is not None:
        arc.relay.throw("session not found", status=404, code="session_not_found")

    await arc.relay.save("_sessions", {"id": session["id"], "revoked_at": utcnow()})
    return {"ok": True}


@arc.relay.whitelist(methods=["POST"], roles=["Guest"], path="/logout")
async def logout(cookies: dict[str, str] | None = None) -> dict:
    token = (cookies or {}).get("arc_session")
    if token:
        token_hash = hash_token(token)
        session = await arc.relay.get(
            "_sessions", {"token_hash": token_hash}, ["id", "revoked_at"]
        )
        if session is not None and session["revoked_at"] is None:
            await arc.relay.save("_sessions", {"id": session["id"], "revoked_at": utcnow()})
            await arc.authn.invalidate_session_cache(token_hash)
    return _cleared_session_response({"ok": True})


@arc.relay.whitelist(methods=["GET"], roles=["*"], path="/whoami")
async def whoami(identity=None) -> dict:
    identity = _require_identity(identity)
    user = await arc.relay.get(
        "_users", identity.user_id, ["email", "username", "full_name", "theme"]
    )
    if user is None:
        arc.relay.throw("authentication required", status=401, code="unauthorized")
    return _profile(user)


@arc.relay.whitelist(methods=["POST"], roles=["*"], path="/me/theme")
async def set_my_theme(theme: str, identity=None) -> dict:
    """`theme` is an open-ended preset NAME (e.g. "Late Night"), not a
    fixed light/dark enum — admin's console owns the actual known set
    (admin/ui/src/theme/presets.ts) and is meant to be extendable there
    with no change needed here. This only enforces shape (matches
    _users.theme's own STRING(60) column), never a specific value list —
    same "DB/API tier enforces shape, app tier enforces meaning" split
    psqldb's SELECT fields already use elsewhere."""
    identity = _require_identity(identity)
    theme = theme.strip()
    if not theme or len(theme) > 60:
        arc.relay.throw("theme must be a non-empty name, 60 characters or fewer", code="invalid_theme")
    await arc.relay.save("_users", {"id": identity.user_id, "theme": theme})
    return {"ok": True, "theme": theme}


def _impersonate_html(ticket: str) -> str:
    """A tiny, self-submitting shell page — no external resources, so it
    works standalone with no build step. `ticket_json` is a properly
    JSON-escaped JS string literal (json.dumps), with `</` additionally
    neutralized so a ticket value can never prematurely close the
    surrounding <script> tag — `ticket` arrives straight from the query
    string, i.e. is attacker-controlled input being embedded into HTML."""
    ticket_json = json.dumps(ticket).replace("</", "<\\/")
    return (
        '<!doctype html><html><head><meta charset="utf-8">'
        "<title>Signing in…</title></head><body>"
        "<p>Signing you in…</p>"
        "<script>"
        # A stray, leftover arc_session cookie from some earlier/unrelated
        # session (this browser doesn't have to be "fresh") makes
        # gateway's csrf_middleware demand a matching X-CSRF-Token the
        # instant one's present — same double-submit rule client.ts's own
        # headers() helper already follows for every mutating call, echoed
        # here by hand since this page has no bundle to share it from.
        "var m=document.cookie.match(/(?:^|;\\s*)csrf_token=([^;]+)/);"
        "var h={'Content-Type':'application/json'};"
        "if(m)h['X-CSRF-Token']=decodeURIComponent(m[1]);"
        "fetch('/impersonate/consume',{method:'POST',credentials:'same-origin',headers:h,"
        "body:JSON.stringify({ticket:" + ticket_json + "})})"
        ".then(function(res){if(!res.ok)throw new Error('invalid');window.location.replace('/');})"
        ".catch(function(){document.body.textContent="
        "'This impersonation link is invalid or has expired.';});"
        "</script></body></html>"
    )


@arc.relay.whitelist(methods=["GET"], roles=["Guest"], path="/impersonate")
async def impersonate(ticket: str) -> Any:
    """Serves the self-submitting shell above — reached by a plain browser
    navigation (the CLI opens this URL directly, and a browser can only
    ever GET a typed/clicked link). Deliberately does NO writes itself:
    relay treats every GET as dry-run for arc.relay.save/delete calls
    (_wire_gateway_route's own safety net, see relay/__init__.py) — an
    earlier version of this endpoint tried to consume the ticket and
    create the session directly here, and relay silently rolled both
    writes back, leaving a Set-Cookie pointing at a session that was
    never actually persisted. The real work happens from this page's own
    POST to /impersonate/consume below, which is a genuine, non-dry-run
    write."""
    if not hasattr(arc, "gateway"):
        arc.relay.throw(
            "this endpoint requires the gateway plugin", status=501, code="not_implemented"
        )
    from gateway.request import Response

    return Response(content=_impersonate_html(ticket), media_type="text/html")


@arc.relay.whitelist(methods=["POST"], roles=["Guest"], path="/impersonate/consume")
async def impersonate_consume(ticket: str) -> Any:
    """Consumes a single-use ticket minted by `arc authn browse-as` and
    establishes a real session for the target user. Deliberately skips
    login()'s own max_sessions check — this is an admin/support action
    against another account's session budget, not a login on that user's
    own behalf, and shouldn't be blocked by however many sessions that
    user already has open elsewhere."""
    ticket_hash = hash_token(ticket)
    row = await arc.relay.get(
        "_impersonation_tickets",
        {"token_hash": ticket_hash},
        ["id", "used_at", "expires_at", "user"],
    )
    if row is None or row["used_at"] is not None or row["expires_at"] <= utcnow():
        arc.relay.throw("invalid or expired impersonation link", status=400, code="invalid_ticket")

    user = await arc.relay.get("_users", row["user"], ["id", "status"])
    if user is None or user["status"] != "Active":
        arc.relay.throw("invalid or expired impersonation link", status=400, code="invalid_ticket")

    # Marked used BEFORE the session is created — a ticket is single-use
    # regardless of whether session creation below succeeds; a failure
    # partway through must never leave a still-consumable ticket behind.
    await arc.relay.save("_impersonation_tickets", {"id": row["id"], "used_at": utcnow()})

    token = secrets.token_urlsafe(32)
    ttl = arc.authn.session_ttl_seconds("Fixed")
    expires_at = utcnow() + timedelta(seconds=ttl)
    await arc.relay.save(
        "_sessions",
        {
            "user": user["id"],
            "token_hash": hash_token(token),
            "session_type": "Fixed",
            "expires_at": expires_at,
        },
    )
    return _session_response({"ok": True}, token=token, ttl_seconds=ttl)


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
    login() already takes (docs/review-2026-07-14.md S3). Deliberately NOT extended to
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

    user = await arc.relay.get("_users", {"email": email}, ["id", "status"])
    if user is None or user["status"] != "Active":
        return generic

    raw_token = secrets.token_urlsafe(32)
    ttl_seconds = arc.authn.reset_token_ttl_seconds()
    await arc.relay.save(
        "_password_resets",
        {
            "user": user["id"],
            "token_hash": hash_token(raw_token),
            "expires_at": utcnow() + timedelta(seconds=ttl_seconds),
        },
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

    reset = await arc.relay.get(
        "_password_resets",
        {"token_hash": hash_token(token)},
        ["id", "used_at", "expires_at", "user"],
    )
    if reset is None or reset["used_at"] is not None or reset["expires_at"] <= utcnow():
        arc.relay.throw("invalid or expired reset link", status=400, code="invalid_reset_token")

    user = await arc.relay.get("_users", reset["user"], ["id", "email"])
    if user is None:
        arc.relay.throw("invalid or expired reset link", status=400, code="invalid_reset_token")

    try:
        # to_thread for the same event-loop reason as login()'s verify —
        # zxcvbn scoring is real CPU on a Guest-reachable path.
        await asyncio.to_thread(
            validate_password_strength,
            new_password,
            min_score=arc.authn.min_password_score(),
            user_inputs=[user["email"]],
        )
    except PasswordPolicyError as exc:
        arc.relay.throw(str(exc), status=400, code="weak_password")

    from argon2 import PasswordHasher

    new_hash = await asyncio.to_thread(PasswordHasher().hash, new_password)
    await arc.relay.save(
        "_users",
        {
            "id": user["id"],
            "password_hash": new_hash,
            "failed_login_count": 0,
            "locked_until": None,
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
    sessions = await arc.relay.list(
        "_sessions",
        fields=["id", "token_hash"],
        filters={"user": user["id"], "revoked_at": {"is_null": True}},
    )
    for s in sessions:
        await arc.relay.save("_sessions", {"id": s["id"], "revoked_at": utcnow()})
        await arc.authn.invalidate_session_cache(s["token_hash"])

    return {"ok": True}


@arc.relay.whitelist(methods=["POST"], roles=["Guest"], path="/refresh")
async def refresh(cookies: dict[str, str] | None = None) -> dict:
    token = (cookies or {}).get("arc_session")
    if not token:
        arc.relay.throw("invalid or expired session", status=401, code="invalid_session")
    token_hash = hash_token(token)
    session = await arc.relay.get(
        "_sessions",
        {"token_hash": token_hash},
        ["id", "revoked_at", "expires_at", "user", "session_type"],
    )
    if session is None or session["revoked_at"] is not None or session["expires_at"] <= utcnow():
        arc.relay.throw("invalid or expired session", status=401, code="invalid_session")

    # The user must still be Active to keep the chain alive — without this,
    # a disabled/locked account could rotate a valid session indefinitely
    # (resolve_identity blocks actual use meanwhile, but a stolen token
    # chain kept warm survives until the account is re-activated). Same
    # generic error as an invalid token, deliberately — no state oracle.
    user = await arc.relay.get("_users", session["user"], ["id", "status"])
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
            "user": session["user"],
            "token_hash": hash_token(new_token),
            "session_type": session["session_type"],
            "expires_at": expires_at,
        },
    )
    return _session_response({"ok": True}, token=new_token, ttl_seconds=ttl)


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
        {
            "user": identity.user_id,
            "key_prefix": prefix,
            "key_hash": hash_token(raw_key),
            "label": label,
            "scopes": scopes,
        },
    )
    return {"key": raw_key, "key_prefix": prefix, "id": str(row["id"])}


@arc.relay.whitelist(methods=["POST"], roles=["*"], path="/access-keys/revoke")
async def revoke_access_key(key_id: str, identity=None) -> dict:
    identity = _require_identity(identity)
    row = await arc.relay.get("_access_keys", key_id, ["user", "key_prefix"])
    if row is None or str(row["user"]) != identity.user_id:
        arc.relay.throw("no such access key", status=404, code="not_found")
    await arc.relay.save("_access_keys", {"id": key_id, "revoked_at": utcnow()})
    await arc.authn.invalidate_access_key_cache(row["key_prefix"])
    return {"ok": True}


@arc.relay.whitelist(methods=["GET"], roles=["*"], path="/sessions")
async def list_sessions(identity=None) -> list[dict]:
    identity = _require_identity(identity)
    rows = await arc.relay.list(
        "_sessions",
        fields=["id", "session_type", "expires_at", "revoked_at", "ip_address"],
        filters={"user": identity.user_id},
        order_by=["-expires_at"],
    )
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
    row = await arc.relay.get("_sessions", session_id, ["user", "token_hash"])
    if row is None or str(row["user"]) != identity.user_id:
        arc.relay.throw("no such session", status=404, code="not_found")
    await arc.relay.save("_sessions", {"id": session_id, "revoked_at": utcnow()})
    await arc.authn.invalidate_session_cache(row["token_hash"])
    return {"ok": True}
