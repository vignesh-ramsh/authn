"""authn.cli — `arc authn ...` commands.

Mounted onto the core `arc` CLI via the `arc.plugins.cli` entry point (see
arc.plugin_cli in the kernel), same mechanism as psqldb.cli/redix.cli/
gateway.cli.

Unlike psqldb/redix's connectivity commands (status/connect, a lightweight
DSN-only probe), every command here does real writes against live tables
(_users/_roles/_sessions/_access_keys) — so this always does a real
`arc.boot()` + opens psqldb (and redix, when installed) rather than reading
connection settings straight off disk. Every mutation goes through
`arc.relay.save()`/`arc.authn`'s own cache-invalidation helpers, never a
raw psqldb write — a raw write would bypass the has_roles/scopes
validation hooks (authn/hooks/) or leave a stale redix cache entry behind
(docs/arc.MD §3.13, docs/Review.MD S1).
"""

from __future__ import annotations

import asyncio
import contextlib
import secrets
import warnings
from datetime import timedelta

import arc
import typer
from rich.console import Console
from rich.table import Table

# The same narrow, precedented exception relay/relay/__init__.py already
# takes on psqldb's internals ("the only dependency Relay takes on
# psqldb's internals") — ValidationError is only re-exported at
# psqldb's own MODULE level (psqldb/__init__.py), not attached to the
# arc.psqldb capability INSTANCE the way relay.RelayError deliberately is
# (docs/arc.MD §3.11), so there's no `arc.psqldb.ValidationError` to catch
# without this.
from psqldb.validation import ValidationError

from authn import PasswordPolicyError, SUPERUSER_ROLE_NAME, utcnow, validate_password_strength

app = typer.Typer(help="Commands for the authn provider.")
console = Console()
err_console = Console(stderr=True, style="bold red")


# ------------------------------------------------------------------------ #
# Shared plumbing
# ------------------------------------------------------------------------ #
def _run(coro) -> None:
    async def _main():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", arc.ArcAdvisory)
            arc.boot()
        await arc.psqldb.open()
        has_redix = hasattr(arc, "redix")
        if has_redix:
            await arc.redix.open()
        try:
            await coro
        finally:
            if has_redix:
                await arc.redix.close()
            await arc.psqldb.close()

    asyncio.run(_main())


@contextlib.contextmanager
def _friendly_errors():
    """Domain validation problems (bad email format, unique-constraint
    violation, a role that doesn't validate, a weak password) are user
    mistakes to fix, not internal failures — report them as a clean
    one-line error, not a Python traceback. Mirrors psqldb.cli's own
    _friendly_errors() exactly."""
    try:
        yield
    except (ValidationError, PasswordPolicyError) as exc:
        err_console.print(str(exc))
        raise typer.Exit(code=1) from exc


def _parse_csv(raw: str) -> list[str]:
    seen: list[str] = []
    for part in raw.split(","):
        name = part.strip()
        if name and name not in seen:
            seen.append(name)
    return seen


async def _get_user_or_exit(email: str) -> dict:
    user = await arc.relay.get("_users", {"email": email.strip().lower()})
    if user is None:
        err_console.print(f"no user with email '{email}'.")
        raise typer.Exit(code=1)
    return user


async def _known_roles(names: list[str]) -> list[str]:
    """Filters `names` down to ones that actually exist in _roles, printing
    a warning for each one dropped — a silent skip would hide a typo'd
    role name from the operator with nothing telling them a grant didn't
    happen (docs/Review.MD proposal discussion)."""
    known = []
    for name in names:
        if await arc.relay.get("_roles", {"name": name}) is None:
            console.print(f"[yellow]warning:[/yellow] role '{name}' does not exist, skipping.")
        else:
            known.append(name)
    return known


def _warn_if_superuser(granted: list[str], *, email: str) -> None:
    if SUPERUSER_ROLE_NAME in granted:
        console.print(
            f"[bold red]WARNING:[/bold red] granting '{SUPERUSER_ROLE_NAME}' to {email} — "
            f"this bypasses every role check in the system."
        )


def _confirm_or_abort(yes: bool, prompt: str = "Proceed?") -> None:
    if not yes and not typer.confirm(prompt, default=False):
        console.print("[dim]Aborted — nothing changed.[/dim]")
        raise typer.Exit(code=1)


# ------------------------------------------------------------------------ #
# Users
# ------------------------------------------------------------------------ #
@app.command(name="create-user")
def create_user(
    email: str = typer.Option(..., "--email"),
    role: str = typer.Option("", "--role", help="Comma-separated role names."),
    random_password: bool = typer.Option(False, "--random-password", help="Generate a password instead of prompting; printed once."),
    superuser: bool = typer.Option(False, "--superuser", help=f"Also grant the '{SUPERUSER_ROLE_NAME}' role, creating it first if it doesn't exist yet."),
) -> None:
    """Create a user. Prompts for a password (hidden input) unless
    --random-password is given."""
    email_norm = email.strip().lower()

    if random_password:
        password = secrets.token_urlsafe(18)
    else:
        password = typer.prompt("Password", hide_input=True, confirmation_prompt=True)

    async def _do() -> None:
        if await arc.relay.get("_users", {"email": email_norm}) is not None:
            err_console.print(f"a user with email '{email_norm}' already exists.")
            raise typer.Exit(code=1)

        roles = await _known_roles(_parse_csv(role))
        if superuser:
            if await arc.relay.get("_roles", {"name": SUPERUSER_ROLE_NAME}) is None:
                await arc.relay.save(
                    "_roles", {"name": SUPERUSER_ROLE_NAME, "description": "Bypasses all role checks."}
                )
                console.print(f"[dim]role '{SUPERUSER_ROLE_NAME}' did not exist — created it.[/dim]")
            if SUPERUSER_ROLE_NAME not in roles:
                roles.append(SUPERUSER_ROLE_NAME)

        validate_password_strength(password, min_score=arc.authn.min_password_score(), user_inputs=[email_norm])

        from argon2 import PasswordHasher

        user = await arc.relay.save(
            "_users", {"email": email_norm, "password_hash": PasswordHasher().hash(password), "has_roles": roles}
        )
        _warn_if_superuser(roles, email=email_norm)
        if random_password:
            console.print(f"[bold]generated password (shown once):[/bold] {password}")
        console.print(f"[bold green]created user {user['email']}[/bold green] (id={user['id']}, roles={roles or []})")

    with _friendly_errors():
        _run(_do())


@app.command(name="set-status")
def set_status(
    email: str = typer.Option(..., "--email"),
    status: str = typer.Argument(..., help="Active, Inactive, or Locked."),
) -> None:
    """Set a user's account status. Setting Active also clears any
    brute-force lockout (failed_login_count/locked_until) — those are a
    separate mechanism from status, and a status change alone wouldn't
    otherwise unlock an account still under an active lockout timer."""
    normalized = status.strip().capitalize()
    if normalized not in ("Active", "Inactive", "Locked"):
        err_console.print(f"status must be one of Active/Inactive/Locked, got '{status}'.")
        raise typer.Exit(code=1)

    async def _do() -> None:
        user = await _get_user_or_exit(email)
        payload = {"id": user["id"], "status": normalized}
        if normalized == "Active":
            payload["failed_login_count"] = 0
            payload["locked_until"] = None
        await arc.relay.save("_users", payload)
        console.print(f"[bold green]{user['email']}: status set to {normalized}.[/bold green]")

    with _friendly_errors():
        _run(_do())


@app.command(name="add-role")
def add_role(email: str = typer.Option(..., "--email"), role: str = typer.Option(..., "--role", help="Comma-separated role names.")) -> None:
    """Add role(s) to a user — additive only, existing roles are kept.
    Idempotent: a role the user already has is silently skipped."""

    async def _do() -> None:
        user = await _get_user_or_exit(email)
        requested = await _known_roles(_parse_csv(role))
        current = list(user.get("has_roles") or [])
        added = [r for r in requested if r not in current]
        if not added:
            console.print("[dim]no changes — user already has every requested (existing) role.[/dim]")
            return
        await arc.relay.save("_users", {"id": user["id"], "has_roles": [*current, *added]})
        _warn_if_superuser(added, email=user["email"])
        console.print(f"[bold green]{user['email']}: added {added}.[/bold green]")

    with _friendly_errors():
        _run(_do())


@app.command(name="remove-role")
def remove_role(email: str = typer.Option(..., "--email"), role: str = typer.Option(..., "--role", help="Comma-separated role names.")) -> None:
    """Remove role(s) from a user, without needing to know/re-type their
    full current role list. Idempotent: a role the user doesn't have is
    silently skipped."""

    async def _do() -> None:
        user = await _get_user_or_exit(email)
        requested = set(_parse_csv(role))
        current = list(user.get("has_roles") or [])
        removed = [r for r in current if r in requested]
        if not removed:
            console.print("[dim]no changes — user doesn't have any of the given role(s).[/dim]")
            return
        remaining = [r for r in current if r not in requested]
        await arc.relay.save("_users", {"id": user["id"], "has_roles": remaining})
        console.print(f"[bold green]{user['email']}: removed {removed}.[/bold green]")

    with _friendly_errors():
        _run(_do())


@app.command(name="set-password")
def set_password(email: str = typer.Option(..., "--email")) -> None:
    """Set (or reset) a user's password. Also clears any brute-force
    lockout and revokes every one of the user's active sessions — a
    password reset is exactly the moment you don't want an already-live
    session (possibly under the old, compromised password) to keep working."""
    password = typer.prompt("New password", hide_input=True, confirmation_prompt=True)

    async def _do() -> None:
        user = await _get_user_or_exit(email)
        validate_password_strength(password, min_score=arc.authn.min_password_score(), user_inputs=[user["email"]])

        from argon2 import PasswordHasher

        await arc.relay.save(
            "_users",
            {
                "id": user["id"], "password_hash": PasswordHasher().hash(password),
                "failed_login_count": 0, "locked_until": None,
            },
        )
        sessions = await arc.relay.list("_sessions", filters={"user": user["id"], "revoked_at": {"is_null": True}})
        for s in sessions:
            await arc.relay.save("_sessions", {"id": s["id"], "revoked_at": utcnow()})
            await arc.authn.invalidate_session_cache(s["token_hash"])
        console.print(
            f"[bold green]{user['email']}: password updated, {len(sessions)} active session(s) revoked.[/bold green]"
        )

    with _friendly_errors():
        _run(_do())


@app.command(name="list-users")
def list_users(
    role: str = typer.Option(None, "--role", help="Only show users granted this role."),
    query: str = typer.Option(None, "-q", "--query", help="Only show users whose email starts with this (case-insensitive)."),
) -> None:
    async def _do() -> None:
        users = await arc.relay.list("_users", order_by=["email"])
        if role:
            users = [u for u in users if role in (u.get("has_roles") or [])]
        if query:
            q = query.strip().lower()
            users = [u for u in users if u["email"].lower().startswith(q)]

        table = Table()
        for col in ("Email", "Status", "Roles", "Max Sessions", "Locked Until"):
            table.add_column(col)
        for u in users:
            table.add_row(
                u["email"], u["status"], ", ".join(u.get("has_roles") or []) or "-",
                str(u["max_sessions"]) if u["max_sessions"] is not None else "unlimited",
                str(u["locked_until"]) if u["locked_until"] else "-",
            )
        console.print(table)
        console.print(f"[dim]{len(users)} user(s)[/dim]")

    with _friendly_errors():
        _run(_do())


# ------------------------------------------------------------------------ #
# Roles
# ------------------------------------------------------------------------ #
@app.command(name="create-role")
def create_role(name: str = typer.Argument(...), description: str = typer.Option(None, "--description")) -> None:
    async def _do() -> None:
        if await arc.relay.get("_roles", {"name": name}) is not None:
            err_console.print(f"role '{name}' already exists.")
            raise typer.Exit(code=1)
        row = await arc.relay.save("_roles", {"name": name, "description": description})
        console.print(f"[bold green]created role '{row['name']}'.[/bold green]")

    with _friendly_errors():
        _run(_do())


@app.command(name="delete-role")
def delete_role(name: str = typer.Argument(...), yes: bool = typer.Option(False, "--yes", "-y")) -> None:
    """Delete a role, and strip it from every user who currently has it.
    Each user's has_roles update goes through arc.relay.save(), which
    fires the existing after_save hook (authn/hooks/_users.py) — so every
    affected user's session/access-key cache is invalidated automatically,
    the same path a normal role change already uses. Not fully atomic
    (each user update is its own transaction) but idempotent/safely
    re-runnable if interrupted partway through."""

    async def _do() -> None:
        role = await arc.relay.get("_roles", {"name": name})
        if role is None:
            err_console.print(f"no role named '{name}'.")
            raise typer.Exit(code=1)

        all_users = await arc.relay.list("_users", fields=["id", "email", "has_roles"])
        affected = [u for u in all_users if name in (u.get("has_roles") or [])]

        if affected:
            console.print(f"This will remove role '{name}' from {len(affected)} user(s):")
            for u in affected:
                console.print(f"  - {u['email']}")
        console.print(f"...and permanently delete the role '{name}'.")
        _confirm_or_abort(yes)

        for u in affected:
            remaining = [r for r in (u["has_roles"] or []) if r != name]
            await arc.relay.save("_users", {"id": u["id"], "has_roles": remaining})

        await arc.relay.delete("_roles", role["id"])
        console.print(f"[bold green]deleted role '{name}'[/bold green] (removed from {len(affected)} user(s)).")

    with _friendly_errors():
        _run(_do())


@app.command(name="list-roles")
def list_roles() -> None:
    async def _do() -> None:
        roles = await arc.relay.list("_roles", order_by=["name"])
        table = Table()
        table.add_column("Name")
        table.add_column("Description")
        for r in roles:
            table.add_row(r["name"], r.get("description") or "-")
        console.print(table)
        console.print(f"[dim]{len(roles)} role(s)[/dim]")

    with _friendly_errors():
        _run(_do())


# ------------------------------------------------------------------------ #
# Sessions
# ------------------------------------------------------------------------ #
@app.command(name="clear-sessions")
def clear_sessions(
    email: str = typer.Option(None, "--email"),
    all_: bool = typer.Option(False, "--all"),
    yes: bool = typer.Option(False, "--yes", "-y"),
) -> None:
    """Revoke active sessions — --email for one user, --all for every
    active session in the system. Revokes (sets revoked_at), never hard-
    deletes — _sessions stays an honest history log; see prune-sessions
    for actually removing old rows."""
    if bool(email) == all_:
        err_console.print("specify exactly one of --email or --all.")
        raise typer.Exit(code=1)

    async def _do() -> None:
        filters = {"revoked_at": {"is_null": True}}
        scope_desc = "ALL users"
        if email:
            user = await _get_user_or_exit(email)
            filters["user"] = user["id"]
            scope_desc = user["email"]

        sessions = await arc.relay.list("_sessions", filters=filters)
        if not sessions:
            console.print("[dim]no active sessions to clear.[/dim]")
            return
        console.print(f"About to revoke {len(sessions)} active session(s) for {scope_desc}.")
        _confirm_or_abort(yes)

        for s in sessions:
            await arc.relay.save("_sessions", {"id": s["id"], "revoked_at": utcnow()})
            await arc.authn.invalidate_session_cache(s["token_hash"])
        console.print(f"[bold green]revoked {len(sessions)} session(s).[/bold green]")

    with _friendly_errors():
        _run(_do())


@app.command(name="prune-sessions")
def prune_sessions(
    older_than_days: int = typer.Option(30, "--older-than-days"),
    yes: bool = typer.Option(False, "--yes", "-y"),
) -> None:
    """Permanently DELETE inactive sessions older than the cutoff — a
    session counts as inactive if it was explicitly revoked (revoked_at
    set) OR has simply timed out (expires_at passed), whichever signal
    applies; most sessions are never explicitly logged out, so expires_at
    alone has to count too, or the majority of sessions would never
    actually get cleaned up. Never touches a still-active session."""

    async def _do() -> None:
        cutoff = utcnow() - timedelta(days=older_than_days)
        rows = await arc.relay.sql(
            "SELECT id FROM _sessions WHERE (revoked_at IS NOT NULL AND revoked_at <= $1) OR (expires_at <= $1)",
            cutoff,
        )
        if not rows:
            console.print("[dim]nothing to prune.[/dim]")
            return
        console.print(f"About to permanently delete {len(rows)} inactive session row(s) older than {older_than_days} day(s).")
        _confirm_or_abort(yes)

        ids = [r["id"] for r in rows]
        await arc.relay.sql("DELETE FROM _sessions WHERE id = ANY($1::uuid[])", ids)
        console.print(f"[bold green]pruned {len(ids)} session row(s).[/bold green]")

    with _friendly_errors():
        _run(_do())


# ------------------------------------------------------------------------ #
# Password resets
# ------------------------------------------------------------------------ #
@app.command(name="prune-password-resets")
def prune_password_resets(
    older_than_days: int = typer.Option(7, "--older-than-days"),
    yes: bool = typer.Option(False, "--yes", "-y"),
) -> None:
    """Permanently DELETE password-reset tokens older than the cutoff — a
    row counts as inactive once it's been used (used_at set) OR its own
    expiry has passed, whichever applies; most reset tokens are never
    consumed at all, so expires_at alone has to count too. Never touches a
    still-valid, unused token. Shorter default cutoff than prune-sessions
    (7 days, not 30) — reset tokens are already short-lived (15 minutes by
    default, authn_reset_token_ttl_minutes), so there's no reason to let
    dead rows accumulate as long."""

    async def _do() -> None:
        cutoff = utcnow() - timedelta(days=older_than_days)
        rows = await arc.relay.sql(
            "SELECT id FROM _password_resets WHERE (used_at IS NOT NULL AND used_at <= $1) OR (expires_at <= $1)",
            cutoff,
        )
        if not rows:
            console.print("[dim]nothing to prune.[/dim]")
            return
        console.print(
            f"About to permanently delete {len(rows)} inactive password-reset row(s) older than {older_than_days} day(s)."
        )
        _confirm_or_abort(yes)

        ids = [r["id"] for r in rows]
        await arc.relay.sql("DELETE FROM _password_resets WHERE id = ANY($1::uuid[])", ids)
        console.print(f"[bold green]pruned {len(ids)} password-reset row(s).[/bold green]")

    with _friendly_errors():
        _run(_do())


# ------------------------------------------------------------------------ #
# Access keys
# ------------------------------------------------------------------------ #
@app.command(name="clear-access-key")
def clear_access_key(
    key: str = typer.Argument(None, help="A specific key's prefix (the safe-to-display 'ak_...' identifier)."),
    email: str = typer.Option(None, "--email"),
    all_: bool = typer.Option(False, "--all"),
    yes: bool = typer.Option(False, "--yes", "-y"),
) -> None:
    """Revoke access key(s) — a specific KEY prefix, --email for all of one
    user's keys, or --all for every active key in the system. Same
    revoke-not-delete distinction as clear-sessions; see prune-access-keys."""
    if sum([bool(key), bool(email), all_]) != 1:
        err_console.print("specify exactly one of KEY (prefix), --email, or --all.")
        raise typer.Exit(code=1)

    async def _do() -> None:
        if key:
            row = await arc.relay.get("_access_keys", {"key_prefix": key})
            if row is None:
                err_console.print(f"no access key with prefix '{key}'.")
                raise typer.Exit(code=1)
            keys = [row] if row["revoked_at"] is None else []
            scope_desc = f"key '{key}'"
        else:
            filters = {"revoked_at": {"is_null": True}}
            scope_desc = "ALL access keys"
            if email:
                user = await _get_user_or_exit(email)
                filters["user"] = user["id"]
                scope_desc = f"{user['email']}'s access keys"
            keys = await arc.relay.list("_access_keys", filters=filters)

        if not keys:
            console.print("[dim]no active access keys to clear.[/dim]")
            return
        console.print(f"About to revoke {len(keys)} access key(s) for {scope_desc}.")
        _confirm_or_abort(yes)

        for k in keys:
            await arc.relay.save("_access_keys", {"id": k["id"], "revoked_at": utcnow()})
            await arc.authn.invalidate_access_key_cache(k["key_prefix"])
        console.print(f"[bold green]revoked {len(keys)} access key(s).[/bold green]")

    with _friendly_errors():
        _run(_do())


@app.command(name="prune-access-keys")
def prune_access_keys(
    older_than_days: int = typer.Option(30, "--older-than-days"),
    yes: bool = typer.Option(False, "--yes", "-y"),
) -> None:
    """Same as prune-sessions, for _access_keys. A key with no expiry
    (expires_at NULL, still valid forever) and never revoked is never
    pruned, by construction — only revoked-or-expired-and-old rows qualify."""

    async def _do() -> None:
        cutoff = utcnow() - timedelta(days=older_than_days)
        rows = await arc.relay.sql(
            "SELECT id FROM _access_keys WHERE (revoked_at IS NOT NULL AND revoked_at <= $1) "
            "OR (expires_at IS NOT NULL AND expires_at <= $1)",
            cutoff,
        )
        if not rows:
            console.print("[dim]nothing to prune.[/dim]")
            return
        console.print(f"About to permanently delete {len(rows)} inactive access key row(s) older than {older_than_days} day(s).")
        _confirm_or_abort(yes)

        ids = [r["id"] for r in rows]
        await arc.relay.sql("DELETE FROM _access_keys WHERE id = ANY($1::uuid[])", ids)
        console.print(f"[bold green]pruned {len(ids)} access key row(s).[/bold green]")

    with _friendly_errors():
        _run(_do())
