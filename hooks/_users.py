"""has_roles validation — every role name a user is granted must exist in
_roles. Same pattern as example_hr/hooks/Employee.py's check_salary.

Also: cache invalidation for authn's session/access-key caches (see
authn/__init__.py's module docstring) — those caches hold a snapshot of
has_roles/status/allowed_ips, so a change to any of those on a user has to
invalidate every cache entry for that user, or a revoked role (etc.) stays
silently honored until the cache entry's TTL expires."""

import arc

# Case-insensitive — checked against the lowercased, stripped value below,
# so "Admin"/"ADMIN"/" admin " are all caught too. These are the identities
# a self-hosted deployment's own bootstrap/ops tooling and conventions
# already assume are special; letting a user claim one as their login
# username would be confusing at best.
_RESERVED_USERNAMES = frozenset({"admin", "administrator", "guest"})


@arc.relay.validate
async def check_has_roles(ctx):
    for role_name in ctx.payload.get("has_roles") or []:
        if await arc.relay.get("_roles", {"name": role_name}, ["id"]) is None:
            arc.relay.throw(f"no role named '{role_name}'", code="unknown_role")


@arc.relay.validate
async def check_username(ctx):
    """Normalizes and validates `username` whenever it's part of this
    write. Lowercased the same way `email` already is (auth_api.login) —
    both are login identifiers, and case-insensitive-by-convention avoids
    "JohnDoe" and "johndoe" silently being different rows. Unique at the DB
    level (schemas/_users.json) tolerates any number of NULLs, so a user
    with no username set is unaffected — this only runs when a write
    actually touches the field."""
    if "username" not in ctx.payload:
        return
    username = ctx.payload["username"]
    if username is None:
        return  # explicit clear — fine
    username = username.strip()
    if not username:
        arc.relay.throw(
            "username cannot be blank — omit the field (or send null) to leave it unset",
            code="invalid_username",
        )
    if any(ch.isspace() for ch in username):
        arc.relay.throw("username cannot contain whitespace", code="invalid_username")
    if username.lower() in _RESERVED_USERNAMES:
        arc.relay.throw(
            f"'{username}' is a reserved username and cannot be used", code="reserved_username"
        )
    ctx.payload["username"] = username.lower()


@arc.relay.after_save
async def invalidate_cache_on_change(ctx):
    if ctx.doc.old is None:
        return  # brand new user — no session/access-key cache entries could exist yet
    changed = any(
        ctx.doc.old.get(field) != ctx.doc.get(field)
        for field in ("has_roles", "status", "allowed_ips")
    )
    if changed:
        await arc.authn.invalidate_user_cache(ctx.new["id"])
