"""has_roles validation — every role name a user is granted must exist in
_roles. Same pattern as example_hr/hooks/Employee.py's check_salary.

Also: cache invalidation for authn's session/access-key caches (see
authn/__init__.py's module docstring) — those caches hold a snapshot of
has_roles/status/allowed_ips, so a change to any of those on a user has to
invalidate every cache entry for that user, or a revoked role (etc.) stays
silently honored until the cache entry's TTL expires."""

import arc


@arc.relay.validate
async def check_has_roles(ctx):
    for role_name in ctx.payload.get("has_roles") or []:
        if await arc.relay.get("_roles", {"name": role_name}) is None:
            arc.relay.throw(f"no role named '{role_name}'", code="unknown_role")


@arc.relay.after_save
async def invalidate_cache_on_change(ctx):
    if ctx.doc.old is None:
        return  # brand new user — no session/access-key cache entries could exist yet
    changed = any(
        ctx.doc.old.get(field) != ctx.doc.get(field) for field in ("has_roles", "status", "allowed_ips")
    )
    if changed:
        await arc.authn.invalidate_user_cache(ctx.new["id"])
