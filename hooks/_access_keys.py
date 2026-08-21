"""scopes validation — an access key's scopes must be a subset of its
owning user's has_roles. Shares the actual subset-check with
api/auth_api.py's create_access_key() via authn.has_roles_subset, rather
than duplicating the logic."""

import arc

from authn import SUPERUSER_ROLE_NAME, has_roles_subset


@arc.relay.validate
async def check_scopes(ctx):
    if "scopes" not in ctx.payload:
        return
    scopes = ctx.payload.get("scopes") or []
    # Enforced HERE, not only at the two HTTP call sites (auth_api.py's
    # own create_access_key, admin's access_keys_api.py) — this hook is
    # the one gate every write to _access_keys crosses (CLI, a direct
    # arc.relay.save, either HTTP endpoint), and the invariant is
    # absolute: a Superuser's own identity.roles legitimately contains
    # "*" (resolve_identity injects it for a session), which makes
    # has_roles_subset(["Superuser"], identity.roles) pass the check
    # below — an access key is a weaker, more exposed credential than a
    # session and must never be able to carry the bypass regardless of
    # how privileged the creating user is.
    if "*" in scopes or SUPERUSER_ROLE_NAME in scopes:
        arc.relay.throw(
            "scopes cannot include the Superuser bypass — access keys may never carry it",
            code="scopes_forbidden",
        )
    user_id = ctx.doc.user
    user = await arc.relay.get("_users", user_id, ["id", "has_roles"])
    if user is None:
        arc.relay.throw("access key must reference an existing user", code="unknown_user")
    if not has_roles_subset(scopes, user.get("has_roles")):
        arc.relay.throw(
            "scopes must be a subset of the owning user's has_roles", code="invalid_scopes"
        )
