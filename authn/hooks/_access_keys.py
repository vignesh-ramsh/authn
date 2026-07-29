"""scopes validation — an access key's scopes must be a subset of its
owning user's has_roles. Shares the actual subset-check with
api/auth_api.py's create_access_key() via authn.has_roles_subset, rather
than duplicating the logic."""

import arc

from authn import has_roles_subset


@arc.relay.validate
async def check_scopes(ctx):
    if "scopes" not in ctx.payload:
        return
    user_id = ctx.doc.user
    user = await arc.relay.get("_users", user_id, ["id", "has_roles"])
    if user is None:
        arc.relay.throw("access key must reference an existing user", code="unknown_user")
    if not has_roles_subset(ctx.payload.get("scopes"), user.get("has_roles")):
        arc.relay.throw(
            "scopes must be a subset of the owning user's has_roles", code="invalid_scopes"
        )
