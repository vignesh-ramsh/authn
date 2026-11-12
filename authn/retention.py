"""Deletes the rows of authn's own tables that nobody needs any more, every night.

Only `_job_log` and soft-deleted files were ever purged; `_sessions` (one row per login), `_password_resets`
and `_impersonation_tickets` grew for ever. Each table has a setting for how long a finished row is kept, and
`0` keeps everything:

  * `_sessions`: revoked or expired for more than `authn_session_retention_days` (default 30)
  * `_password_resets`, `_impersonation_tickets`: used or expired for more than `authn_token_retention_days`
    (default 7)

A session or token that is still live is never touched, whatever the setting says. Deletes run in batches, one
short statement each, so no long lock is held and no table's worth of ids sits in memory.
"""

from __future__ import annotations

import arc

SESSION_RETENTION_DAYS_KEY = "authn_session_retention_days"
TOKEN_RETENTION_DAYS_KEY = "authn_token_retention_days"

#: rows deleted per statement
BATCH_SIZE = 5_000

#: table -> (the setting holding its retention in days, the condition for "finished"; $1 is the cutoff).
#: A closed list: the table names are interpolated into SQL.
_PURGES: dict[str, tuple[str, str]] = {
    "_sessions": (SESSION_RETENTION_DAYS_KEY, "revoked_at < $1 OR expires_at < $1"),
    "_password_resets": (TOKEN_RETENTION_DAYS_KEY, "used_at < $1 OR expires_at < $1"),
    "_impersonation_tickets": (TOKEN_RETENTION_DAYS_KEY, "used_at < $1 OR expires_at < $1"),
}


async def _delete_in_batches(table: str, finished: str, cutoff) -> int:
    deleted = 0
    while True:
        rows = await arc.relay.sql(
            f'DELETE FROM "{table}" WHERE id IN (SELECT id FROM "{table}" WHERE {finished} LIMIT $2) RETURNING id',
            cutoff,
            BATCH_SIZE,
        )
        deleted += len(rows)
        if len(rows) < BATCH_SIZE:
            return deleted


async def purge_finished_auth_rows() -> dict[str, int]:
    """Registered as a nightly task (02:10, queue `default`) by authn's register(). How many rows were deleted from
    each table; a table whose setting is 0 (or less) is left alone."""
    result: dict[str, int] = {}
    for table, (setting, finished) in _PURGES.items():
        days = int(arc.settings.get(setting) or 0)
        result[table] = await _delete_in_batches(table, finished, arc.tz.ago(days=days)) if days > 0 else 0
    return result
