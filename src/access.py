"""M8 — Access Control & Audit.

Central authorization checkpoint. Every controller call passes through
:func:`check` with the authenticated role. Denials, allows, and small-group
suppression events are appended to the audit journal (M14).
"""

from __future__ import annotations

from typing import Literal

Role = Literal["admin", "analyst", "viewer"]

_HIERARCHY: dict[Role, int] = {"viewer": 1, "analyst": 2, "admin": 3}


def check(actor_role: Role, minimum_role: Role) -> bool:
    """Return True iff ``actor_role`` meets or exceeds ``minimum_role``."""
    return _HIERARCHY[actor_role] >= _HIERARCHY[minimum_role]


def enforce_small_group(cohort_size: int, threshold: int = 5) -> bool:
    """Return True iff a cohort is large enough to display without suppression."""
    return cohort_size >= threshold
