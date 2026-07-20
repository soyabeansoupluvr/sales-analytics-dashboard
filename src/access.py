"""M8 - Access Control and Audit.

Provides the dashboard's authorization checkpoint. Controllers call this module before reading
views or aggregating cohorts. Allow, deny, and small-group suppression decisions are written to
the M14 audit log.

Entry points:

* check compares two roles without writing an audit event.
* authorize checks whether an actor can access a view and audits the decision.
* enforce_small_group suppresses cohorts below the configured threshold and audits the suppression.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Literal

from src.logs import AuditAction, AuditEvent, AuditLog, AuditOutcome, Database

if TYPE_CHECKING:
    from src.config import Settings


# --------------------------------------------------------------------------- #
# Roles and views
# --------------------------------------------------------------------------- #

Role = Literal["admin", "analyst", "viewer"]

# The dashboard exposes four canonical views. The manager persona (aka "viewer") sees the
# aggregate views only; per-customer breakdowns require analyst or admin.
ViewName = Literal["revenue", "products", "time", "customers"]

_ROLE_RANK: dict[Role, int] = {"viewer": 1, "analyst": 2, "admin": 3}

_VIEW_MINIMUM_ROLE: dict[ViewName, Role] = {
    "revenue": "viewer",
    "products": "viewer",
    "time": "viewer",
    "customers": "analyst",
}


# --------------------------------------------------------------------------- #
# Value objects
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class Actor:
    """The authenticated caller on whose behalf an action is being taken."""

    username: str
    role: Role

    def __post_init__(self) -> None:
        if not isinstance(self.username, str) or not self.username:
            raise ValueError("Actor.username must be a non-empty string")
        if self.role not in _ROLE_RANK:
            raise ValueError(f"Actor.role must be one of {sorted(_ROLE_RANK)}")


@dataclass(frozen=True, slots=True)
class Decision:
    """Result of an authorization check.

    reason is a short code such as role_ok, role_insufficient, or small_group_suppressed.
    It is written to the audit log so decisions can be filtered by cause.
    """

    allowed: bool
    reason: str


class AccessError(Exception):
    """Raised for M8 contract violations (unknown view, malformed actor)."""


# --------------------------------------------------------------------------- #
# Low-level predicate
# --------------------------------------------------------------------------- #


def check(actor_role: Role, minimum_role: Role) -> bool:
    """Return whether actor_role meets the required role.

    This is a pure predicate and does not write to the audit log. Use authorize when the check
    guards a real action.
    """

    if actor_role not in _ROLE_RANK:
        raise AccessError(f"unknown actor role: {actor_role!r}")
    if minimum_role not in _ROLE_RANK:
        raise AccessError(f"unknown minimum role: {minimum_role!r}")
    return _ROLE_RANK[actor_role] >= _ROLE_RANK[minimum_role]


# --------------------------------------------------------------------------- #
# Full checkpoint
# --------------------------------------------------------------------------- #


def authorize(
    actor: Actor,
    view: ViewName,
    settings: "Settings",
    *,
    audit_log: AuditLog | None = None,
) -> Decision:
    """Authorize an actor to read a view and audit the decision.

    Args:
        actor: Authenticated caller.
        view: Canonical view name the caller wants to read.
        settings: Application settings used to open the audit database.
        audit_log: Optional audit log injection point for tests.

    Returns:
        Decision with allowed set to True when the actor's role meets the view's minimum role.

    Notes:
        Each call writes one audit entry. Allowed reads are recorded as PROTECTED_STORE_READ,
        and denied reads are recorded as ACCESS_DENIED.
    """

    if view not in _VIEW_MINIMUM_ROLE:
        raise AccessError(f"unknown view: {view!r}")

    minimum_role = _VIEW_MINIMUM_ROLE[view]
    allowed = check(actor.role, minimum_role)
    decision = Decision(
        allowed=allowed,
        reason="role_ok" if allowed else "role_insufficient",
    )

    log = audit_log if audit_log is not None else AuditLog(Database(settings))
    log.append(
        AuditEvent(
            timestamp=datetime.now(timezone.utc),
            actor=actor.username,
            action=(AuditAction.PROTECTED_STORE_READ if allowed else AuditAction.ACCESS_DENIED),
            outcome=AuditOutcome.SUCCESS if allowed else AuditOutcome.FAILURE,
            resource=view,
            details={
                "actor_role": actor.role,
                "minimum_role": minimum_role,
                "reason": decision.reason,
            },
        )
    )
    return decision


# --------------------------------------------------------------------------- #
# k-anonymity / small-group suppression
# --------------------------------------------------------------------------- #


def enforce_small_group(
    cohort_size: int,
    settings: "Settings | None" = None,
    *,
    threshold: int | None = None,
    actor: Actor | None = None,
    resource: str | None = None,
    audit_log: AuditLog | None = None,
) -> bool:
    """Return whether a cohort meets the small-group threshold.

    Args:
        cohort_size: Number of distinct customers in the aggregated cohort.
        settings: Application settings. Required unless threshold is supplied.
        threshold: Explicit threshold override.
        actor: Actor requesting the aggregation. When supplied for a suppressed
            cohort, the denial is audited.
        resource: Resource label recorded on the audit event.
        audit_log: Optional audit log injection point for tests.

    Returns:
        True when cohort_size is greater than or equal to the threshold.

    Notes:
        Suppressed cohorts with an actor are audited as ACCESS_DENIED with reason
        small_group_suppressed. Checks without an actor are silent.
    """

    if threshold is None:
        if settings is None:
            raise AccessError("enforce_small_group requires either settings or threshold")
        threshold = settings.small_group_threshold

    if threshold <= 0:
        raise AccessError(f"threshold must be positive, got {threshold}")
    if cohort_size < 0:
        raise AccessError(f"cohort_size must be non-negative, got {cohort_size}")

    allowed = cohort_size >= threshold
    if allowed or actor is None:
        return allowed

    # Suppressed cohort with a known actor: audit the denial.
    if settings is None:  # pragma: no cover - guarded above via threshold path
        return allowed
    log = audit_log if audit_log is not None else AuditLog(Database(settings))
    log.append(
        AuditEvent(
            timestamp=datetime.now(timezone.utc),
            actor=actor.username,
            action=AuditAction.ACCESS_DENIED,
            outcome=AuditOutcome.FAILURE,
            resource=resource,
            details={
                "actor_role": actor.role,
                "reason": "small_group_suppressed",
                "cohort_size": cohort_size,
                "threshold": threshold,
            },
        )
    )
    return allowed


__all__ = [
    "AccessError",
    "Actor",
    "Decision",
    "Role",
    "ViewName",
    "authorize",
    "check",
    "enforce_small_group",
]
