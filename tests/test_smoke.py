"""Smoke tests for the scaffold.

These tests verify that every module can be imported without side effects.
Concrete unit tests for each M-module are added alongside the corresponding
feature branches (feature/m6-pseudonymize, feature/m9-ingestion, ...).
"""

import importlib

import pytest


@pytest.mark.parametrize(
    "module",
    [
        "src.config",
        "src.ingestion",
        "src.cleaning",
        "src.pseudonymize",
        "src.storage",
        "src.access",
        "src.analytics",
        "src.visualization",
        "src.logs",
    ],
)
def test_module_imports(module: str) -> None:
    importlib.import_module(module)


def test_access_hierarchy() -> None:
    from src import access

    assert access.check("admin", "viewer") is True
    assert access.check("viewer", "admin") is False
    assert access.enforce_small_group(4, threshold=5) is False
    assert access.enforce_small_group(5, threshold=5) is True
