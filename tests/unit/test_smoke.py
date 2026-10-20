"""Smoke tests: verify the package imports and exposes __version__."""

from uacp import __version__


def test_version_exists() -> None:
    """__version__ should be a non-empty string."""
    assert isinstance(__version__, str)
    assert len(__version__) > 0


def test_version_format() -> None:
    """__version__ should look like semver (X.Y.Z)."""
    parts = __version__.split(".")
    assert len(parts) == 3
    assert all(part.isdigit() for part in parts)
