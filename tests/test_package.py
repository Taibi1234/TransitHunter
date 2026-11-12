"""Initial package integrity tests."""

from transithunter import __version__


def test_version_is_defined() -> None:
    """The installed package exposes its development version."""
    assert __version__ == "0.1.0"