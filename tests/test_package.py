"""Package integrity tests."""

from transithunter import __version__


def test_version_is_defined() -> None:
    """The installed package exposes its release version."""
    assert __version__ == "1.0.0"