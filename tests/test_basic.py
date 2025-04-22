"""
Basic tests for spryx_http package.
"""

import os


def test_package_structure():
    """Test that the package files exist."""
    assert os.path.isdir("spryx_http")
    assert os.path.isfile("spryx_http/__init__.py")


def test_package_modules():
    """Test that the core modules exist."""
    module_files = [
        "base.py",
        "auth.py",
        "pagination.py",
        "exceptions.py",
        "retry.py",
        "settings.py",
    ]
    for module in module_files:
        assert os.path.isfile(
            f"spryx_http/{module}"
        ), f"Module file {module} should exist"
