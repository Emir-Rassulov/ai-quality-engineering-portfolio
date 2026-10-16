import os

import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "network: calls a live httpbin server (deselect with -m 'not network')",
    )


@pytest.fixture(scope="session")
def base_url():
    """
    httpbin address. Defaults to the public server; set HTTPBIN_URL to point
    the same tests at a local copy (the public one is sometimes slow or down).
    """
    return os.environ.get("HTTPBIN_URL", "https://httpbin.org").rstrip("/")
