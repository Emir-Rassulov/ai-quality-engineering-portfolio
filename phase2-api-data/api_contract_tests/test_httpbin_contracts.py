import json

import httpx2
import pytest

pytestmark = pytest.mark.network


# ==================================================
# 1. Auth
# ==================================================

@pytest.mark.parametrize(
    ("auth", "expected_status"),
    [
        (("user", "pass"), 200),
        (("wrong", "pass"), 401),
        (("user", "wrong"), 401),
        (None, 401),  # no credentials at all
    ],
    ids=["correct", "wrong-user", "wrong-password", "no-credentials"],
)
def test_basic_auth_status(base_url, auth, expected_status):
    response = httpx2.get(
        f"{base_url}/basic-auth/user/pass", auth=auth, timeout=10.0
    )

    assert response.status_code == expected_status


def test_basic_auth_success_body(base_url):
    response = httpx2.get(
        f"{base_url}/basic-auth/user/pass", auth=("user", "pass"), timeout=10.0
    )

    assert response.json() == {"authenticated": True, "user": "user"}


def test_basic_auth_rejection_asks_for_credentials(base_url):
    """A 401 must tell the client how to authenticate."""
    response = httpx2.get(f"{base_url}/basic-auth/user/pass", timeout=10.0)

    assert "basic" in response.headers["www-authenticate"].lower()


# ==================================================
# 2. Status codes (the server side of the contract)
# ==================================================

@pytest.mark.parametrize("status_code", [200, 400, 404, 429, 500])
def test_status_endpoint_returns_the_requested_code(base_url, status_code):
    response = httpx2.get(f"{base_url}/status/{status_code}", timeout=10.0)

    assert response.status_code == status_code


# ==================================================
# 3. Slow responses
# ==================================================

def test_client_timeout_shorter_than_delay_raises_read_timeout(base_url):
    """Connected fine, but the body took too long: ReadTimeout, not ConnectTimeout."""
    with pytest.raises(httpx2.ReadTimeout):
        httpx2.get(f"{base_url}/delay/2", timeout=0.5)


def test_same_delay_succeeds_with_a_longer_timeout(base_url):
    """Guard: proves the timeout, not the server, caused the failure above."""
    response = httpx2.get(f"{base_url}/delay/1", timeout=10.0)

    assert response.status_code == 200


# ==================================================
# 4. Streaming
# ==================================================

def test_stream_yields_five_json_lines_in_order_and_ends_cleanly(base_url):
    received_ids = []

    with httpx2.Client(timeout=10.0) as client:
        with client.stream("GET", f"{base_url}/stream/5") as response:
            assert response.status_code == 200

            # One line at a time: each line is checked as it arrives.
            for line in response.iter_lines():
                assert line.strip(), "stream contained a blank line"
                payload = json.loads(line)  # raises if the line is not valid JSON
                assert isinstance(payload, dict)
                received_ids.append(payload["id"])

        # The loop ended without an error, so the stream finished,
        # and leaving the block must have closed it.
        assert response.is_closed

    assert received_ids == [0, 1, 2, 3, 4]
