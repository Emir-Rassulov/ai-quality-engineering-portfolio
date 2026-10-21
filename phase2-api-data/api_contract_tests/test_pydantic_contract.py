import httpx2
import pytest
from pydantic import ValidationError

from models import BasicAuthResponse


@pytest.mark.network
def test_live_response_matches_the_contract(base_url):
    response = httpx2.get(
        f"{base_url}/basic-auth/user/pass", auth=("user", "pass"), timeout=10.0
    )

    parsed = BasicAuthResponse.model_validate(response.json())

    assert parsed.authenticated is True
    assert parsed.user == "user"


@pytest.mark.parametrize(
    ("payload", "error_type", "field"),
    [
        ({"authenticated": True}, "missing", "user"),
        ({"user": "user"}, "missing", "authenticated"),
        ({"authenticated": "true", "user": "user"}, "bool_type", "authenticated"),
        ({"authenticated": True, "user": 123}, "string_type", "user"),
    ],
    ids=[
        "missing-user",
        "missing-authenticated",
        "string-instead-of-bool",
        "int-instead-of-string",
    ],
)
def test_malformed_payload_is_rejected_for_the_right_reason(
    payload, error_type, field
):
    with pytest.raises(ValidationError) as exc_info:
        BasicAuthResponse.model_validate(payload)

    errors = exc_info.value.errors()
    assert [(e["type"], e["loc"]) for e in errors] == [(error_type, (field,))]


def test_strict_mode_rejects_a_value_that_lax_mode_would_quietly_convert():
    """Why strict matters: lax mode turns the string "true" into True."""
    payload = {"authenticated": "true", "user": "user"}

    assert BasicAuthResponse.model_validate(payload, strict=False).authenticated is True
    with pytest.raises(ValidationError):
        BasicAuthResponse.model_validate(payload)
