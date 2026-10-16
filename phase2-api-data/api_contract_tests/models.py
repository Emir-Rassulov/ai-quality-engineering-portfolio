from pydantic import BaseModel, ConfigDict


class BasicAuthResponse(BaseModel):
    """
    Contract for GET /basic-auth/{user}/{passwd} on success.

    strict=True: "true" is not a bool and "1" is not an int. If the service
    quietly changes a field's type, the contract test fails.
    Extra fields are ignored on purpose: adding a field is not a breaking change.
    """

    model_config = ConfigDict(strict=True)

    authenticated: bool
    user: str
