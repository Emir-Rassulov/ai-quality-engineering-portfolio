import pytest
import requests


@pytest.fixture(scope="session")
def api_session():
    session = requests.Session()
    session.headers.update({
        "Content-Type": "application/json"
    })

    yield session

    session.close()


@pytest.fixture(scope="session")
def base_url():
    return "https://jsonplaceholder.typicode.com"