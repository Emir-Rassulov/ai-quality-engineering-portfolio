import pytest


# Test 1: Get post by ID
@pytest.mark.parametrize("post_id", [1, 2, 3])

def test_get_post_by_id(api_session, base_url, post_id):
    response = api_session.get(
        f"{base_url}/posts/{post_id}",
        timeout=5
    )

    assert response.status_code == 200


# Test 2: Validate post structure
def test_get_post_has_required_fields(api_session, base_url):
    response = api_session.get(
        f"{base_url}/posts/1",
        timeout=5
    )

    post = response.json()

    assert {"userId", "id", "title", "body"}.issubset(post.keys())


# Test 3: Unknown endpoint returns 404
def test_unknown_endpoint_returns_404(api_session, base_url):
    response = api_session.get(
        f"{base_url}/unknown-endpoint",
        timeout=5
    )

    assert response.status_code == 404

# Test 4: Create a new post
def test_create_post(api_session, base_url):
    """Verify that creating a post returns the expected data.

    JSONPlaceholder does not persist changes, so this test validates
    the creation response rather than verifying persisted state.
    """
    payload = {
        "title": "QA test",
        "body": "Testing POST request",
        "userId": 1
    }

    response = api_session.post(
        f"{base_url}/posts",
        json=payload,
        timeout=5
    )

    assert response.status_code == 201
    assert response.json()["title"] == payload["title"]

# Test 5: Update an existing post
def test_update_post(api_session, base_url):
    payload = {
        "id": 1,
        "title": "Updated QA test",
        "body": "Updated post body",
        "userId": 1
    }

    response = api_session.put(
        f"{base_url}/posts/1",
        json=payload,
        timeout=5
    )

    assert response.status_code == 200

# Test 6: Delete an existing post
def test_delete_post(api_session, base_url):
    response = api_session.delete(
        f"{base_url}/posts/1",
        timeout=5
    )

    assert response.status_code == 200