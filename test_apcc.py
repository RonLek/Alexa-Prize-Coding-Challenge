import os
import tempfile

import pytest
from server import create_app

@pytest.fixture
def client():
    app = create_app()
    app.config['TESTING'] = True

    with app.app_context():
        with app.test_client() as client:
            yield client

def test_status(client):
    """Start with a blank database."""

    rv = client.get('/status')
    assert b'alive' in rv.status
