import pytest
import time
from web import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


def test_homepage_structure(client):
    # Ensure the homepage loads and contains the React mount point and CSS link
    response = client.get('/')
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert '<div id="react-chat-app"></div>' in html
    assert 'static/css/chat.css' in html
    # Check that Babel script is included
    assert 'babel.min.js' in html


def test_homepage_response_time(client):
    # Measure GET / performance
    start = time.time()
    response = client.get('/')
    duration = time.time() - start
    assert response.status_code == 200
    # Ensure response time is under 1 second
    assert duration < 1
