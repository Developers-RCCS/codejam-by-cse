import pytest
import time
from web import app

@pytest.fixture

def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


def test_ask_endpoint_structure(client):
    response = client.post('/ask', json={
        'query': 'What is history?',
        'chat_history': [],
        'language': 'english'
    })
    assert response.status_code == 200
    data = response.get_json()
    assert 'answer' in data
    assert 'translated' in data
    assert 'chat_history' in data


def test_ask_response_time(client):
    start = time.time()
    response = client.post('/ask', json={
        'query': 'Who was Alexander the Great?',
        'chat_history': [],
        'language': 'english'
    })
    duration = time.time() - start
    assert response.status_code == 200
    # ensure response time is under 3 seconds
    assert duration < 3


def test_translate_endpoint(client):
    response = client.post('/translate', json={
        'text': 'Hello',
        'target_language': 'sinhala'
    })
    assert response.status_code == 200
    data = response.get_json()
    assert 'translated_text' in data
    # translated text should not be empty
    assert data['translated_text'] != ''