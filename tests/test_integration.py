import os
from fastapi.testclient import TestClient
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.app import app

client = TestClient(app)

def test_database_connection():
    mongo_uri = os.getenv("MONGO_URI")
    assert mongo_uri is not None and mongo_uri.startswith("mongodb")
    
    response = client.get("/status")  # exemplo: rota que verifica conexão
    assert response.status_code == 200
