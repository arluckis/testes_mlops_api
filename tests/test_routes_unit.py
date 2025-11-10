from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import pytest

from app.app import app

client = TestClient(app)

# ------------------------------
# TESTES UNITÁRIOS DAS ROTAS
# ------------------------------

def test_root_route():
    """Deve retornar status 200 e mensagem da aplicação"""
    response = client.get("/")
    assert response.status_code == 200
    body = response.json()
    assert "message" in body
    assert "Basic ML App" in body["message"]


def test_predict_route_mocked():
    """
    Testa a rota /predict sem precisar carregar modelo real nem acessar banco.
    Usa MagicMock para simular comportamento do modelo.
    """
    fake_model = MagicMock()
    fake_model.predict.return_value = ("intent_teste", {"intent_teste": 1.0})

    # Mock do get_mongo_collection para evitar inserção real e ObjectId
    fake_collection = MagicMock()
    fake_collection.insert_one.return_value = {"_id": "fake_id_123"}

    with patch("app.app.get_mongo_collection", return_value=fake_collection):
        with patch("app.app.MODELS", {"mock_model": fake_model}):
            r = client.post("/predict", params={"text": "olá"})
            assert r.status_code == 200
            data = r.json()
            assert "predictions" in data
            assert "mock_model" in data["predictions"]
            assert data["predictions"]["mock_model"]["top_intent"] == "intent_teste"


@patch("app.app.get_mongo_collection")
def test_predict_inserts_into_db(mock_get_collection):
    """
    Garante que a função collection.insert_one() é chamada ao prever algo.
    """
    fake_collection = MagicMock()
    mock_get_collection.return_value = fake_collection

    fake_model = MagicMock()
    fake_model.predict.return_value = ("intent_x", {"intent_x": 1.0})

    with patch("app.app.MODELS", {"mock_model": fake_model}):
        r = client.post("/predict", params={"text": "oi"})
        assert r.status_code == 200

    # Verifica se insert_one foi chamado
    fake_collection.insert_one.assert_called()
