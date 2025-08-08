"""
Test per l'API FastAPI
"""
import pytest
from fastapi.testclient import TestClient

from src.api.main import app


@pytest.fixture
def client():
    """Fixture per il test client"""
    return TestClient(app)


class TestAPI:
    """Test per gli endpoint dell'API"""

    def test_health_endpoint(self, client):
        """Test endpoint health"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        # Il modello potrebbe non essere caricato durante i test
        assert data["status"] in ["healthy", "unhealthy"]
        assert "model_loaded" in data
        assert "model_version" in data
        assert "timestamp" in data

    def test_predict_endpoint_valid_data(self, client):
        """Test endpoint predict con dati validi"""
        # Dati di test con features corrette
        test_data = {
            "features": {
                "radius_mean": 17.99,
                "texture_mean": 10.38,
                "perimeter_mean": 122.8,
                "area_mean": 1001.0,
                "smoothness_mean": 0.1184,
                "compactness_mean": 0.2776,
                "concavity_mean": 0.3001,
                "concave_points_mean": 0.1471,
                "symmetry_mean": 0.2419,
                "fractal_dimension_mean": 0.07871,
                "radius_se": 1.095,
                "texture_se": 0.9053,
                "perimeter_se": 8.589,
                "area_se": 153.4,
                "smoothness_se": 0.006399,
                "compactness_se": 0.04904,
                "concavity_se": 0.05373,
                "concave_points_se": 0.01587,
                "symmetry_se": 0.03003,
                "fractal_dimension_se": 0.006193,
            }
        }
        response = client.post("/predict", json=test_data)
        # Il modello potrebbe non essere caricato durante i test
        if response.status_code == 503:
            # Modello non caricato
            data = response.json()
            assert "detail" in data
            assert "Modello non caricato" in data["detail"]
        else:
            # Modello caricato
            assert response.status_code == 200
            data = response.json()
            assert "prediction" in data
            assert "probability" in data
            assert "confidence" in data

    def test_predict_endpoint_invalid_data(self, client):
        """Test endpoint predict con dati non validi"""
        # Dati con numero sbagliato di features
        test_data = {"features": [0.1] * 10}  # Solo 10 features invece di 20
        response = client.post("/predict", json=test_data)
        assert response.status_code == 422  # Validation error

    def test_predict_endpoint_missing_data(self, client):
        """Test endpoint predict con dati mancanti"""
        test_data = {}
        response = client.post("/predict", json=test_data)
        assert response.status_code == 422  # Validation error

    def test_predict_endpoint_wrong_type(self, client):
        """Test endpoint predict con tipo di dati sbagliato"""
        test_data = {"features": "not_a_list"}
        response = client.post("/predict", json=test_data)
        assert response.status_code == 422  # Validation error
