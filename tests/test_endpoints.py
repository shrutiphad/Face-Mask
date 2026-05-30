from __future__ import annotations

import io
import numpy as np
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch

from app.main import create_app




FIXED_EMBEDDING = np.ones(512, dtype=np.float32) / np.sqrt(512)  # unit vector

# Tiny valid 1×1 pixel JPEG (base64-decoded inline)
# This is a real JPEG so cv2 can decode it; the face detection is mocked.
TINY_JPEG = bytes([
    0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46, 0x49, 0x46, 0x00, 0x01,
    0x01, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00, 0xFF, 0xDB, 0x00, 0x43,
    0x00, 0x08, 0x06, 0x06, 0x07, 0x06, 0x05, 0x08, 0x07, 0x07, 0x07, 0x09,
    0x09, 0x08, 0x0A, 0x0C, 0x14, 0x0D, 0x0C, 0x0B, 0x0B, 0x0C, 0x19, 0x12,
    0x13, 0x0F, 0x14, 0x1D, 0x1A, 0x1F, 0x1E, 0x1D, 0x1A, 0x1C, 0x1C, 0x20,
    0x24, 0x2E, 0x27, 0x20, 0x22, 0x2C, 0x23, 0x1C, 0x1C, 0x28, 0x37, 0x29,
    0x2C, 0x30, 0x31, 0x34, 0x34, 0x34, 0x1F, 0x27, 0x39, 0x3D, 0x38, 0x32,
    0x3C, 0x2E, 0x33, 0x34, 0x32, 0xFF, 0xC0, 0x00, 0x0B, 0x08, 0x00, 0x01,
    0x00, 0x01, 0x01, 0x01, 0x11, 0x00, 0xFF, 0xC4, 0x00, 0x1F, 0x00, 0x00,
    0x01, 0x05, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08,
    0x09, 0x0A, 0x0B, 0xFF, 0xC4, 0x00, 0xB5, 0x10, 0x00, 0x02, 0x01, 0x03,
    0x03, 0x02, 0x04, 0x03, 0x05, 0x05, 0x04, 0x04, 0x00, 0x00, 0x01, 0x7D,
    0xFF, 0xDA, 0x00, 0x08, 0x01, 0x01, 0x00, 0x00, 0x3F, 0x00, 0xFB, 0xD2,
    0x8A, 0x28, 0x03, 0xFF, 0xD9,
])


@pytest.fixture
def client():
    """
    Returns a TestClient with both FaceEmbedder and VectorStore mocked.
    The app goes through its full lifespan (startup) with patched singletons.
    """
    mock_embedder = MagicMock()
    mock_embedder.embed.return_value = FIXED_EMBEDDING

    mock_store = MagicMock()
    mock_store.count.return_value = 0
    mock_store.search.return_value = []

    with (
        patch("app.main.FaceEmbedder", return_value=mock_embedder),
        patch("app.main.VectorStore", return_value=mock_store),
    ):
        app = create_app()
        with TestClient(app) as c:
            c._mock_embedder = mock_embedder
            c._mock_store = mock_store
            yield c


#

class TestHealth:
    def test_health_returns_ok(self, client):
        client._mock_store.count.return_value = 3
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["enrolled_count"] == 3

    def test_health_has_process_time_header(self, client):
        r = client.get("/health")
        assert "X-Process-Time-Ms" in r.headers




class TestEnroll:
    def test_enroll_success(self, client):
        client._mock_store.count.return_value = 1

        r = client.post(
            "/enroll",
            files={"image": ("face.jpg", io.BytesIO(TINY_JPEG), "image/jpeg")},
            data={"identity_id": "person_01"},
        )
        assert r.status_code == 201
        body = r.json()
        assert body["identity_id"] == "person_01"
        assert body["enrolled_count"] == 1
        assert "enrolled" in body["message"].lower()

    def test_enroll_calls_upsert_with_correct_id(self, client):
        client.post(
            "/enroll",
            files={"image": ("face.jpg", io.BytesIO(TINY_JPEG), "image/jpeg")},
            data={"identity_id": "alice"},
        )
        call_args = client._mock_store.upsert.call_args
        assert call_args[0][0] == "alice"           # identity_id positional arg
        np.testing.assert_array_almost_equal(       # embedding matches mock output
            call_args[0][1], FIXED_EMBEDDING
        )

    def test_enroll_empty_file_returns_422(self, client):
        r = client.post(
            "/enroll",
            files={"image": ("empty.jpg", io.BytesIO(b""), "image/jpeg")},
            data={"identity_id": "person_01"},
        )
        assert r.status_code == 422

    def test_enroll_no_face_returns_422(self, client):
        client._mock_embedder.embed.side_effect = ValueError("No face detected")
        r = client.post(
            "/enroll",
            files={"image": ("noface.jpg", io.BytesIO(TINY_JPEG), "image/jpeg")},
            data={"identity_id": "person_01"},
        )
        assert r.status_code == 422
        assert "No face detected" in r.json()["detail"]

    def test_enroll_missing_identity_id_returns_422(self, client):
        r = client.post(
            "/enroll",
            files={"image": ("face.jpg", io.BytesIO(TINY_JPEG), "image/jpeg")},
            # identity_id field missing entirely
        )
        assert r.status_code == 422

    def test_enroll_upsert_semantics(self, client):
        """Re-enrolling the same id twice should call upsert twice — no error."""
        for _ in range(2):
            r = client.post(
                "/enroll",
                files={"image": ("face.jpg", io.BytesIO(TINY_JPEG), "image/jpeg")},
                data={"identity_id": "person_01"},
            )
            assert r.status_code == 201
        assert client._mock_store.upsert.call_count == 2




class TestSearch:
    def _setup_hit(self, client, identity_id: str, score: float):
        client._mock_store.count.return_value = 5
        client._mock_store.search.return_value = [
            {"identity_id": identity_id, "score": score}
        ]

    def test_search_match_above_threshold(self, client):
        self._setup_hit(client, "person_01", 0.72)
        r = client.post(
            "/search",
            files={"image": ("query.jpg", io.BytesIO(TINY_JPEG), "image/jpeg")},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["is_match"] is True
        assert body["top_match"]["identity_id"] == "person_01"
        assert body["top_match"]["score"] == pytest.approx(0.72, abs=1e-4)
        assert body["threshold_used"] == pytest.approx(0.35, abs=1e-4)
        assert body["latency_ms"] >= 0

    def test_search_hard_negative_below_threshold(self, client):
        """
        Look-alike scores 0.20 — below our 0.35 threshold — should be rejected.
        This is the hard-negative test case.
        """
        self._setup_hit(client, "person_02", 0.20)
        r = client.post(
            "/search",
            files={"image": ("lookalike.jpg", io.BytesIO(TINY_JPEG), "image/jpeg")},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["is_match"] is False
        assert body["top_match"]["score"] == pytest.approx(0.20, abs=1e-4)

    def test_search_exactly_at_threshold_is_match(self, client):
        """Boundary condition: score == threshold → is_match should be True."""
        self._setup_hit(client, "person_03", 0.35)
        r = client.post(
            "/search",
            files={"image": ("face.jpg", io.BytesIO(TINY_JPEG), "image/jpeg")},
        )
        assert r.json()["is_match"] is True

    def test_search_empty_collection_returns_422(self, client):
        client._mock_store.count.return_value = 0
        r = client.post(
            "/search",
            files={"image": ("face.jpg", io.BytesIO(TINY_JPEG), "image/jpeg")},
        )
        assert r.status_code == 422
        assert "No faces enrolled" in r.json()["detail"]

    def test_search_no_face_in_query_returns_422(self, client):
        client._mock_store.count.return_value = 5
        client._mock_embedder.embed.side_effect = ValueError("No face detected")
        r = client.post(
            "/search",
            files={"image": ("blank.jpg", io.BytesIO(TINY_JPEG), "image/jpeg")},
        )
        assert r.status_code == 422

    def test_search_response_includes_enrolled_count(self, client):
        self._setup_hit(client, "person_01", 0.55)
        r = client.post(
            "/search",
            files={"image": ("face.jpg", io.BytesIO(TINY_JPEG), "image/jpeg")},
        )
        assert r.json()["enrolled_count"] == 5

    def test_search_passes_ef_search_to_vector_store(self, client):
        """
        Verify that search() is called (not a Python loop).
        If the unit test shows search() was never called, we caught a regression.
        """
        self._setup_hit(client, "person_01", 0.55)
        client.post(
            "/search",
            files={"image": ("face.jpg", io.BytesIO(TINY_JPEG), "image/jpeg")},
        )
        assert client._mock_store.search.called, (
            "VectorStore.search() was not called — this means the endpoint "
            "is using a Python loop instead of Qdrant ANN.  Fix this."
        )