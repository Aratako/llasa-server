"""FastAPI APIエンドポイントのテスト"""

from io import BytesIO
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from llasa_server.api import app


@pytest.fixture
def client():
    """TestClientのフィクスチャ"""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(autouse=True)
def reset_mock(mock_tts_instance):
    """各テスト後にモックをリセット"""
    yield
    mock_tts_instance.reset_mock()
    # side_effectもクリア
    mock_tts_instance.generate_speech.side_effect = None


class TestHealthEndpoint:
    """ヘルスチェックエンドポイントのテスト"""

    def test_health_check_without_server(self, client):
        """サーバー未初期化時のヘルスチェック"""
        with patch("llasa_server.api.tts_server", None):
            response = client.get("/health")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "ok"
            assert data["server_initialized"] is False

    def test_health_check_with_server(self, client, mock_tts_instance):
        """サーバー初期化済み時のヘルスチェック"""
        with patch("llasa_server.api.tts_server", mock_tts_instance):
            response = client.get("/health")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "ok"
            assert data["server_initialized"] is True


class TestTTSEndpoint:
    """TTSエンドポイントのテスト"""

    def test_tts_without_server_initialization(self, client):
        """サーバー未初期化時のTTSリクエスト"""
        with patch("llasa_server.api.tts_server", None):
            response = client.post("/tts", data={"text": "こんにちは"})
            assert response.status_code == 503
            assert "初期化されていません" in response.json()["detail"]

    def test_tts_simple_text(self, client, mock_tts_instance):
        """シンプルなテキストでのTTS生成"""
        with patch("llasa_server.api.tts_server", mock_tts_instance):
            response = client.post("/tts", data={"text": "こんにちは"})
            assert response.status_code == 200
            assert response.headers["content-type"] == "audio/wav"
            assert (
                "filename=generated_speech.wav"
                in response.headers["content-disposition"]
            )

            # generate_speechが正しいパラメータで呼ばれたことを確認
            mock_tts_instance.generate_speech.assert_called_once()
            call_kwargs = mock_tts_instance.generate_speech.call_args.kwargs
            assert call_kwargs["text"] == "こんにちは"
            assert call_kwargs["reference_audio_path"] is None
            assert call_kwargs["reference_text"] is None
            assert call_kwargs["temperature"] == 0.8
            assert call_kwargs["top_p"] == 1.0
            assert call_kwargs["repetition_penalty"] == 1.1
            assert call_kwargs["max_tokens"] == 2048

    def test_tts_with_custom_parameters(self, client, mock_tts_instance):
        """カスタムパラメータでのTTS生成"""
        with patch("llasa_server.api.tts_server", mock_tts_instance):
            response = client.post(
                "/tts",
                data={
                    "text": "テスト",
                    "temperature": 0.9,
                    "top_p": 0.95,
                    "repetition_penalty": 1.2,
                    "max_tokens": 1024,
                },
            )
            assert response.status_code == 200

            # パラメータが正しく渡されたことを確認
            call_kwargs = mock_tts_instance.generate_speech.call_args.kwargs
            assert call_kwargs["text"] == "テスト"
            assert call_kwargs["temperature"] == 0.9
            assert call_kwargs["top_p"] == 0.95
            assert call_kwargs["repetition_penalty"] == 1.2
            assert call_kwargs["max_tokens"] == 1024

    def test_tts_with_reference_audio(self, client, mock_tts_instance):
        """リファレンス音声付きでのTTS生成"""
        with patch("llasa_server.api.tts_server", mock_tts_instance):
            # モックの音声ファイルを作成
            fake_audio = b"fake audio content"
            files = {
                "reference_audio": ("reference.wav", BytesIO(fake_audio), "audio/wav")
            }
            data = {"text": "こんにちは", "reference_text": "リファレンス"}

            response = client.post("/tts", data=data, files=files)
            assert response.status_code == 200

            # generate_speechが呼ばれたことを確認
            mock_tts_instance.generate_speech.assert_called_once()
            call_kwargs = mock_tts_instance.generate_speech.call_args.kwargs
            assert call_kwargs["text"] == "こんにちは"
            assert call_kwargs["reference_text"] == "リファレンス"
            # reference_audio_pathが設定されていることを確認（実際のパスは一時ファイル）
            assert call_kwargs["reference_audio_path"] is not None

    def test_tts_missing_text(self, client):
        """テキストなしでのTTSリクエスト"""
        response = client.post("/tts", data={})
        assert response.status_code == 422  # Validation error

    def test_tts_generation_error(self, client, mock_tts_instance):
        """音声生成時のエラー処理"""
        # generate_speechでエラーを発生させる
        mock_tts_instance.generate_speech.side_effect = RuntimeError("生成エラー")
        with patch("llasa_server.api.tts_server", mock_tts_instance):
            response = client.post("/tts", data={"text": "こんにちは"})
            assert response.status_code == 500
            assert "音声生成エラー" in response.json()["detail"]

    def test_tts_response_is_valid_audio(self, client, mock_tts_instance):
        """レスポンスが有効な音声データであることを確認"""
        with patch("llasa_server.api.tts_server", mock_tts_instance):
            response = client.post("/tts", data={"text": "こんにちは"})
            assert response.status_code == 200
            assert len(response.content) > 0
            # WAVファイルのヘッダーを確認（RIFFヘッダー）
            assert response.content[:4] == b"RIFF"


class TestAPIDocumentation:
    """API ドキュメンテーションのテスト"""

    def test_openapi_schema_available(self, client):
        """OpenAPIスキーマが利用可能であることを確認"""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        schema = response.json()
        assert "openapi" in schema
        assert "paths" in schema
        assert "/tts" in schema["paths"]
        assert "/health" in schema["paths"]
