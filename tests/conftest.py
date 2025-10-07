"""pytest設定とグローバルフィクスチャ

LlasaTTSServerのインスタンス化時にモデルがロードされないようにモックします。
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest


@pytest.fixture(scope="session")
def mock_tts_instance():
    """モックTTSサーバーインスタンス（全テストで共有）"""
    mock_instance = MagicMock()
    mock_instance.generate_speech = MagicMock(
        return_value=(
            np.zeros(16000, dtype=np.float32),  # 1秒のサイレント音声
            16000,  # サンプルレート
        )
    )
    return mock_instance


@pytest.fixture(scope="session", autouse=True)
def mock_model_loading(mock_tts_instance):
    """モデルロードを防ぐため、LlasaTTSServerの__init__をモック（全テストで自動適用）"""
    # LlasaTTSServerクラス自体をモック
    with patch("llasa_server.server.LlasaTTSServer") as mock_class:
        mock_class.return_value = mock_tts_instance
        # api.pyでインポートされる場所もモック
        with patch("llasa_server.api.LlasaTTSServer", mock_class):
            yield mock_class
