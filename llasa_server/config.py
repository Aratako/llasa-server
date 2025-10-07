"""サーバー設定を管理するモジュール"""

from contextvars import ContextVar
from dataclasses import dataclass
from typing import Optional


# 定数定義
class AudioConstants:
    """音声処理に関する定数"""

    SAMPLE_RATE = 16000  # サンプリングレート (Hz)
    MAX_REFERENCE_AUDIO_DURATION = 15.0  # リファレンス音声の最大長 (秒)
    MAX_REFERENCE_AUDIO_SIZE = 50 * 1024 * 1024  # 50MB
    ALLOWED_AUDIO_FORMATS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
    ALLOWED_MIME_TYPES = {
        "audio/wav",
        "audio/wave",
        "audio/x-wav",
        "audio/mpeg",
        "audio/mp3",
        "audio/flac",
        "audio/ogg",
        "audio/x-m4a",
        "audio/mp4",
    }


class TextConstants:
    """テキスト処理に関する定数"""

    MAX_TEXT_LENGTH = 300  # 最大テキスト長（文字数）


@dataclass
class ServerConfig:
    """サーバー設定"""

    # Llasaモデル設定
    llasa_model_id: str = "NandemoGHS/Anime-Llasa-3B"
    xcodec2_model_id: str = "NandemoGHS/Anime-XCodec2"

    # vLLM設定
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.8
    max_model_len: int = 2048

    # デバイス設定
    device: str = "cuda"

    # サーバー設定
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False


# コンテキスト変数を使用してスレッドセーフな設定管理を実現
_config_context: ContextVar[Optional[ServerConfig]] = ContextVar("config", default=None)


def get_config() -> ServerConfig:
    """設定を取得する

    Returns:
        ServerConfig: 現在の設定インスタンス

    Note:
        設定が未設定の場合はデフォルト設定を作成して返す
    """
    config = _config_context.get()
    if config is None:
        config = ServerConfig()
        _config_context.set(config)
    return config


def set_config(new_config: ServerConfig) -> None:
    """設定を設定する

    Args:
        new_config: 新しい設定インスタンス
    """
    _config_context.set(new_config)
