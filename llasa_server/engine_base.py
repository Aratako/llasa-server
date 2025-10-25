"""推論エンジンの抽象基底クラス"""

from abc import ABC, abstractmethod
from typing import Optional


class BaseLlasaEngine(ABC):
    """Llasa推論エンジンの抽象基底クラス"""

    @abstractmethod
    def generate_speech_tokens(
        self,
        text: str,
        reference_speech_ids: Optional[list[int]] = None,
        reference_text: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.8,
        top_p: float = 1.0,
        repetition_penalty: float = 1.1,
        max_tokens: int = 2048,
    ) -> list[int]:
        """テキストからspeech tokenを生成する

        Args:
            text: 生成するテキスト
            reference_speech_ids: リファレンス音声のspeech ID（オプション）
            reference_text: リファレンス音声のテキスト（オプション）
            system_prompt: システムプロンプト（オプション、メタデータを含む）
            temperature: サンプリング温度
            top_p: Top-pサンプリングの閾値
            repetition_penalty: 繰り返しペナルティ
            max_tokens: 最大トークン数

        Returns:
            生成されたspeech IDのリスト

        Raises:
            ValueError: テキストが空、またはパラメータが無効な場合
            RuntimeError: 生成に失敗した場合
        """
        pass
