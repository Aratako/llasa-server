"""Llasa TTS推論サーバのメインクラス"""

import logging
import time
from typing import Optional

import numpy as np
import torch

from .codec import XCodec2Wrapper
from .config import AudioConstants
from .engine_base import BaseLlasaEngine

logger = logging.getLogger(__name__)


class LlasaTTSServer:
    """Llasa TTS推論サーバ"""

    def __init__(
        self,
        llasa_model_id: str = "NandemoGHS/Anime-Llasa-3B",
        xcodec2_model_id: str = "NandemoGHS/Anime-XCodec2",
        backend: str = "vllm",
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.8,
        max_model_len: int = 2048,
        device: str = "cuda",
    ):
        """
        Args:
            llasa_model_id: LlasaモデルのHugging Face ID
            xcodec2_model_id: XCodec2モデルのHugging Face ID
            backend: 推論バックエンド ("vllm", "sglang", or "transformers")
            tensor_parallel_size: vLLM/SGLangのテンソル並列サイズ
            gpu_memory_utilization: vLLM/SGLangのGPUメモリ使用率
            max_model_len: モデルの最大長
            device: 使用するデバイス
        """
        # バックエンドに応じてエンジンを初期化（遅延インポート）
        if backend == "vllm":
            logger.info("vLLMエンジンを初期化中...")
            from .engine_vllm import VLLMLlasaEngine

            self.llasa_engine: BaseLlasaEngine = VLLMLlasaEngine(
                model_id=llasa_model_id,
                tensor_parallel_size=tensor_parallel_size,
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=max_model_len,
            )
        elif backend == "sglang":
            logger.info("SGLangエンジンを初期化中...")
            from .engine_sglang import SGLangLlasaEngine

            self.llasa_engine: BaseLlasaEngine = SGLangLlasaEngine(
                model_id=llasa_model_id,
                tensor_parallel_size=tensor_parallel_size,
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=max_model_len,
            )
        elif backend == "transformers":
            logger.info("Transformersエンジンを初期化中...")
            from .engine_transformers import TransformersLlasaEngine

            self.llasa_engine: BaseLlasaEngine = TransformersLlasaEngine(
                model_id=llasa_model_id,
                device=device,
                torch_dtype=torch.bfloat16,
            )
        else:
            raise ValueError(
                f"無効なバックエンド: {backend} (選択肢: 'vllm', 'sglang', 'transformers')"
            )

        logger.info("XCodec2を初期化中...")
        self.codec = XCodec2Wrapper(
            model_id=xcodec2_model_id,
            device=device,
        )

    def generate_speech(
        self,
        text: str,
        reference_audio_path: Optional[str] = None,
        reference_text: Optional[str] = None,
        temperature: float = 0.8,
        top_p: float = 1.0,
        repetition_penalty: float = 1.1,
        max_tokens: int = 2048,
    ) -> tuple[np.ndarray, int]:
        """テキストから音声を生成する

        Args:
            text: 生成するテキスト
            reference_audio_path: リファレンス音声のパス（オプション）
            reference_text: リファレンス音声のテキスト（オプション）
            temperature: サンプリング温度
            top_p: Top-pサンプリング
            repetition_penalty: 繰り返しペナルティ
            max_tokens: 最大トークン数

        Returns:
            (音声波形, サンプリングレート)のタプル
        """
        start = time.perf_counter()
        # リファレンス音声がある場合はエンコード
        reference_speech_ids = None
        if reference_audio_path is not None:
            logger.debug("リファレンス音声をエンコード中...")
            reference_speech_ids = self.codec.encode_audio(
                reference_audio_path
            ).tolist()

        # Speech tokenを生成
        logger.debug("Speech tokenを生成中...")
        speech_ids = self.llasa_engine.generate_speech_tokens(
            text=text,
            reference_speech_ids=reference_speech_ids,
            reference_text=reference_text,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            max_tokens=max_tokens,
        )

        if not speech_ids:
            raise ValueError("音声生成に失敗しました")

        # Speech tokenを音声にデコード
        logger.debug("音声波形にデコード中...")
        audio_waveform = self.codec.decode_speech_tokens(speech_ids)

        end = time.perf_counter()
        logger.debug(f"音声生成に{end - start:.2f}秒かかりました")

        return audio_waveform, AudioConstants.SAMPLE_RATE
