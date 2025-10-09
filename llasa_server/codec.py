"""XCodec2音声コーデックのラッパー"""

import logging
import os

import numpy as np
import torch
import torchaudio
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from xcodec2.configuration_bigcodec import BigCodecConfig
from xcodec2.modeling_xcodec2 import XCodec2Model

from .config import AudioConstants

logger = logging.getLogger(__name__)


class XCodec2Wrapper:
    """XCodec2モデルのラッパークラス"""

    def __init__(
        self, model_id: str = "NandemoGHS/Anime-XCodec2", device: str = "cuda"
    ):
        """
        Args:
            model_id: Hugging FaceのモデルID
            device: 使用するデバイス（cuda/cpu）
        """
        self.device = device
        # transformers > 4.49.0 での不具合対応
        # https://github.com/zhenye234/X-Codec-2.0/issues/24#issuecomment-2911159706
        ckpt_path = hf_hub_download(repo_id=model_id, filename="model.safetensors")
        ckpt = {}
        with safe_open(ckpt_path, framework="pt", device="cpu") as f:
            for k in f.keys():
                ckpt[k.replace(".beta", ".bias")] = f.get_tensor(k)
        codec_config = BigCodecConfig.from_pretrained(model_id)
        self.model = XCodec2Model.from_pretrained(
            None, config=codec_config, state_dict=ckpt
        )
        self.model.eval().to(device)

    def encode_audio(
        self,
        audio_path: str,
        max_duration: float = AudioConstants.MAX_REFERENCE_AUDIO_DURATION,
    ) -> torch.Tensor:
        """音声ファイルをspeech IDにエンコードする

        Args:
            audio_path: 音声ファイルのパス
            max_duration: 最大長（秒）

        Returns:
            エンコードされたspeech ID

        Raises:
            FileNotFoundError: 音声ファイルが存在しない場合
            ValueError: 音声ファイルの読み込みや処理に失敗した場合
            RuntimeError: GPUメモリ不足などの実行時エラー
        """
        # ファイルの存在確認
        if not os.path.exists(audio_path):
            logger.error(f"音声ファイルが見つかりません: {audio_path}")
            raise FileNotFoundError(f"音声ファイルが見つかりません: {audio_path}")

        try:
            # 音声を読み込む
            waveform, sample_rate = torchaudio.load(audio_path)
            logger.debug(
                f"音声を読み込みました: shape={waveform.shape}, sample_rate={sample_rate}"
            )
        except Exception as e:
            logger.error(f"音声ファイルの読み込みに失敗: {audio_path}, エラー: {e}")
            raise ValueError(f"音声ファイルの読み込みに失敗しました: {str(e)}") from e

        try:
            # 最大長でトリミング
            max_samples = int(sample_rate * max_duration)
            if waveform.shape[1] > max_samples:
                logger.debug(f"音声を{max_duration}秒にトリミングします")
                waveform = waveform[:, :max_samples]

            # ステレオの場合はモノラルに変換
            if waveform.size(0) > 1:
                logger.debug("ステレオ音声をモノラルに変換します")
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            # 16kHzにリサンプリング
            if sample_rate != AudioConstants.SAMPLE_RATE:
                logger.debug(
                    f"音声を{AudioConstants.SAMPLE_RATE}Hzにリサンプリングします"
                )
                prompt_wav = torchaudio.transforms.Resample(
                    orig_freq=sample_rate, new_freq=AudioConstants.SAMPLE_RATE
                )(waveform)
            else:
                prompt_wav = waveform

        except Exception as e:
            logger.error(f"音声の前処理に失敗: {e}")
            raise ValueError(f"音声の前処理に失敗しました: {str(e)}") from e

        try:
            # GPUに転送してエンコード
            prompt_wav = prompt_wav.to(self.device)

            with torch.no_grad():
                vq_code_prompt = self.model.encode_code(input_waveform=prompt_wav)[
                    0, 0, :
                ]

            logger.debug(f"音声をエンコードしました: {len(vq_code_prompt)}トークン")
            return vq_code_prompt

        except torch.cuda.OutOfMemoryError as e:
            logger.error("GPU メモリ不足エラー")
            raise RuntimeError(
                "GPU メモリが不足しています。音声ファイルを短くするか、"
                "別のデバイスで実行してください。"
            ) from e
        except Exception as e:
            logger.error(f"音声のエンコードに失敗: {e}")
            raise RuntimeError(f"音声のエンコードに失敗しました: {str(e)}") from e

    def decode_speech_tokens(self, speech_ids: list[int]) -> np.ndarray:
        """Speech IDを音声波形にデコードする

        Args:
            speech_ids: Speech IDのリスト

        Returns:
            音声波形（numpy配列、16kHz）

        Raises:
            ValueError: speech_idsが空または無効な場合
            RuntimeError: デコードに失敗した場合
        """
        # 入力の検証
        if not speech_ids:
            logger.error("speech_idsが空です")
            raise ValueError("speech_idsが空です")

        if not all(isinstance(id, int) for id in speech_ids):
            logger.error("speech_idsに整数以外の値が含まれています")
            raise ValueError("speech_idsは整数のリストである必要があります")

        logger.debug(f"{len(speech_ids)}個のspeech tokenをデコードします")

        try:
            # テンソルに変換して形状を(1, 1, length)にする
            speech_tokens = (
                torch.tensor(speech_ids).unsqueeze(0).unsqueeze(0).to(self.device)
            )

            with torch.no_grad():
                gen_wav = self.model.decode_code(speech_tokens)

            audio_array = gen_wav[0, 0, :].cpu().numpy()
            logger.debug(f"音声をデコードしました: {len(audio_array)}サンプル")
            return audio_array

        except torch.cuda.OutOfMemoryError as e:
            logger.error("GPU メモリ不足エラー")
            raise RuntimeError(
                "GPU メモリが不足しています。より短いテキストで試すか、"
                "別のデバイスで実行してください。"
            ) from e
        except Exception as e:
            logger.error(f"音声のデコードに失敗: {e}")
            raise RuntimeError(f"音声のデコードに失敗しました: {str(e)}") from e
