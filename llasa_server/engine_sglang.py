"""SGLangを使用したLlasa推論エンジン"""

import logging
import re
from typing import Optional

import nest_asyncio
import sglang as sgl
from transformers import AutoTokenizer

from .engine_base import BaseLlasaEngine
from .utils import build_llasa_prompt, extract_speech_ids

logger = logging.getLogger(__name__)


class SGLangLlasaEngine(BaseLlasaEngine):
    """SGLangを使用したLlasa TTSエンジン"""

    def __init__(
        self,
        model_id: str = "NandemoGHS/Anime-Llasa-3B",
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: int = 2048,
    ):
        """
        Args:
            model_id: Hugging FaceのモデルID
            tensor_parallel_size: テンソル並列サイズ
            gpu_memory_utilization: GPU メモリ使用率
            max_model_len: モデルの最大長
        """
        # SGLangが内部でasyncioを使用するため、nest_asyncioを適用
        # モジュールインポート時ではなく、初期化時に適用することで
        # uvloopとの競合を回避（run_server.pyでloop="asyncio"を使用）
        nest_asyncio.apply()

        self.model_id = model_id

        # トークナイザーを読み込む
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        # SGLangエンジンを初期化
        logger.info(f"SGLangエンジンを初期化中: {model_id}")
        self.llm = sgl.Engine(
            model_path=model_id,
            tp_size=tensor_parallel_size,
            mem_fraction_static=gpu_memory_utilization,
            context_length=max_model_len,
        )

        # SGLangが内部でloggingレベルを変更することがあるため、
        # エンジン初期化後に明示的にログレベルを復元
        import os

        log_level = os.getenv("LOG_LEVEL", "INFO")
        logging.getLogger().setLevel(log_level)
        logger.setLevel(log_level)

        # 特殊トークンのIDを取得
        self.speech_end_id = self.tokenizer.convert_tokens_to_ids(
            "<|SPEECH_GENERATION_END|>"
        )

        logger.info("SGLangエンジンの初期化が完了しました")

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
            RuntimeError: SGLangでの生成に失敗した場合
        """
        # 入力の検証
        if not text or not text.strip():
            logger.error("テキストが空です")
            raise ValueError("生成するテキストを入力してください")

        # パラメータの検証
        if not 0.0 <= temperature <= 2.0:
            logger.warning(f"temperatureが範囲外です: {temperature}")
            raise ValueError("temperatureは0.0から2.0の範囲で指定してください")

        if not 0.0 <= top_p <= 1.0:
            logger.warning(f"top_pが範囲外です: {top_p}")
            raise ValueError("top_pは0.0から1.0の範囲で指定してください")

        if repetition_penalty < 1.0:
            logger.warning(f"repetition_penaltyが範囲外です: {repetition_penalty}")
            raise ValueError("repetition_penaltyは1.0以上で指定してください")

        # リファレンス音声のバリデーション
        if (reference_speech_ids is None) != (reference_text is None):
            logger.error(
                "リファレンス音声とテキストは両方指定するか、両方省略してください"
            )
            raise ValueError(
                "リファレンス音声を使用する場合は、"
                "reference_speech_idsとreference_textの両方を指定してください"
            )

        try:
            # 共通のプロンプト構築関数を使用
            formatted_text, assistant_content, reference_length = build_llasa_prompt(
                text=text,
                reference_speech_ids=reference_speech_ids,
                reference_text=reference_text,
            )

            # チャットテンプレートを適用
            chat = []
            if system_prompt is not None:
                chat.append({"role": "system", "content": system_prompt})

            chat.extend(
                [
                    {
                        "role": "user",
                        "content": "Convert the text to speech:" + formatted_text,
                    },
                    {"role": "assistant", "content": assistant_content},
                ]
            )

            prompt = self.tokenizer.apply_chat_template(
                chat,
                tokenize=False,
                add_generation_prompt=False,
                continue_final_message=True,
            )

            logger.debug(f"生成プロンプト: {prompt[:500]}...")

            input_ids = self.tokenizer.encode(
                prompt,
                add_special_tokens=False,
            )

            logger.debug(f"入力トークン数: {len(input_ids)}")

        except Exception as e:
            logger.error(f"プロンプトの構築に失敗: {e}")
            raise ValueError(f"プロンプトの構築に失敗しました: {str(e)}") from e

        try:
            # サンプリングパラメータを設定
            sampling_params = {
                "temperature": temperature,
                "top_p": top_p,
                "frequency_penalty": repetition_penalty
                - 1.0,  # SGLangではfrequency_penaltyを使用
                "max_new_tokens": max_tokens
                - len(self.tokenizer.encode(prompt, add_special_tokens=False))
                - 2,
                "stop_token_ids": [self.speech_end_id],
            }

            logger.debug("SGLangで生成を開始します")
            # 生成
            outputs = self.llm.generate(
                input_ids=input_ids, sampling_params=sampling_params
            )

            if not outputs:
                logger.error("SGLangが出力を返しませんでした")
                raise RuntimeError("音声生成に失敗しました（出力が空です）")

            # 生成されたテキストを取得
            generated_text = outputs["text"]

        except Exception as e:
            logger.error(f"SGLangでの生成に失敗: {e}")
            raise RuntimeError(f"音声生成に失敗しました: {str(e)}") from e

        try:
            # <|s_xxxxx|>パターンを全て抽出してリストに変換
            speech_token_pattern = r"<\|s_\d+\|>"
            speech_tokens_list = re.findall(speech_token_pattern, generated_text)

            if not speech_tokens_list:
                logger.warning("生成されたテキストにspeech tokenが見つかりませんでした")
                logger.debug(f"生成テキスト: {generated_text[:200]}")
                raise RuntimeError(
                    "音声生成に失敗しました（speech tokenが見つかりません）"
                )

            # speech IDを抽出
            speech_ids = extract_speech_ids(speech_tokens_list)

            if not speech_ids:
                logger.error("speech IDの抽出に失敗しました")
                raise RuntimeError("speech IDの抽出に失敗しました")

            logger.debug(f"{len(speech_ids)}個のspeech tokenを生成しました")
            if reference_speech_ids:
                logger.debug(
                    f"リファレンス音声の長さ: {len(reference_speech_ids)}トークン"
                )
                speech_ids = reference_speech_ids + speech_ids
            return speech_ids

        except Exception as e:
            if isinstance(e, RuntimeError):
                raise
            logger.error(f"speech tokenの抽出に失敗: {e}")
            raise RuntimeError(f"speech tokenの抽出に失敗しました: {str(e)}") from e
