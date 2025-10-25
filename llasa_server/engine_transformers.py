"""Transformersを使用したLlasa推論エンジン"""

import logging
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .engine_base import BaseLlasaEngine
from .utils import build_llasa_prompt, extract_speech_ids

logger = logging.getLogger(__name__)


class TransformersLlasaEngine(BaseLlasaEngine):
    """Transformersを使用したLlasa TTSエンジン"""

    def __init__(
        self,
        model_id: str = "NandemoGHS/Anime-Llasa-3B",
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
    ):
        """
        Args:
            model_id: Hugging FaceのモデルID
            device: 使用するデバイス ('cuda' or 'cpu')
            torch_dtype: モデルの精度
        """
        self.model_id = model_id
        self.device = device

        # トークナイザーを読み込む
        logger.info(f"トークナイザーを読み込み中: {model_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        # モデルを読み込む
        logger.info(f"モデルを読み込み中: {model_id}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
        )
        self.model.eval().to(device)

        # 特殊トークンのIDを取得
        self.speech_end_id = self.tokenizer.convert_tokens_to_ids(
            "<|SPEECH_GENERATION_END|>"
        )

        logger.info(f"モデルの初期化が完了しました (device: {device})")

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

            input_ids = self.tokenizer.apply_chat_template(
                chat,
                tokenize=True,
                return_tensors="pt",
                continue_final_message=True,
            ).to(self.device)

            logger.debug(f"入力トークン数: {input_ids.shape[1]}")

        except Exception as e:
            logger.error(f"プロンプトの構築に失敗: {e}")
            raise ValueError(f"プロンプトの構築に失敗しました: {str(e)}") from e

        try:
            logger.debug("Transformersで生成を開始します")

            # 生成
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids,
                    max_length=max_tokens,
                    eos_token_id=self.speech_end_id,
                    do_sample=True,
                    top_p=top_p,
                    temperature=temperature,
                    repetition_penalty=repetition_penalty,
                )

            # 生成されたトークンIDを取得（入力部分を除く）
            if reference_length > 0:
                # リファレンス音声がある場合は、reference_lengthを考慮
                generated_ids = outputs[0, input_ids.shape[1] - reference_length : -1]
            else:
                generated_ids = outputs[0, input_ids.shape[1] : -1]

            # トークンIDをテキストにデコード
            speech_tokens = self.tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True
            )

        except Exception as e:
            logger.error(f"Transformersでの生成に失敗: {e}")
            raise RuntimeError(f"音声生成に失敗しました: {str(e)}") from e

        try:
            # speech IDを抽出
            speech_ids = extract_speech_ids(speech_tokens)

            if not speech_ids:
                logger.warning("生成されたテキストにspeech tokenが見つかりませんでした")
                logger.debug(f"生成トークン: {speech_tokens[:10]}")
                raise RuntimeError(
                    "音声生成に失敗しました（speech tokenが見つかりません）"
                )

            logger.debug(f"{len(speech_ids)}個のspeech tokenを生成しました")
            return speech_ids

        except Exception as e:
            if isinstance(e, RuntimeError):
                raise
            logger.error(f"speech tokenの抽出に失敗: {e}")
            raise RuntimeError(f"speech tokenの抽出に失敗しました: {str(e)}") from e
