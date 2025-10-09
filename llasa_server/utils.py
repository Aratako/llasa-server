"""ユーティリティ関数"""

import logging
import re

logger = logging.getLogger(__name__)

# テキスト正規化のための定数
REPLACE_MAP: dict[str, str] = {
    r"\t": "",
    r"\[n\]": "",
    r" ": "",
    r"　": "",
    r"[;▼♀♂《》≪≫①②③④⑤⑥]": "",
    r"[\u02d7\u2010-\u2015\u2043\u2212\u23af\u23e4\u2500\u2501\u2e3a\u2e3b]": "",
    r"[\uff5e\u301C]": "ー",
    r"？": "?",
    r"！": "!",
    r"[●◯〇]": "○",
    r"♥": "♡",
}

FULLWIDTH_ALPHA_TO_HALFWIDTH = str.maketrans(
    {
        chr(full): chr(half)
        for full, half in zip(
            list(range(0xFF21, 0xFF3B)) + list(range(0xFF41, 0xFF5B)),
            list(range(0x41, 0x5B)) + list(range(0x61, 0x7B)),
        )
    }
)

# 半角カタカナから全角カタカナへの正しいマッピング
# 濁点・半濁点付きは複雑なので、基本文字のみマップ（濁点は別途処理が必要）
_HALFWIDTH_KATAKANA_CHARS = "ｦｧｨｩｪｫｬｭｮｯｰｱｲｳｴｵｶｷｸｹｺｻｼｽｾｿﾀﾁﾂﾃﾄﾅﾆﾇﾈﾉﾊﾋﾌﾍﾎﾏﾐﾑﾒﾓﾔﾕﾖﾗﾘﾙﾚﾛﾜﾝ"
_FULLWIDTH_KATAKANA_CHARS = "ヲァィゥェォャュョッーアイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワン"

HALFWIDTH_KATAKANA_TO_FULLWIDTH = str.maketrans(
    _HALFWIDTH_KATAKANA_CHARS, _FULLWIDTH_KATAKANA_CHARS
)

FULLWIDTH_DIGITS_TO_HALFWIDTH = str.maketrans(
    {
        chr(full): chr(half)
        for full, half in zip(range(0xFF10, 0xFF1A), range(0x30, 0x3A))
    }
)

INVALID_PATTERN = re.compile(
    r"[^\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF\u3400-\u4DBF\u3005"
    r"\u0041-\u005A\u0061-\u007A"
    r"\u0030-\u0039"
    r"。、!?…♪♡○]"
)


def normalize_text(text: str) -> str:
    """テキストを正規化する

    Args:
        text: 入力テキスト

    Returns:
        正規化されたテキスト
    """
    for pattern, replacement in REPLACE_MAP.items():
        text = re.sub(pattern, replacement, text)

    text = text.translate(FULLWIDTH_ALPHA_TO_HALFWIDTH)
    text = text.translate(FULLWIDTH_DIGITS_TO_HALFWIDTH)
    text = text.translate(HALFWIDTH_KATAKANA_TO_FULLWIDTH)

    text = re.sub(r"…{3,}", "……", text)

    return text


def ids_to_speech_tokens(speech_ids: list[int]) -> list[str]:
    """Speech IDをトークン文字列に変換する

    Args:
        speech_ids: Speech IDのリスト

    Returns:
        Speech tokenの文字列リスト
    """
    speech_tokens_str = []
    for speech_id in speech_ids:
        speech_tokens_str.append(f"<|s_{speech_id}|>")
    return speech_tokens_str


def extract_speech_ids(speech_tokens_str: list[str]) -> list[int]:
    """Speech tokenの文字列からIDを抽出する

    Args:
        speech_tokens_str: Speech tokenの文字列リスト

    Returns:
        Speech IDのリスト
    """
    speech_ids = []
    for token_str in speech_tokens_str:
        if token_str.startswith("<|s_") and token_str.endswith("|>"):
            num_str = token_str[4:-2]
            num = int(num_str)
            speech_ids.append(num)
        else:
            logger.warning(f"Unexpected token: {token_str}")
    return speech_ids


def build_llasa_prompt(
    text: str,
    reference_speech_ids: list[int] | None = None,
    reference_text: str | None = None,
) -> tuple[str, str, int]:
    """Llasaモデル用のプロンプトを構築する

    Args:
        text: 生成するテキスト
        reference_speech_ids: リファレンス音声のspeech ID（オプション）
        reference_text: リファレンス音声のテキスト（オプション）

    Returns:
        (input_text, assistant_content, reference_length)のタプル
        - input_text: フォーマット済みの入力テキスト
        - assistant_content: アシスタントの応答プレフィックス
        - reference_length: リファレンスspeech tokenの長さ（ない場合は0）
    """
    # テキストを正規化
    text = normalize_text(text)
    logger.debug(f"正規化後のテキスト: {text}")

    # プロンプトを構築
    if reference_speech_ids is not None and reference_text is not None:
        # リファレンス音声がある場合
        reference_text = normalize_text(reference_text)
        speech_ids_prefix = ids_to_speech_tokens(reference_speech_ids)
        input_text = reference_text + " " + text
        assistant_content = "<|SPEECH_GENERATION_START|>" + "".join(speech_ids_prefix)
        reference_length = len(speech_ids_prefix)
        logger.debug(f"リファレンス音声付きで生成: {reference_length}トークン")
    else:
        # リファレンス音声がない場合
        input_text = text
        assistant_content = "<|SPEECH_GENERATION_START|>"
        reference_length = 0
        logger.debug("リファレンス音声なしで生成")

    formatted_text = (
        f"<|TEXT_UNDERSTANDING_START|>{input_text}<|TEXT_UNDERSTANDING_END|>"
    )

    return formatted_text, assistant_content, reference_length
