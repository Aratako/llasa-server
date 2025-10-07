"""ユーティリティ関数のテスト"""

from llasa_server.utils import (
    extract_speech_ids,
    ids_to_speech_tokens,
    normalize_text,
)


class TestNormalizeText:
    """normalize_text関数のテスト"""

    def test_remove_special_characters(self):
        """特殊文字の削除をテスト"""
        text = "こんにちは▼世界♀です♂《test》"
        expected = "こんにちは世界ですtest"
        assert normalize_text(text) == expected

    def test_fullwidth_alpha_to_halfwidth(self):
        """全角英字を半角に変換するテスト"""
        text = "ＡＢＣＤ１２３４"
        expected = "ABCD1234"
        assert normalize_text(text) == expected

    def test_halfwidth_katakana_to_fullwidth(self):
        """半角カタカナを全角に変換するテスト"""
        text = "ｶﾀｶﾅ"
        expected = "カタカナ"
        assert normalize_text(text) == expected

    def test_fullwidth_digits_to_halfwidth(self):
        """全角数字を半角に変換するテスト"""
        text = "０１２３４５６７８９"
        expected = "0123456789"
        assert normalize_text(text) == expected

    def test_replace_fullwidth_tilde(self):
        """全角チルダを長音符に変換するテスト"""
        text = "あ〜い～う"
        expected = "あーいーう"
        assert normalize_text(text) == expected

    def test_replace_fullwidth_punctuation(self):
        """全角句読点を半角に変換するテスト"""
        text = "こんにちは？世界！"
        expected = "こんにちは?世界!"
        assert normalize_text(text) == expected

    def test_multiple_ellipsis(self):
        """三点リーダーの正規化をテスト"""
        text = "あ…………い"
        expected = "あ……い"
        assert normalize_text(text) == expected

    def test_remove_spaces(self):
        """スペースの削除をテスト"""
        text = "こんにちは 世界　です"
        expected = "こんにちは世界です"
        assert normalize_text(text) == expected

    def test_combined_normalization(self):
        """複合的な正規化をテスト"""
        text = "こんにちは？　ＡＢＣＤ▼１２３４♀ｶﾀｶﾅ！"
        expected = "こんにちは?ABCD1234カタカナ!"
        assert normalize_text(text) == expected

    def test_empty_string(self):
        """空文字列のテスト"""
        assert normalize_text("") == ""

    def test_only_valid_characters(self):
        """有効な文字のみの場合のテスト"""
        text = "こんにちは世界"
        assert normalize_text(text) == text


class TestIdsToSpeechTokens:
    """ids_to_speech_tokens関数のテスト"""

    def test_single_id(self):
        """単一IDの変換をテスト"""
        speech_ids = [42]
        expected = ["<|s_42|>"]
        assert ids_to_speech_tokens(speech_ids) == expected

    def test_multiple_ids(self):
        """複数IDの変換をテスト"""
        speech_ids = [0, 1, 2, 3]
        expected = ["<|s_0|>", "<|s_1|>", "<|s_2|>", "<|s_3|>"]
        assert ids_to_speech_tokens(speech_ids) == expected

    def test_large_ids(self):
        """大きなIDの変換をテスト"""
        speech_ids = [1000, 9999, 100000]
        expected = ["<|s_1000|>", "<|s_9999|>", "<|s_100000|>"]
        assert ids_to_speech_tokens(speech_ids) == expected

    def test_empty_list(self):
        """空リストのテスト"""
        assert ids_to_speech_tokens([]) == []


class TestExtractSpeechIds:
    """extract_speech_ids関数のテスト"""

    def test_single_token(self):
        """単一トークンの抽出をテスト"""
        tokens = ["<|s_42|>"]
        expected = [42]
        assert extract_speech_ids(tokens) == expected

    def test_multiple_tokens(self):
        """複数トークンの抽出をテスト"""
        tokens = ["<|s_0|>", "<|s_1|>", "<|s_2|>", "<|s_3|>"]
        expected = [0, 1, 2, 3]
        assert extract_speech_ids(tokens) == expected

    def test_large_ids(self):
        """大きなIDの抽出をテスト"""
        tokens = ["<|s_1000|>", "<|s_9999|>", "<|s_100000|>"]
        expected = [1000, 9999, 100000]
        assert extract_speech_ids(tokens) == expected

    def test_invalid_token_warning(self, caplog):
        """不正なトークンの警告をテスト"""
        tokens = ["<|s_42|>", "invalid_token", "<|s_100|>"]
        expected = [42, 100]
        result = extract_speech_ids(tokens)
        assert result == expected
        assert "Unexpected token: invalid_token" in caplog.text

    def test_empty_list(self):
        """空リストのテスト"""
        assert extract_speech_ids([]) == []

    def test_round_trip(self):
        """変換の往復をテスト"""
        original_ids = [0, 42, 100, 999, 10000]
        tokens = ids_to_speech_tokens(original_ids)
        recovered_ids = extract_speech_ids(tokens)
        assert recovered_ids == original_ids
