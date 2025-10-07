"""APIクライアントの使用例"""

import time

import requests


def simple_tts_example():
    """シンプルなTTS生成の例"""
    url = "http://localhost:8000/tts"

    # テキストのみ
    data = {
        "text": "こんにちは、私はAIです。これは音声合成のテストです。",
        "temperature": 0.8,
        "top_p": 1.0,
        "repetition_penalty": 1.1,
    }

    print("音声を生成中...")
    start = time.perf_counter()
    response = requests.post(url, data=data)
    end = time.perf_counter()

    if response.status_code == 200:
        print(f"音声生成にかかった時間: {end - start:.2f}秒")
        with open("output_simple.wav", "wb") as f:
            f.write(response.content)
        print("音声を保存しました: output_simple.wav")
    else:
        print(f"エラー: {response.status_code}")
        print(response.json())


def reference_audio_example():
    """リファレンス音声を使用したTTS生成の例"""
    url = "http://localhost:8000/tts"

    # リファレンス音声付き
    data = {
        "text": "これは生成したい新しいテキストです。",
        "reference_text": "これはリファレンス音声のテキストです。",
        "temperature": 0.8,
        "top_p": 1.0,
        "repetition_penalty": 1.1,
    }

    print("リファレンス音声を使用して音声を生成中...")

    with open("reference.wav", "rb") as f:
        files = {
            "reference_audio": f,
        }
        response = requests.post(url, data=data, files=files)

    if response.status_code == 200:
        with open("output_with_reference.wav", "wb") as f:
            f.write(response.content)
        print("音声を保存しました: output_with_reference.wav")
    else:
        print(f"エラー: {response.status_code}")
        print(response.json())


if __name__ == "__main__":
    # シンプルな例
    simple_tts_example()

    # リファレンス音声を使う例（reference.wavがある場合）
    # reference_audio_example()
