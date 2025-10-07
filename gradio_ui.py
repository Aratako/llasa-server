"""Gradioを使用したLlasa TTS Web UI"""

import argparse
import io
import os
from typing import Optional

import gradio as gr
import numpy as np
import requests
import soundfile as sf

from llasa_server.config import TextConstants


def generate_speech(
    ref_audio_path: Optional[str],
    ref_text: Optional[str],
    target_text: str,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    server_url: str = "http://localhost:8000",
    progress=gr.Progress(),
) -> Optional[tuple[int, np.ndarray]]:
    """APIサーバーにリクエストを送信して音声を生成

    Args:
        ref_audio_path: リファレンス音声のパス
        ref_text: リファレンス音声のテキスト
        target_text: 生成するテキスト
        temperature: サンプリング温度
        top_p: Top-pサンプリング
        repetition_penalty: 繰り返しペナルティ
        server_url: APIサーバーのURL
        progress: Gradioの進捗表示

    Returns:
        (sample_rate, audio_array) のタプル、またはNone
    """
    if not target_text or not target_text.strip():
        gr.Warning("生成するテキストを入力してください。")
        return None

    # リファレンス音声がある場合、リファレンステキストを必須にする
    if ref_audio_path and (not ref_text or not ref_text.strip()):
        gr.Warning(
            "リファレンス音声を使用する場合は、リファレンステキストを入力してください。"
        )
        return None

    if len(target_text) > TextConstants.MAX_TEXT_LENGTH:
        gr.Warning(
            f"テキストが長すぎます。{TextConstants.MAX_TEXT_LENGTH}文字以内にしてください。"
        )
        target_text = target_text[: TextConstants.MAX_TEXT_LENGTH]

    try:
        progress(0, "リクエストを準備中...")

        # リクエストデータを準備
        data = {
            "text": target_text,
            "temperature": temperature,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
        }

        files = {}

        # リファレンステキストがある場合
        if ref_text and ref_text.strip():
            data["reference_text"] = ref_text

        # リファレンス音声がある場合
        if ref_audio_path:
            progress(0.25, "音声をアップロード中...")
            with open(ref_audio_path, "rb") as f:
                files["reference_audio"] = f

                # APIにリクエスト送信
                progress(0.5, "音声を生成中...")
                response = requests.post(
                    f"{server_url}/tts",
                    data=data,
                    files=files,
                    timeout=120,
                )
        else:
            # リファレンス音声なしの場合
            progress(0.5, "音声を生成中...")
            response = requests.post(
                f"{server_url}/tts",
                data=data,
                timeout=120,
            )

        # レスポンスチェック
        if response.status_code != 200:
            gr.Error(f"エラー: {response.status_code} - {response.text}")
            return None

        progress(0.75, "音声を処理中...")

        # WAVファイルを読み込み
        audio_data = io.BytesIO(response.content)
        audio_array, sample_rate = sf.read(audio_data)

        progress(1, "完了！")

        return (sample_rate, audio_array)

    except requests.exceptions.ConnectionError:
        gr.Error("サーバーに接続できません。サーバーが起動しているか確認してください。")
        return None
    except requests.exceptions.Timeout:
        gr.Error("リクエストがタイムアウトしました。")
        return None
    except Exception as e:
        gr.Error(f"エラーが発生しました: {str(e)}")
        return None


def create_ui(default_server_url: str = "http://localhost:8000"):
    """Gradio UIを作成

    Args:
        default_server_url: デフォルトのAPIサーバーURL
    """

    with gr.Blocks(title="Anime Llasa 3B TTS") as app:
        gr.Markdown(
            """
# Llasa Web UI

このWeb UIは、Llasa TTSモデルを使用して音声を生成します。

問題が発生した場合は、リファレンス音声をWAVまたはMP3に変換し、15秒以内にクリップし、プロンプトを短くしてみてください。
"""
        )

        with gr.Tab("TTS"):
            with gr.Row():
                with gr.Column():
                    server_url_input = gr.Textbox(
                        label="APIサーバーURL",
                        value=default_server_url,
                        placeholder="http://localhost:8000",
                    )
                    ref_audio_input = gr.Audio(
                        label="リファレンス音声（オプション）",
                        type="filepath",
                    )
                    ref_text_input = gr.Textbox(
                        label="リファレンステキスト"
                        "（リファレンス音声を使用する場合は必須）",
                        placeholder="リファレンス音声を提供する場合、"
                        "その文字起こしをここに入力してください。",
                        lines=3,
                    )
                    gen_text_input = gr.Textbox(
                        label="生成するテキスト",
                        placeholder="ここに生成したいテキストを入力してください...",
                        lines=10,
                    )

                    with gr.Row():
                        temperature_slider = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=0.8,
                            step=0.05,
                            label="Temperature",
                        )
                        top_p_slider = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=1.0,
                            step=0.05,
                            label="Top-p",
                        )
                        repetition_penalty_slider = gr.Slider(
                            minimum=1.0,
                            maximum=1.5,
                            value=1.1,
                            step=0.05,
                            label="Repetition Penalty",
                        )

                    generate_btn = gr.Button("音声を生成", variant="primary")

                with gr.Column():
                    audio_output = gr.Audio(label="生成された音声")

            # ボタンクリック時の処理
            generate_btn.click(
                lambda ref_audio, ref_text, gen_text, temp, top_p, rep_pen, url: (
                    generate_speech(
                        ref_audio,
                        ref_text,
                        gen_text,
                        temp,
                        top_p,
                        rep_pen,
                        url,
                    )
                ),
                inputs=[
                    ref_audio_input,
                    ref_text_input,
                    gen_text_input,
                    temperature_slider,
                    top_p_slider,
                    repetition_penalty_slider,
                    server_url_input,
                ],
                outputs=[audio_output],
            )

        with gr.Tab("Credits"):
            gr.Markdown(
                """
# Credits

* [zhenye234](https://github.com/zhenye234) for the original [repo](
  https://github.com/zhenye234/LLaSA_training)
* [mrfakename](https://huggingface.co/mrfakename) for the [gradio demo code](
  https://huggingface.co/spaces/mrfakename/E2-F5-TTS)
* [SunderAli17](https://huggingface.co/SunderAli17) for the [gradio demo code](
  https://huggingface.co/spaces/SunderAli17/llasa-3b-tts)
"""
            )

    return app


if __name__ == "__main__":
    # コマンドライン引数のパース
    parser = argparse.ArgumentParser(description="Llasa TTS Gradio Web UI")
    parser.add_argument(
        "--server-url",
        type=str,
        default=os.getenv("LLASA_SERVER_URL", "http://localhost:8000"),
        help="APIサーバーのURL (デフォルト: http://localhost:8000, "
        "環境変数 LLASA_SERVER_URL で設定可能)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Web UIのホスト (デフォルト: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Web UIのポート (デフォルト: 7860)",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Gradio公開リンクを生成",
    )

    args = parser.parse_args()

    # UIを作成して起動
    app = create_ui(default_server_url=args.server_url)
    app.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
    )
