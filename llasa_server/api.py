"""FastAPIベースのTTS APIサーバー"""

import io
import logging
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import soundfile as sf
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from .config import AudioConstants, TextConstants, get_config
from .server import LlasaTTSServer

# ロガーの設定
logger = logging.getLogger(__name__)

# グローバルなサーバーインスタンス
tts_server: Optional[LlasaTTSServer] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """アプリケーションのライフサイクル管理"""
    global tts_server
    logger.info("Llasa推論サーバーを起動します...")

    # 設定を取得
    config = get_config()

    # 設定を使ってサーバーを初期化
    tts_server = LlasaTTSServer(
        llasa_model_id=config.llasa_model_id,
        xcodec2_model_id=config.xcodec2_model_id,
        backend=config.backend,
        tensor_parallel_size=config.tensor_parallel_size,
        gpu_memory_utilization=config.gpu_memory_utilization,
        max_model_len=config.max_model_len,
        device=config.device,
        output_sample_rate=config.output_sample_rate,
    )
    logger.info("サーバー起動完了！")

    yield

    # シャットダウン処理（必要に応じて）
    logger.info("サーバーをシャットダウン中...")


app = FastAPI(title="Llasa TTS API Server", lifespan=lifespan)


def validate_audio_file(audio_file: UploadFile) -> None:
    """音声ファイルのバリデーション

    Args:
        audio_file: アップロードされた音声ファイル

    Raises:
        HTTPException: バリデーションエラーの場合
    """
    # ファイル名の検証
    if not audio_file.filename:
        logger.warning("ファイル名が空です")
        raise HTTPException(status_code=400, detail="ファイル名が指定されていません")

    # ファイル拡張子の検証
    file_ext = Path(audio_file.filename).suffix.lower()
    if file_ext not in AudioConstants.ALLOWED_AUDIO_FORMATS:
        logger.warning(f"非対応の音声フォーマット: {file_ext}")
        raise HTTPException(
            status_code=400,
            detail=f"非対応の音声フォーマットです。対応フォーマット: "
            f"{', '.join(AudioConstants.ALLOWED_AUDIO_FORMATS)}",
        )

    # MIMEタイプの検証
    if audio_file.content_type:
        if audio_file.content_type not in AudioConstants.ALLOWED_MIME_TYPES:
            logger.warning(f"非対応のMIMEタイプ: {audio_file.content_type}")
            raise HTTPException(
                status_code=400,
                detail=f"非対応のMIMEタイプです: {audio_file.content_type}",
            )

    # ファイルサイズの検証（オプション：事前チェック）
    # 注: FastAPIはファイル全体を読み込む前にサイズをチェックできない場合があるため、
    # 実際の読み込み時にもサイズチェックを行う
    logger.debug(
        f"音声ファイルのバリデーション成功: {audio_file.filename}, "
        f"MIMEタイプ: {audio_file.content_type}"
    )


@app.post("/tts")
async def generate_tts(
    text: str = Form(..., description="生成するテキスト"),
    reference_audio: Optional[UploadFile] = File(
        None, description="リファレンス音声ファイル（オプション）"
    ),
    reference_text: Optional[str] = Form(
        None, description="リファレンス音声のテキスト（オプション）"
    ),
    temperature: float = Form(0.8, description="サンプリング温度"),
    top_p: float = Form(1.0, description="Top-pサンプリング"),
    repetition_penalty: float = Form(1.1, description="繰り返しペナルティ"),
    max_tokens: int = Form(2048, description="最大トークン数"),
):
    """テキストから音声を生成するエンドポイント

    Args:
        text: 生成するテキスト
        reference_audio: リファレンス音声ファイル（オプション）
        reference_text: リファレンス音声のテキスト（オプション）
        temperature: サンプリング温度
        top_p: Top-pサンプリング
        repetition_penalty: 繰り返しペナルティ
        max_tokens: 最大トークン数

    Returns:
        WAV形式の音声データ
    """
    if tts_server is None:
        raise HTTPException(
            status_code=503, detail="TTSサーバーがまだ初期化されていません"
        )

    # テキスト長のバリデーション
    if len(text) > TextConstants.MAX_TEXT_LENGTH:
        logger.warning(f"テキストが長すぎます: {len(text)}文字")
        raise HTTPException(
            status_code=400,
            detail=f"テキストが長すぎます。{TextConstants.MAX_TEXT_LENGTH}文字以内にしてください。",
        )

    reference_audio_path = None
    try:
        # リファレンス音声がある場合は一時ファイルに保存
        if reference_audio is not None:
            # 音声ファイルのバリデーション
            validate_audio_file(reference_audio)

            # ファイルサイズの検証
            content = await reference_audio.read()
            if len(content) > AudioConstants.MAX_REFERENCE_AUDIO_SIZE:
                logger.warning(f"ファイルサイズが大きすぎます: {len(content)}バイト")
                raise HTTPException(
                    status_code=400,
                    detail=f"ファイルサイズが大きすぎます。"
                    f"{AudioConstants.MAX_REFERENCE_AUDIO_SIZE / (1024 * 1024)}MB以内にしてください。",
                )

            # 一時ファイルに保存
            file_ext = (
                Path(reference_audio.filename).suffix
                if reference_audio.filename
                else ".wav"
            )
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=file_ext
            ) as temp_file:
                temp_file.write(content)
                reference_audio_path = temp_file.name
                logger.debug(
                    f"リファレンス音声を一時ファイルに保存: {reference_audio_path}"
                )

        # 音声を生成
        audio_waveform, sample_rate = tts_server.generate_speech(
            text=text,
            reference_audio_path=reference_audio_path,
            reference_text=reference_text,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            max_tokens=max_tokens,
        )

        # メモリ上でWAVファイルを作成
        wav_buffer = io.BytesIO()
        sf.write(wav_buffer, audio_waveform, sample_rate, format="WAV")
        wav_buffer.seek(0)

        return StreamingResponse(
            wav_buffer,
            media_type="audio/wav",
            headers={
                "Content-Disposition": "attachment; filename=generated_speech.wav"
            },
        )

    except ValueError as e:
        # 入力バリデーションエラー（パラメータが不正など）
        logger.warning(f"入力バリデーションエラー: {e}")
        raise HTTPException(status_code=400, detail=f"入力エラー: {str(e)}") from e
    except RuntimeError as e:
        # 音声生成処理のエラー
        logger.error(f"音声生成処理エラー: {e}")
        raise HTTPException(status_code=500, detail=f"処理エラー: {str(e)}") from e
    except HTTPException:
        # FastAPIのHTTPExceptionはそのまま再送出
        raise
    except Exception as e:
        # その他の予期しないエラー
        logger.exception("予期しないエラーが発生しました")
        raise HTTPException(status_code=500, detail=f"音声生成エラー: {str(e)}") from e

    finally:
        # 一時ファイルをクリーンアップ
        if reference_audio_path and os.path.exists(reference_audio_path):
            try:
                os.unlink(reference_audio_path)
            except Exception as e:
                logger.warning(f"一時ファイルの削除に失敗: {e}")


@app.get("/health")
async def health_check():
    """ヘルスチェックエンドポイント"""
    return {
        "status": "ok",
        "server_initialized": tts_server is not None,
    }
