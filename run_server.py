"""APIサーバーを起動するスクリプト"""

import argparse
import logging
import os

import uvicorn

from llasa_server.config import ServerConfig, set_config

# ロギングの基本設定
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def parse_args():
    """コマンドライン引数をパースする"""
    parser = argparse.ArgumentParser(
        description="Llasa TTS推論サーバーを起動",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # モデル設定
    parser.add_argument(
        "--llasa-model-id",
        type=str,
        default="NandemoGHS/Anime-Llasa-3B",
        help="LlasaモデルのHugging Face ID",
    )
    parser.add_argument(
        "--xcodec2-model-id",
        type=str,
        default="NandemoGHS/Anime-XCodec2",
        help="XCodec2モデルのHugging Face ID",
    )

    # vLLM設定
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="vLLMのテンソル並列サイズ",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.8,
        help="vLLMのGPUメモリ使用率 (0.0-1.0)",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=2048,
        help="モデルの最大長",
    )

    # デバイス設定
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="使用するデバイス (cuda/cpu)",
    )

    # サーバー設定
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="バインドするホスト",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="バインドするポート",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="ファイル変更時に自動リロード",
    )

    return parser.parse_args()


def main():
    """APIサーバーを起動"""
    args = parse_args()

    # 設定を作成
    config = ServerConfig(
        llasa_model_id=args.llasa_model_id,
        xcodec2_model_id=args.xcodec2_model_id,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        device=args.device,
        host=args.host,
        port=args.port,
        reload=args.reload,
    )

    # グローバル設定を設定
    set_config(config)

    # サーバーを起動
    uvicorn.run(
        "llasa_server.api:app",
        host=config.host,
        port=config.port,
        reload=config.reload,
    )


if __name__ == "__main__":
    main()
