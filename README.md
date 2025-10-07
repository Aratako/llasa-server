# Llasa TTS 推論サーバー

vLLMを使用したLlasa TTSモデルのシンプルな高速推論サーバーです。

## Llasaとは

**Llasa**は、LLaMAアーキテクチャをベースにしたText-to-Speech (TTS) モデルです。テキストを入力として受け取り、対応する**speech token**を出力します。これらのspeech tokenは、**XCodec2**によって音声波形に変換されます。

### データフロー

```text
テキスト入力
  ↓
[テキスト正規化]
  ↓
[vLLM + Llasa] → Speech token生成
  ↓
[XCodec2] → 音声波形デコード
  ↓
16kHz WAV音声出力
```

## 特徴

- **高速推論**: vLLMによる最適化されたLLM推論
- **音声クローニング**: リファレンス音声による声質制御のサポート
- **RESTful API**: FastAPIベースの使いやすいAPI
- **Web UI**: Gradio製のブラウザインターフェースを利用可能
- **柔軟な設定**: GPU設定、モデルパラメータの細かい調整が可能

## インストール

WSL2のUbuntu 24.04、Python 3.12、CUDA 12.9の環境で動作確認済みです。

```bash
# XCodec2のインストール
uv pip install xcodec2==0.1.5

# vLLMのインストール
uv pip install vllm==0.10.1.1 --torch-backend=auto

# その他の依存関係をインストール
uv pip install -r requirements.txt

# 開発環境が必要な場合
uv pip install -r requirements-dev.txt
```

## 使い方

### 推論サーバーの起動

#### 基本的な起動

```bash
python run_server.py
```

サーバーは `http://localhost:8000` で起動します。

#### コマンドライン引数での設定

```bash
# FP8モデルを使用
python run_server.py --llasa-model-id NandemoGHS/Anime-Llasa-3B-FP8

# カスタムポートで起動
python run_server.py --port 9000

# vLLMの設定を調整 (マルチGPUなど)
python run_server.py --gpu-memory-utilization 0.7 --tensor-parallel-size 2

# すべてのオプションを確認
python run_server.py --help
```

**利用可能な引数:**

- `--llasa-model-id`: Llasaモデル ID（デフォルト: NandemoGHS/Anime-Llasa-3B）
- `--xcodec2-model-id`: XCodec2モデル ID（デフォルト: NandemoGHS/Anime-XCodec2）
- `--tensor-parallel-size`: vLLMのTensor Parallelのサイズ（デフォルト: 1、マルチGPU時に使用）
- `--gpu-memory-utilization`: GPU メモリ使用率（デフォルト: 0.8）
- `--max-model-len`: コンテキスト長（デフォルト: 2048）
- `--device`: 仕様デバイス（デフォルト: cuda）
- `--host`: バインドホスト（デフォルト: 0.0.0.0）
- `--port`: バインドポート（デフォルト: 8000）
- `--reload`: 自動リロード有効化

### Gradio Web UIの起動

上記の推論サーバを使い、ブラウザから簡単に音声生成ができる簡易的なWeb UIも提供しています。

```bash
python gradio_ui.py
```

Web UIは `http://localhost:7860` で起動します。

#### Web UIのコマンドライン引数

```bash
# 推論サーバーのURLを指定
python gradio_ui.py --server-url http://192.168.1.100:8000

# カスタムポートでWeb UIを起動
python gradio_ui.py --port 8080

# 公開リンクを生成（外部アクセス可能）
python gradio_ui.py --share

# 環境変数で推論サーバーのURLを設定
LLASA_SERVER_URL=http://192.168.1.100:8000 python gradio_ui.py
```

**利用可能な引数:**

- `--server-url`: APIサーバーのURL（デフォルト: <http://localhost:8000>、
  環境変数 `LLASA_SERVER_URL` でも設定可能）
- `--host`: Web UIのホスト（デフォルト: 0.0.0.0）
- `--port`: Web UIのポート（デフォルト: 7860）
- `--share`: Gradio公開リンクを生成

### API仕様

#### POST /tts

テキストから音声を生成します。

**パラメータ:**

- `text` (必須): 生成するテキスト
- `reference_audio` (オプション): リファレンス音声ファイル（WAV形式推奨）
- `reference_text` (オプション): リファレンス音声のテキスト
- `temperature` (デフォルト: 0.8): Llasaのtemperature
- `top_p` (デフォルト: 1.0): Llasaのtop-p
- `repetition_penalty` (デフォルト: 1.1): Llasaのrepetition penalty
- `max_tokens` (デフォルト: 2048): 生成する最大トークン数

**レスポンス:** WAV形式の音声データ（wav、16kHz、モノラル）

### 使用例

#### シンプルな音声合成の例（Python）

```python
import requests

url = "http://localhost:8000/tts"
data = {
    "text": "こんにちは、これはテスト音声です。",
}

response = requests.post(url, data=data)

with open("output.wav", "wb") as f:
    f.write(response.content)
```

#### リファレンス音声を使用する例（Python）

```python
import requests

url = "http://localhost:8000/tts"
data = {
    "text": "これは生成したい新しいテキストです。",
    "reference_text": "これはリファレンス音声のテキストです。",
}
files = {
    "reference_audio": open("reference.wav", "rb"),
}

response = requests.post(url, data=data, files=files)

with open("output.wav", "wb") as f:
    f.write(response.content)
```

#### cURLでの使用例

```bash
# シンプルなTTS
curl -X POST "http://localhost:8000/tts" \
  -F "text=こんにちは、これはテスト音声です。" \
  -o output.wav

# リファレンス音声付き
curl -X POST "http://localhost:8000/tts" \
  -F "text=これは生成したい新しいテキストです。" \
  -F "reference_text=これはリファレンス音声のテキストです。" \
  -F "reference_audio=@reference.wav;type=audio/wav" \
  -o output.wav
```

### Web UIでの使用

1. APIサーバーを起動:

   ```bash
   python run_server.py
   ```

2. 別のターミナルでWeb UIを起動:

   ```bash
   python gradio_ui.py
   ```

3. ブラウザで `http://localhost:7860` にアクセス

4. UIで以下を設定:
   - **APIサーバーURL**: 必要に応じて変更（デフォルト: <http://localhost:8000>）
   - **リファレンス音声**: オプションで音声ファイルをアップロード
   - **リファレンステキスト**: リファレンス音声を使用する場合は必須
   - **生成するテキスト**: 生成したいテキストを入力
   - **パラメータ**: Temperature、Top-p、Repetition Penaltyを調整

5. 「音声を生成」ボタンをクリック

### ヘルスチェック

```bash
curl http://localhost:8000/health
```

## アーキテクチャ

### コンポーネント

- **vLLM**: Llasa LLMモデル（NandemoGHS/Anime-Llasa-3B）の推論
- **XCodec2**: Speech tokenから音声波形への変換（NandemoGHS/Anime-XCodec2）
- **FastAPI**: REST APIサーバー
- **Gradio**: Web UIフロントエンド

### ディレクトリ構成

```text
llasa-server/
├── llasa_server/
│   ├── __init__.py
│   ├── api.py           # FastAPI エンドポイント
│   ├── codec.py         # XCodec2 ラッパー
│   ├── config.py        # サーバー設定管理
│   ├── engine.py        # vLLM エンジン
│   ├── server.py        # メインサーバークラス
│   └── utils.py         # ユーティリティ類
├── run_server.py        # APIサーバー起動スクリプト
├── gradio_ui.py         # Gradio Web UI
├── example_client.py    # 推論のサンプルスクリプト
├── requirements.txt     # 必要ライブラリ
├── requirements-dev.txt # 開発用ライブラリ
└── README.md            # このドキュメント
```

## ライセンス

このリポジトリのコードはMITライセンスの下で提供されています。

利用しているモデルのライセンスについては、各モデルの提供元を確認してください。デフォルトで使われているAnime-Llasa-3BおよびAnime-XCodec2はCC BY-NC 4.0ライセンスです。
