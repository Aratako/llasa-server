"""Gradioを使用したLlasa TTS Web UI（メタデータ対応版）"""

import argparse
import io
import os
from typing import Optional

import gradio as gr
import numpy as np
import requests
import soundfile as sf

from llasa_server.config import TextConstants
from llasa_server.utils import build_system_prompt


def generate_speech(
    ref_audio_path: Optional[str],
    ref_text: Optional[str],
    target_text: str,
    caption: str,
    emotion: str,
    profile: str,
    mood: str,
    speed: str,
    prosody: str,
    pitch_timbre: str,
    style: str,
    notes: str,
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
        caption: 音声のキャプション（必須）
        emotion: 感情タグ
        profile: 話者プロファイル
        mood: ムード
        speed: 話速
        prosody: 抑揚
        pitch_timbre: ピッチ・声質
        style: スタイル
        notes: 特記事項
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

    # システムプロンプトを構築（captionがあれば）
    system_prompt = None
    if caption and caption.strip():
        system_prompt = build_system_prompt(
            caption=caption.strip(),
            emotion=emotion.strip() if emotion else "",
            profile=profile.strip() if profile else "",
            mood=mood.strip() if mood else "",
            speed=speed.strip() if speed else "",
            prosody=prosody.strip() if prosody else "",
            pitch_timbre=pitch_timbre.strip() if pitch_timbre else "",
            style=style.strip() if style else "",
            notes=notes.strip() if notes else "",
        )

    try:
        progress(0, "リクエストを準備中...")

        # リクエストデータを準備
        data = {
            "text": target_text,
            "temperature": temperature,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
        }

        # システムプロンプトがある場合
        if system_prompt:
            data["system_prompt"] = system_prompt

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


# プリセット例
EXAMPLES = {
    "ex0": {
        "text": "今思えば、あの時すでに運命の歯車は狂っていたのだろう",
        "caption": "落ち着いた中低音の女性声。シリアスな雰囲気で、張りのある声で断定的に話す。ナレーションのようなスタイル。",
        "emotion": "serious",
        "profile": "落ち着いた女性声",
        "mood": "シリアス、深刻",
        "speed": "一定",
        "prosody": "メリハリがある、断定的",
        "pitch_timbre": "中低音、張りのある声",
        "style": "ナレーション風",
        "notes": "",
    },
    "ex1": {
        "text": "ちょっと、何触ってるの！？痴漢はれっきとした犯罪よ！この変態！",
        "caption": "大人の女性の声。冷静に問い詰め始め、次第に語気を強めていく。最後には怒りを込めて張りのある声で言い放つ。",
        "emotion": "angry",
        "profile": "大人の女性声",
        "mood": "詰問、怒り",
        "speed": "普通",
        "prosody": "メリハリが強く、最後は語気が強まる",
        "pitch_timbre": "中低音、張りのある声",
        "style": "会話調",
        "notes": "事実を突きつけるような強い口調。",
    },
    "ex2": {
        "text": "あ、あのね……ずっと言えなかったけど…。私、ずっとあなたのことが好きでした。つ、付き合ってください！",
        "caption": "恥ずかしそうに話す若い女性の声。泣き出しそうな震え声で、途切れ途切れに想いを伝える。切なさがこもっている。",
        "emotion": "shy",
        "profile": "若い女性声",
        "mood": "恥ずかしさ、切なさ",
        "speed": "遅い",
        "prosody": "途切れがち、感情がこもっている",
        "pitch_timbre": "震え声、高め、息多め",
        "style": "告白",
        "notes": "泣き出しそうな震え声。",
    },
    "ex3": {
        "text": "（腹を押しつぶされて）（うめき声）うっ…！（すすり泣き）やめて…もう殴らないでぇ…",
        "caption": "幼い少女がお腹を殴られ苦しそうにうめく。高くてか細いロリ声で、泣き出しそうな震え声。",
        "emotion": "sad",
        "profile": "ロリ声",
        "mood": "怯え、悲しみ",
        "speed": "",
        "prosody": "震え声",
        "pitch_timbre": "高め、か細い",
        "style": "嗚咽、うめき声",
        "notes": "泣き出しそうなか細いうめき声。",
    },
    "ex4": {
        "text": "（すすり泣き）…っ…うぅ…ぁぁ…",
        "caption": "若い女性が言葉にならず、感情を押し殺してすすり泣いている。悲しみや悔しさがこもった嗚咽。",
        "emotion": "sad",
        "profile": "若い女性声",
        "mood": "悲しみ、悔しさ",
        "speed": "",
        "prosody": "",
        "pitch_timbre": "鼻にかかった声",
        "style": "嗚咽、すすり泣き",
        "notes": "言葉にならず、感情を押し殺してすすり泣いている様子。",
    },
    "ex5": {
        "text": "そういえばあの時、庭で彼女を見かけたような…。",
        "caption": "落ち着いた低めの若い男性声。思い出すように、淡々とした口調で話す。一定のペースと抑揚。",
        "emotion": "worried",
        "profile": "若い男性声",
        "mood": "落ち着き",
        "speed": "",
        "prosody": "落ち着いている",
        "pitch_timbre": "やや低め",
        "style": "会話調、独り言",
        "notes": "思い出すように話している。",
    },
    "ex6": {
        "text": "（耳舐め）はむっ、れろっ…ちゅっ…（含み笑い）ふふっ…（耳元で）どう？（囁き）お耳、気持ちいい？",
        "caption": "若い女性のセクシーな耳舐めと囁き声。吐息を多く含んだ低めのトーンで、ゆっくりと誘うように問いかける。耳元で話しているような距離感。",
        "emotion": "seductive",
        "profile": "成熟した女性声",
        "mood": "誘惑的、セクシー",
        "speed": "とても遅い",
        "prosody": "語尾が上がる",
        "pitch_timbre": "低め、息多め、囁き",
        "style": "耳舐め、囁き",
        "notes": "耳舐めをしながら、非常に近い距離感で話す。",
    },
    "ex7": {
        "text": "（喘ぎ）ん、はぁんっ…あっ、んんぅ！",
        "caption": "若い女性の喘ぎ声。セリフはなく、苦悶と快感が混じったうめき声が続く。高めの声で、絶頂に至るような様子。",
        "emotion": "ecstatic",
        "profile": "若い女性声",
        "mood": "快楽、絶頂",
        "speed": "",
        "prosody": "",
        "pitch_timbre": "高めの喘ぎ声",
        "style": "喘ぎ、うめき声",
        "notes": "セリフはなく、苦悶と快感が混じった喘ぎ声のみ。",
    },
    "ex8": {
        "text": "（喘ぎ）はあんっ、あっ、あふっ、ふああっ、ふあっ、あっ、ああんっ！そんなに激しく動いたら…あんっ、イク、イっちゃう！",
        "caption": "若い女性のリズミカルな喘ぎ声。高い声で息を漏らしながら、連続して喘いでいる。快感が続いている様子。",
        "emotion": "ecstatic",
        "profile": "若い女性声",
        "mood": "快楽、絶頂",
        "speed": "",
        "prosody": "リズミカルな喘ぎ",
        "pitch_timbre": "高め、息多め",
        "style": "喘ぎ",
        "notes": "連続した喘ぎ声。ピストン運動を想起させるようなリズミカルな息遣い。",
    },
    "ex9": {
        "text": "（囁き）ほぉら、もっといっぱい突いてぇ…（喘ぎ）んっ、あんっ…あっ、んん…。良いわよ、君のおちんちん、気持ちいい…（喘ぎ）あんっ",
        "caption": "お姉さんのような低めの囁きと喘ぎ声。大人びた声で、余裕ありげに快楽の声を上げる。",
        "emotion": "aroused",
        "profile": "お姉さん的な女性声",
        "mood": "快楽",
        "speed": "ゆっくり",
        "prosody": "吐息混じり",
        "pitch_timbre": "低め",
        "style": "喘ぎ",
        "notes": "甘い囁きと喘ぎ声。",
    },
    "ex10": {
        "text": "（フェラ音）あむっ、ちゅっ、（チュパ音）ちゅぱっ、ちゅぷっ……。（含み笑い）ふふっ、すごくビクビクしてる",
        "caption": "若い女性の声。愛情を込めたフェラチオを思わせるウェットなチュパ音が続く。官能的な雰囲気。",
        "emotion": "aroused",
        "profile": "若い女性声",
        "mood": "官能的、愛情",
        "speed": "とても遅い",
        "prosody": "語尾が上がる",
        "pitch_timbre": "息多め、ウェットな音質",
        "style": "キス音、チュパ音",
        "notes": "キスやフェラチオを想起させるチュパ音が続く。ウェットなリップノイズが特徴的。",
    },
}


def apply_example(example_key: str):
    """プリセット例を適用"""
    ex = EXAMPLES.get(example_key, EXAMPLES["ex0"])
    return (
        ex.get("text", ""),
        ex.get("caption", ""),
        ex.get("emotion", ""),
        ex.get("profile", ""),
        ex.get("mood", ""),
        ex.get("speed", ""),
        ex.get("prosody", ""),
        ex.get("pitch_timbre", ""),
        ex.get("style", ""),
        ex.get("notes", ""),
    )


def create_ui(default_server_url: str = "http://localhost:8000"):
    """Gradio UIを作成

    Args:
        default_server_url: デフォルトのAPIサーバーURL
    """

    with gr.Blocks(title="Anime Llasa 3B TTS (Captions)") as app:
        gr.Markdown(
            """
# Anime Llasa 3B TTS (メタデータ対応版)

このWeb UIは、Llasa TTSモデル（キャプション対応版）を使用して音声を生成します。

新しいモデルでは、システムプロンプトに音声のキャプションやメタデータを入力することで、
それに従った音声が生成されます。

問題が発生した場合は、リファレンス音声をWAVまたはMP3に変換し、15秒以内にクリップし、
プロンプトを短くしてみてください。
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
                        lines=8,
                    )

                with gr.Column():
                    gr.Markdown("### メタデータ（caption以外はオプション）")
                    caption_input = gr.Textbox(
                        label="caption（推奨）",
                        placeholder="音声を説明する短いキャプション（例：落ち着いた女性声、シリアスな雰囲気）",
                        lines=2,
                    )
                    emotion_input = gr.Textbox(
                        label="emotion（オプション）",
                        placeholder="感情タグ（例：happy, sad, angry, seriousなど）",
                    )
                    profile_input = gr.Textbox(
                        label="profile（オプション）",
                        placeholder="話者プロファイル（例：お姉さん的な女性声、若い男性声など）",
                    )
                    mood_input = gr.Textbox(
                        label="mood（オプション）",
                        placeholder="ムード（例：恥ずかしさ、悲しみ、愛情的など）",
                    )
                    speed_input = gr.Textbox(
                        label="speed（オプション）",
                        placeholder="話速（例：ゆっくり、速い、一定など）",
                    )
                    prosody_input = gr.Textbox(
                        label="prosody（オプション）",
                        placeholder="抑揚・リズム（例：震え声、平坦、語尾が上がるなど）",
                    )
                    pitch_timbre_input = gr.Textbox(
                        label="pitch_timbre（オプション）",
                        placeholder="ピッチ・声質（例：高め、中低音、息多め、囁きなど）",
                    )
                    style_input = gr.Textbox(
                        label="style（オプション）",
                        placeholder="スタイル（例：ナレーション風、会話調、囁き、喘ぎなど）",
                    )
                    notes_input = gr.Textbox(
                        label="notes（オプション）",
                        placeholder="特記事項（例：距離感、吐息などの追加事項）",
                        lines=2,
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

            audio_output = gr.Audio(label="生成された音声")

            # ボタンクリック時の処理
            generate_btn.click(
                generate_speech,
                inputs=[
                    ref_audio_input,
                    ref_text_input,
                    gen_text_input,
                    caption_input,
                    emotion_input,
                    profile_input,
                    mood_input,
                    speed_input,
                    prosody_input,
                    pitch_timbre_input,
                    style_input,
                    notes_input,
                    temperature_slider,
                    top_p_slider,
                    repetition_penalty_slider,
                    server_url_input,
                ],
                outputs=[audio_output],
            )

        with gr.Tab("Examples"):
            gr.Markdown("## プリセット例\n例を選択してTTSタブに適用できます。")

            # ドロップダウンのラベルを作成
            labels = [
                (
                    f"{k}: {EXAMPLES[k]['caption'][:50]}..."
                    if len(EXAMPLES[k]["caption"]) > 50
                    else f"{k}: {EXAMPLES[k]['caption']}"
                )
                for k in EXAMPLES.keys()
            ]
            keys = list(EXAMPLES.keys())
            label_to_key = {labels[i]: keys[i] for i in range(len(keys))}

            example_dropdown = gr.Dropdown(
                choices=labels, value=labels[0], label="例を選択"
            )
            apply_btn = gr.Button("TTSタブに適用", variant="primary")

            # 適用ボタンクリック時
            apply_btn.click(
                lambda label: apply_example(label_to_key[label]),
                inputs=[example_dropdown],
                outputs=[
                    gen_text_input,
                    caption_input,
                    emotion_input,
                    profile_input,
                    mood_input,
                    speed_input,
                    prosody_input,
                    pitch_timbre_input,
                    style_input,
                    notes_input,
                ],
            )

        with gr.Tab("README"):
            gr.Markdown(
                """
# README（推論用メタデータの書き方）

このデモは、systemメタデータ（`emotion / profile / mood / speed / prosody / pitch_timbre / style / notes / caption`）を
systemメッセージとして渡すことで、TTS の読み方・声色を制御します。

---

## 各フィールドの意味と記述例

- **caption（推奨）**
  後段の TTS 制御に使う**短い日本語キャプション**。**セリフ本文は書かない**。
  - 推奨：**1〜2文、全角 30〜80 文字**。
  - 例：「落ち着いた中低音の女性声。シリアスで張りがあり、断定的に語るナレーション調。」

- **emotion**（1つ選択／英語タグ）
  例：`angry, sad, excited, surprised, ecstatic, shy, aroused, serious, relaxed, joyful, ...`
  学習時のリストと整合する英語タグを 1 つ。

- **profile**（話者プロファイル）
  例：「お姉さん的な女性声」「若い男性声」「落ち着いた男性声」「大人の女性声」

- **mood**（感情・ムードの自然文）
  例：「シリアス」「快楽」「恥ずかしさ」「落ち着き」「官能的」

- **speed**（話速／自由記述）
  例：「とても遅い」「やや速い」「一定」「(1.2×)」

- **prosody**（抑揚・リズム）
  例：「メリハリがある」「語尾が上がる」「ため息混じり」「平坦」「震え声」

- **pitch_timbre**（ピッチ／声質）
  例：「高め」「低め」「中低音」「息多め」「張りのある」「囁き」「鼻にかかった声」

- **style**（発話スタイル）
  例：「ナレーション風」「会話調」「朗読調」「囁き」「喘ぎ」「嗚咽」「告白」

- **notes**（特記事項）
  間・ブレス・笑い、効果音の有無、距離感（耳元／遠くから）など。必要なければ空でOK。

---

## emotionの一覧

以下のようなemotionを利用可能です。ただしemotionは自動アノテーションなので、
ものによっては出現していない・学習データ量が極端に少ないものもあると思われます。
その場合の効果は薄いです。

"angry", "sad", "disdainful", "excited", "surprised", "satisfied", "unhappy",
"anxious", "hysterical", "delighted", "scared", "worried", "indifferent",
"upset", "impatient", "nervous", "guilty", "scornful", "frustrated",
"depressed", "panicked", "furious", "empathetic", "embarrassed", "reluctant",
"disgusted", "keen", "moved", "proud", "relaxed", "grateful", "confident",
"interested", "curious", "confused", "joyful", "disapproving", "negative",
"denying", "astonished", "serious", "sarcastic", "conciliative", "comforting",
"sincere", "sneering", "hesitating", "yielding", "painful", "awkward",
"amused", "loving", "dating", "longing", "aroused", "seductive", "ecstatic", "shy"

---

## 特殊タグの挿入（text）

読み上げテキストについて、全角括弧を使った以下のような制御タグを利用できます。
ただしタグは自動アノテーションなので出現していないものもあると思われ、効果はものによると思われます。

### 1. 声の変化（スタイル・感情・意図）
- 感情/トーン：`（優しく）` `（囁き）` `（自信なさげに）` `（からかうように）`
  `（挑発するように）` `（独り言のように）`
- 感情の推移：`（徐々に怒りを込めて）` `（だんだん悲しげに）` `（喜びを爆発させて）`
- 声の状態：`（声が震えて）` `（眠そうに）` `（酔っ払って）` `（声を枯らして）`

### 2. 非言語的な発声
- 感情的な発声：`（うめき声）` `（吐息）` `（息切れ）` `（嗚咽）`
  `（くすくす笑い）` `（小さな悲鳴）`
- 呼吸：`（息をのむ）` `（深い溜息）` `（荒い息遣い）`
- 口の音：`（舌打ち）` `（リップノイズ）` `（唾を飲み込む音）`

### 3. アクション
- 話者の動作：`（笑いながら）` `（泣きながら）` `（咳き込みながら）`
  `（勢いに任せて攻撃）`
- 受ける動作：`（持ち上げられて）` `（首を絞められて）` `（腹を押しつぶされて）`

### 4. 音響・効果音
- 接触音：`（キス音）` `（耳舐め）` `（打撃音）` `（衣擦れの音）`
- NSFW関連音：`（チュパ音）` `（フェラ音）` `（ピストン音）` `（射精音）`
  `（粘着質な水音）`
- 環境音：`（ドアの開閉音）` `（足音）` `（雨音）`
- 音響効果：`（電話越しに）` `（スピーカー越しに）` `（エコー）`

### 5. 発話のリズム・間
- ペース：`（早口で）` `（ゆっくりと強調して）` `（一気にまくしたてて）`
- 間：`（少し間を置いて）` `（一呼吸おいて）` `（沈黙）`

### 6. 距離感・位置関係
- 位置：`（遠くから）` `（耳元で）` `（背後から）` `（ドア越しに）`

> 例：
> `（囁き）ふふ…今日はよく頑張ったね。（キス音）`
> `（徐々に怒りを込めて）もう一度言う。`

---

## 使い方（要点）

1. TTS タブでテキストとメタデータを入力（caption は推奨）。
2. 参照音声があれば読み込み（最大 15 秒、モノラル化・自動リサンプル）。
   参照音声とメタデータの組み合わせは未検証です。
3. 例を使いたいときは Examples タブで選択→TTSタブに適用。
4. 「音声を生成」を押して音声を生成。
"""
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
    parser = argparse.ArgumentParser(description="Llasa TTS Gradio Web UI (Captions版)")
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
