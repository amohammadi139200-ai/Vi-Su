import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import gradio as gr
import torch

from faster_whisper import WhisperModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# =========================================================
# Vi-Su AI
# Professional AI Subtitle Studio
# =========================================================

APP_NAME = "Vi-Su AI"

WHISPER_MODEL = "small"

# مدل ترجمه چندزبانه
TRANSLATION_MODEL = "facebook/nllb-200-distilled-600M"

whisper_model = None
translation_tokenizer = None
translation_model = None


# =========================================================
# LANGUAGES
# =========================================================

LANGUAGES = {
    "فارسی": {
        "whisper": "fa",
        "nllb": "pes_Latn",
    },
    "English": {
        "whisper": "en",
        "nllb": "eng_Latn",
    },
    "Español": {
        "whisper": "es",
        "nllb": "spa_Latn",
    },
    "العربية": {
        "whisper": "ar",
        "nllb": "arb_Arab",
    },
    "Türkçe": {
        "whisper": "tr",
        "nllb": "tur_Latn",
    },
    "Русский": {
        "whisper": "ru",
        "nllb": "rus_Cyrl",
    },
    "Français": {
        "whisper": "fr",
        "nllb": "fra_Latn",
    },
    "Deutsch": {
        "whisper": "de",
        "nllb": "deu_Latn",
    },
    "中文": {
        "whisper": "zh",
        "nllb": "zho_Hans",
    },
    "हिन्दी": {
        "whisper": "hi",
        "nllb": "hin_Deva",
    },
}


# =========================================================
# LOAD WHISPER
# =========================================================

def get_whisper_model():

    global whisper_model

    if whisper_model is None:

        print("\nLoading Whisper model...")

        whisper_model = WhisperModel(
            WHISPER_MODEL,
            device="cpu",
            compute_type="int8"
        )

        print("Whisper loaded.")

    return whisper_model


# =========================================================
# LOAD TRANSLATION MODEL
# =========================================================

def get_translation_model():

    global translation_tokenizer
    global translation_model

    if (
        translation_tokenizer is None
        or translation_model is None
    ):

        print("\nLoading NLLB translation model...")

        translation_tokenizer = AutoTokenizer.from_pretrained(
            TRANSLATION_MODEL
        )

        translation_model = AutoModelForSeq2SeqLM.from_pretrained(
            TRANSLATION_MODEL
        )

        translation_model.eval()

        print("NLLB translation model loaded.")

    return (
        translation_tokenizer,
        translation_model
    )


# =========================================================
# FFMPEG
# =========================================================

def find_ffmpeg():

    candidates = [
        shutil.which("ffmpeg"),
        "/usr/bin/ffmpeg",
        "/usr/local/bin/ffmpeg",
    ]

    for path in candidates:

        if path and os.path.exists(path):
            return path

    return None


# =========================================================
# VIDEO PATH
# =========================================================

def get_video_path(video):

    if video is None:
        return None

    if isinstance(video, str):
        return video

    if isinstance(video, dict):

        return (
            video.get("path")
            or video.get("name")
        )

    return None


# =========================================================
# TEXT CLEANING
# =========================================================

def clean_text(text):

    if not text:
        return ""

    text = str(text)

    text = text.replace("\n", " ")

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# =========================================================
# TIME
# =========================================================

def format_srt_time(seconds):

    total_ms = int(
        round(
            float(seconds) * 1000
        )
    )

    hours = total_ms // 3600000

    total_ms %= 3600000

    minutes = total_ms // 60000

    total_ms %= 60000

    seconds_value = total_ms // 1000

    milliseconds = total_ms % 1000

    return (
        f"{hours:02d}:"
        f"{minutes:02d}:"
        f"{seconds_value:02d},"
        f"{milliseconds:03d}"
    )


def format_ass_time(seconds):

    total_cs = int(
        round(
            float(seconds) * 100
        )
    )

    hours = total_cs // 360000

    total_cs %= 360000

    minutes = total_cs // 6000

    total_cs %= 6000

    seconds_value = total_cs // 100

    centiseconds = total_cs % 100

    return (
        f"{hours}:"
        f"{minutes:02d}:"
        f"{seconds_value:02d}."
        f"{centiseconds:02d}"
    )


# =========================================================
# ASS ESCAPE
# =========================================================

def ass_escape(text):

    text = str(text)

    text = text.replace(
        "\\",
        "\\\\"
    )

    text = text.replace(
        "{",
        "\\{"
    )

    text = text.replace(
        "}",
        "\\}"
    )

    return text


# =========================================================
# TRANSLATION
# =========================================================

def translate_texts(
    texts,
    source_language,
    target_language
):

    if not texts:
        return []

    if source_language == target_language:

        return [
            clean_text(text)
            for text in texts
        ]

    tokenizer, model = get_translation_model()

    tokenizer.src_lang = source_language

    translated_results = []

    batch_size = 4

    for start in range(
        0,
        len(texts),
        batch_size
    ):

        batch = texts[
            start:start + batch_size
        ]

        batch = [
            clean_text(text)
            for text in batch
        ]

        encoded = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )

        with torch.no_grad():

            generated_tokens = model.generate(
                **encoded,
                forced_bos_token_id=
                    tokenizer.convert_tokens_to_ids(
                        target_language
                    ),
                max_length=512,
                num_beams=4,
                early_stopping=True
            )

        decoded = tokenizer.batch_decode(
            generated_tokens,
            skip_special_tokens=True
        )

        for result in decoded:

            result = clean_text(
                result
            )

            if not result:

                raise RuntimeError(
                    "مدل ترجمه پاسخ خالی تولید کرد."
                )

            translated_results.append(
                result
            )

    return translated_results


# =========================================================
# SRT
# =========================================================

def build_srt(
    segments,
    output_path
):

    with open(
        output_path,
        "w",
        encoding="utf-8-sig"
    ) as file:

        for index, segment in enumerate(
            segments,
            start=1
        ):

            file.write(
                f"{index}\n"
            )

            file.write(
                f"{format_srt_time(segment['start'])} --> "
                f"{format_srt_time(segment['end'])}\n"
            )

            file.write(
                segment["text"].strip()
            )

            file.write(
                "\n\n"
            )


# =========================================================
# ASS
# =========================================================

def build_ass(
    segments,
    output_path,
    target_name
):

    header = """[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Noto Sans,52,&H00FFFFFF,&H000000FF,&H00111111,&H90000000,0,0,0,0,100,100,0,0,1,3,1,2,70,70,55,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(header)

        for segment in segments:

            start = format_ass_time(
                segment["start"]
            )

            end = format_ass_time(
                segment["end"]
            )

            text = ass_escape(
                segment["text"]
            )

            if target_name == "فارسی":

                text = (
                    "{\\q2}"
                    + text
                )

            file.write(
                f"Dialogue: 0,"
                f"{start},"
                f"{end},"
                f"Default,,"
                f"0,0,55,,"
                f"{text}\n"
            )


# =========================================================
# BURN SUBTITLES
# =========================================================

def burn_subtitles(
    video_path,
    ass_path,
    output_path
):

    ffmpeg = find_ffmpeg()

    if not ffmpeg:

        raise RuntimeError(
            "FFmpeg نصب نیست."
        )

    ass_filename = Path(
        ass_path
    ).name

    command = [
        ffmpeg,

        "-y",

        "-i",
        video_path,

        "-vf",
        f"subtitles={ass_filename}",

        "-c:v",
        "libx264",

        "-preset",
        "veryfast",

        "-crf",
        "20",

        "-c:a",
        "aac",

        "-b:a",
        "192k",

        "-movflags",
        "+faststart",

        output_path
    ]

    result = subprocess.run(
        command,
        cwd=os.path.dirname(
            ass_path
        ),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    if result.returncode != 0:

        raise RuntimeError(
            "خطا در ساخت ویدئوی زیرنویس‌دار:\n\n"
            + result.stderr[-7000:]
        )

    if not os.path.exists(
        output_path
    ):

        raise RuntimeError(
            "فایل خروجی ساخته نشد."
        )

    return output_path


# =========================================================
# MAIN PROCESS
# =========================================================

def process_video(
    video,
    source_name,
    target_name,
    progress=gr.Progress()
):

    video_path = get_video_path(
        video
    )

    if not video_path:

        return (
            None,
            None,
            "❌ لطفاً یک ویدئو انتخاب کن."
        )

    if not os.path.exists(
        video_path
    ):

        return (
            None,
            None,
            "❌ فایل ویدئو پیدا نشد."
        )

    source_config = LANGUAGES.get(
        source_name
    )

    target_config = LANGUAGES.get(
        target_name
    )

    if not source_config or not target_config:

        return (
            None,
            None,
            "❌ زبان انتخاب‌شده معتبر نیست."
        )

    ffmpeg = find_ffmpeg()

    if not ffmpeg:

        return (
            None,
            None,
            "❌ FFmpeg پیدا نشد."
        )

    try:

        # =================================================
        # WORKSPACE
        # =================================================

        workdir = tempfile.mkdtemp(
            prefix="visu_"
        )

        base_name = Path(
            video_path
        ).stem

        srt_path = os.path.join(
            workdir,
            f"{base_name}_ViSu.srt"
        )

        ass_path = os.path.join(
            workdir,
            f"{base_name}_ViSu.ass"
        )

        output_path = os.path.join(
            workdir,
            f"{base_name}_ViSu_Final.mp4"
        )

        # =================================================
        # 1. WHISPER
        # =================================================

        progress(
            0.05,
            desc="🎬 آماده‌سازی ویدئو..."
        )

        progress(
            0.12,
            desc="🎙️ تشخیص گفتار با Whisper..."
        )

        model = get_whisper_model()

        segments_generator, info = model.transcribe(
            video_path,
            language=source_config["whisper"],
            beam_size=5,
            vad_filter=True
        )

        segments = []

        for item in segments_generator:

            text = clean_text(
                item.text
            )

            if text:

                segments.append({
                    "start": float(
                        item.start
                    ),
                    "end": float(
                        item.end
                    ),
                    "text": text
                })

        if not segments:

            raise RuntimeError(
                "هیچ گفتاری در ویدئو پیدا نشد."
            )

        progress(
            0.35,
            desc=(
                f"✅ {len(segments)} "
                "جمله تشخیص داده شد"
            )
        )

        # =================================================
        # 2. TRANSLATION
        # =================================================

        progress(
            0.40,
            desc="🌍 بارگذاری موتور ترجمه AI..."
        )

        original_texts = [
            item["text"]
            for item in segments
        ]

        translated_texts = translate_texts(
            original_texts,
            source_config["nllb"],
            target_config["nllb"]
        )

        if len(translated_texts) != len(
            segments
        ):

            raise RuntimeError(
                "تعداد ترجمه‌ها با جمله‌ها برابر نیست."
            )

        translated_segments = []

        for segment, translation in zip(
            segments,
            translated_texts
        ):

            translated_segments.append({

                "start":
                    segment["start"],

                "end":
                    segment["end"],

                "original":
                    segment["text"],

                "text":
                    translation
            })

        progress(
            0.65,
            desc="✅ ترجمه با مدل AI انجام شد"
        )

        # =================================================
        # 3. SRT
        # =================================================

        progress(
            0.70,
            desc="📄 ساخت فایل SRT..."
        )

        build_srt(
            translated_segments,
            srt_path
        )

        # =================================================
        # 4. ASS
        # =================================================

        progress(
            0.76,
            desc="🎨 طراحی زیرنویس..."
        )

        build_ass(
            translated_segments,
            ass_path,
            target_name
        )

        # =================================================
        # 5. HARD SUBTITLE
        # =================================================

        progress(
            0.82,
            desc="🔥 چسباندن زیرنویس روی ویدئو..."
        )

        burn_subtitles(
            video_path,
            ass_path,
            output_path
        )

        # =================================================
        # DONE
        # =================================================

        progress(
            1.0,
            desc="🎉 پروژه با موفقیت تمام شد!"
        )

        return (

            output_path,

            srt_path,

            (
                "🎉 Vi-Su AI با موفقیت کار را تمام کرد!\n\n"

                f"🎙️ زبان اصلی: {source_name}\n"
                f"🌍 زبان مقصد: {target_name}\n"
                f"📝 تعداد جمله‌ها: {len(segments)}\n\n"

                "✅ گفتار استخراج شد\n"
                "✅ ترجمه AI انجام شد\n"
                "✅ فایل SRT ساخته شد\n"
                "✅ زیرنویس روی ویدئو حک شد"
            )
        )

    except Exception as error:

        print(
            "\n========== Vi-SU ERROR ==========\n"
        )

        print(
            repr(error)
        )

        print(
            "\n=================================\n"
        )

        return (

            None,

            None,

            "❌ پردازش متوقف شد.\n\n"
            + str(error)
        )


# =========================================================
# CLEAR
# =========================================================

def clear_all():

    return (
        None,
        None,
        None,
        "آماده دریافت ویدئو..."
    )


# =========================================================
# PREMIUM DESIGN
# =========================================================

css = """

body {

    background:
        radial-gradient(
            circle at 10% 0%,
            rgba(255,106,0,0.12),
            transparent 30%
        ),
        radial-gradient(
            circle at 90% 10%,
            rgba(255,106,0,0.08),
            transparent 25%
        ),
        #050505 !important;
}


/* Main */

.gradio-container {

    max-width: 1320px !important;

    background:
        transparent !important;

    padding-bottom: 40px !important;
}


/* Hero */

#hero {

    margin-top: 15px;

    padding: 32px;

    border-radius: 24px;

    border:
        1px solid
        rgba(255,106,0,0.22);

    background:
        linear-gradient(
            135deg,
            rgba(255,106,0,0.14),
            rgba(18,18,18,0.96)
        );

    box-shadow:
        0 15px 50px
        rgba(0,0,0,0.45);
}


#hero h1 {

    font-size: 42px !important;

    font-weight: 800 !important;

    margin-bottom: 5px !important;
}


#hero p {

    color:
        #b7b7b7 !important;

    font-size: 16px !important;
}


/* Panels */

.panel {

    border:
        1px solid
        rgba(255,255,255,0.07) !important;

    border-radius: 22px !important;

    background:
        rgba(13,13,13,0.94) !important;

    padding: 20px !important;

    box-shadow:
        0 12px 40px
        rgba(0,0,0,0.35) !important;
}


/* Buttons */

button {

    border-radius:
        13px !important;

    min-height:
        48px !important;

    transition:
        0.2s ease !important;
}


.primary {

    background:
        linear-gradient(
            135deg,
            #ff6a00,
            #ff7d1a
        ) !important;

    border:
        none !important;

    color:
        white !important;

    font-weight:
        800 !important;

    box-shadow:
        0 8px 25px
        rgba(255,106,0,0.22) !important;
}


.primary:hover {

    transform:
        translateY(-1px);

    box-shadow:
        0 12px 30px
        rgba(255,106,0,0.32) !important;
}


/* Inputs */

input,
textarea,
select {

    border-radius:
        12px !important;
}


/* Status */

#status textarea {

    font-size:
        14px !important;

    line-height:
        1.8 !important;
}


/* Footer */

#footer {

    text-align:
        center;

    color:
        #666;

    padding:
        25px;
}

"""


# =========================================================
# UI
# =========================================================

with gr.Blocks(
    title="Vi-Su AI"
) as demo:

    # -----------------------------------------------------
    # HERO
    # -----------------------------------------------------

    gr.Markdown(
        """
        <div id="hero">

            <h1>🟠 Vi-Su AI</h1>

            <p>
                Professional AI Video Subtitle Studio
            </p>

            <p>
                🎙️ Speech Recognition
                &nbsp; • &nbsp;
                🌍 AI Translation
                &nbsp; • &nbsp;
                🎬 Hardcoded Subtitles
            </p>

        </div>
        """,
        elem_id="hero"
    )


    # -----------------------------------------------------
    # MAIN
    # -----------------------------------------------------

    with gr.Row():

        # ===============================================
        # VIDEO STUDIO
        # ===============================================

        with gr.Column(
            scale=1,
            elem_classes="panel"
        ):

            gr.Markdown(
                "## 🎬 Video Studio"
            )

            gr.Markdown(
                "ویدئو را انتخاب کن و زبان مبدأ و مقصد را مشخص کن."
            )

            video_input = gr.Video(
                label="ویدئوی ورودی",
                sources=["upload"]
            )

            with gr.Row():

                source_dropdown = gr.Dropdown(

                    choices=
                        list(
                            LANGUAGES.keys()
                        ),

                    value="Español",

                    label=
                        "🎙️ زبان ویدئو"
                )

                target_dropdown = gr.Dropdown(

                    choices=
                        list(
                            LANGUAGES.keys()
                        ),

                    value="فارسی",

                    label=
                        "🌍 زبان ترجمه"
                )


            process_button = gr.Button(

                "🚀 شروع پردازش",

                variant="primary"
            )


            clear_button = gr.Button(
                "🗑️ پاک کردن"
            )


        # ===============================================
        # OUTPUT
        # ===============================================

        with gr.Column(
            scale=1,
            elem_classes="panel"
        ):

            gr.Markdown(
                "## ✨ خروجی پروژه"
            )

            status_box = gr.Textbox(

                label=
                    "📊 وضعیت پردازش",

                value=
                    "آماده دریافت ویدئو...",

                lines=9,

                elem_id="status"
            )


            video_output = gr.Video(

                label=
                    "🎥 ویدئوی نهایی"
            )


            srt_output = gr.File(

                label=
                    "📄 فایل زیرنویس SRT"
            )


    # -----------------------------------------------------
    # FOOTER
    # -----------------------------------------------------

    gr.Markdown(
        """
        <div id="footer">
            🟠 Vi-Su AI — AI Video Subtitle Studio
        </div>
        """,
        elem_id="footer"
    )


    # -----------------------------------------------------
    # EVENTS
    # -----------------------------------------------------

    process_button.click(

        fn=process_video,

        inputs=[
            video_input,
            source_dropdown,
            target_dropdown
        ],

        outputs=[
            video_output,
            srt_output,
            status_box
        ]
    )


    clear_button.click(

        fn=clear_all,

        inputs=[],

        outputs=[
            video_input,
            video_output,
            srt_output,
            status_box
        ]
    )


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":

    demo.launch(

        server_name="0.0.0.0",

        server_port=7860,

        theme=gr.themes.Base(),

        css=css
    )