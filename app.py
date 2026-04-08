import os
import sys
import sqlite3
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
import concurrent.futures
from datetime import datetime

import gradio as gr

from modules.pipelines.p import (
    ensure_pipeline_ready,
    speech_to_text,
    predict_emotion,
    choose_strategy,
    build_prompt_messages,
    generate_llm_response,
)
from modules.safety import (
    is_crisis,
    is_violence_toward_others,
    CRISIS_REPLY,
    VIOLENCE_RISK_REPLY,
    crisis_and_violence_reply,
)

DISCLAIMER_MD = """
### MindCare — please read

- **This service is not medical or psychological counselling.** It does not diagnose or treat any condition and does not replace in-person care from licensed professionals.
- If you have thoughts of **self-harm, suicide, or harming others or a group**, or you are in immediate danger, contact **local emergency services or a crisis line** right away (e.g. **120 / 110** in mainland China; crisis support e.g. **010-82951332**—verify current numbers). Do not rely on this chat alone. If there is a **concrete plan to hurt others**, this service **cannot** promise confidentiality and may encourage you to contact police or a crisis team.
- How your conversation may be used depends on your deployment **privacy policy**. By default, **nothing is written to a database** unless you set `MINDCARE_LOG_INTERACTIONS=1`.
- On **Hugging Face Spaces**, storage is often **ephemeral**; caches may be cleared after restarts, and the first request may re-download models.
"""

_DB_PATH = os.environ.get("MINDCARE_DB_PATH", "mindcare_logs.db")


def _logging_enabled():
    return os.environ.get("MINDCARE_LOG_INTERACTIONS", "").lower() in ("1", "true", "yes")


def init_db():
    conn = sqlite3.connect(_DB_PATH)
    c = conn.cursor()
    c.execute(
        """CREATE TABLE IF NOT EXISTS interactions
                 (timestamp TEXT, user_text TEXT, emotion TEXT, strategy TEXT, llm_reply TEXT)"""
    )
    conn.commit()
    conn.close()


def log_interaction(user_text, emotion, strategy, reply):
    if not _logging_enabled():
        return
    conn = sqlite3.connect(_DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT INTO interactions VALUES (?, ?, ?, ?, ?)",
        (datetime.now().isoformat(), user_text, emotion, strategy, reply),
    )
    conn.commit()
    conn.close()


def process_interaction(audio_path, text_input, history):
    history = history or []
    user_text = ""
    emotion = "neutral"

    if audio_path:
        parallel = os.environ.get("MINDCARE_PARALLEL_AUDIO", "").lower() in ("1", "true", "yes")
        emo_mode = (os.environ.get("MINDCARE_EMOTION_MODE") or "model").strip().lower()
        
        if parallel and emo_mode != "text":
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                fut_text = executor.submit(speech_to_text, audio_path)
                fut_emo = executor.submit(predict_emotion, audio_path, None)
                user_text = fut_text.result()
                emotion = fut_emo.result()
        else:
            user_text = speech_to_text(audio_path)
            emotion = predict_emotion(audio_path, user_text=user_text)
    elif text_input and text_input.strip():
        user_text = text_input.strip()
        emotion = predict_emotion(None, user_text=user_text)

    if not user_text:
        return history, history, ""

    history_messages = []
    if history:
        for turn in history:
            if isinstance(turn, (list, tuple)) and len(turn) >= 2:
                history_messages.append({"role": "user", "content": turn[0]})
                history_messages.append({"role": "assistant", "content": turn[1]})
            elif isinstance(turn, dict):
                history_messages.append(turn)

    if is_crisis(user_text):
        if is_violence_toward_others(user_text):
            reply = crisis_and_violence_reply()
            log_tag = "crisis_and_violence_risk"
        else:
            reply = CRISIS_REPLY
            log_tag = "crisis_intervention"
        history = history + [(user_text, reply)]
        log_interaction(user_text, emotion, log_tag, reply)
        return history, history, ""

    if is_violence_toward_others(user_text):
        history = history + [(user_text, VIOLENCE_RISK_REPLY)]
        log_interaction(user_text, emotion, "violence_risk_intervention", VIOLENCE_RISK_REPLY)
        return history, history, ""

    strategy = choose_strategy(emotion)
    
    messages = build_prompt_messages(user_text, emotion, strategy, history_messages)
    
    reply = generate_llm_response(messages)

    history = history + [(user_text, reply)]
    log_interaction(user_text, emotion, strategy, reply)

    return history, history, ""


if _logging_enabled():
    init_db()

with gr.Blocks(theme=gr.themes.Soft()) as app:
    gr.Markdown("# MindCare")
    gr.Markdown(DISCLAIMER_MD)

    chatbot = gr.Chatbot(label="Conversation")
    state_history = gr.State([])

    with gr.Row():
        with gr.Column(scale=8):
            text_input = gr.Textbox(
                placeholder="Type your message here, or use audio only…",
                show_label=False,
            )
        with gr.Column(scale=1):
            audio_input = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Audio")

    submit_btn = gr.Button("Send")

    submit_btn.click(
        process_interaction,
        inputs=[audio_input, text_input, state_history],
        outputs=[chatbot, state_history, text_input],
    )


def _is_hf_space() -> bool:
    if os.environ.get("SYSTEM") == "spaces":
        return True
    return bool(os.environ.get("SPACE_REPO_NAME") or os.environ.get("SPACE_ID"))


def _is_colab() -> bool:
    return bool(os.environ.get("COLAB_RELEASE_TAG"))


app.queue(max_size=20)

if __name__ == "__main__":
    if not _is_hf_space() and os.environ.get("MINDCARE_EAGER_LOAD", "").lower() in (
        "1",
        "true",
        "yes",
    ):
        ensure_pipeline_ready()

    user = os.environ.get("GRADIO_AUTH_USERNAME")
    pwd = os.environ.get("GRADIO_AUTH_PASSWORD")
    auth = (user, pwd) if user and pwd else None

    if _is_hf_space():
        # Spaces inject port and reverse proxy; do not force server_name/server_port.
        app.launch(auth=auth)
    elif _is_colab():
        # Colab iframe is often unreliable; default to Gradio share link.
        share = os.environ.get("GRADIO_SHARE", "1").lower() not in ("0", "false", "no")
        app.launch(share=share, auth=auth)
    else:
        app.launch(
            server_name=os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"),
            server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")),
            share=os.environ.get("GRADIO_SHARE", "").lower() in ("1", "true", "yes"),
            auth=auth,
        )
